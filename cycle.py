

from numpy import interp
import torch
import torch.nn as nn
from utils import *
from transformers.optimization import Adafactor, AdafactorSchedule
from basic_model import *
from parameter import *
 #
class CycleGAN():
    #G_AB       ->       gumbel softmax       ->       D_A      ->       G_BA     ->      gumbel softmax      ->      D_B
    def __init__(self,args,GAB,GBA,DA,DB,tokenizer) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if(args.load_G == 1):
            print("G_B and G_A are loaded")
            self.G_AB = None
            self.G_AB = torch.load('./model/G_AB.pt').to(self.device)
            self.G_BA = None
            self.G_BA = torch.load('./model/G_BA.pt').to(self.device)
        else:
            self.G_AB = G(args=args,pretrained=GAB,name="G_AB",tokenizer=tokenizer,prefix='translate English to '+full_language+': ').to(self.device)
            self.G_BA = G(args=args,pretrained=GBA,name="G_BA",tokenizer=tokenizer,prefix='translate '+full_language+' to English: ').to(self.device)
        if(args.load_D == 1):
            print("D_A and D_B are loaded")
            self.D_A = None
            self.D_A = torch.load('./model/D_A.pt').to(self.device)
            self.D_B = None
            self.D_B = torch.load('./model/D_B.pt').to(self.device)
        else:
            self.D_A = D(args=args,pretrained=DA,name="D_A",tokenizer=tokenizer,prefix='classification: ').to(self.device)
            self.D_B = D(args=args,pretrained=DB,name="D_B",tokenizer=tokenizer,prefix='classification: ').to(self.device)
        self.tokenizer = tokenizer
        self.args = args
        self.bs = args.batch_size
        self.num_beam = args.num_beam
        self.GB_cycle_meter = AvgrageMeter()
        self.GA_cycle_meter = AvgrageMeter()
        self.GAB_once_meter = AvgrageMeter()
        self.GBA_once_meter = AvgrageMeter()
        self.DA_meter = AvgrageMeter()
        self.DB_meter = AvgrageMeter()
        self.DA_GP_meter = AvgrageMeter()
        self.DB_GP_meter = AvgrageMeter()
        self.DA_meter2 = AvgrageMeter()
        self.DB_meter2 = AvgrageMeter()
        self.DA_GP_meter2 = AvgrageMeter()
        self.DB_GP_meter2 = AvgrageMeter()
        self.fake_A_pool = Pool(args.poolsize)  
        self.fake_B_pool = Pool(args.poolsize)  
        self.criterionGAN = torch.nn.MSELoss()
        self.criterionCycle = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.criterionIdt = torch.nn.CrossEntropyLoss(ignore_index=0)
        # self.optimizer_G_AB = Adafactor(self.G_AB.parameters(), lr = args.G_lr ,scale_parameter=False, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
        # self.optimizer_G_BA = Adafactor(self.G_BA.parameters(), lr = args.G_lr ,scale_parameter=False, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
        # self.optimizer_D_A = Adafactor(self.D_A.parameters(), lr = args.D_lr ,scale_parameter=False, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
        # self.optimizer_D_B = Adafactor(self.D_B.parameters(), lr = args.D_lr ,scale_parameter=False, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
        # self.optimizer_G_AB = torch.optim.RMSprop(self.G_AB.parameters(),  lr= args.G_lr , weight_decay=args.G_weight_decay)
        # self.optimizer_G_BA = torch.optim.RMSprop(self.G_BA.parameters(),  lr= args.G_lr , weight_decay=args.G_weight_decay)
        # self.optimizer_D_A = torch.optim.RMSprop(self.D_A.parameters(),  lr= args.D_lr , weight_decay=args.D_weight_decay)
        # self.optimizer_D_B = torch.optim.RMSprop(self.D_B.parameters(),  lr= args.D_lr, weight_decay=args.D_weight_decay)
        self.optimizer_G_AB = torch.optim.Adam(self.G_AB.parameters(),  lr= args.G_lr ,  betas=(0, 0.9)  )
        self.optimizer_G_BA = torch.optim.Adam(self.G_BA.parameters(),  lr= args.G_lr ,  betas=(0, 0.9)  )
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(),  lr= args.D_lr,  betas=(0, 0.9)  )
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(),  lr= args.D_lr,  betas=(0, 0.9)  )
        self.scheduler_G_AB =torch.optim.lr_scheduler.StepLR(self.optimizer_G_AB, 1, gamma=args.G_gamma)
        self.scheduler_G_BA = torch.optim.lr_scheduler.StepLR(self.optimizer_G_BA, 1, gamma=args.G_gamma)
        self.scheduler_D_A =torch.optim.lr_scheduler.StepLR(self.optimizer_D_A, 1, gamma=args.D_gamma)
        self.scheduler_D_B = torch.optim.lr_scheduler.StepLR(self.optimizer_D_B, 1, gamma=args.D_gamma)
    
    def forward(self):#TODO: prefix + gumblesoftmax
        self.fake_B,self.fake_B_attn = self.G_AB.gumbel_generate(self.real_A,self.real_A_attn)  # G_A(A) (batchsize*numbeam,sentencelength,vocabsize)
        self.rec_A,self.rec_A_attn = self.G_BA.gumbel_generate_soft(self.fake_B,self.fake_B_attn)   # G_B(G_A(A)) (batchsize*numbeam,sentencelength,vocabsize)
        self.fake_A,self.fake_A_attn = self.G_BA.gumbel_generate(self.real_B,self.real_B_attn)  # G_B(B)
        self.rec_B,self.rec_B_attn = self.G_AB.gumbel_generate_soft(self.fake_A,self.fake_A_attn)   # G_A(G_B(B))
    def set_input(self,A,A_attn,B,B_attn):
        self.real_A,self.real_A_attn = A,A_attn
        self.real_B,self.real_B_attn = B,B_attn
    
    def optimize_parameters(self,trainD=True,trainG=True):
        if(trainD and trainG):
            self.forward()
            self.set_requires_grad([self.D_A, self.D_B], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G_AB.zero_grad()  # set G_A and G_B's gradients to zero
            self.optimizer_G_BA.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G()             # calculate gradients for G_A and G_B
            self.optimizer_G_AB.step()       # update G_A and G_B's weights
            self.optimizer_G_BA.step()       # update G_A and G_B's weights
            self.set_requires_grad([self.D_A, self.D_B], True)
            self.optimizer_D_A.zero_grad()   # set D_A and D_B's gradients to zero
            self.optimizer_D_B.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()      # calculate graidents for D_B
            self.optimizer_D_A.step()  # update D_A and D_B's weights
            self.optimizer_D_B.step()  # update D_A and D_B's weights
        elif(trainG):
            self.forward()
            self.set_requires_grad([self.D_A, self.D_B], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G_AB.zero_grad()  # set G_A and G_B's gradients to zero
            self.optimizer_G_BA.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G()             # calculate gradients for G_A and G_B
            self.optimizer_G_AB.step()       # update G_A and G_B's weights
            self.optimizer_G_BA.step()       # update G_A and G_B's weights
            self.set_requires_grad([self.D_A, self.D_B], True)
        elif(trainD):
            self.set_requires_grad([self.G_AB, self.G_BA], False)
            self.forward()
            self.optimizer_D_A.zero_grad()   # set D_A and D_B's gradients to zero
            self.optimizer_D_B.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()      # calculate graidents for D_B
            self.optimizer_D_A.step()  # update D_A and D_B's weights
            self.optimizer_D_B.step()  # update D_A and D_B's weights
            self.set_requires_grad([self.G_AB, self.G_BA], True)
            
        

    def trainmode(self):
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.args.lambda_identity
        lambda_A = self.args.lambda_A
        lambda_B = self.args.lambda_B
        lambda_once = self.args.lambda_once

        '''
        # Identity loss TODO: is it useless for MT?
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.G_AB(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.G_BA(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        '''

        # GAN loss D_A(G_A(A))
        # self.loss_G_A = self.criterionGAN(self.D_A(self.fake_B,self.fake_B_attn), torch.ones((self.fake_B.shape[0],1),device=self.device))*lambda_once
        self.loss_G_A = torch.mean(-self.D_A(self.fake_B,self.fake_B_attn))*lambda_once
        # GAN loss D_B(G_B(B))
        # self.loss_G_B = self.criterionGAN(self.D_B(self.fake_A,self.fake_A_attn), torch.ones((self.fake_A.shape[0],1),device=self.device))*lambda_once
        self.loss_G_B = torch.mean(-self.D_B(self.fake_A,self.fake_A_attn))*lambda_once

        tile_A = tile(self.real_A,0,self.num_beam)
        # Forward cycle loss || G_B(G_A(A)) - A||
        if(tile_A.shape[1]>self.rec_A.shape[1]):#realsize>rec -> add tail to the rec
            self.tail = torch.zeros(tile_A.shape[1]-self.rec_A.shape[1],device=self.device,requires_grad=False).long()
            self.tail = self.tail.repeat(tile_A.shape[0],1)
            self.one_hot = torch.zeros((self.tail.shape[0], self.tail.shape[1],self.rec_A.shape[-1]),device=self.device,requires_grad=False)
            self.tail = self.one_hot.scatter_(-1, self.tail.unsqueeze(-1), 1.).float()
            self.temp = torch.hstack((self.rec_A,self.tail))
            self.loss_cycle_A = self.criterionCycle(self.temp.reshape(-1,self.temp.shape[-1]), tile_A.reshape(-1)).mean() * lambda_A
        else:#realsize<rec -> add tail to the real
            self.tail = torch.zeros(tile_A.shape[0],self.rec_A.shape[1]-tile_A.shape[1],device=self.device,requires_grad=False).long()
            self.temp  = torch.hstack((tile_A,self.tail))
            self.loss_cycle_A = self.criterionCycle(self.rec_A.reshape(-1,self.rec_A.shape[-1]), self.temp.reshape(-1)).mean() *lambda_A


        tile_B = tile(self.real_B,0,self.num_beam)
        # Backward cycle loss || G_A(G_B(B)) - B||
        if(tile_B.shape[1]>self.rec_B.shape[1]):
            self.tail = torch.zeros(tile_B.shape[1]-self.rec_B.shape[1],device=self.device,requires_grad=False).long()
            self.tail =self.tail.repeat(tile_B.shape[0],1)
            self.one_hot = torch.zeros((self.tail.shape[0], self.tail.shape[1],self.rec_B.shape[-1]),device=self.device,requires_grad=False)
            self.tail = self.one_hot.scatter_(-1, self.tail.unsqueeze(-1), 1.).float()
            self.temp = torch.hstack((self.rec_B,self.tail))
            self.loss_cycle_B = self.criterionCycle(self.temp.reshape(-1,self.temp.shape[-1]), tile_B.reshape(-1)).mean() *lambda_B
        else:
            self.tail = torch.zeros(tile_B.shape[0],self.rec_B.shape[1]-tile_B.shape[1],device=self.device,requires_grad=False).long()
            self.temp  = torch.hstack((tile_B,self.tail))
            self.loss_cycle_B = self.criterionCycle(self.rec_B.reshape(-1,self.rec_B.shape[-1]), self.temp.reshape(-1)).mean() *lambda_B
            


        # combined loss and calculate gradients
        # temp = torch.ones(1,requires_grad=False,device=self.device)*200
        self.loss_G =  self.loss_G_A + self.loss_G_B + self.loss_cycle_A  + self.loss_cycle_B# + self.loss_idt_A + self.loss_idt_B#
        self.GA_cycle_meter.update(self.loss_cycle_A.item(),self.bs)
        self.GB_cycle_meter.update(self.loss_cycle_B.item(),self.bs)
        self.GAB_once_meter.update(self.loss_G_A.item(),self.bs)
        self.GBA_once_meter.update(self.loss_G_B.item(),self.bs)
        self.loss_G.backward()
        


    def backward_D_basic(self, D, real,real_attn, fake,fake_attn):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = D(real,real_attn)
        # loss_D_real = self.criterionGAN(pred_real, torch.ones((pred_real.shape[0],1),device=self.device,requires_grad=False))
        # Fake
        pred_fake = D(fake.detach(),fake_attn)
        # loss_D_fake = self.criterionGAN(pred_fake, torch.zeros((pred_fake.shape[0],1),device=self.device,requires_grad=False))
        pred_fake = torch.mean(pred_fake.reshape(pred_real.shape[0],self.num_beam,-1),1)
        loss_D = torch.mean(pred_fake - pred_real)
        # loss_D = (loss_D_real + loss_D_fake) * 0.5
        '''
        fake_ = fake.clone()
        alpha = torch.clip(torch.rand((real.shape[0], 1, 1),device=torch.device('cuda:0')),min=1e-1,max=9e-1)

        onehot = torch.zeros((real.shape[0],real.shape[1],fake.shape[-1]), device=torch.device('cuda:0'))
        onehot_real = onehot.scatter_(-1,real.unsqueeze(-1),1).float()
        

        if(onehot_real.shape[1]>fake.shape[1]):
            onehot_real = alpha.expand(onehot_real.size()) *onehot_real
            fake = (1-alpha.expand(fake.size())) *fake
            onehot_real[:,:fake.shape[1],:] = onehot_real[:,:fake.shape[1],:]+fake
            interpolates=onehot_real.clone()
        else:
            fake = alpha.expand(fake.size()) *fake
            onehot_real = (1-alpha.expand(onehot_real.size())) *onehot_real
            fake[:,:onehot_real.shape[1],:] = fake[:,:onehot_real.shape[1],:]+onehot_real
            interpolates=fake.clone()
        '''
        onehot = torch.zeros((real.shape[0],real.shape[1],fake.shape[-1]), device=torch.device('cuda:0'))
        onehot_real = onehot.scatter_(-1,real.unsqueeze(-1),1).float()

        
        temp = onehot_real.clone()
        temp = torch.autograd.Variable(temp, requires_grad=True)
        output = D(temp,1-(temp[:, :,0]>1e-4).long())
        gradient = torch.autograd.grad(outputs=output, inputs=temp,grad_outputs=torch.ones(output.size()).cuda(),create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty1 = ((gradient.mean(-1).norm(2,1)) ** 2).mean() * self.args.lambda_GP#TODO

        temp = fake.clone()
        temp = torch.autograd.Variable(temp, requires_grad=True)
        output = D(temp,1-(temp[:, :,0]>1e-4).long())
        gradient = torch.autograd.grad(outputs=output, inputs=temp,grad_outputs=torch.ones(output.size()).cuda(),create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty2 = ((gradient.mean(-1).norm(2,1)) ** 2).mean() * self.args.lambda_GP#TODO
        
        gradient_penalty = (gradient_penalty1+gradient_penalty2) /2
        if(gradient_penalty.item()>100):
            torch.save(real,'./checkpoint/real.pt')
            # torch.save(fake_,'./checkpoint/fake_.pt')
            # torch.save(alpha,'./checkpoint/alpha.pt')
            torch.save(onehot_real,'./checkpoint/onehot_real.pt')
            # torch.save(fake,'./checkpoint/fake.pt')
            torch.save(temp,'./checkpoint/interpolates.pt')
            torch.save(output,'./checkpoint/output.pt')
            torch.save(gradient,'./checkpoint/gradient.pt')
            torch.save(gradient_penalty,'./checkpoint/gradient_penalty.pt')
        
        ret = loss_D+gradient_penalty
        ret.backward()
        return loss_D.item(),gradient_penalty.item()
    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B =self.fake_B
        loss_D,loss_GP = self.backward_D_basic(self.D_A, self.real_B, self.real_B_attn, fake_B,1-(fake_B[:, :,0]>0.5).long())
        # onehot = torch.zeros((self.real_A.shape[0],self.real_A.shape[1],fake_B.shape[-1]), device=torch.device('cuda:0'))
        # onehot_real_A = onehot.scatter_(-1,self.real_A.unsqueeze(-1),1).float()
        # loss_D2,loss_GP2 = self.backward_D_basic(self.D_A, self.real_B, self.real_B_attn,onehot_real_A,self.real_A_attn)
        self.DA_meter.update(loss_D,self.bs)
        self.DA_GP_meter.update(loss_GP,self.bs)
        # self.DA_meter2.update(loss_D2,self.bs)
        # self.DA_GP_meter2.update(loss_GP2,self.bs)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        fake_A = self.fake_A
        loss_D,loss_GP  = self.backward_D_basic(self.D_B, self.real_A,self.real_A_attn,  fake_A,1-(fake_A[:, :,0]>0.5).long())
        # onehot = torch.zeros((self.real_B.shape[0],self.real_B.shape[1],fake_A.shape[-1]), device=torch.device('cuda:0'))
        # onehot_real_B = onehot.scatter_(-1,self.real_B.unsqueeze(-1),1).float()
        # loss_D2,loss_GP2 = self.backward_D_basic(self.D_B, self.real_A, self.real_A_attn,onehot_real_B,self.real_B_attn)
        self.DB_meter.update(loss_D,self.bs)
        self.DB_GP_meter.update(loss_GP,self.bs)
        # self.DB_meter2.update(loss_D2,self.bs)
        # self.DB_GP_meter2.update(loss_GP2,self.bs)

        
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                net.set_require_grad(requires_grad)
    def getLoss(self,withG=True):
        ret = {}
        if(withG):
            ret['GB_cycle_meter'] = self.GB_cycle_meter.avg
            ret['GA_cycle_meter'] = self.GA_cycle_meter.avg 
            ret['GAB_once_meter'] = self.GAB_once_meter.avg
            ret['GBA_once_meter'] = self.GBA_once_meter.avg 
            self.GB_cycle_meter.reset()
            self.GA_cycle_meter.reset() 
            self.GAB_once_meter.reset()
            self.GBA_once_meter.reset() 
        ret['DA_meter'] = self.DA_meter.avg 
        ret['DB_meter'] = self.DB_meter.avg 
        ret['DA_GP_meter'] = self.DA_GP_meter.avg 
        ret['DB_GP_meter'] = self.DB_GP_meter.avg 
        # ret['DA_meter2'] = self.DA_meter2.avg 
        # ret['DB_meter2'] = self.DB_meter2.avg 
        # ret['DA_GP_meter2'] = self.DA_GP_meter2.avg 
        # ret['DB_GP_meter2'] = self.DB_GP_meter2.avg 
        self.DA_meter.reset()
        self.DB_meter.reset() 
        self.DA_GP_meter.reset()
        self.DB_GP_meter.reset() 
        # self.DA_meter2.reset()
        # self.DB_meter2.reset() 
        # self.DA_GP_meter2.reset()
        # self.DB_GP_meter2.reset() 
        return ret
