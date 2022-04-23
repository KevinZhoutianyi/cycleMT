
import torch
import torch.nn as nn
from utils import *
from transformers.optimization import Adafactor, AdafactorSchedule
from basic_model import *

class CycleGAN():
    #G_AB       ->       gumbel softmax       ->       D_A      ->       G_BA     ->      gumbel softmax      ->      D_B
    def __init__(self,args,GAB,GBA,DA,DB,tokenizer) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G_AB = G(args=args,pretrained=GAB,name="G_AB",tokenizer=tokenizer,prefix='translate English to Portuguese: ').to(self.device)
        self.G_BA = G(args=args,pretrained=GBA,name="G_BA",tokenizer=tokenizer,prefix='translate Portuguese to English: ').to(self.device)
        self.D_A = D(args=args,pretrained=DA,name="D_A").to(self.device)
        self.D_B = D(args=args,pretrained=DB,name="D_B").to(self.device)
        self.tokenizer = tokenizer
        self.args = args
        self.bs = args.batch_size
        self.GB_cycle_meter = AvgrageMeter()
        self.GA_cycle_meter = AvgrageMeter()
        self.GAB_once_meter = AvgrageMeter()
        self.GBA_once_meter = AvgrageMeter()
        self.DA_meter = AvgrageMeter()
        self.DB_meter = AvgrageMeter()
        self.fake_A_pool = Pool()  
        self.fake_B_pool = Pool()  
        self.criterionGAN = torch.nn.MSELoss()
        self.criterionCycle = torch.nn.CrossEntropyLoss( )
        self.criterionIdt = torch.nn.CrossEntropyLoss( )
        self.optimizer_G_AB = Adafactor(self.G_AB.parameters(), lr = args.G_lr ,scale_parameter=True, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
        self.optimizer_G_BA = Adafactor(self.G_BA.parameters(), lr = args.G_lr ,scale_parameter=True, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
        self.optimizer_D_A = Adafactor(self.D_A.parameters(), lr = args.D_lr ,scale_parameter=True, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
        self.optimizer_D_B = Adafactor(self.D_B.parameters(), lr = args.D_lr ,scale_parameter=True, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
    def forward(self):#TODO: prefix + gumblesoftmax
        self.fake_B,self.fake_B_attn = self.G_AB.gumble_generate(self.real_A,self.real_A_attn)  # G_A(A)
        self.rec_A,self.rec_A_attn = self.G_BA.gumble_generate(self.fake_B,self.fake_B_attn)   # G_B(G_A(A))
        self.fake_A,self.fake_A_attn = self.G_BA.gumble_generate(self.real_B,self.real_B_attn)  # G_B(B)
        self.rec_B,self.rec_B_attn = self.G_AB.gumble_generate(self.fake_A,self.fake_A_attn)   # G_A(G_B(B))
    def set_input(self,A,A_attn,B,B_attn):
        self.real_A,self.real_A_attn = A,A_attn
        self.real_B,self.real_B_attn = B,B_attn
    def optimize_parameters(self):
        self.forward()    
        self.set_requires_grad([self.D_A, self.D_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G_AB.zero_grad()  # set G_A and G_B's gradients to zero
        self.optimizer_G_BA.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G_AB.step()       # update G_A and G_B's weights
        self.optimizer_G_BA.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.D_A, self.D_B], True)
        self.optimizer_D_A.zero_grad()   # set D_A and D_B's gradients to zero
        self.optimizer_D_B.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D_A.step()  # update D_A and D_B's weights
        self.optimizer_D_B.step()  # update D_A and D_B's weights

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

        self.loss_G_A = self.criterionGAN(self.D_A(self.fake_B), torch.ones((self.fake_B.shape[0],1),device=self.device))
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.D_B(self.fake_A), torch.ones((self.fake_A.shape[0],1),device=self.device))


        # Forward cycle loss || G_B(G_A(A)) - A||
        self.tail = torch.zeros(self.real_A.shape[1]-self.rec_A.shape[1],device=self.device,requires_grad=False).long()
        self.tail = self.tail.repeat(self.real_A.shape[0],1)
        self.one_hot = torch.zeros((self.tail.shape[0], self.tail.shape[1],self.rec_A.shape[-1]),device=self.device,requires_grad=False)
        self.tail = self.one_hot.scatter_(-1, self.tail.unsqueeze(-1), 1.).float()
        self.temp = torch.hstack((self.rec_A,self.tail))
        self.loss_cycle_A = self.criterionCycle(self.temp.reshape(-1,self.temp.shape[-1]), self.real_A.reshape(-1)).mean() * lambda_A


        # Backward cycle loss || G_A(G_B(B)) - B||
        self.tail = torch.zeros(self.real_B.shape[1]-self.rec_B.shape[1],device=self.device,requires_grad=False).long()
        self.tail =self.tail.repeat(self.real_B.shape[0],1)
        self.one_hot = torch.zeros((self.tail.shape[0], self.tail.shape[1],self.rec_B.shape[-1]),device=self.device,requires_grad=False)
        self.tail = self.one_hot.scatter_(-1, self.tail.unsqueeze(-1), 1.).float()
        self.temp = torch.hstack((self.rec_B,self.tail))
        self.loss_cycle_B = self.criterionCycle(self.temp.reshape(-1,self.temp.shape[-1]), self.real_B.reshape(-1)).mean() * lambda_B


        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B# + self.loss_idt_A + self.loss_idt_B
        self.GA_cycle_meter.update(self.loss_cycle_A.item(),self.bs)
        self.GB_cycle_meter.update(self.loss_cycle_B.item(),self.bs)
        self.GAB_once_meter.update(self.loss_G_A.item(),self.bs)
        self.GBA_once_meter.update(self.loss_G_B.item(),self.bs)
        self.loss_G.backward()
        


    def backward_D_basic(self, D, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = D(real)
        loss_D_real = self.criterionGAN(pred_real, torch.ones((pred_real.shape[0],1),device=self.device,requires_grad=False))
        # Fake
        pred_fake = D(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros((pred_fake.shape[0],1),device=self.device,requires_grad=False))
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.D_A, self.real_B, fake_B)
        self.DA_meter.update(self.loss_D_A.item(),self.bs)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)#TODO:make leak here!
        self.loss_D_B = self.backward_D_basic(self.D_B, self.real_A, fake_A)

        self.DB_meter.update(self.loss_D_B.item(),self.bs)

        
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                net.set_require_grad(requires_grad)
    def getLoss(self):
        ret = {}
        ret['GB_cycle_meter'] = self.GB_cycle_meter.avg
        ret['GA_cycle_meter'] = self.GA_cycle_meter.avg 
        ret['GAB_once_meter'] = self.GAB_once_meter.avg
        ret['GBA_once_meter'] = self.GBA_once_meter.avg 
        ret['DA_meter'] = self.DA_meter.avg 
        ret['DB_meter'] = self.DB_meter.avg 
        self.GB_cycle_meter.reset()
        self.GA_cycle_meter.reset() 
        self.GAB_once_meter.reset()
        self.GBA_once_meter.reset() 
        self.DA_meter.reset()
        self.DB_meter.reset() 
        return ret
