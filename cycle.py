
import torch
import torch.nn as nn
from utils import *
from transformers.optimization import Adafactor, AdafactorSchedule

class D(nn.Module):
    def __init__(self,args,name='D') -> None:
        super(D, self).__init__()
        self.encoder = torch.load(args.D_model_name.replace('/','')+'.pt').get_encoder()
        self.classifier = torch.nn.Linear(in_features=512, out_features=2, bias=True)
    def forward(self,x):
        return self.classifier(self.encoder(x).last_hidden_state)


class G(nn.Module):
    def __init__(self,args,name='G') -> None:
        super(G, self).__init__()
        self.G = torch.load(args.G_model_name.replace('/','')+'.pt')
    def forward(self,x):
        return self.G(x)


class CycleGAN():
    #G_AB       ->       gumbel softmax       ->       D_A      ->       G_BA     ->      gumbel softmax      ->      D_B
    def __init__(self,args) -> None:
        self.G_AB = G(args=args,name="G_AB")
        self.G_BA = G(args=args,name="G_BA")
        self.D_A = D(args=args,name="D_A")
        self.D_B = D(args=args,name="D_B")
        self.fake_A_pool = Pool()  
        self.fake_B_pool = Pool()  
        self.criterionGAN = torch.nn.MSELoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.optimizer_G_AB = Adafactor(self.G_AB.parameters(), lr = args.G_lr ,scale_parameter=True, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
        self.optimizer_G_BA = Adafactor(self.G_BA.parameters(), lr = args.G_lr ,scale_parameter=True, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
        self.optimizer_D_A = Adafactor(self.D_A.parameters(), lr = args.D_lr ,scale_parameter=True, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
        self.optimizer_D_B = Adafactor(self.D_B.parameters(), lr = args.D_lr ,scale_parameter=True, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
        
    def forward(self):#TODO: prefix + gumblesoftmax
        self.fake_B = self.G_AB(self.real_A)  # G_A(A)
        self.rec_A = self.G_BA(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.G_BA(self.real_B)  # G_B(B)
        self.rec_B = self.G_AB(self.fake_A)   # G_A(G_B(B))


    def set_input(self,A,B):
        self.real_A = A
        self.real_B = B
    def optimize_parameters(self):
        self.forward()    
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G_AB.zero_grad()  # set G_A and G_B's gradients to zero
        self.optimizer_G_BA.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights


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
        self.loss_G_A = self.criterionGAN(self.D_A(self.fake_B), torch.ones((self.fake_B.shape[0],1)))
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.D_B(self.fake_A), torch.ones((self.fake_A.shape[0],1)))
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A# + self.loss_idt_B
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
        loss_D_real = self.criterionGAN(pred_real, torch.ones((self.pred_real.shape[0],1)))
        # Fake
        pred_fake = D(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros((self.pred_fake.shape[0],1)))
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.D_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.D_B, self.real_A, fake_A)

        
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad