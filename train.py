import torch
from utils import *
from cycle import *
from basic_model import *
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
def my_train(loader,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for step,batch in enumerate(loader):
        a = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)
        a_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
        b = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)    
        b_attn = Variable(batch[3], requires_grad=False).to(device, non_blocking=False)
        model.set_input(a,a_attn,b,b_attn)
        model.optimize_parameters()
