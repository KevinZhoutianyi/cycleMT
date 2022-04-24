import torch
from utils import *
from cycle import *
from basic_model import *
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from test import *
def my_train(loader,model,total_iter,args,logging,valid_loader,tokenizer,wandb):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.trainmode()
    for step,batch in enumerate(loader):
        total_iter += args.batch_size
        a = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)
        a_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
        b = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)    
        b_attn = Variable(batch[3], requires_grad=False).to(device, non_blocking=False)
        model.set_input(a,a_attn,b,b_attn)
        model.forward()
        if(total_iter>args.D_pretrain_iter):
            model.optimize_parameters_G()
        model.optimize_parameters_D()
        if(total_iter%args.rep_iter == 0):
            loss_dict = model.getLoss()
            logging.info(loss_dict)
            wandb.log(loss_dict)
        if(total_iter%args.test_iter == 0 and total_iter>args.D_pretrain_iter):
            my_test(valid_loader,model,tokenizer,logging,wandb)
