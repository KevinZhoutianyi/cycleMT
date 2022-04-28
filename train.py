import torch
from utils import *
from cycle import *
from basic_model import *
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from test import *
def my_train(loader,model,total_iter,args,logging,valid_loader,tokenizer,wandb):
    logging.info(f"total iter:{total_iter}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.trainmode()
    for step,batch in enumerate(loader):
        total_iter[0] += args.batch_size
        a = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)
        a_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
        b = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)    
        b_attn = Variable(batch[3], requires_grad=False).to(device, non_blocking=False)
        model.set_input(a,a_attn,b,b_attn)
        if(total_iter[0]<args.D_pretrain_iter):
            model.optimize_parameters(trainD=True,trainG=False)
        else:
            model.optimize_parameters(trainD=True,trainG=True)


        if(total_iter[0]%args.rep_iter == 0):
            if(total_iter[0]>args.D_pretrain_iter):
                loss_dict = model.getLoss(withG=False)
            else:
                loss_dict = model.getLoss(withG=True)
            logging.info(loss_dict)
            logging.info(f"{step/len(loader)*100}%")
            wandb.log(loss_dict)


        if(total_iter[0]%args.test_iter == 0 and total_iter[0]>args.D_pretrain_iter):
            torch.save(model.D_A,'./checkpoint/D_A.pt')
            torch.save(model.D_B,'./checkpoint/D_B.pt')
            my_test(valid_loader,model,tokenizer,logging,wandb)
