# %%

import os
os.getcwd() 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from test import *
warnings.filterwarnings("ignore")
from datasets import load_dataset,load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch_optimizer as optim
from transformers.optimization import Adafactor, AdafactorSchedule
import torch.backends.cudnn as cudnn
from utils import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from torch.autograd import Variable
import logging
import sys
import transformers
from basic_model import *
import time
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import string
from cycle import *
from train import *
from parameter import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
if(True):
    parser = argparse.ArgumentParser("main")

    parser.add_argument('--valid_num_points', type=int,             default = 100, help='validation data number')
    parser.add_argument('--train_num_points', type=int,             default = 500, help='train data number')

    parser.add_argument('--batch_size', type=int,                   default=4,     help='Batch size')
    parser.add_argument('--max_length', type=int,                   default=128,     help='max_length')

    parser.add_argument('--gpu', type=int,                          default=0,      help='gpu device id')
    parser.add_argument('--G_AB_model_name', type=str,              default='t5-small',      help='model_name')
    parser.add_argument('--G_BA_model_name', type=str,              default='Onlydrinkwater/T5-small-de-en',      help='model_name')
    parser.add_argument('--D_A_model_name', type=str,               default='t5-small',      help='model_name')
    parser.add_argument('--D_B_model_name', type=str,               default='Onlydrinkwater/T5-small-de-en',      help='model_name')
    parser.add_argument('--exp_name', type=str,                     default='CYCLE!',      help='experiment name')
    parser.add_argument('--rep_iter', type=int,                     default=100,      help='report times for 1 epoch')
    parser.add_argument('--test_iter', type=int,                    default=500,      help='report times for 1 epoch')

    parser.add_argument('--epochs', type=int,                       default=50,     help='num of training epochs')

    parser.add_argument('--G_lr', type=float,                       default=5e-6,   help='learning rate for G')
    parser.add_argument('--G_weight_decay', type=float,             default=1e-3,   help='learning de for G')
    parser.add_argument('--G_gamma', type=float,                    default=0.9,    help='lr*gamma after each test')
    parser.add_argument('--G_grad_clip', type=float,                default=1,   help='grad_clip')
    parser.add_argument('--D_lr', type=float,                       default=5e-5,   help='learning rate for D')
    parser.add_argument('--D_weight_decay', type=float,             default=1e-3,   help='learning de for D')
    parser.add_argument('--D_gamma', type=float,                    default=1,    help='lr*gamma after each test')
    parser.add_argument('--D_grad_clip', type=float,                default=1e-2,   help='grad_clip')
    parser.add_argument('--lambda_identity', type=float,            default=0.5,   help='')
    parser.add_argument('--lambda_A', type=float,                   default=100,   help='')
    parser.add_argument('--lambda_B', type=float,                   default=100,   help='')
    parser.add_argument('--lambda_once', type=float,                default=1,   help='')
    parser.add_argument('--smoothing', type=float,                  default=0.5,    help='labelsmoothing')

    parser.add_argument('--load_D', type=int,                       default=0,      help='load pretrained D')
    parser.add_argument('--load_G', type=int,                       default=0,      help='load pretrained D')
    parser.add_argument('--num_workers', type=int,                  default=0,      help='num_workers')
    parser.add_argument('--valid_begin', type=int,                  default=1,      help='whether valid before train')
    parser.add_argument('--train_G', type=int,                      default=1,      help='whether valid before train')
    parser.add_argument('--train_D', type=int,                      default=1,      help='whether valid before train')
    parser.add_argument('--D_pretrain_iter', type=int,              default=500,      help='whether valid before train')


    args = parser.parse_args()#(args=['--batch_size', '8',  '--no_cuda'])#used in ipynb
    args.test_iter = args.test_iter//args.batch_size * args.batch_size
    args.rep_iter = args.rep_iter//args.batch_size * args.batch_size
    print('args.test_iter',args.test_iter)
    print('args.rep_iter',args.rep_iter)#1

# %%
import wandb
os.environ['WANDB_API_KEY']='a166474b1b7ad33a0549adaaec19a2f6d3f91d87'
os.environ['WANDB_NAME']=args.exp_name
wandb.init(project="CYCLEGAN",config=args)

# %%
#logging file
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 

log_format = '%(asctime)s |\t  %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join("./log/", now+'.txt'),'w',encoding = "UTF-8")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info(args)

# %%
GABmodelname = args.G_AB_model_name
GBAmodelname = args.G_BA_model_name
DAmodelname = args.D_A_model_name
DBmodelname = args.D_B_model_name
GABpretrained  =  AutoModelForSeq2SeqLM.from_pretrained(GABmodelname)
GBApretrained  =  AutoModelForSeq2SeqLM.from_pretrained(GBAmodelname)
DApretrained  =  AutoModelForSeq2SeqLM.from_pretrained(DAmodelname)
DBpretrained  =  AutoModelForSeq2SeqLM.from_pretrained(DBmodelname)
logging.info(f'Gmodelsize:{count_parameters_in_MB(GABpretrained)}MB')
logging.info(f'Dmodelsize:{count_parameters_in_MB(DApretrained)}MB')

tokenizer = AutoTokenizer.from_pretrained(GABmodelname)
# tokenizerBA = AutoTokenizer.from_pretrained(GBAmodelname)#its the same


# %%
dataset = load_dataset('wmt16',language+'-en')
train = dataset['train']['translation'][:args.train_num_points]
valid = dataset['train']['translation'][(args.train_num_points-args.valid_num_points):args.train_num_points]#TODO:


train_data = get_Dataset_chaos(train, tokenizer,max_length=args.max_length)
train_dataloader = DataLoader(train_data, sampler= RandomSampler(train_data), 
                        batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
valid_data = get_Dataset(valid, tokenizer,max_length=args.max_length)
valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), 
                        batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)

# %%
cycleGAN = CycleGAN(args,GABpretrained,GBApretrained,DApretrained,DBpretrained,tokenizer)

# %%
if(args.valid_begin==1):
    my_test(valid_dataloader,cycleGAN,tokenizer,logging,wandb)
total_iter = [0]  
for epoch in range(args.epochs):

    logging.info(f"\n\n  ----------------epoch:{epoch}----------------")
    my_train(train_dataloader,cycleGAN,total_iter,args,logging,valid_dataloader,tokenizer,wandb)



# %%



