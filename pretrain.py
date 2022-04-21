# %%

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset,load_metric
import torch
import logging
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import sys
import time
from transformers.optimization import Adafactor
import os
import gc


# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length= 512
test_step = 1000
report_step = 100
seed = 2
bs =4 
lr = 1e-4
train_num = 1000
valid_num = 200

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
log_format = '%(asctime)s |\t  %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join("./log/", now+'.txt'),'w',encoding = "UTF-8")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# Setting the seeds
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
torch.manual_seed(seed)
cudnn.enabled=True
torch.cuda.manual_seed(seed)

# %%
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")

# input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
# labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids
# loss = model(input_ids=input_ids, labels=labels).loss

# %%
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val*n #TODO:its just for W
        self.cnt += n
        self.avg = self.sum / self.cnt

def tokenize(text_data, tokenizer, max_length, padding = True):
    
    encoding = tokenizer(text_data, return_tensors='pt', padding=padding, truncation = True, max_length = max_length)

    input_ids = encoding['input_ids']
    
    attention_mask = encoding['attention_mask']
    
    return input_ids, attention_mask
def get_Dataset(dataset, tokenizer):
    train_sentence = [x['de'] for x in dataset]
    train_target = [x['en'] for x in dataset]

  
    model1_input_ids, model1_input_attention_mask = tokenize(train_sentence, tokenizer, max_length = max_length)
  
    model1_target_ids, model1_target_attention_mask = tokenize(train_target, tokenizer, max_length = max_length)
 
    train_data = TensorDataset(model1_input_ids, model1_input_attention_mask, model1_target_ids, model1_target_attention_mask)
   
    return train_data

# %%
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
optimizer = Adafactor(model.parameters(), lr = lr ,scale_parameter=False, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))

# %%

dataset = load_dataset('wmt16','de-en')
# dataset = dataset.shuffle(seed=2)
train = dataset['train']['translation'][:train_num]
valid = dataset['train']['translation'][train_num:(train_num+valid_num)]

def preprocess(dat):
    for t in dat:
        t['de'] = "translate German to English: " + t['de']  #needed for T5
preprocess(train)
preprocess(valid)

train_data = get_Dataset(train, tokenizer)
train_dataloader = DataLoader(train_data, sampler= SequentialSampler(train_data), 
                        batch_size=bs, pin_memory=True, num_workers=4)
valid_data = get_Dataset(valid, tokenizer)
valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), 
                        batch_size=bs, pin_memory=True, num_workers=4)

# %%
def my_train(_dataloader,model,optimizer):
    objs = AvgrageMeter()
    for step,batch in enumerate(_dataloader):
        optimizer.zero_grad()
        train_x = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)
        train_x_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
        train_y = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)    
        train_y_attn = Variable(batch[3], requires_grad=False).to(device, non_blocking=False)    
        train_y[train_y == tokenizer.pad_token_id] = -100
        loss = model(input_ids=train_x, attention_mask=train_x_attn, labels=train_y).loss
        loss.backward()
        optimizer.step()
        objs.update(loss.item(), bs)
        if(step%report_step==0 and step!=0):
            logging.info(f'step:{step}\t,avgloss:{objs.avg}')
            objs.reset()

# %%
import copy
@torch.no_grad()
def my_test(_dataloader,model,epoch):
    # logging.info(f"GPU mem before test:{getGPUMem(device)}%")
    acc = 0
    counter = 0
    model.eval()
    metric_sacrebleu =  load_metric('sacrebleu')
    metric_bleu =  load_metric('bleu')

    # for step, batch in enumerate(tqdm(_dataloader,desc ="test for epoch"+str(epoch))):
    for step, batch in enumerate(_dataloader):
        
        test_dataloaderx = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)
        test_dataloaderx_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
        test_dataloadery = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)
        test_dataloadery_attn = Variable(batch[3], requires_grad=False).to(device, non_blocking=False)
        target_ids = copy.deepcopy(test_dataloadery)
        target_ids[target_ids == tokenizer.pad_token_id] = -100
        ls = model(input_ids=test_dataloaderx, attention_mask=test_dataloaderx_attn, labels=target_ids).loss
        acc+= ls.item()
        counter+= 1
        pre = model.generate(test_dataloaderx)
        x_decoded = tokenizer.batch_decode(test_dataloaderx,skip_special_tokens=True)
        pred_decoded = tokenizer.batch_decode(pre,skip_special_tokens=True)
        label_decoded =  tokenizer.batch_decode(test_dataloadery,skip_special_tokens=True)
        
        pred_str = [x  for x in pred_decoded]
        label_str = [[x] for x in label_decoded]
        pred_list = [x.split()  for x in pred_decoded]
        label_list = [[x.split()] for x in label_decoded]
        metric_sacrebleu.add_batch(predictions=pred_str, references=label_str)
        metric_bleu.add_batch(predictions=pred_list, references=label_list)
        if  step%100==0:
            logging.info(f'x_decoded[:2]:{x_decoded[:2]}')
            logging.info(f'pred_decoded[:2]:{pred_decoded[:2]}')
            logging.info(f'label_decoded[:2]:{label_decoded[:2]}')
            
            
    sacrebleu_score = metric_sacrebleu.compute()
    bleu_score = metric_bleu.compute()
    logging.info('sacreBLEU : %f',sacrebleu_score['score'])#TODO:bleu may be wrong cuz max length
    logging.info('BLEU : %f',bleu_score['bleu'])
    logging.info('test loss : %f',acc/(counter))
    
    del test_dataloaderx,acc,counter,test_dataloaderx_attn,sacrebleu_score,bleu_score,test_dataloadery,test_dataloadery_attn,ls,pre,x_decoded,pred_decoded,label_decoded,pred_str,label_str,pred_list,label_list
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    
    
    # logging.info(f"GPU mem after test:{getGPUMem(device)}%")
        

# %%

my_test(valid_dataloader,model,-1)
for epoch in range(10):

    logging.info(f"\n\n  ----------------epoch:{epoch}----------------")
    my_train(train_dataloader,model,optimizer )
    my_test(valid_dataloader,model,epoch) 

torch.save(model,'./model/'+now+'model.pt')


# %%



