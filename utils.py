

import os
import torch
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from parameter import seed_,language
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Pool():

    def __init__(self, pool_size=1):
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num = 0
            self.buffer = []

    def query(self, inputs):
        if self.pool_size == 1:  # if the buffer size is 0, do nothing
            return inputs
        ret = []
        maxx = 0
        for input in inputs:
            if self.num < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num = self.num + 1
                self.buffer.append(input)
                maxx = max(maxx,input.shape[0])
                ret.append(input)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.buffer[random_id].clone()
                    self.buffer[random_id] = input
                    maxx = max(maxx,tmp.shape[0])
                    ret.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    maxx = max(maxx,input.shape[0])
                    ret.append(input)

        mat = torch.zeros((len(ret),maxx,input.shape[-1]),device=device,requires_grad=False)
        mat[:,:,0] = 1
        for i,r in enumerate(ret):
            mat[i,:r.shape[0],:] = r
        return mat

def tokenize(text_data, tokenizer, max_length, padding = True):
    
    encoding = tokenizer(text_data, return_tensors='pt', padding=padding, truncation = True, max_length = max_length)

    input_ids = encoding['input_ids']
    
    attention_mask = encoding['attention_mask']
    
    return input_ids, attention_mask
def get_Dataset_chaos(dataset, tokenizer,max_length):
    b_sentence = [x[language] for x in dataset]
    a_sentence = [x['en'] for x in dataset]
    a_ids, a_ids_attn = tokenize(a_sentence, tokenizer, max_length = max_length)
    shuffle_index = torch.randperm(a_ids.shape[0])
    a_ids=a_ids[shuffle_index]
    a_ids_attn=a_ids_attn[shuffle_index]
    
    b_ids, b_ids_attn = tokenize(b_sentence, tokenizer, max_length = max_length)
    train_data = TensorDataset(a_ids, a_ids_attn, b_ids, b_ids_attn)
    return train_data
def get_Dataset(dataset, tokenizer,max_length):
    b_sentence = [x[language] for x in dataset]
    a_sentence = [x['en'] for x in dataset]
    a_ids, a_ids_attn = tokenize(a_sentence, tokenizer, max_length = max_length)
    b_ids, b_ids_attn = tokenize(b_sentence, tokenizer, max_length = max_length)
    train_data = TensorDataset(a_ids, a_ids_attn, b_ids, b_ids_attn)
    return train_data #needed for T5
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('t5-small')
def d(l):
    return tokenizer.batch_decode(l,skip_special_tokens=True)
def en(l):
    return tokenize(l,tokenizer,512,True)