

import os
import torch
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from parameter import seed_
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

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num = 0
            self.buffer = []

    def query(self, inputs):
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return inputs
        ret = []
        for input in inputs:
            input = torch.unsqueeze(input.data, 0)
            if self.num < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_ = self.num + 1
                self.buffer.append(input)
                ret.append(input)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = input
                    ret.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    ret.append(input)
        ret = torch.cat(ret, 0)   # collect all the images and return
        return ret

def tokenize(text_data, tokenizer, max_length, padding = True):
    
    encoding = tokenizer(text_data, return_tensors='pt', padding=padding, truncation = True, max_length = max_length)

    input_ids = encoding['input_ids']
    
    attention_mask = encoding['attention_mask']
    
    return input_ids, attention_mask
def get_train_Dataset(dataset, tokenizer):
    print('get train data start')
    train_sentence = [x['en'] for x in dataset]
    train_target = [x[target_language] for x in dataset]

  
    model1_input_ids, model1_input_attention_mask = tokenize(train_sentence, tokenizer, max_length = max_length)
  
    model1_target_ids, model1_target_attention_mask = tokenize(train_target, tokenizer, max_length = max_length)
 
    train_data = TensorDataset(model1_input_ids, model1_input_attention_mask, model1_target_ids, model1_target_attention_mask)
    
    print('get train data end')
   
    return train_data