from cProfile import label
from math import dist
import os
import random
import numpy as np
import gc
import copy
from transformers import  T5ForConditionalGeneration
from transformers import T5Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from parameter import seed_
from utils import *

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

seed_torch(seed_)
class Embedding_(torch.nn.Module):
    def __init__(self, embedding_layer):
        super(Embedding_, self).__init__()
        self.embedding = embedding_layer.cuda()
        #https://github.com/huggingface/transformers/issues/4875
    def forward(self, mask):
        if mask.ndim == 2:
            assert mask.dtype == torch.long
            return self.embedding(mask)
        
        assert mask.dtype == torch.float
        return torch.matmul(mask[:,:,:32100], self.embedding.weight[:32100,:])
class D(nn.Module):
    def __init__(self,args,pretrained,name='D') -> None:
        super(D, self).__init__()
        self.name = name
        self.args = args
        self.encoder = pretrained.get_encoder()
        self.embedding = Embedding_(self.encoder.embed_tokens).requires_grad_()
        self.classifier = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        
    def set_require_grad(self,require):
        for param in self.encoder.parameters():
            param.requires_grad = require
        self.embedding.requires_grad_ = require
        self.classifier.requires_grad_ = require
    def forward(self,x):
        x_emb = self.embedding(x)
        distr = self.encoder(inputs_embeds=x_emb).last_hidden_state
        distr = torch.mean(distr,1).squeeze()
        ret =  self.classifier(distr)
        return ret

class G(nn.Module):
    def __init__(self,args,pretrained,tokenizer,prefix,name='G') -> None:
        super(G, self).__init__()
        self.name = name
        self.tokenizer = tokenizer
        self.tokenzied_prefix = tokenize([prefix],tokenizer,512,True)[0].squeeze()
        self.tokenzied_prefix.require_grad = False
        self.tokenzied_prefix_attn = tokenize([prefix],tokenizer,512,True)[1].squeeze()
        self.tokenzied_prefix_attn.require_grad = False
        self.args = args
        self.model = pretrained
        self.encoder = self.model.get_encoder()
        self.embedding = Embedding_(self.encoder.embed_tokens).requires_grad_()
    def set_require_grad(self,require):
        self.embedding.requires_grad_ = require
        for p in self.model.parameters():
            p.requires_grad = require
    def gumble_generate(self,x,x_attn):
        '''
        input is (batchsize,sentencelength)
        1. add prefix
        2. gumble trick
        3. return the generated one_hot output(batchsize,sentencelength,vocabsize)
        '''
        x,x_attn = self.add_prefix(x,x_attn)
        x_ = x
        if(len(x.shape)==3):
            x = torch.argmax(x,-1)
        generate_id = self.model.generate(x,num_beams=1)
        att = (generate_id>0.5).long()
        x_emb = self.embedding(x_)
        distr = self.model(inputs_embeds=x_emb, attention_mask=x_attn, labels = generate_id, decoder_attention_mask = torch.ones_like(generate_id,requires_grad=False).long()).logits
        one_hot = torch.zeros(generate_id.shape[0], generate_id.shape[1],distr.shape[-1], device=torch.device('cuda:0'),requires_grad=False)
        one_hot_output = one_hot.scatter_(-1, generate_id.unsqueeze(-1), 1.).float().detach() + distr - distr.detach()
        return one_hot_output,att
    def add_prefix(self,x,x_attn):
        prefix = self.tokenzied_prefix.repeat(x.shape[0],1).cuda()
        if(len(x.shape)==2):
            x = torch.hstack((prefix,x))
        else:
            one_hot = torch.zeros(x.shape[0], prefix.shape[1],x.shape[-1], device=torch.device('cuda:0'),requires_grad=False)
            prefix = one_hot.scatter_(-1, prefix.unsqueeze(-1), 1.).float()
            
            x = torch.hstack((prefix,x))
        prefix = self.tokenzied_prefix_attn.repeat(x.shape[0],1).cuda()
        x_attn = torch.hstack((prefix,x_attn))
        return x,x_attn
    def generate(self, input_ids, num_beams = 4, max_length=512):#long training time!
        output_ids = self.model.generate( input_ids = input_ids, num_beams = num_beams, early_stopping = True, max_length = max_length, length_penalty =0.6, repetition_penalty = 0.8)
        return output_ids
    def test_generate(self, x, num_beams = 4, max_length=512):
        prefix = self.tokenzied_prefix.repeat(x.shape[0],1).cuda()
        x = torch.hstack((prefix,x))
        output_ids = self.model.generate( input_ids = x, num_beams = num_beams, early_stopping = True, max_length = max_length, length_penalty =0.6, repetition_penalty = 0.8)
        return output_ids
    def forward(self, input_ids, input_attn, target_ids = None, target_attn = None):
        inp_emb = self.embedding(input_ids)
        out = self.model(inputs_embeds = inp_emb, attention_mask = input_attn, labels = target_ids, decoder_attention_mask = target_attn, return_dict=True)
        return out
