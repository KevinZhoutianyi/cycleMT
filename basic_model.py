from cProfile import label
from dis import dis
from lib2to3.pgen2.tokenize import generate_tokens
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
import torch.nn.functional as F

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
    def __init__(self,args,pretrained,name='D',tokenizer=None,prefix='') -> None:
        super(D, self).__init__()
        self.name = name
        self.args = args
        self.tokenizer = tokenizer
        self.tokenzied_prefix = tokenize([prefix],tokenizer,args.max_length,True)[0].squeeze().cuda()
        self.tokenzied_prefix.require_grad = False
        self.tokenzied_prefix_attn = tokenize([prefix],tokenizer,args.max_length,True)[1].squeeze().cuda()
        self.tokenzied_prefix_attn.require_grad = False
        self.encoder = pretrained.get_encoder()
        self.embedding = Embedding_(self.encoder.embed_tokens).requires_grad_()
        # self.dropout = nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        self.relu = nn.ReLU()
        
    def set_require_grad(self,require):
        for param in self.encoder.parameters():
            param.requires_grad = require
        self.embedding.requires_grad_ = require
        self.classifier.requires_grad_ = require
    def forward(self,x,x_attn):
        x,x_attn =  self.add_prefix(x,x_attn)
        x_emb = self.embedding(x)
        distr = self.encoder(inputs_embeds=x_emb,attention_mask=x_attn).last_hidden_state
        x_attn= x_attn.unsqueeze(-1)
        distr = torch.mul(distr,x_attn)#previously,even the word is 0, their will be some value in the context vector, the model will make them large to classifier.
        distr = distr[:,0,:]#torch.sum(distr,1)/torch.sum(x_attn,1)
        ret =  self.classifier(distr)#(bs,1)
        # ret = self.relu(ret)#(bs,1)no for WGAN
        return ret
    def add_prefix(self,x,x_attn):
        prefix = self.tokenzied_prefix.repeat(x.shape[0],1)#.cuda()
        if(len(x.shape)==2):
            x = torch.hstack((prefix,x))
        else:
            one_hot = torch.zeros(x.shape[0], prefix.shape[1],x.shape[-1], device=torch.device('cuda:0'),requires_grad=False)
            prefix = one_hot.scatter_(-1, prefix.unsqueeze(-1), 1.).float()
            
            x = torch.hstack((prefix,x))
        prefix = self.tokenzied_prefix_attn.repeat(x.shape[0],1)#.cuda()
        x_attn = torch.hstack((prefix,x_attn))
        return x,x_attn

class G(nn.Module):
    def __init__(self,args,pretrained,tokenizer,prefix,name='G') -> None:
        super(G, self).__init__()
        self.name = name
        self.tokenizer = tokenizer
        self.tokenzied_prefix = tokenize([prefix],tokenizer,args.max_length,True)[0].squeeze().cuda()
        self.tokenzied_prefix.require_grad = False
        self.tokenzied_prefix_attn = tokenize([prefix],tokenizer,args.max_length,True)[1].squeeze().cuda()
        self.tokenzied_prefix_attn.require_grad = False
        self.args = args
        self.num_beam = args.num_beam
        self.model = pretrained
        self.softmax = torch.nn.Softmax(dim=-1)
        self.encoder = self.model.get_encoder()
        self.embedding = Embedding_(self.encoder.embed_tokens).requires_grad_()
    def set_require_grad(self,require):
        self.embedding.requires_grad_ = require
        for p in self.model.parameters():
            p.requires_grad = require


    def gumbel_generate(self,x,x_attn):
        ''' A -> B
        input is (batchsize,sentencelength) without prefix and start padding
        1. add prefix
        2. gumble trick
        3. return the generated one_hot output(batchsize*numbeam,sentencelength,vocabsize)
        '''
        #assume BS =32
        num_beam = self.num_beam
        x,x_attn = self.add_prefix(x,x_attn)#add translate a to b:
        x_ = x#made copy (BS,seqlen)
        if(len(x.shape)==3):
            x = torch.argmax(x,-1)#change logit to index if needed
        generate_id = self.generate(x,num_beams=num_beam,num_return_sequences=num_beam)[:,1:].contiguous()#get rid of start padding  shapewill be (32*4,seqlen)
        att = (generate_id>0.5).long()# (32*4,seqlen)
        x_ = tile(x_,0,num_beam)#(BS*num_beam,seqlen)
        x_attn = tile(x_attn,0,num_beam)#(BS*num_beam,seqlen)
        x_emb = self.embedding(x_)#(BS,seqlen,512)
        distr = self.model(inputs_embeds=x_emb, attention_mask=x_attn, labels = generate_id, decoder_attention_mask =att).logits
        # one_hot_output,att = distr,1-(distr[:, :,0]>0.5).long()
        distr_softmax = self.softmax(distr)
        one_hot = torch.zeros(generate_id.shape[0], generate_id.shape[1],distr_softmax.shape[-1], device=torch.device('cuda:0'),requires_grad=False)
        one_hot_output = one_hot.scatter_(-1, generate_id.unsqueeze(-1), 1.).float().detach() + distr_softmax - distr_softmax.detach()
        return one_hot_output,att# not start with padding

    def gumbel_generate_soft(self,x,x_attn):
        ''' B -> cycleA
        input is (batchsize*numbeam,sentencelength) without prefix and start padding
        1. add prefix
        2. gumble trick
        3. return the generated one_hot output(batchsize*numbeam,sentencelength,vocabsize)
        '''
        x,x_attn = self.add_prefix(x,x_attn)#add translate a to b:
        x_ = x#made copy
        if(len(x.shape)==3):
            x = torch.argmax(x,-1)#change logit to index if needed
        generate_id = self.generate(x,num_beams=2,num_return_sequences=1)[:,1:].contiguous()#get rid of start padding 
        att = (generate_id>0.5).long()
        x_emb = self.embedding(x_)
        distr = self.model(inputs_embeds=x_emb, attention_mask=x_attn, labels = generate_id, decoder_attention_mask =att).logits
        # one_hot_output,att = distr,1-(distr[:, :,0]>0.5).long()
        distr_softmax = self.softmax(distr)
        return distr_softmax,att# not start with padding


    def add_prefix(self,x,x_attn):
        prefix = self.tokenzied_prefix.repeat(x.shape[0],1)#.cuda()
        if(len(x.shape)==2):
            x = torch.hstack((prefix,x))
        else:
            one_hot = torch.zeros(x.shape[0], prefix.shape[1],x.shape[-1], device=torch.device('cuda:0'),requires_grad=False)
            prefix = one_hot.scatter_(-1, prefix.unsqueeze(-1), 1.).float()
            
            x = torch.hstack((prefix,x))
        prefix = self.tokenzied_prefix_attn.repeat(x.shape[0],1)#.cuda()
        x_attn = torch.hstack((prefix,x_attn))
        return x,x_attn
    def generate(self, input_ids, num_beams = 2, max_length=512, num_return_sequences=2):#long training time!
        max_length = self.args.max_length
        output_ids = self.model.generate( input_ids = input_ids, num_beams = num_beams, early_stopping = True, max_length = max_length, length_penalty =0.6, repetition_penalty = 1,num_return_sequences=num_return_sequences )
        return output_ids
    def test_generate(self, x, num_beams = 2, max_length=512):
        max_length = self.args.max_length
        prefix = self.tokenzied_prefix.repeat(x.shape[0],1)#.cuda()
        x = torch.hstack((prefix,x))
        output_ids = self.model.generate( input_ids = x, num_beams = num_beams, early_stopping = True, max_length = max_length, length_penalty =0.6, repetition_penalty = 1)
        return output_ids
    def forward(self, input_ids, input_attn, target_ids = None, target_attn = None):
        input_ids, input_attn = self.add_prefix(input_ids,input_attn)
        inp_emb = self.embedding(input_ids)
        out = self.model(inputs_embeds = inp_emb, attention_mask = input_attn, labels = target_ids, decoder_attention_mask = target_attn, return_dict=True)
        return out
