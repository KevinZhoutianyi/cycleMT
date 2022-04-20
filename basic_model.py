from cProfile import label
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
        return torch.matmul(mask, self.embedding.weight[:32100,:])


class T5(nn.Module):
    
    def __init__(self, criterion, tokenizer,args=None,name='unknown'):
        super(T5, self).__init__()
        self.name = name
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        
        self._criterion = criterion
        self.args = args
        self.model = torch.load(args.model_name.replace('/','')+'.pt')
        self.encoder = self.model.get_encoder()
        self.embedding = Embedding_(self.encoder.embed_tokens).requires_grad_()#convert token to 512dimensions vector
        self.enc_emb_scale = 1

    def forward(self, input_ids, input_attn, target_ids = None, target_attn = None):
        inp_emb = self.embedding(input_ids)/self.enc_emb_scale
        out = self.model(inputs_embeds = inp_emb, attention_mask = input_attn, labels = target_ids, decoder_attention_mask = target_attn, return_dict=True)
        return out
    
    def loss(self, input_ids, input_attn, target_ids, target_attn):

        batch_size = target_ids.shape[0]
        out_emb = self.embedding(target_ids)/self.enc_emb_scale
        inp_emb = self.embedding(input_ids)/self.enc_emb_scale
        logits = self.model(inputs_embeds = inp_emb, attention_mask = input_attn, decoder_inputs_embeds   = out_emb, decoder_attention_mask = target_attn, return_dict=True).logits
        logits = logits[:,:,:32100]
        loss = self._criterion(logits,target_ids)
        loss = loss[target_ids[:, 0] != 1]#get rid of padding loss
        loss = torch.mean(loss)
        return loss



    def get_loss_vec(self, input_ids, input_attn, target_ids = None, target_attn = None):
        '''
        only count the loss when attn is 1(ie:mask the model output logits)
        reason:
        1. we need loss vector, so we cant use self().loss
        2. we will use logits with criterion to get loss, so we cannot use CE(ignoreindex==padindex)
        '''
        batch_size = target_ids.shape[0]
        target_ids_ = copy.deepcopy(target_ids)
        target_ids_[target_ids == self.tokenizer.pad_token_id] = -100
        temp = (self(input_ids, input_attn, target_ids = target_ids_))
        logits = temp.logits
        loss_seq = self._criterion(logits.view(-1,logits.shape[-1]), target_ids_.view(-1)).view(batch_size,-1)
        count = torch.sum(target_attn,-1).squeeze_()
        loss_vec_ = torch.sum(loss_seq,-1)/count
        return loss_vec_


    # used for generation of summaries from articles
    def generate(self, input_ids, num_beams = 4, max_length=512):
        
        output_ids = self.model.generate( input_ids = input_ids, num_beams = num_beams, early_stopping = True, max_length = max_length, length_penalty =0.6, repetition_penalty = 0.8)
        return output_ids

    def new(self):

        model_new = T5(self._criterion, self.tokenizer, args = self.args).cuda()
        model_new.model.load_state_dict(self.model.state_dict())        
        return model_new
