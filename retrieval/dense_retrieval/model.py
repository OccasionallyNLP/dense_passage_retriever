# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 17:40:42 2021

@author: OK
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import tensor as T
from transformers import PreTrainedModel, RobertaModel, BertModel, T5EncoderModel

class Encoder(PreTrainedModel):
    def __init__(self, config, pool, model_class):
        super().__init__(config)
        self.pool = pool
        self.pretrained_model = model_class(config)
        
    def init_pretrained_model(self, state_dict):
        self.pretrained_model.load_state_dict(state_dict) 
    
    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        if isinstance(self.pretrained_model, BertModel):
            output = self.pretrained_model(input_ids, attention_mask, token_type_ids)
            
        else: 
            output = self.pretrained_model(input_ids, attention_mask) 
        
        if self.pool=='cls':
            # T5 Encoder만 따로 만들어놔야함.
            if isinstance(self.pretrained_model, T5EncoderModel):
                out = output.last_hidden_state[:,0,:] # bs, seq_len, dim
            else:
                out = output['pooler_output'] # bs, dim
        elif self.pool == 'mean':
            out = output['last_hidden_states'].masked_fill(attention_mask.unsqueeze(2).repeat(1,1,self.config.hidden_size)==0,0) # bs, seq_len, dim
            out = out.sum(dim=1) # bs, dim
            s = attention_mask.sum(-1, keepdim=True) # bs, 1
            out = out/(s+1e-12)
        return out
    
class DprEncoder(nn.Module):
    def __init__(self, encoder_p:PreTrainedModel, encoder_q:PreTrainedModel, fix_p_encoder : bool = False, fix_q_encoder : bool = False):
        super().__init__()
        self.passage_encoder = encoder_p
        self.question_encoder = encoder_q
        self.fix_p_encoder = fix_p_encoder
        self.fix_q_encoder = fix_q_encoder
        
    def forward(self, passage_input_ids:T, passage_attention_mask:T, passage_token_type_ids:T, question_input_ids:T, question_attention_mask:T, question_token_type_ids:T, labels:None):
        if self.fix_p_encoder:
            with torch.no_grad():
                ep = self.passage_encoder(input_ids = passage_input_ids, attention_mask = passage_attention_mask, token_type_ids =  passage_token_type_ids)
        else:
            ep = self.passage_encoder(input_ids = passage_input_ids, attention_mask = passage_attention_mask, token_type_ids = passage_token_type_ids)
        if self.fix_q_encoder:
            with torch.no_grad():
                eq = self.question_encoder(input_ids = question_input_ids, attention_mask = question_attention_mask, token_type_ids = question_token_type_ids)
        else:
            eq = self.question_encoder(input_ids = question_input_ids, attention_mask = question_attention_mask, token_type_ids = question_token_type_ids)
        if labels is None:
            return ep, eq # (bs, dim), (bs,dim)
        else:
            score = eq.matmul(ep.T) # (bs,bs) or (bs, 2bs)
            loss = F.nll_loss(F.log_softmax(score,dim=-1),labels)
            return ep, eq, score, loss
        
