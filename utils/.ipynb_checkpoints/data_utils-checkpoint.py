# -*- coding: utf-8 -*-
# data_utils
import json
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import tensor as T
from typing import List
import random
import copy
from transformers import BatchEncoding

class DprTrainDataset(Dataset):
    def __init__(self, args, data:List[dict], tokenizer):
        super().__init__()
        self.data = data
        self.args = args
        self.tokenizer = tokenizer 
        self.passage_max_length = args.passage_max_length
        self.question_max_length = args.question_max_length        
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)  
    
    def get_feature(self, data_i):
        question = data_i['question']
        # 첫번째 것만 활용.
        positive_ctx = data_i['positive_ctxs'][0]['title']+' '+data_i['positive_ctxs'][0]['context'] if self.args.contain_title else data_i['positive_ctxs'][0]['context']
        positive_ctx_id = data_i['positive_ctxs_ids'][0]
        hard_negative_ctxs = []
        hard_negative_ctxs_ids = []
        if self.args.n_hard_negative_ctxs>0:
            assert len(data_i['negative_ctxs'])>=self.args.n_hard_negative_ctxs
            hard_negative_ctxs=[i['title']+' '+i['context'] if self.args.contain_title  else i['context'] for i in data_i['negative_ctxs'][:self.args.n_hard_negative_ctxs]]
            hard_negative_ctxs_ids=[i for i in data_i['negative_ctxs_ids'][:self.args.n_hard_negative_ctxs]]
        return {'question':question, 'positive_ctx':positive_ctx, 'hard_negative_ctxs':hard_negative_ctxs, 'positive_ctx_id':positive_ctx_id,  'hard_negative_ctxs_ids':hard_negative_ctxs_ids}

    def _collate_fn(self, batch):
        batch = [self.get_feature(i) for i in batch]
        questions=[]
        passages=[]
        passages_indices=[]
        sampled_indices=[]
        for d in batch:
            if d['positive_ctx_id'] in sampled_indices:
                continue
            questions.append(d['question'])
            passages_indices.append(len(passages))
            passages.append(d['positive_ctx'])
            sampled_indices.append(d['positive_ctx_id'])
            # 이 부분에 대해서 생각할 것이 많네
            if d['hard_negative_ctxs_ids']:
                for a,b in zip(d['hard_negative_ctxs_ids'],d['hard_negative_ctxs']):
                    if a not in sampled_indices:
                        passages.append(b)
                        sampled_indices.append(a)
        questions = self.tokenizer(questions, max_length = self.question_max_length, padding = True, truncation = True, return_tensors = 'pt')
        passages = self.tokenizer(passages, max_length = self.passage_max_length, padding = True, truncation = True, return_tensors = 'pt')
        if questions.get('token_type_ids') is not None:
            question_token_type_ids = questions.token_type_ids
        if passages.get('token_type_ids') is not None:
            passage_token_type_ids = passages.token_type_ids
        labels = torch.tensor(passages_indices)
        if question_token_type_ids is not None and passage_token_type_ids is not None:
            output = BatchEncoding(dict(question_input_ids = questions.input_ids, question_attention_mask = questions.attention_mask, question_token_type_ids = question_token_type_ids, passage_input_ids = passages.input_ids, passage_attention_mask = passages.attention_mask, passage_token_type_ids =passage_token_type_ids, labels=labels))
        else:
            output = BatchEncoding(dict(question_input_ids = questions.input_ids, question_attention_mask = questions.attention_mask, passage_input_ids = passages.input_ids, passage_attention_mask = passages.attention_mask, labels=labels))
        return output