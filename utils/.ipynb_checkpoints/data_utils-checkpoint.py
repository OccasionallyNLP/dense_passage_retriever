# -*- coding: utf-8 -*-
# data utils 로 빼두기.
import json
import os
import hashlib
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import tensor as T
from typing import List,Optional
import random
from collections import deque

class DprPassageDataset(Dataset):
    def __init__(self, args, data:List[dict], tokenizer):
        super().__init__()
        self.args = args
        self.data = data
        self.tokenizer = tokenizer 
        self.passage_max_length = args.passage_max_length

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def _collate_fn(self, batch):
        passages = []
        labels = []
        for b in batch:
            if self.args.contain_title:
                passages.append(b['title']+' '+b['context'])
            else:
                passages.append(b['context'])
        output = self.tokenizer(passages, max_length = self.passage_max_length, padding = 'max_length', truncation = True, return_tensors = 'pt')
        return output
    
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
    
    def get_feature(self,data_i):
        question = data_i['question']#self.tokenizer(data_i['question'], max_length=self.question_max_length, padding = 'max_length', truncation = True)
        positive_ctx = data_i['positive_ctxs'][0]['title']+' '+data_i['positive_ctxs'][0]['context'] if self.args.contain_title else data_i['positive_ctxs'][0]['context']
        positive_ctx_id = data_i['positive_ctxs_ids'][0]
        hard_negative_ctxs = []
        hard_negative_ctxs_ids = []
        #data_i['negative_ctxs_ids'][0]
        if self.args.n_hard_negative_ctxs>0:
            assert len(data_i['negative_ctxs'])>=self.args.n_hard_negative_ctxs
            hard_negative_ctxs=[i['title']+' '+i['context'] if self.args.contain_title  else i['context'] for i in data_i['negative_ctxs'][:self.args.n_hard_negative_ctxs]]
            hard_negative_ctxs_ids=[i for i in data_i['negative_ctxs_ids'][:self.args.n_hard_negative_ctxs]]
        return {'question':question, 'positive_ctx':positive_ctx, 'hard_negative_ctxs':hard_negative_ctxs, 'positive_ctx_id':positive_ctx_id,  'hard_negative_ctxs_ids':hard_negative_ctxs_ids}

    def _collate_fn(self, batch):
        #print(batch)
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
            if d['hard_negative_ctxs_ids']:
                for a,b in zip(d['hard_negative_ctxs_ids'],d['hard_negative_ctxs']):
                    if a not in sampled_indices:
                        passages.append(b)
                        sampled_indices.append(a)
        questions = self.tokenizer(questions, max_length = self.question_max_length, padding = 'max_length', truncation = True, return_tensors = 'pt')
        passages = self.tokenizer(passages, max_length = self.passage_max_length, padding = 'max_length', truncation = True, return_tensors = 'pt')
        labels = T(passages_indices)
        return dict(question_input_ids = questions.input_ids, question_attention_mask = questions.attention_mask, question_token_type_ids = questions.token_type_ids, passage_input_ids = passages.input_ids, passage_attention_mask = passages.attention_mask, passage_token_type_ids = passages.token_type_ids, labels=labels)

    
# class DprTrainDatasetWithHistory(Dataset):
#     history_prefix='<history>'
#     query_prefix='<query>'
#     apprentice_prefix='<apprentice>'
#     wizard_prefix='<wizard>'
#     title_prefix='<title>'
#     context_prefix='<context>'
    
#     def __init__(self, args, data:List[dict], tokenizer):
#         super().__init__()
#         self.data = data
#         self.args = args
#         self.tokenizer = tokenizer 
#         self.passage_max_length = args.passage_max_length
#         self.question_max_length = args.question_max_length        
    
#     def __getitem__(self, index):
#         return self.data[index]
    
#     def __len__(self):
#         return len(self.data)  
    
#     def _get_history(self, history:List[dict])->List:
#         output = []
#         if history:
#             output = []
#             for i in history:
#                 if i['speaker']=='user':
#                     output.append(self.apprentice_prefix+i['utterance'])
#                 else:
#                     if self.args.just_user:
#                         continue
#                     output.append(self.wizard_prefix+i['utterance'])
#         return output
    
#     def _get_history_n(self, history:List[str], n:int=None):
#         if n is None:
#              output = self.history_prefix+''.join(history)
#         else:
#             output = self.history_prefix+''.join(history[-n:])
#         return output
    
#     def _get_query(self, query):
#         output = self.query_prefix+self.apprentice_prefix+query
#         return output
    
#     def get_feature(self,data_i):
#         question = self._get_query(data_i['question'])
#         history = self._get_history_n(self._get_history(data_i['history']),self.args.history_n)
#         positive_ctx = data_i['positive_ctxs'][0]['title']+' '+data_i['positive_ctxs'][0]['context'] if self.args.contain_title else data_i['positive_ctxs'][0]['context']
#         positive_ctx_id = data_i['positive_ctxs_ids'][0]
#         hard_negative_ctxs = []
#         hard_negative_ctxs_ids = []
#         if self.args.n_hard_negative_ctxs>0:
#             assert len(data_i['negative_ctxs'])>=self.args.n_hard_negative_ctxs
#             hard_negative_ctxs=[i['title']+' '+i['context'] if self.args.contain_title  else i['context'] for i in data_i['negative_ctxs'][:self.args.n_hard_negative_ctxs]]
#             hard_negative_ctxs_ids=[i in data_i['negative_ctxs_ids'][:self.args.n_hard_negative_ctxs]]
#         return {'question':question, 'positive_ctx':positive_ctx, 'hard_negative_ctxs':hard_negative_ctxs, 'positive_ctx_id':positive_ctx_id,  'hard_negative_ctxs_ids':hard_negative_ctxs_ids, 'history':history}

#     def _collate_fn(self, batch):
#         #print(batch)
#         batch = [self.get_feature(i) for i in batch]
#         questions=[]
#         passages=[]
#         passages_indices=[]
#         sampled_indices=[]
#         for d in batch:
#             if d['positive_ctx_id'] in sampled_indices:
#                 continue
#             if d['hard_negative_ctxs_ids']:
#                 for j in d['hard_negative_ctxs_ids']:
#                     if j in sampled_indices:
#                         continue
#             questions.append(d['history']+d['question'])
#             passages.append(d['positive_ctx'])
#             passages_indices.append(len(passages_indices))
#             sampled_indices.append(d['positive_ctx_id'])
#             if d['hard_negative_ctxs_ids']:
#                 for a,b in zip(d['hard_negative_ctxs_ids'],d['hard_negative_ctxs']):
#                     if a not in sampled_indices:
#                         passages.append(b)
#                         sampled_indices.append(a)
#         self.tokenizer.truncation_side = 'left'                
#         questions = self.tokenizer(questions, max_length = self.question_max_length, padding = 'max_length', truncation = True, return_tensors = 'pt')
#         self.tokenizer.truncation_side = 'right'
#         passages = self.tokenizer(passages, max_length = self.passage_max_length, padding = 'max_length', truncation = True, return_tensors = 'pt')
#         labels = T(passages_indices)
#         return dict(question_input_ids = questions.input_ids, question_attention_mask = questions.attention_mask, question_token_type_ids = questions.token_type_ids, passage_input_ids = passages.input_ids, passage_attention_mask = passages.attention_mask, passage_token_type_ids = passages.token_type_ids, labels=labels)
    
# class Collater(object):
#     history_prefix='<history>'
#     query_prefix='<query>'
#     apprentice_prefix='<apprentice>'
#     wizard_prefix='<wizard>'
#     title_prefix='<title>'
#     context_prefix='<context>'
    
#     def __init__(self, args, tokenizer):
#         self.args = args
#         self.tokenizer = tokenizer
    
#     def _get_history(self, history):
#         output = []
#         if history:
#             output = []
#             for i in history:
#                 if i['speaker']=='user':
#                     output.append(self.apprentice_prefix+i['utterance'])
#                 else:
#                     if self.args.just_user:
#                         continue
#                     output.append(self.wizard_prefix+i['utterance'])
#         return output

#     def _get_history_n(self, history:List[str], n:int=None):
#         if n is None:
#              output = self.history_prefix+''.join(history)
#         else:
#             output = self.history_prefix+''.join(history[-n:])
#         return output
                    
#     def _collate_fn(self,batch):
#         questions = []
#         histories = []
#         positive_ctxs_ids = []
#         indices = []
#         for data in batch:
#             if 'positive_ctxs_ids' in data:
#                 positive_ctxs_ids.extend([data['positive_ctxs_ids'][0]])
#             question = data['question']
#             if self.args.include_history:
#                 history = self._get_history_n(self._get_history(data['history']),self.args.history_n)
#                 question = history + question
#             indices.append(data['_id'])
#             questions.append(question)
#         self.tokenizer.truncation_side = 'left'                
#         questions = self.tokenizer(questions, max_length = self.args.question_max_length, padding = 'max_length', truncation = True, return_tensors = 'pt')
#         self.tokenizer.truncation_side = 'right'                
#         return questions, positive_ctxs_ids, indices

class Collater(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
    
    def _collate_fn(self, batch):
        questions = []
        positive_ctxs_ids = []
        indices = []
        for data in batch:
            if 'positive_ctxs_ids' in data:
                positive_ctxs_ids.extend([data['positive_ctxs_ids'][0]]) # OK - question이 1개 이므로.
            question = data['question']
            indices.append(data['_id'])
            questions.append(question)
        questions = self.tokenizer(questions, max_length = self.args.question_max_length, padding = 'max_length', truncation = True, return_tensors = 'pt')
        return questions, positive_ctxs_ids, indices