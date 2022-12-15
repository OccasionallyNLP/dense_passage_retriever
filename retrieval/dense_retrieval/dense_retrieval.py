# -*- coding: utf-8 -*-

import torch
from torch import tensor as T
from utils.data_utils import DprContextDataset
from retrieval.retrieval_base import Retrieval
import os
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset,SequentialSampler

class DprRetrieval(Retrieval):
    def __init__(self, args, tokenizer, passage_encoder, question_encoder):
        super().__init__(args, tokenizer)
        self.context_dataset = DprContextDataset(self.args, self.contexts, self.tokenizer)
        self.passage_encoder = passage_encoder
        self.question_encoder = question_encoder
        self.doc_count = len(self.contexts)

    def exec_embedding(self): # 고도화 가능 -> multigpu
        if torch.cuda.is_available():
            self.passage_encoder=self.passage_encoder.cuda()
        context_dataloader = DataLoader(self.context_dataset, batch_size = self.args.batch_size, sampler = SequentialSampler(self.context_dataset))
        output = []
        indices = []
        for data in context_dataloader:
            self.passage_encoder.eval()
            with torch.no_grad():
                if torch.cuda.is_available():
                    data = {i:j.cuda() for i,j in data.items()}
                passage_embedding = self.passage_encoder(**data)
                output.extend(passage_embedding.cpu().tolist())
                indices.extend(data['labels'].cpu().tolist())
        # indices와 output을 묶어서 저장
        f = open(self.args.passage_embedding_path, 'w', encoding='utf-8')
        output_join = {'passage_embedding':output,'passage_indice':indices}
        del output
        del indices
        json.dump(output_join,f)
        self.passage_embedding = output_join
        return self.passage_embedding
        
    def load_embedding(self):
        if os.path.isfile(self.args.passage_embedding_path):
            with open(self.args.passage_embedding_path,'r', encoding='utf-8') as f:
                try:
                    self.passage_embedding = json.load(f)
                except:
                    self.passage_embedding = self.exec_embedding()
            if self.passage_embedding is None:
                self.passage_embedding = self.exec_embedding()
        else:
            self.passage_embedding = self.exec_embedding()
        return self.passage_embedding
        
    def get_score(self, query:str or List[str]): # inference
        if isinstance(query,str):
            question_input_ids, question_attention_mask, question_token_type_ids = DprContextDataset.tensorize(query, None, self.tokenizer, self.args.question_max_length)
            question_input_ids, question_attention_mask, question_token_type_ids = T(question_input_ids), T(question_attention_mask), T(question_token_type_ids)
            
        elif isinstance(query, list):
            #print('has')
            tmp = list(map(lambda i : DprContextDataset.tensorize(i, None, self.tokenizer, self.args.question_max_length),query))
            tmp = T(tmp)
            question_input_ids, question_attention_mask, question_token_type_ids = tmp[:,0],tmp[:,1],tmp[:,2]
            
        if question_input_ids.dim()==1:
            question_input_ids = question_input_ids.unsqueeze(0) 
            question_attention_mask = question_attention_mask.unsqueeze(0) 
            question_token_type_ids = question_token_type_ids.unsqueeze(0)
        data = dict(input_ids=question_input_ids, attention_mask = question_attention_mask, token_type_ids = question_token_type_ids)
        if torch.cuda.is_available():
            self.question_encoder = self.question_encoder.cuda()
            data = {i:j.cuda() for i,j in data.items()}
        
        with torch.no_grad():
            self.question_encoder.eval()
            question_embedding = self.question_encoder(**data)
            
        if self.passage_embedding is None:
            self.passage_embedding = self.load_embedding()
        # matmul
        tmp = T(self.passage_embedding['passage_embedding']).T
        if torch.cuda.is_available():
            tmp = tmp.cuda()
        score = question_embedding.matmul(tmp)
        return score
    
    def get_top_n(self,query,n):
        score = self.get_score(query)
        assert n<=self.doc_count, 'document의 길이가 n보다 작습니다.'
        values, indices =torch.topk(score,n,dim=-1)
        top_n = [[self.passage_embedding['passage_indice'][j] for j in i] for i in indices.tolist()]
        return top_n

    def retrieve(self, query, n):
        return self.get_top_n(query, n) 
