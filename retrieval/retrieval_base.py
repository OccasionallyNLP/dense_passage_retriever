# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:36:41 2021

@author: OK
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:01:05 2021

@author: OK
"""

import os
import json

from tqdm import tqdm

class Retrieval:
    def __init__(self, args, tokenizer):
        self.args = args
        self.passage_encoder = None
        self.query_encoder = None 
        self.passage_embedding = None 
        self.tokenizer = tokenizer
        
        with open(self.args.data_path,"r", encoding='utf-8') as f:
            context = json.load(f)
        self.contexts = context

    def exec_embedding(self):
        raise NotImplementedError

    def load_embedding(self):
        raise NotImplementedError

    def get_score(self, query):
        raise NotImplementedError

    def retrieve(self, query_or_dataset, topk=20):
        assert self.passage_embedding is not None, "exec_embedding()을 먼저 수행한 후에 retrieve()를 작동시켜 주세요. "
        raise NotImplementedError
        
        