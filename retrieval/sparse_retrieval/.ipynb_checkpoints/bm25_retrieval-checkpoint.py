# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:42:03 2021

@author: OK
"""

import numpy as np
import os
from tqdm import tqdm

import json
import math

from typing import List
import argparse

from rank_bm25 import BM25Okapi

from collections import defaultdict, Counter

from retrieval.retrieval_base import Retrieval

from utils.tools import *

class BM25(Retrieval):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.doc_count = len(self.contexts)
        
    def exec_embedding(self):
        tokenized_contexts = [self.tokenizer.tokenize(j['title']+' '+j['context']) if self.args.include_title else self.tokenizer.tokenize(j['context']) for i,j in tqdm(self.contexts.items(),desc='tokenize')]
        self.db_id_to_wiki_id = {i:j for i,j in enumerate(self.contexts.keys())}
        bm25 = BM25Okapi(tokenized_contexts) # use basic value
        self.passage_encoder = bm25
        return self.passage_encoder
    
    def get_score(self, query):
        tokenized_query = self.tokenizer.tokenize(query)
        if self.passage_encoder is not None:
            pass
        else:
            self.passage_encoder = self.exec_embedding()
        doc_scores = self.passage_encoder.get_scores(tokenized_query)
        sorted_idx = np.argsort(-doc_scores)
        result = [(self.db_id_to_wiki_id[i],doc_scores[i]) for i in sorted_idx]
        return result
    
    def get_top_n(self,query,n):
        result = self.get_score(query)
        if n is not None:
            assert n<=self.doc_count, 'document의 길이가 n보다 작습니다.'
            output = result[:n]
        else:
            output = result
        return output

    def retrieve(self, query, n):
        return self.get_top_n(query, n) 
    
    # hard negative
    def retrieve_without(self, query, answer, n , m):
        result = self.get_top_n(query,n=None)  # query와 유사한 context를 가져옴
        positives = result[:n]
        negatives = []
        for i,score in result:
            if answer not in self.contexts[i]['context']:
                if len(negatives)==m:
                    break
                negatives.append(i)
        return positives, negatives
