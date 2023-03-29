# -*- coding: utf-8 -*-
import numpy as np
import os
from tqdm import tqdm
import json
import math
from typing import List, Optional
import argparse
from rank_bm25 import BM25Okapi
from collections import defaultdict, Counter

class BM25(object):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer        
        self.bm25 = None
        
    def exec_embedding(self, contexts):
        tokenized_contexts = [self.tokenizer(i['tc']) for i in contexts]
        self.db_id_to_contexts_id = {i:j['doc_id'] for i,j in enumerate(contexts)}
        bm25 = BM25Okapi(tokenized_contexts) # use basic value
        self.bm25 = bm25        
    
    def get_score(self, query):
        tokenized_query = self.tokenizer(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        sorted_idx = np.argsort(-doc_scores)
        result = [(self.db_id_to_contexts_id[i],doc_scores[i]) for i in sorted_idx]
        return result
    
    def get_top_n(self,query, n):
        result = self.get_score(query)
        if n is not None:
            output = result[:n]
        else:
            output = result
        return output

    def retrieve(self, query, n):
        return self.get_top_n(query, n) 
    
    # hard negative
    def retrieve_without(self, query, answer, contexts, n , m):
        result = self.get_top_n(query, n=None)  # query와 유사한 context를 가져옴
        positives = result[:n]
        negatives = []
        for i,score in result:
            if answer not in contexts[i]['tc']:
                if len(negatives)==m:
                    break
                negatives.append(i)
        return positives, negatives
