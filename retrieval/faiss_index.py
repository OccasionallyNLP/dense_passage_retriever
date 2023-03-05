# -*- coding: utf-8 -*-
# faiss 활용
import faiss
import logging
import numpy as np
import os
import pickle

from typing import List, Tuple

logger = logging.getLogger()

class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def init_index(self, vector_dim: int):
        raise NotImplementedError

    def indexing(self, passage_vectors:np.array, db_ids:List[int]):
        raise NotImplementedError
        
    def search(self, query_vectors: np.array, k: int):
        raise NotImplementedError

    def save(self, index_name:str):
        faiss.write_index(self.index, index_name)
    
    def load(self, index_name: str):
        self.index = faiss.read_index(index_name)
        
class DenseFlatIndexer(DenseIndexer):
    def __init__(self, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)

    def init_index(self, vector_dim: int):
        index = faiss.IndexFlatIP(vector_dim)
        self.index = faiss.IndexIDMap2(index)
    
    def indexing(self, passage_vectors, db_ids):
        n = len(passage_vectors)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids_i = db_ids[i : i + self.buffer_size]
            passage_vectors_i = passage_vectors[i : i + self.buffer_size]
            self.index.add_with_ids(passage_vectors_i, db_ids_i)
            logger.info("data indexed %d"%(i+1))
         
    def search(self, query_vectors: np.array, k: int):
        distances, indices = self.index.search(query_vectors, k)
        return distances, indices

class DenseIVFPQIndexer(DenseIndexer):
    pass
