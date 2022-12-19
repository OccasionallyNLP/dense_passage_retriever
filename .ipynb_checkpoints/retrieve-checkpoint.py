# -*- coding: utf-8 -*-
import os
import json
import torch
import pickle
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,DistributedSampler,RandomSampler,SequentialSampler
import torch.nn.functional as F
import logging
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertConfig
import argparse
from retrieval.dense_retrieval.model import *
from retrieval.faiss_index import *
from utils.data_utils import *
from utils.utils import *
from utils.tools import str2bool
from utils.distribute_utils import *
from utils.model import get_back_bone_model
from utils.metrics import compute_topk_accuracy
import faiss

def get_args():
    # parser
    parser = argparse.ArgumentParser()
    # index
    # 가능한 option - with faiss, w/o faiss
    #parser.add_argument('--index_name', type=str)
    #parser.add_argument('--buffer_size', type=int, default=50000)
    parser.add_argument('--vector_dim', type=int, default=768)
    parser.add_argument('--passage_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--output_dir', type=str)
    #parser.add_argument('--with_faiss', type=str2bool, default = False)
    parser.add_argument('--local_rank', type=int, default = 0)
    parser.add_argument('--distributed', type=str2bool, default = False)
    parser.add_argument('--model', type=str, default = 'bert', choices = ['bert','roberta'], help = 'back bone model')
    parser.add_argument('--pool', type=str, default = 'cls', choices = ['cls','mean'], help = 'sentence representation') # second option 가능
    parser.add_argument('--shared', type= str2bool, default = False, help = 'share query encoder and passage encoder')
    parser.add_argument('--question_max_length',type= int)
    #parser.add_argument('--include_history',type= str2bool) #XXX
    #parser.add_argument('--just_user', type=str2bool, default = False) # XXX
    #parser.add_argument('--history_n',type=int) # XXX
    #parser.add_argument('--n_shards', type = int, default = 1)
    #parser.add_argument('--k', type = int, default = 100)
    parser.add_argument('--test_data', type = str)
    args = parser.parse_args()
    return args

def prepare_model(config, args, model_type):
    passage_encoder = Encoder(config, args, model_type)
    # question encoder
    question_encoder = Encoder(config, args, model_type)
    if args.shared:
        model = DprEncoder(passage_encoder, passage_encoder)
    else:
        model = DprEncoder(passage_encoder,question_encoder)
    model.load_state_dict(torch.load(args.model_path, map_location = 'cpu'))
    return model

def load_passage_vectors_db_ids(passage_path):
    total_passage_vectors = []
    total_passages = pickle.load(open(passage_path,'rb'))
    db_ids = [i for i in total_passages.keys()]
    passage_vectors = [i for i in total_passages.values()]
    return db_ids, passage_vectors

def inference(args, model, tokenizer, index, db_ids, passage_embeddings):
    # data
    test_data = load_data(args.test_data, local_rank = args.local_rank, distributed = args.distributed, drop_last = False)
    test_sampler = SequentialSampler(test_data)
    collater = Collater(args, tokenizer)
    test_dataloader = DataLoader(test_data, batch_size = args.batch_size, sampler = test_sampler, collate_fn = collater._collate_fn)
    iter_bar = tqdm(test_dataloader, desc='step', disable=args.local_rank not in [-1,0])
    model.eval()
    predicts = []
    actuals = []
    passage_embeddings = torch.tensor(np.array(passage_embeddings).astype('float32')).cuda()
    with torch.no_grad():
        for data in iter_bar:
            questions, positive_ctxs_ids, _ids = data
            data = {i:j.cuda() for i,j in questions.items()}
            question_embeddings = model.module.question_encoder(**data) if args.distributed else model.question_encoder(**data)
            if args.with_faiss:
                # cpu
                question_embeddings = question_embeddings.cpu().tolist()
                question_embeddings = np.array(question_embeddings).astype('float32')
                # 단일 machine 에서 동작
                distances, indices = index.search(question_embeddings, args.k)
            else:
                score = question_embeddings.matmul(passage_embeddings.T)
                distances, indices = torch.topk(score, args.k, dim = -1)
                indices = [[db_ids[j.item()] for j in i] for i in indices]
            predicts.extend(indices)
            if positive_ctxs_ids:
                actuals.extend(positive_ctxs_ids)    
    if args.distributed:
        predicts_ = gather_tensors(np.array(predicts))
        total_predicts = []
        for i in predicts_:
            total_predicts.extend(i)
        actuals_ = gather_tensors(np.array(actuals))
        total_actuals = []
        for i in actuals_:
            total_actuals.extend(i)
        if args.local_rank == 0:
            if total_actuals:
                scores = compute_topk_accuracy(total_actuals, total_predicts)
                print(np.round(scores, 2))
                with open(os.path.join(args.output_dir,'result.txt'),'w') as f:
                    f.write(str(np.round(scores, 2)))
            # tagging
            test_data = load_data(args.test_data, local_rank = 0, distributed = False)
            for i in test_data:
                i['predicted_ctxs_ids']=total_predicts[i['_id']].tolist()
            save_jsonl(args.output_dir,test_data,'attached')
    else:
        if actuals:
            scores = compute_topk_accuracy(actuals, predicts)
            print(np.round(scores, 2))
            with open(os.path.join(args.output_dir,'result.txt'),'w') as f:
                f.write(str(np.round(scores, 2)))
        # tagging
        test_data = load_data(args.test_data, local_rank = 0, distributed = False)
        for i in test_data:
            i['predicted_ctxs_ids']=predicts[i['_id']]
        save_jsonl(args.output_dir,test_data,'attached')

def main():
    args = get_args()
    seed_everything(42)
    # sanity check
    os.makedirs(args.output_dir, exist_ok = True)
    if args.local_rank in [-1,0]:
        logger = log()
    tokenizer, config, model, model_type = get_back_bone_model(args)
    model = prepare_model(config, args, model_type)
    # distributed 관련
    if args.distributed:
        model = prepare_for_distributed(args, model) 
    else:
        model.cuda()
    # index 관련
    if args.index_name:
        if os.path.exists(args.index_name):
            index = DenseFlatIndexer(args.buffer_size)
            index.load(args.index_name)
    else:
        db_ids, passage_embeddings = load_passage_vectors_db_ids(args.n_shards, args.passage_path)
        if args.with_faiss:
            index = DenseFlatIndexer(args.buffer_size)
            index.init_index(args.vector_dim)
            index.index_data(passage_vectors, db_ids)
        else:
            index = None
    inference(args, model, tokenizer, index, db_ids, passage_embeddings)
    
        
if __name__ == '__main__':
    main()
