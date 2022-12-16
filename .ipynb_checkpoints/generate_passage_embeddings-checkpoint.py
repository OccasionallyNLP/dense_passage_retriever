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
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertConfig, BertModel
import argparse
from retrieval.dense_retrieval.model import *
#from retrieval.dense_retrieval.dense_retrieval import *
from utils.model import get_back_bone_model
from utils.data_utils import *
from utils.utils import *
from utils.tools import str2bool
from utils.distribute_utils import prepare_for_distributed, distributed_load_data, get_global, gather_tensors
import math

def get_args():
    # parser
    parser = argparse.ArgumentParser()

    # index name
    parser.add_argument('--shard_id', type=int, default = 0) 
    parser.add_argument('--n_shards', type=int, default = 1) 
    # data
    parser.add_argument('--passage_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--output_dir', type=str)

    # distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--distributed', type=str2bool, default = False)

    # model 관련
    parser.add_argument('--model', type=str, default = 'bert', choices = ['bert','roberta'], help = 'back bone model')
    parser.add_argument('--pool', type=str, default = 'cls', choices = ['cls','mean'], help = 'sentence representation') # second option 가능
    parser.add_argument('--shared', type= str2bool, default = False, help = 'share query encoder and passage encoder')
    #parser.add_argument('--include_history', type= str2bool, default = False)

    # 데이터 관련
    parser.add_argument('--passage_max_length',type= int)
    parser.add_argument('--contain_title', type=str2bool, default = True)
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

# generate passage embedding for each shard.
def generate_passage_embeddings(args, model, shard_passages, tokenizer, shard_id, data):
    # shard 개수 만큼 data를 불러오고 나서 distributed 진행하면 될 듯.
    shard_passages = load_data(shard_passages, args.local_rank, args.distributed)
    passage_dataset = DprPassageDataset(args, shard_passages, tokenizer)
    passage_sampler = SequentialSampler(passage_dataset)
    passage_dataloader = DataLoader(passage_dataset, batch_size = args.batch_size, sampler = passage_sampler, collate_fn = passage_dataset._collate_fn)
    iter_bar = tqdm(passage_dataloader, desc='step', disable=args.local_rank not in [-1,0])
    model.eval()
    db_indices = []
    passage_embeddings = []
    with torch.no_grad():
        for data in iter_bar:
            labels = data['labels']
            db_indices.extend(labels)
            data = {i:j.cuda() for i,j in data.items() if i!='labels'}
            if args.distributed:
                out = model.module.passage_encoder(**data).cpu().tolist()
            else:
                out = model.passage_encoder(**data).cpu().tolist()
            passage_embeddings.extend(out)
    if args.distributed:
        db_indices_ = gather_tensors(np.array(db_indices))
        passage_embeddings_ = gather_tensors( np.array(passage_embeddings))
        if args.local_rank == 0:
            total = []
            for i,j in zip(db_indices_, passage_embeddings_):
                for a,b in zip(i,j):
                    total.append((int(a),b))
            total = sorted(total, key = lambda i : i[0])
            with open(os.path.join(args.output_dir,'passage_embeddings_%d')%shard_id,'wb') as f:
                pickle.dump(total, f)
    else:
        total = [(int(i),j) for i,j in zip(db_indices, passage_embeddings)]
        with open(os.path.join(args.output_dir,'passage_embeddings_%d')%shard_id,'wb') as f:
            pickle.dump(total, f)
        
def main():
    args = get_args()
    seed_everything(42)
    # sanity check
    os.makedirs(args.output_dir, exist_ok = True)
    if args.local_rank in [-1,0]:
        logger = log()
    tokenizer, config, model, model_type = get_back_bone_model(args)
    # passage encoder
    model = prepare_model(config, args, model_type)
    # distributed 관련
    if args.distributed:
        model = prepare_for_distributed(args, model) 
    else:
        model.cuda()
    
    passages = json.load(open(args.passage_path,'rb'))
    # passages = make_index(passages)
    for i,j in passages.items():
        j['_id']=i
    shard_size = math.ceil(len(passages) / args.n_shards)
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    shard_passages = []
    keys = list(passages.keys())[start_idx:end_idx]
    for i in keys:
        shard_passages.append(passages[i])
    shard_passages = distributed_load_data(shard_passages, args.local_rank, args.distributed)
    generate_passage_embeddings(args, model, shard_passages, tokenizer, args.shard_id, shard_passages)
    
if __name__ == '__main__':
    main()
    
