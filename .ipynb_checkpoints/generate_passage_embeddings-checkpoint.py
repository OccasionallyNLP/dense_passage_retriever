# -*- coding: utf-8 -*-
import os
import json
import pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertConfig, BertModel
import argparse
from utils.utils import *
from utils.get_models import get_back_bone_model
from retrieval.dense_retrieval.model import *
import math

def get_args():
    # parser
    parser = argparse.ArgumentParser()

    # index name
    parser.add_argument('--shard_id', type=int, default = 0) 
    parser.add_argument('--n_shards', type=int, default = 1) 
    
    # data
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--check_point_dir', type=str)
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--output_dir', type=str)

    # 데이터 관련
    parser.add_argument('--max_length',type= int, default = 512)
    
    # distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    args = parser.parse_args()
    return args

def generate_embeddings(data, model, tokenizer, batch_size, device, max_length=None):
    result = []
    model.eval()
    with torch.no_grad():
        for i,batch_start in enumerate(tqdm(range(0, len(data), batch_size))):
            batch = data[batch_start:batch_start+batch_size]
            batch_doc_id = [j['doc_id'] for j in batch]
            batch_text = [j['tc'] for j in batch]
            model_input = tokenizer(batch_text, padding=True, truncation=True, max_length = max_length, return_tensors='pt').to(device)
            output = model(**model_input)
            output = output.cpu()
            result.extend([batch_doc_id[i], output[i].numpy()] for i in range(len(batch_doc_id)))
    return result

def prepare_model(config, pool, model_type):
    passage_encoder = Encoder(config, pool, model_type)
    # question encoder
    question_encoder = Encoder(config, pool, model_type)
    model = DprEncoder(passage_encoder, question_encoder)
    model.load_state_dict(torch.load(args.model_state_dict, map_location = 'cpu'))
    return model

def main():
    args = get_args()
    seed_everything(42)
    # sanity check
    os.makedirs(args.output_dir, exist_ok = True)
    
    with open(os.path.join(args.check_point_dir,'args.txt'), 'r') as f:
        check_point_args = json.load(f)
    
    ############################## data ##################################
    data = load_jsonl(args.data_path)
    shard_size = int(len(data) / args.n_shards)
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    data = data[start_idx:end_idx]
    ######################################################################
    
    ############################## model #################################
    ############################ model #######################################################
    tokenizer, config, model, model_type = get_back_bone_model(check_point_args['model'])
    
    # passage encoder
    encoder_p = Encoder(config, check_point_args['pool'], model_type)
    # question encoder
    encoder_q = Encoder(config, check_point_args['pool'], model_type)
    retriever_model = DprEncoder(encoder_p, encoder_q)
    retriever_model.load_state_dict(torch.load(os.path.join(os.path.join(args.check_point_dir,'best_model')), map_location='cpu'))
    model = retriever_model.passage_encoder
    # single node
    if args.local_rank == -1:  # single-node multi-gpu (or cpu) mode
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    # multi gpu
    else:
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device) 
        model.to(device)
    ######################################################################
    
    ############################## generate ##############################
    embeddings = generate_embeddings(data, model, tokenizer, args.batch_size, device, args.max_length)
    ######################################################################
    
    ############################## save ##################################
    file_name = os.path.join(args.output_dir,f'embedding_{args.shard_id}.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(embeddings, f)
    ######################################################################
    
if __name__ == '__main__':
    main()
    