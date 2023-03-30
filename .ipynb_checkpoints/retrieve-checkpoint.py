# -*- coding: utf-8 -*-
import os
import json
import torch
import pickle
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import AutoModel, AutoTokenizer, AutoConfig
import argparse
from retrieval.dense_retrieval.model import *
from utils.data_utils import *
from utils.utils import *
from utils.distributed_utils import *
from utils.get_models import get_back_bone_model
from utils.metrics import compute_topk_accuracy
from typing import List, Dict

def get_args():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--passage_embedding_path', type=str) # folder
    parser.add_argument('--check_point_dir', type=str)
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--local_rank', type=int, default = 0)
    parser.add_argument('--max_length', type=int, default = 64)
    parser.add_argument('--top_k', type = int, default = 100)
    parser.add_argument('--test_data', type = str)
    args = parser.parse_args()
    return args

def get_passage_embeddings_db_ids(path):
    data = []
    for i in os.listdir(path):
        if i.endswith('pkl'):
            cur_path = os.path.join(path,i)
            d = pickle.load(open(cur_path,'rb'))
            data.append(d)
    db_ids = []
    passage_embeddings = []
    for d in data:
        for i in d:
            idx, embedding = i
            db_ids.append(idx)
            passage_embeddings.append(embedding)
    return db_ids, passage_embeddings

def inference(data, model, tokenizer, batch_size, device, db_ids, passage_embeddings, max_length=None, top_k=100):
    passage_embeddings = torch.tensor(passage_embeddings) # n, dim
    passage_embeddings = passage_embeddings.T # dim, n
    retrieved_ctxs_ids = []
    positive_ctxs_ids = []
    model.eval()
    with torch.no_grad():
        for i,batch_start in enumerate(tqdm(range(0, len(data), batch_size))):
            batch = data[batch_start:batch_start+batch_size]
            batch_positive_ctxs_ids = [j['positive_ctxs_ids'] for j in batch]
            batch_text = [j['question'] for j in batch]
            model_input = tokenizer(batch_text, padding=True, truncation=True, max_length = max_length, return_tensors='pt').to(device)
            output = model(**model_input) # bs, dim
            output = output.cpu() # bs, dim
            # MIPS
            matmul = torch.matmul(output, passage_embeddings) # bs, n
            _, indices = torch.topk(matmul, k=top_k, dim=-1) # bs,
            indices = [[db_ids[j.item()] for j in i] for i in indices]
            retrieved_ctxs_ids.extend(indices) # indexs - bs, k
            positive_ctxs_ids.extend(batch_positive_ctxs_ids)
    for i,j in zip(data, retrieved_ctxs_ids):
        i['retrieved_ctxs_ids']=j
    return data, positive_ctxs_ids, retrieved_ctxs_ids

def hit(actual:List[int],predict:List[List[int]])->dict:
    from collections import defaultdict
    result = defaultdict(list)
    for i,j in zip(actual, predict):
        for k in range(1,101):
            result[k].append(i in j[:k])
    output = dict()
    for i,j in result.items():
        output[i]=sum(j)/len(j)
    print(output)
    return output

def main():
    args = get_args()
    seed_everything(42)
    # sanity check
    os.makedirs(args.output_dir, exist_ok = True)
    
    with open(os.path.join(args.check_point_dir,'args.txt'), 'r') as f:
        check_point_args = json.load(f)
    
    ############################ model #######################################################
    tokenizer, config, model, model_type = get_back_bone_model(check_point_args['model'], False)
    # passage encoder
    encoder_p = Encoder(config, check_point_args['pool'], model_type)
    # question encoder
    encoder_q = Encoder(config, check_point_args['pool'], model_type)
    retriever_model = DprEncoder(encoder_p, encoder_q)
    retriever_model.load_state_dict(torch.load(os.path.join(os.path.join(args.check_point_dir,'best_model')), map_location='cpu'))
    model = retriever_model.question_encoder
    # device
    # single node
    if args.local_rank == -1:  # single-node multi-gpu (or cpu) mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    # multi node - TODO
    else:
        device = torch.device("cuda", args.local_rank) 
        model.to(device)
    
    ######## TODO #####################################
    # distributed 관련
    # if args.distributed:
    #     model = prepare_for_distributed(args, model) 
    # else:
    #     model.cuda()
    #########################################################################################
    
    #########################  data #########################################################
    test_data = load_jsonl(args.test_data)
    db_ids, passage_embeddings = get_passage_embeddings_db_ids(args.passage_embedding_path)
    ##########################################################################################
    
    #########################  retrieve #########################################################
    test_data, positive_ctxs_ids, retrieved_ctxs_ids = inference(test_data, model, tokenizer, args.batch_size, device, db_ids, passage_embeddings, max_length=args.max_length, top_k=args.top_k)
    result = hit(positive_ctxs_ids, retrieved_ctxs_ids)
    print(result)
    pickle.dump(result, open(os.path.join(args.output_dir,'result.pkl'),'wb'))
    
        
if __name__ == '__main__':
    main()    