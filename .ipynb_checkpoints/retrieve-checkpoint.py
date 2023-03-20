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

def get_args():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--passage_embedding_path', type=str)
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
    data = pickle.load(open(path,'rb'))
    db_ids = []
    passage_embeddings = []
    for i in data:
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
    return positive_ctxs_ids, retrieved_ctxs_ids

def main():
    args = get_args()
    seed_everything(42)
    # sanity check
    os.makedirs(args.output_dir, exist_ok = True)
    
    with open(os.path.join(args.check_point_dir,'args.txt'), 'r') as f:
        check_point_args = json.load(f)
    
    ############################ model #######################################################
    tokenizer, config, model, model_type = get_back_bone_model(check_point_args['model'])
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
    positive_ctxs_ids, retrieved_ctxs_ids = inference(test_data, model, tokenizer, args.batch_size, device, db_ids, passage_embeddings, max_length=args.max_length, top_k=args.top_k)
    #########################################################################################
#     if args.distributed:
#         positive_ctxs_ids_ = gather_tensors(np.array(positive_ctxs_ids))
        
#         total_positive_ctxs_ids = []
#         for i in positive_ctxs_ids_:
#             total_positive_ctxs_ids.extend(i)
#         retrieved_ctxs_ids_ = gather_tensors(np.array(retrieved_ctxs_ids))
#         total_retrieved_ctxs_ids = []
#         for i in retrieved_ctxs_ids_:
#             total_retrieved_ctxs_ids.extend(i)
            
#         if args.local_rank == 0:
#             scores = compute_topk_accuracy(total_positive_ctxs_ids, total_retrieved_ctxs_ids)
#             print(np.round(scores, 2))
#             with open(os.path.join(args.output_dir,'result.txt'),'w') as f:
#                 f.write(str(np.round(scores, 2)))
#             # tagging
#             test_data = load_data(args.test_data, local_rank = 0, distributed = False)
#             for i in test_data:
#                 i['predicted_ctxs_ids']=total_predicts[i['_id']].tolist()
#             save_jsonl(args.output_dir,test_data,'attached')
#     else:
#         if actuals:
#             scores = compute_topk_accuracy(actuals, predicts)
#             print(np.round(scores, 2))
#             with open(os.path.join(args.output_dir,'result.txt'),'w') as f:
#                 f.write(str(np.round(scores, 2)))
#         # tagging
#         test_data = load_data(args.test_data, local_rank = 0, distributed = False)
#         for i in test_data:
#             i['predicted_ctxs_ids']=predicts[i['_id']]
#         save_jsonl(args.output_dir,test_data,'attached')
    acc = compute_topk_accuracy(positive_ctxs_ids, retrieved_ctxs_ids)
    print(np.round(acc, 3))
    with open(os.path.join(args.output_dir,'result.txt'),'w') as f:
        f.write(str(np.round(acc, 3)))
    # for i,j in zip(test_data, retrieved_ctxs_ids):
    #     i['predicted_ctxs_ids']=total_predicts[i['_id']].tolist()
    #     save_jsonl(args.output_dir,test_data,'attached')
        
if __name__ == '__main__':
    main()    