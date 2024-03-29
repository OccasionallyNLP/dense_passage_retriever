# -*- coding: utf-8 -*-
import os
import json
import pickle
from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import logging
from transformers import AutoConfig, AutoModel, AutoTokenizer
import argparse
from tqdm import tqdm

from utils.data_utils import *
from utils.metrics import compute_topk_accuracy
from utils.distributed_utils import *
from utils.utils import *
from utils.get_models import get_back_bone_model

from retrieval.dense_retrieval.model import *

# parser
def get_args():
    parser = argparse.ArgumentParser()
    # test name
    parser.add_argument('--test_name', type=str)
    # data
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--val_data', type=str)
    parser.add_argument('--n_hard_negative_ctxs', type=int, default=0)
    parser.add_argument('--output_dir', type=str)
    # logging 관련
    parser.add_argument('--logging_term', type=int, default = 1000)
    # 학습 관련
    parser.add_argument('--epochs', type=int, default = 20)
    parser.add_argument('--eval_epoch', type = int, default = 1, help = 'term of evaluation')
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--lr', type=float, default = 2e-5)
    parser.add_argument('--warmup', type=float, default = 1000)
    parser.add_argument('--decay', type=float, default = 0.05)
    parser.add_argument('--fp16', type=str2bool, default = True)

    # model 관련
    parser.add_argument('--model', type=str, default = 'bert', help = 'back bone model')
    parser.add_argument('--pool', type=str, default = 'cls', choices = ['cls','mean'], help = 'sentence representation') # second option 가능
    parser.add_argument('--shared', type= str2bool, default = False, help = 'share query encoder and passage encoder')

    # further train
    # model path
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--further_train', type=str2bool, default = False)
    
    # 데이터 관련
    parser.add_argument('--passage_max_length',type= int, default = 512)
    parser.add_argument('--question_max_length', type=int, default = 64)
    parser.add_argument('--contain_title', type=str2bool, default = True)
    
    # distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--distributed', type=str2bool, default = False)
    
    # early stop 관련
    parser.add_argument('--early_stop', type=str2bool, default = True)
    parser.add_argument('--patience', type=int, default = 3)
    args = parser.parse_args()
    return args

def evaluation(args, model, tokenizer, eval_dataloader):
    model.eval()
    Val_loss = 0
    preds = []
    actuals = []
    with torch.no_grad():
        for data in tqdm(eval_dataloader, desc = 'evaluate', disable =  args.local_rank not in [-1,0]):
            data.to('cuda')
            _,_,score,loss = model.forward(**data) #
            pred = score.argmax(dim=-1) 
            preds.extend(pred.cpu().tolist())
            actuals.extend(data['labels'].cpu().tolist())
            Val_loss+=loss.item()
        Val_Acc = (np.array(actuals)==np.array(preds)).sum()
        cnt = len(actuals)
    return dict(Loss=Val_loss/len(eval_dataloader), cnt=cnt, acc=Val_Acc)

def get_scores(scores):
    if args.distributed:
        cnt = sum([j.item() for j in get_global(args, torch.tensor([scores['cnt']]).cuda())])
        acc = sum([j.item() for j in get_global(args, torch.tensor([scores['acc']]).cuda())])/cnt
        total_loss = [j.item() for j in get_global(args, torch.tensor([scores['Loss']]).cuda())]
        total_loss = sum(total_loss)/len(total_loss) 
    else:
        acc = scores['acc']/scores['cnt']
        total_loss = scores['Loss']
    return dict(Loss=np.round(total_loss,3), acc=np.round(acc,3))

def train():
    optimizer_grouped_parameters = make_optimizer_group(model, args.decay)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.decay)
    scheduler = get_linear_scheduler(len(train_dataloader)*args.epochs, args.warmup, optimizer, train_dataloader)
    
    if args.fp16:
        scaler = GradScaler()
    global_step = 0
    if args.distributed:
        flag_tensor = torch.zeros(1).to(args.local_rank)
    else:
        flag_tensor = torch.zeros(1).to(0)
    early_stop = EarlyStopping(args.patience, args.output_dir, max = False, min_difference=1e-5)
    
    for epoch in range(1, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        Loss = 0.
        c = 0
        iter_bar = tqdm(train_dataloader, desc='step', disable=args.local_rank not in [-1,0])
        for data in iter_bar:
            optimizer.zero_grad()
            data.to('cuda')
            if args.fp16:
                with autocast():
                    _,_,_,loss = model.forward(**data)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                _,_,_,loss = model.forward(**data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            
            global_step+=1           
            c+=1
            scheduler.step()
            
            if args.distributed:
                Loss_i = loss.mean().item()
            else:
                Loss_i = loss.item()
            Loss+=Loss_i
            iter_bar.set_postfix({'epoch':epoch, 'global_step':global_step, 'lr':f"{scheduler.get_last_lr()[0]:.5f}", 'last_loss':f'{loss.item():.5f}','epoch_loss':f'{Loss/c:.5f}'})
            
            if global_step%args.logging_term == 0:
                if args.local_rank in [-1,0]:
                    logger1.info(iter_bar)
                    logger2.info(iter_bar)
                    
        
        if args.local_rank in [-1,0]:
            logger1.info(f'epoch : {epoch} ----- Train_Loss : {Loss/len(train_dataloader):.5f}')
            logger2.info(f'epoch : {epoch} ----- Train_Loss : {Loss/len(train_dataloader):.5f}')
        
        if args.eval_epoch:
            if epoch%args.eval_epoch==0:
                # TODO
                scores_ = evaluation(args, model, tokenizer, val_dataloader)
                scores = get_scores(scores_)
                
                if args.local_rank in [-1,0]:
                    logger1.info(f'epoch : {epoch} ----- {scores}')
                    logger2.info(f'epoch : {epoch} ----- {scores}')
                    model_to_save = model.module if args.distributed else model
                    early_stop.check(model_to_save, scores['Loss'])
                    if args.early_stop:
                        if early_stop.timetobreak:
                            flag_tensor += 1
                            if args.distributed:
                                torch.distributed.broadcast(flag_tensor, 0)
                if args.distributed:
                    torch.distributed.barrier()          
                        
                     # 저장시 - gpu 0번 것만 저장 - barrier 필수
        if flag_tensor:
            logger1.info('early stop')
            logger2.info('early stop')
            break
    # 저장시 - gpu 0번 것만 저장 - barrier 필수
    if args.local_rank in [-1,0]:
        torch.save(early_stop.best_model, os.path.join(early_stop.save_dir,'best_model'))
        logger1.info('train_end')
        logger2.info('train end')
    
    

 #########################################################################################
if __name__=='__main__':
    args = get_args()
    if args.model_path is not None:
        args.further_train = True
    if args.local_rank in [-1,0]:
        with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    seed_everything(42)
    if args.local_rank in [-1,0]:
        logger1, logger2 = get_log(args)
        logger1.info(args)
        logger2.info(args)

    # sanity check
    os.makedirs(args.output_dir, exist_ok = True)
    ############################ model #######################################################
    tokenizer, config, model, model_type = get_back_bone_model(args.model, args.further_train)
    
    # passage encoder
    encoder_p = Encoder(config, args.pool, model_type)
    # question encoder
    encoder_q = Encoder(config, args.pool, model_type)
    
    if not args.further_train:
    # initiate
        encoder_p.init_pretrained_model(model.state_dict())
        encoder_q.init_pretrained_model(model.state_dict())
    if args.shared:
        model = DprEncoder(encoder_p, encoder_p)
    else:
        model = DprEncoder(encoder_p, encoder_q)
    if args.further_train:
        model.load_state_dict(torch.load(args.model_path))
    # distributed 관련
    if args.distributed:
        model = prepare_for_distributed(args, model)
    else:
        model.cuda()
    #########################################################################################
    
    ############################ data #########################################################
    # data
    train_data = load_jsonl(args.train_data)
    train_dataset = DprTrainDataset(args, train_data, tokenizer)
    train_sampler = DistributedSampler(train_dataset) if args.distributed else RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, sampler = train_sampler, collate_fn = train_dataset._collate_fn)
    val_data = load_data(args.val_data, local_rank = args.local_rank, distributed = args.distributed)
    # val
    val_dataset = DprTrainDataset(args, val_data, tokenizer)# if not args.include_history else DprTrainDatasetWithHistory(args, train_data, tokenizer)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, collate_fn=val_dataset._collate_fn, sampler = val_sampler)
    ##########################################################################################
    
    ################################ train ##################################################
    train()
   
    
    
