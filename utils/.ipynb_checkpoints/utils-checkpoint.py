# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import logging
import os
import hashlib
from tqdm import tqdm
import json
import copy

def seed_everything(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def log():
    logger = logging.getLogger('stream') # 적지 않으면 root로 생성
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] >> %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger    
    
def get_log(args):
    #global logger1, logger2
    logger1 = logging.getLogger('file') # 적지 않으면 root로 생성
    logger2 = logging.getLogger('stream') # 적지 않으면 root로 생성

    # 2. logging level 지정 - 기본 level Warning
    logger1.setLevel(logging.INFO)
    logger2.setLevel(logging.INFO)

    # 3. logging formatting 설정 - 문자열 format과 유사 - 시간, logging 이름, level - messages
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] >> %(message)s')

    # 4. handler : log message를 지정된 대상으로 전달하는 역할.
    # SteamHandler : steam(terminal 같은 console 창)에 log message를 보냄
    # FileHandler : 특정 file에 log message를 보내 저장시킴.
    # handler 정의
    stream_handler = logging.StreamHandler()
    # handler에 format 지정
    stream_handler.setFormatter(formatter)
    # logger instance에 handler 삽입
    logger2.addHandler(stream_handler)
    os.makedirs(args.output_dir,exist_ok=True)
    if args.test_name is not None:
        file_handler = logging.FileHandler(os.path.join(args.output_dir,'%s.txt'%(args.test_name)), encoding='utf-8')
    else:
        file_handler = logging.FileHandler(os.path.join(args.output_dir,'report.txt'), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger1.addHandler(file_handler)
    return logger1, logger2

def save_jsonl(address,data,name):
    f = open(os.path.join(address,name+'.jsonl'),'w',encoding = 'utf-8')
    for i in data:
        f.write(json.dumps(i, ensure_ascii=False)+'\n')
        
def load_jsonl(path):
    result = []
    f = open(path,'r',encoding = 'utf-8')
    for i in tqdm(f):
        result.append(json.loads(i))
    return result 

def compute_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# for distributed
def make_index(data):
    for _,i in enumerate(data):
        i['_id']=_
    return data       

def load_data(data_path, local_rank, distributed, drop_last = True):
    data = load_jsonl(data_path)
    data = make_index(data)
    samples = []
    if distributed:
        world_size = torch.distributed.get_world_size()
        if drop_last:
            data = data[:len(data)//world_size*world_size] # drop last 효과
        else:
            num_samples = math.ceil(len(data)/world_size)
            total_size = num_samples*world_size
            padding_size = total_size - num_samples
            if padding_size <= len(data):
                data += data[:padding_size]
            else:
                data += (data*math.ceil(padding_size/len(data)))[:padding_size] 
        num_samples = math.ceil(len(data)/world_size)
        samples = data[local_rank:local_rank+num_samples]
        return samples
    return data

class EarlyStopping(object):
    def __init__(self, patience, save_dir, max = True, min_difference=1e-5):
        self.patience = patience
        self.min_difference = min_difference
        self.max = max
        self.score = -float('inf') if max else float('inf')
        self.best_model = None
        self.best_count = 0
        self.timetobreak = False
        self.save_dir = save_dir
    
    def check(self, model, calc_score):
        if self.max:
            if self.score-calc_score<self.min_difference:
                self.score = calc_score
                self.best_count = 0
                self.best_model = copy.deepcopy(model.state_dict())
            else:
                self.best_count+=1
                if self.best_count>=self.patience:
                    self.timetobreak=True
                    torch.save(self.best_model, os.path.join(self.save_dir,'best_model'))
        else:
            if self.score-calc_score>self.min_difference:
                self.score = calc_score
                self.best_count = 0
                self.best_model = copy.deepcopy(model.state_dict())
            else:
                self.best_count+=1
                if self.best_count>=self.patience:
                    self.timetobreak=True
                    torch.save(self.best_model, os.path.join(self.save_dir,'best_model'))