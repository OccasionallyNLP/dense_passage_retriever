import numpy as np
from typing import List

def compute_topk_accuracy(answers:List[int], candidates:List[List[int]])->List[float]:
    # answers : 정답지 - 문서의 id로 구성됨.
    # candidates : 2 dimension List
    # shape : (N, k) <- 즉 k개의 정답지에 대한 id를 지니고 있음.
    answers = np.array(answers).reshape(-1,1)
    candidates = np.array(candidates)
    N,k = candidates.shape
    
    acc_score = []
    for j in range(1,k+1):
        a = np.sum(np.sum(candidates[:,:j] == answers, axis=-1)>=1)/N
        acc_score.append(a)
    return acc_score

def compute_topk_precision(answers:List[int], candidates:List[List[int]])->float:
    # answers : 정답지 - 문서의 id로 구성됨.
    # candidates : 2 dimension List
    # shape : (N, k) <- 즉 k개의 정답지에 대한 id를 지니고 있음.
    answers = np.array(answers).reshape(-1,1)
    candidates = np.array(candidates)
    N,k = candidates.shape
    precision_score = []
    for j in range(1,k+1):
        a = (np.sum(candidates[:,:j] == answers, axis=-1))/j
        precision_score.append(np.sum(a)/N)
    return precision_score

## MRR - Mean Reciprocal Rank
## 1/|Q|*sum(1/RANK_i)
def compute_MRR_K(answers:List[int], candidates:List[List[int]],K:int)->float:
    # answers : 정답지 - 문서의 id로 구성됨.
    # candidates : 2 dimension List
    # shape : (N, k) <- 즉 k개의 정답지에 대한 id를 지니고 있음.
    answers = np.array(answers).reshape(-1,1)
    candidates = np.array(candidates)
    candidates = candidates[:,:K]
    N,k = candidates.shape
    assert K<=k
        
    a = (candidates == answers)
    hit = a.sum(axis=-1)>=1
    rank = (np.argmax(a,axis=-1)+1).astype(np.float)
    #print(rank)
    reciprocal = np.reciprocal(rank)*hit
    #print(reciprocal)
    #print(hit)    
    return np.sum(reciprocal)/N