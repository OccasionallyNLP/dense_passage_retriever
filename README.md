# Dense Passage Retrieval for Open-Domain Question Answering - 2020

# 데이터
KorQuAD 1.0으로 Question Answering data로 선정.
contexts로는 2019.02 wikipedia dump file(700MB)을 활용

# component
1. 전처리
    - wikipedia 전처리 
        - original article 개수 : 1,005,594
            - 어절 단위 50으로 article 본문을 쪼개서 passage로 만듦.
        - passage 개수 : 1,346,460
            klue bert tokenizer로 title과 context를 tokenize 했을 때 길이
            |token 수|min|median|95|99|99.9|max|
            |-|-|-|-|-|-|-|
            |-|3|116|155|188|256|1466|
         
             - 향후 passage embedding 만들 때 max length 기준을 300 수준으로 설정.  
         
         
    - korquad 전처리 : wiki version 문제, 어절 단위로 본문을 나누면서 소실되는 answer 등으로 인해 개수 감소.
        - original train 개수 : 60,407
        - original dev 개수 : 5,774
        - preprocessed train 개수 : 41,501 - 이를 다시, 
        - preprocessed dev 개수 : 4,612  
            - question embedding 만들 때 max length는 64 수준으로 설정.  
        
    - tydi 전처리  
    
    
2. passage encoder, question encoder 학습  
    - klue bert 활용
3. passage embedding 생성 및 indexing    
4. inference 

# 진행 사항.
1. 원 논문에서도 wikipedia version이 다름에 따라, 활용하고자 하는 QA 데이터와 context 데이터(wikipedia) 간의 version이 상이하면 데이터를 버렸음.
    - 나 역시도, 원 데이터 버전이 다르다면 이를 버림 (answer clue를 pivot으로 활용함) 
    
2. 논문 구현 - 논문 이해부터 구현까지 1달 정도 시간이 소요됨.(2021.10 ~ 2021.11)
    - 검색에 대해서 배경지식이 없던 터라, 다소 많은 시간이 소요됨.
    
3. in batch negative를 활용하기에, 동일한 batch 내에 같은 passage를 쓰는 경우는 batch에서 제외시킴
    - 이로 인해 batch마다 실제 batch 개수가 상이할 수 있음.
    - 원래는 Sampler에서 이를 변경하고자 했으나, 분산 학습까지 가면, 각각의 gpu에서 len(dataloader)가 동일해야 되는 문제가 있어서 위의 방법으로 진행.
    - TODO - distributed sampling에서의 접근

4. bm25로 hard negative 추가하는 코드 작성. 
- 2022.12.15
    - bm25 학습하는 데에 5분 정도 소요.
    - bm25 inference (dev data) 하는 데에 15시간 소요. 
    - hard negative 찾는 데에 150시간 소요.

5. 학습 시간 - epoch 당 5분. GPU: 3070

6. faiss 관련
    - 아직 적용 안함.

# 수정사항  

## 실험 결과
|no|backbone|data|설명|비고|
|-|-|-|-|-|
|1|bm25|korquad|sparse representation|-|
|2|klue bert|korquad|dense representation|with hard negative|
|3|klue bert|korquad|dense representation|-|

|no|R@1|R@5|R@20|R@100|
|-|-|-|-|-|
|1|0.57|0.78|0.87|0.93|
|2|0.17|0.34|0.51|0.68|
|3|||||

## 필요 library
wikiextractor
- https://github.com/attardi/wikiextractor
faiss
transformers

## hyperparameters
|no|name|comment|
|-|-|-|
|1|batch size|32|
|2|fp16|True|
|3|passage max length|256|
|4|question max length|64|
|5|optimizer|Adam|
|6|learning rate|2e-5|
|7|warm up strategy|linear till 200 step|
|8|back bone|klue bert|