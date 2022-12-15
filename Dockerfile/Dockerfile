FROM huggingface/transformers-pytorch-gpu
WORKDIR /dense_passage_retriever
COPY  ./requirements.txt .
RUN pip install --upgrade pip && pip install -r ./requirements.txt 
