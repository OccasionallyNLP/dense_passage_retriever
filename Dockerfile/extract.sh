#docker build -t dpr .
docker run -it -v D:\jupyter_notebook/dense_passage_retriever:/dense_passage_retriever/ -p 8888:8888 --gpus all dpr /bin/bash
