python3 retrieve.py --passage_path ./output/children --model_path ./output/children/best_model --batch_size 32 --output_dir ./output/children --with_faiss False --distributed False --model bert --pool cls --shared False --question_max_length 64 --include_history True --n_shards 1 --k 100 --test_data ./data/children/test_data_retriever.jsonl