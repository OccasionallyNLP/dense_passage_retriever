python3 generate_passage_embeddings.py --shard_id 0 --n_shards 1 --passage_path ./data/children/contexts.pkl  --model_path ./output/children/best_model --batch_size 16 --output_dir ./output/children --distributed False --local_rank 0 --model bert --pool cls --shared False --include_history True --passage_max_length 200 --contain_title True