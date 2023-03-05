from transformers import AutoModel, AutoTokenizer, BertModel, AutoConfig
import torch
# utils.model
def get_back_bone_model(args):
    # back bone model load
    if args.model == 'bert':
        model = AutoModel.from_pretrained("klue/bert-base")
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        config = AutoConfig.from_pretrained('klue/bert-base')
    if 'bert' in args.model:
        model_type = BertModel
    # if args.include_history:
    #     tokenizer.add_special_tokens(dict(additional_special_tokens=['<wizard>','<apperentice>','<query>','<history>','<knowledge>','<context>','<title>']))
    #     model.resize_token_embeddings(len(tokenizer))
    #     config.vocab_size = len(tokenizer)
    return tokenizer, config, model, model_type