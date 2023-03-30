from transformers import AutoModel, AutoTokenizer, AutoConfig, BertModel, T5EncoderModel

def get_back_bone_model(model_name, further_train:bool=False):
    # back bone model load
    if model_name == 'bert':
        if further_train:
            model = None
        else:
            model = AutoModel.from_pretrained("klue/bert-base")
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        config = AutoConfig.from_pretrained('klue/bert-base')
        model_type = BertModel
    
    elif model_name == 't5':
        if further_train:
            model = None
        else:
            model = T5EncoderModel.from_pretrained("klue/bert-base")
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        config = AutoConfig.from_pretrained('klue/bert-base')
        model_type = T5EncoderModel
    return tokenizer, config, model, model_type