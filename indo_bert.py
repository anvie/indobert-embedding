
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import numpy as np

tokenizer = None
model = None

FIRST_INIT = False

def _ensure_initialized():
    global FIRST_INIT, tokenizer, model
    if FIRST_INIT:
        return
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
    model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
    model.eval()
    FIRST_INIT = True

def get_embedding(text:str):
    _ensure_initialized()
    text_input = f"[CLS] {text.lower()} [SEP]"

    tokenized_text = tokenizer.tokenize(text_input)
    segments = [1] * len(tokenized_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    logging.info("indexed_tokens: %s", indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments])

    with torch.no_grad():
        encoded_layers = model(tokens_tensor, segments_tensors)
    
    last_hidden_state = encoded_layers.last_hidden_state
    
    token_embeddings = torch.stack([last_hidden_state])
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # swap dimensions
    token_embeddings = token_embeddings.permute(1,0,2)

    logging.info("token_embedding.size = {}".format(token_embeddings.size()))

    token_vecs = last_hidden_state[0]

    sentence_embedding = torch.mean(token_vecs, dim=0)

    return [tensor.item() for tensor in sentence_embedding]


def text_similarity(text1, text2):
    embedding1 = np.array(get_embedding(text1)).reshape(1, -1)
    embedding2 = np.array(get_embedding(text2)).reshape(1, -1)
    return torch.cosine_similarity(torch.tensor(embedding1), torch.tensor(embedding2)).item()



    