from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pickle
import json
from tqdm import tqdm
import pickle
import itertools

CXR_BERT_URL = 'microsoft/BiomedVLP-CXR-BERT-specialized'


def get_cxr_bert_tokenizer_and_encoder():
    tokenizer = AutoTokenizer.from_pretrained(CXR_BERT_URL, trust_remote_code=True)
    model = AutoModel.from_pretrained(CXR_BERT_URL, trust_remote_code=True)
    model.train(False)

    return tokenizer, model


def get_text_embeddings(input: torch.Tensor, tokenizer: AutoTokenizer, model: AutoModel,
                        max_pad_len: int = 256) -> torch.Tensor:
    """Wrapper around CXR-BERT's get_projected_embeddings() function to retrieve text embeddings.

    Args:
        tokenizer: Text tokenizer
        model: CXR BERT Model
        max_pad_len: Maximum length to pad each sequence

    Returns:
        Tensor of L2-normalized input text embeddings
    """
    tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=input,
                                                   padding='max_length',
                                                   truncation=True,
                                                   max_length=max_pad_len,
                                                   return_tensors='pt')
    # the tokenized text
    input_ids = tokenizer_output.input_ids
    # because we did padding (0's for padding)
    attention_mask = tokenizer_output.attention_mask

    stacked_tensor = torch.stack((input_ids, attention_mask), dim=2)
    return stacked_tensor





if __name__ == "__main__":

    tokenizer, model = get_cxr_bert_tokenizer_and_encoder()


    with open("data/p10_train_text_clean.json", "r") as file:
        text_dict = json.load(file)

    for dicom in list(text_dict.items()):
        stacked_tensor = get_text_embeddings([text_dict[dicom[0]]], tokenizer, model)
        text_dict[dicom[0]] = stacked_tensor
        #print(stacked_tensor.size())

    with open("data/p10_embeded_dicom.pkl", "wb") as file:
        pickle.dump(text_dict, file)


    



    
    # train texts


