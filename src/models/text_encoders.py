from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from typing import List


CXR_BERT_URL = 'microsoft/BiomedVLP-CXR-BERT-specialized'


def get_cxr_bert_tokenizer_and_encoder():
    tokenizer = AutoTokenizer.from_pretrained(CXR_BERT_URL, trust_remote_code=True)
    model = AutoModel.from_pretrained(CXR_BERT_URL, trust_remote_code=True)
    model.train(False)

    return tokenizer, model


def get_text_embeddings(input: List[str], tokenizer: AutoTokenizer, model: AutoModel,
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
                                                   max_length=max_pad_len,
                                                   truncation=True,
                                                   return_tensors='pt')
    # embeddings = model(input_ids=tokenizer_output.input_ids,
    #                    attention_mask=tokenizer_output.attention_mask,
    #                    output_cls_projected_embedding=False,
    #                    return_dict=False)[0]
    # embeddings = F.normalize(embeddings, dim=1)

    return torch.stack((tokenizer_output.input_ids, tokenizer_output.attention_mask), -1)
