from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


CXR_BERT_URL = 'microsoft/BiomedVLP-CXR-BERT-specialized'


def get_cxr_bert_tokenizer_and_encoder():
    tokenizer = AutoTokenizer.from_pretrained(CXR_BERT_URL, trust_remote_code=True)
    model = AutoModel.from_pretrained(CXR_BERT_URL, trust_remote_code=True)
    model.train(False)

    return tokenizer, model


def get_text_embeddings(input: torch.Tensor, tokenizer: AutoTokenizer, model: AutoModel) -> torch.Tensor:
    """Wrapper around CXR-BERT's get_projected_embeddings() function to retrieve text embeddings.

    Args:
        tokenizer: Text tokenizer
        model: CXR BERT Model

    Returns:
        Tensor of L2-normalized input text embeddings
    """
    tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=input,
                                                   add_special_tokens=True,
                                                   padding='longest',
                                                   return_tensors='pt')
    raw_embeddings = model(input_ids=tokenizer_output.input_ids,
                           attention_mask=tokenizer_output.attention_mask,
                           output_cls_projected_embedding=True,
                           return_dict=False)[2]
    embeddings = F.normalize(raw_embeddings, dim=1)

    return embeddings
