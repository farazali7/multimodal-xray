from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json


CXR_BERT_URL = 'microsoft/BiomedVLP-CXR-BERT-specialized'


def get_cxr_bert_tokenizer_and_encoder(device):
    tokenizer = AutoTokenizer.from_pretrained(CXR_BERT_URL, trust_remote_code=True)
    model = AutoModel.from_pretrained(CXR_BERT_URL, trust_remote_code=True)
    model.to(device)
    model.train(False)

    return tokenizer, model


def get_text_embeddings(input: torch.Tensor, tokenizer: AutoTokenizer, model: AutoModel, device,
                        max_pad_len: int = 512) -> torch.Tensor:
    """Wrapper around CXR-BERT's get_projected_embeddings() function to retrieve text embeddings.

    Args:
        tokenizer: Text tokenizer
        model: CXR BERT Model
        max_pad_len: Maximum length to pad each sequence

    Returns:
        Tensor of L2-normalized input text embeddings
    """
    tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=input,
                                                   add_special_tokens=True,
                                                   padding='max_length',
                                                   truncation=True,
                                                   max_length=max_pad_len,
                                                   return_tensors='pt')
    # the tokenized text
    input_ids = tokenizer_output.input_ids.to(device)
    # because we did padding (0's for padding)
    attention_mask = tokenizer_output.attention_mask.to(device)

    with torch.no_grad():
        embeddings = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_cls_projected_embedding=False,
                        return_dict=False)[0]
    # embeddings = F.normalize(raw_embeddings, dim=1)

    return embeddings

def batch_embeddings(raw_inputs, device, batch_size=16):
    """
    creates a list of all of the embeddings, doing the embeddings a batch 
    so size <batch_size> at a time
    
    """
    tokenizer, model = get_cxr_bert_tokenizer_and_encoder(device)

    inputs_len = len(raw_inputs)
    total_batches = inputs_len // batch_size

    # tokenize in batches to save memory
    tokenized_data = []
    end_idx = 0
    for i in range(total_batches + 1):
        start_idx = i * batch_size
        if end_idx != inputs_len:
            end_idx = start_idx + batch_size
        else:
            end_idx = -1
        batch_text = raw_inputs[start_idx:end_idx]

        batch_embeddings = get_text_embeddings(batch_text, tokenizer, model, device)
        # the print statement it to check the status of the function
        #print(batch_embeddings.shape)

        tokenized_data.extend(batch_embeddings)
        #print(len(tokenized_data))

    return tokenized_data


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = {}
    with open("../../data_preprocess/train.json", 'r') as file:
        text = json.load(file)

    raw_inputs = text["texts"][:5900] # more than 5900 ish I get mem error
     #17590 reports for p10

    tokenized_data = batch_embeddings(raw_inputs, device)
    #print(tokenized_data[0])
    







    #outputs = get_text_embeddings(raw_inputs[:250], tokenizer, model)
    #print(outputs.shape)
    # torch.Size([3, 256, 768])
