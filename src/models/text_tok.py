from transformers import AutoTokenizer, AutoModel
import torch


def txt_encoder():
    """
    Bert model for text 
    raw_inputs is the reports for one patients
    """
    checkpoint = "microsoft/BiomedVLP-CXR-BERT-general"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)



    
    return model, tokenizer

    
def get_text_embeddings(raw_inputs) -> torch.Tensor:
    """Retrieve text embeddings.

    """
    model, tok = txt_encoder()

    inputs = tok(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)


    return outputs


if __name__ == "__main__":
    raw_inputs = ["There is no pneumothorax or pleural effusion", "No pleural effusion or pneumothorax is seen", "The extent of the pleural effusion is constant."]

    outputs = get_text_embeddings(raw_inputs)
    print(outputs.last_hidden_state.shape)

    # #torch.Size([3, 11, 768])


