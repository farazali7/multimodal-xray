from transformers import AutoTokenizer
import torch
import unidecode
import re

def read_file(filename):
    """
    Read the text file corresponding to radiology reports. 
    Normalize the text. 

    """
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    text = ' '.join(re.sub('[^A-Za-z ]+', '', text).lower().split())
    return unidecode.unidecode(text)

text = read_file('../data/physionet.org/files/mimic-cxr/2.0.0/files/p10/p10000032/s50414267.txt')

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")

encoded_input = tokenizer.encode(text, add_special_tokens=True)

tokens_tensor = torch.tensor([encoded_input])


print(tokens_tensor)
