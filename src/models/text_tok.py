# code was edited, but ideas from:
# https://github.com/spro/char-rnn.pytorch,
# and Dr.Colin Rafel HW6 Deep Learning Assignment (Unviversity of Tornto, Fall2023)

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


def create_mappings(text):
    """
    Create the mappings.
    """
    idx_to_char = list(set(text))
    # Maps character to token index
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    return char_to_idx

def tokenize(text, char_to_idx):
    """
    Tokenize the dataset

    """
    return [char_to_idx[char] for char in text]

def process_and_tokenize(filename):
    """
    Process the file and tokenize it

    """
    text = read_file(filename)
    char_to_idx = create_mappings(text)
    corpus_indices = tokenize(text, char_to_idx)
    return corpus_indices, char_to_idx



# sample usage: MIMICXR data
filename = '../data/s50578979.txt'
corpus_indices, char_to_idx = process_and_tokenize(filename)
print(corpus_indices) 