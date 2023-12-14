import os
import json
from sklearn.model_selection import train_test_split
import glob
import re

def create_json(texts_root, output_file):
    data_index = []

    for root, _, files in os.walk(texts_root):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                data_index.append(path)
                
            

    with open(output_file, 'w') as file:
        json.dump(data_index, file, indent=4)





