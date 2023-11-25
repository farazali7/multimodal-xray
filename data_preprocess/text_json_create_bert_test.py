import os
import json
from sklearn.model_selection import train_test_split
import re

def create_json(texts_root, output_file):
    data_index = {'texts': [], 'patient_id': []}
    patient_to_text = {}

    for root, _, files in os.walk(texts_root):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                patient_id = path.split(os.sep)[-2].split('.')[0]
                patient_to_text.setdefault(patient_id, []).append(path)

    for patient_id, texts in patient_to_text.items():
        data_index['patient_id'].append(patient_id)
        text_contents = []
        for text_path in texts:
            try:
                with open(text_path, 'r') as text_file:
                    text = text_file.read().split(":")[-1]
                    text = ''.join(re.sub('[^A-Za-z ]+', '', text).lower().strip())
                    text_contents.append(text)
            except FileNotFoundError:
                print(f"File not found: {text_path}")
        data_index['texts'].append(text_contents)

    with open(output_file, 'w') as file:
        json.dump(data_index, file, indent=4)

texts_root = '../data/downloaded_reports/physionet.org/files/mimic-cxr'
output_file = 'all_data_text.json'

create_json(texts_root, output_file)

# train - validation split 

with open('all_data_text.json', 'r') as file:
    data = json.load(file)

# split by the index of the patient-id
indices = list(range(len(data['patient_id'])))
train_indices, val_indices = train_test_split(indices, test_size=0.2)

def get_items_by_indices(items_list, indices):
    return [items_list[i] for i in indices]



train_data = {
    'texts': get_items_by_indices(data['texts'], train_indices),
    'patient_id': get_items_by_indices(data['patient_id'], train_indices)
}
txts = []


for patient_txt in train_data["texts"]:
    txts.extend(patient_txt)
train_data["texts"] = txts

val_data = {
    'texts': get_items_by_indices(data['texts'], val_indices),
    'patient_id': get_items_by_indices(data['patient_id'], val_indices)
}

txts = []


for patient_txt in val_data["texts"]:
    txts.extend(patient_txt)
val_data["texts"] = txts

with open('train.json', 'w') as file:
    json.dump(train_data, file, indent=4)

with open('val.json', 'w') as file:
    json.dump(val_data, file, indent=4)
