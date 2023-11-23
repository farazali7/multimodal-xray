import os
import json
from sklearn.model_selection import train_test_split
import glob
import re

def create_json(images_root, texts_root, output_file):
    data_index = {'images': [],'labels':[] ,'texts': [], 'patient_id': []}
    patient_to_image = {}
    patient_to_text = {}

    study = []
    for root, _, files in os.walk(images_root):
        for file in files:
            if file.endswith('.jpg'):
                path = os.path.join(root, file)
                patient_id = path.split(os.sep)[-3]  
                study_id = path.split(os.sep)[-2]
                if study_id in study:
                    break
                else:
                    study.append(study_id)

                
                if 'p' in patient_id and len(patient_id) > 3:  # Check if its valid ID for patient
                    patient_to_image.setdefault(patient_id, []).append(path)
                    

    for root, _, files in os.walk(texts_root):
        
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                patient_id = path.split(os.sep)[-2].split('.')[0]
                if patient_id in patient_to_image:  
                    patient_to_text.setdefault(patient_id, []).append(path)

    for patient_id in patient_to_image.keys():
        data_index['patient_id'].append(patient_id)
        data_index['images'].append(patient_to_image[patient_id])
        data_index['texts'].append(patient_to_text.get(patient_id, []))  
    #data_index['labels'] = data_index['images'].copy()

    # get the text from the files

    text_contents = []
    for text_file_path in data_index['texts']:
        text_contents.append([])
        for text in text_file_path:
            try:
                with open(text, 'r') as text_file:
                    text = text_file.read().split(":")[-1]
                    text = ''.join(re.sub('[^A-Za-z ]+', '', text).lower().strip())
                    text_contents[-1].append(text)
            except FileNotFoundError:
                print(f"File not found: {text}")

    data_index['texts'] = text_contents


    with open(output_file, 'w') as file:
        json.dump(data_index, file, indent=4)

images_root = '../data/downloaded_jpgs/physionet.org/files/mimic-cxr-jpg'
texts_root = '../data/downloaded_reports/physionet.org/files/mimic-cxr'
output_file = 'all_data3.json'

create_json(images_root, texts_root, output_file)


# train - validation split 


with open('all_data3.json', 'r') as file:
    data = json.load(file)

# make the text go into the list



# split by the index of the patient-id (same as index for img and text)
indices = list(range(len(data['patient_id'])))

train_indices, val_indices = train_test_split(indices, test_size=0.2)

def get_items_by_indices(items_list, indices):
    return [items_list[i] for i in indices]

train_data = {
    'images': get_items_by_indices(data['images'], train_indices),
    'texts': get_items_by_indices(data['texts'], train_indices),
    'labels': []
}

imgs = []
txts = []

for patient_img in train_data["images"]:   
    imgs.extend(patient_img)
train_data["images"] = imgs

train_data["labels"] = imgs

for patient_txt in train_data["texts"]:
    txts.extend(patient_txt)
train_data["texts"] = txts


val_data = {
    'images': get_items_by_indices(data['images'], val_indices),
    'texts': get_items_by_indices(data['texts'], val_indices),
    'labels': []
}

imgs = []
txts = []

for patient_img in val_data["images"]:
    imgs.extend(patient_img)
val_data["images"] = imgs

for patient_txt in val_data["texts"]:
    txts.extend(patient_txt)
val_data["texts"] = txts

val_data["labels"] = imgs


with open('train.json', 'w') as file:
    json.dump(train_data, file, indent=4)

with open('val.json', 'w') as file:
    json.dump(val_data, file, indent=4)


