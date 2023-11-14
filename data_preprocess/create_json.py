import os
import json
import random
from sklearn.model_selection import train_test_split
import glob

# use for dataloader in train.py 
def create_json(images_root, texts_root, output_file):
    """
    Create JSON to store the paths to the imgs, texts, and also the patient study as the corresponding labels
    """
    data_index = {'images': [], 'texts': [], 'labels': []}

    for root, _, _ in os.walk(images_root):
        jpg_files = glob.glob(os.path.join(root, '*.jpg'))
        if jpg_files:
            data_index['images'].append(jpg_files)
            label = root.split('\\')[-1]
            data_index['labels'].append(label)



    
    for root, _, _ in os.walk(texts_root):
        txt_files = glob.glob(os.path.join(root, '*.txt'))

        for text in txt_files:
            if text:
                txt_label = text.split('\\')[-1].split('.')[-2]
                if txt_label in data_index['labels']:
                    data_index['texts'].append([])
                    data_index['texts'][-1].append(text)
        

#     # Write data to JSON file
    with open(output_file, 'w') as file:
        json.dump(data_index, file, indent=4)

images_root = '../data/physionet.org/files/mimic-cxr-jpg'
texts_root = '../data/physionet.org/files/mimic-cxr'
create_json(images_root, texts_root, 'all_data2.json')


# expected json:
# {   
#  "images": ["../data/../image1.jpg", ...], 
# "texts": ["../data/../report1.txt", ...],
# "labels": ["s56353", ...] 
#  }


# train - validation split 


with open('all_data2.json', 'r') as file:
    data = json.load(file)

# split by the index of the labels (same as index for img and text)
indices = list(range(len(data['labels'])))

train_indices, val_indices = train_test_split(indices, test_size=0.2)

def get_items_by_indices(items_list, indices):
    return [items_list[i] for i in indices]

train_data = {
    'images': get_items_by_indices(data['images'], train_indices),
    'texts': get_items_by_indices(data['texts'], train_indices),
    'labels': get_items_by_indices(data['labels'], train_indices)
}

val_data = {
    'images': get_items_by_indices(data['images'], val_indices),
    'texts': get_items_by_indices(data['texts'], val_indices),
    'labels': get_items_by_indices(data['labels'], val_indices)
}

with open('train.json', 'w') as file:
    json.dump(train_data, file, indent=4)

with open('val.json', 'w') as file:
    json.dump(val_data, file, indent=4)