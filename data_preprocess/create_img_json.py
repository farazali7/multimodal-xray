import os
import json
from sklearn.model_selection import train_test_split
import glob
import re
import more_itertools

# Helper
def create_json(images_root, output_file):
    data_index = {'images': [], 'patient_id': []}
    patient_to_image = {}

    study = []
    for root, _, files in os.walk(images_root):
        for file in files:
            if file.endswith('.jpg'):
                path = os.path.join(root, file)
                patient_id = path.split(os.sep)[-3]  
                
                if 'p' in patient_id and len(patient_id) > 3:  # Check if its valid ID for patient
                    patient_to_image.setdefault(patient_id, []).append(path)

            
                    

   
    for patient_id in patient_to_image.keys():
        data_index['patient_id'].append(patient_id)
        data_index['images'].append(patient_to_image[patient_id])

    flattened_list = list(more_itertools.collapse(data_index['images']))
    print(len(flattened_list))




    with open(output_file, 'w') as file:
        json.dump(data_index, file, indent=4)


def get_items_by_indices(items_list, indices):
    return [items_list[i] for i in indices]

def write_json(images_root, output_file, train_file, test_file, val_file):
    """
    Create 3 json files for test, train and val images
    """
    create_json(images_root, output_file)
    # train - validation split 

    with open(output_file, 'r') as file:
        data = json.load(file)

    # make the text go into the list

    # split by the index of the patient-id 
    indices = list(range(len(data['patient_id'])))

    # train 80% test and val: 10% each
    train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=1)
    # test and val: 10% each
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=1) 




    train_data = {
        'images': get_items_by_indices(data['images'], train_indices)
    }
    # train

    imgs = []

    for patient_img in train_data["images"]:   
        imgs.extend(patient_img)
    train_data["images"] = imgs


    # val
    val_data = {
        'images': get_items_by_indices(data['images'], val_indices)
        }
    imgs = []

    for patient_img in val_data["images"]:
        imgs.extend(patient_img)
    val_data["images"] = imgs

    # test
    test_data = {
        'images': get_items_by_indices(data['images'], test_indices)
        }
    imgs = []
    for patient_img in test_data["images"]:
        imgs.extend(patient_img)
    test_data["images"] = imgs


    with open(train_file, 'w') as file:
        json.dump(train_data, file, indent=4)

    with open(val_file, 'w') as file:
        json.dump(val_data, file, indent=4)

    with open(test_file, 'w') as file:
        json.dump(test_data, file, indent=4)


if __name__ == "__main__":
    images_root = '../data/downloaded_jpgs/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10'
    output_file = 'p10_data_images.json'
    train_file = 'train.json'
    test_file = 'test.json'
    val_file = 'val.json'

    write_json(images_root, output_file, train_file, test_file, val_file)

    with open(train_file, "r") as f:
        data = json.load(f)

    print("train:, ", len(data["images"]))

    with open(test_file, "r") as f:
        data = json.load(f)

    print("test:, ", len(data["images"]))

    with open(val_file, "r") as f:
        data = json.load(f)

    print("val: ", len(data["images"]))

