import os
import json
from sklearn.model_selection import train_test_split
import glob
import re
import more_itertools

def create_json(images_root, output_file):
    data_index = {'images': [], 'disease': []}
    patient_to_image = {}

    study = []
    for root, _, files in os.walk(images_root):
        for file in files:
            if file.endswith('.png'):
                path = os.path.join(root, file)
                disease = path.split("-")[-1].split('.')[0]  
                
                
                patient_to_image.setdefault(disease, []).append(path)

            
                    



    with open(output_file, 'w') as file:
        json.dump(patient_to_image, file, indent=4)

# helper
def to_onehot(one_hot_json):

    diction = {'fracture': [
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0
    ], 'consolidation': [
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ], 'cardiomegaly': [
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ], 'lung lesion': [
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0
    ], 'pneumothorax': [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0
    ], 'edema': [
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0
    ], 'lung opacity': [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0
    ], 'pleural effusion': [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0
    ], 'pneumonia': [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0
    ], 'atelectasis': [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ]}
    

    with open(one_hot_json, 'w') as file:
        json.dump(diction, file, indent=4)
    
    
def synth_encoding(imgs_dict_json, one_hot_json, dump_json):
    with open(imgs_dict_json, "r") as file:
        data_synth = json.load(file)

    

    with open(one_hot_json, "r") as file:
        data_hot = json.load(file)

    path_to_hot = {}
    for disease in data_synth.keys():
        img_paths = data_synth[disease]
        for path in img_paths:
            path_to_hot[path] = data_hot[disease]

    print(len(path_to_hot))

    with open(dump_json, 'w') as file:
        json.dump(path_to_hot, file, indent=4)
    
# helper
def fifty_perc_of_synth():
    with open('synth_data_images.json', "r") as file:
        data_synth = json.load(file)
    out_file = "50_synth.json"
    new_dict = {}
    for disease in data_synth.keys():
        new_dict[disease] = data_synth[disease][:len(data_synth[disease])//2]

    length = 0
    for disease in new_dict.keys():
        length += len(new_dict[disease])
    print(length)

    

    
    with open(out_file, 'w') as file:
        json.dump(new_dict, file, indent=4)
    # 2337

if __name__ == "__main__":
    # images_root = '../data/batch_2023-12-11_21-13-06'
    # output_file = 'synth_data_images.json'
    # imgs_dict_json = "synth_data_images.json"
    # one_hot_json = "one_hot_json.json"
    # dump_json = "50_synthetic_embeddings.json"
    # imgs_dict_json = "50_synth.json"
    # synth_encoding(imgs_dict_json, one_hot_json, dump_json)

    #create_json(images_root, output_file)

    # with open(output_file, 'r') as file:
    #     data = json.load(file)

    # print(data.keys())

    #to_onehot("one_hot_json.json")




    # length = 0
    # for disease in data_synth.keys():
    #     length += len(data_synth[disease])
    # print(length)

    all_data = "50_synth_real.json"

    with open('50_synthetic_embeddings.json', "r") as file:
        data_synth = json.load(file)

    with open('path_to_label_train.json', "r") as file:
        data_real = json.load(file)

    merged_dict = {**data_synth, **data_real}

    print(len(merged_dict))

    with open(all_data, 'w') as file:
        json.dump(merged_dict, file, indent=4)






    