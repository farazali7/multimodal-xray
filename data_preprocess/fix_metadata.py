import pandas as pd
import json

# helper, run only when you have raw metadata csv
def fix_csv(file_path, new_file_path):
    """
    Fix the metadata.csv file to only contain p10 data, and remove columns with all 0's 
    Save the new CSV file in cleaned_mimic-cxr-2.0.0-chexpert.csv

    file_path = 'mimic-cxr-2.0.0-chexpert.csv'
    new_file_path = 'cleaned_mimic-cxr-2.0.0-chexpert.csv'

    """

    
    data = pd.read_csv(file_path)
    # 227827 studies 

    data.fillna(0, inplace=True)
    # remove any rows with -1 (uncertain labels)
    data = data.loc[~(data == -1).any(axis=1)]
    # only get the p10 studies
    data = data[data['subject_id'].astype(str).str.startswith('10')]
    data = data.astype(int)

    # remove the columns that are all 0's
    data = data.loc[:, (data != 0).any(axis=0)]
    #print("shape: ", data.shape[0])

    # only keep the rows in the 10 classes

    columns_to_remove = ["Enlarged Cardiomediastinum", "No Finding", "Pleural Other", "Support Devices"]
    # remove any images classified to the above labels
    data = data[~data[columns_to_remove].any(axis=1)]
    # remove those columns
    data.drop(columns=columns_to_remove, inplace=True)



    # check the columns that are all 0 and remove them


    data.to_csv(new_file_path, index=False)

    data = pd.read_csv(new_file_path)
    #print(len(data))

    # 5672 p10 studies



# helper 
def save_csv_dict(new_file_path):
    """

    new_file_path = 'cleaned_mimic-cxr-2.0.0-chexpert.csv'

    
    """
    images_dict = {}
    # get the dict {"p10../s678..": [0, 1, 0, 0,0]} (labels) 
    with open(new_file_path, 'r') as file:
        next(file)

        for row in file:  
            items = row.strip().split(',')
            patient_id= 'p' + str(items[0])
            study_id= 's' + str(items[1])
            image_name = patient_id + '/' + study_id
            label = items[2:]
            label = [int(i) for i in label]
            images_dict[image_name] = label
    #print("length of {p10../s678..: [0, 1, 0, 0,0]}", len(images_dict))
    return images_dict

# helper
def save_dict_trainval(train_or_val_test):
    #test_imgs_path = "val.json"
    train_imgs_path = f"{train_or_val_test}.json"

    with open(train_imgs_path, "r") as file:
        train_dict = json.load(file)

    imgs_list = train_dict["images"]
        # get the dict {"p10../s678..": "list of full path to img"} 
    #print(len(imgs_list))


    path_img_dict = {}
    for img_path in imgs_list:
        study_id = img_path.split("/")[-2]
        patient_id = img_path.split("/")[-3]
        img = patient_id + '/' + study_id

        path_img_dict.setdefault(img, []).append(img_path)  
    
    
    # flattened_list = [item for sublist in path_img_dict.values() for item in sublist]

    # print("flat", len(flattened_list))
    return path_img_dict

def create_json_dict(new_file_path):
    """
    get the {"p10../s678..": "list of full paths to img"} saved in a json for test, train and val data
    """
    # get the dict {"p10../s678..": [0, 1, 0, 0,0]} for all p10 images from the metadata
    images_dict = save_csv_dict(new_file_path)
    #print("expect 5672", len(images_dict))


    # get the dict {"p10../s678..": "full path to img"} for the train data
    path_img_dict_train = save_dict_trainval("train") # save train.json
    #flattened_list = [item for sublist in path_img_dict_train.values() for item in sublist]
    #print("expect 18884", len(flattened_list))


    path_label_dict = {}
    for short_path in path_img_dict_train.keys():
        if short_path in images_dict:
            #{"full length path": label}
            for path in path_img_dict_train[short_path]:
                path_label_dict[path] = images_dict[short_path]


    destination_path = "path_to_label_train.json"

    with open(destination_path, "w") as f:
        json.dump(path_label_dict, f, indent=4)

    # get the dict {"p10../s678..": "full path to img"} for the test data
    path_img_dict_test = save_dict_trainval("test") # save train.json

    #flattened_list = [item for sublist in path_img_dict_test.values() for item in sublist]
    #print("expect 2160", len(flattened_list))

    path_label_dict = {}
    for short_path in path_img_dict_test.keys():
        if short_path in images_dict:
            #{"full length path": label}
            for path in path_img_dict_test[short_path]:
                path_label_dict[path] = images_dict[short_path]

    destination_path = "path_to_label_test.json"

    with open(destination_path, "w") as f:
        json.dump(path_label_dict, f, indent=4)

    # get the dict {"p10../s678..": "full path to img"} for the val data
    path_img_dict_val = save_dict_trainval("val") # save train.json

    #flattened_list = [item for sublist in path_img_dict_val.values() for item in sublist]
    #print("expect 2567", len(flattened_list))

    path_label_dict = {}
    for short_path in path_img_dict_val.keys():
        if short_path in images_dict:
            #{"full length path": label}
            for path in path_img_dict_val[short_path]:
                path_label_dict[path] = images_dict[short_path]

    destination_path = "path_to_label_val.json"

    with open(destination_path, "w") as f:
        json.dump(path_label_dict, f, indent=4)




if __name__ == "__main__":
    file_path = '../data/mimic-cxr-2.0.0-chexpert.csv'
    new_file_path = '../data/cleaned_mimic-cxr-2.0.0-chexpert.csv'

    #fix_csv(file_path, new_file_path)

    create_json_dict(new_file_path)
    with open("path_to_label_train.json", "r") as f:
        data = json.load(f)

    print("train:, ", len(data))

    with open("path_to_label_test.json", "r") as f:
        data = json.load(f)

    print("test:, ", len(data))

    with open("path_to_label_val.json", "r") as f:
        data = json.load(f)

    print("val: ", len(data))


       









    




