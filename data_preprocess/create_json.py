import os
import json
import random
from sklearn.model_selection import train_test_split

# use for dataloader in train.py 
def create_json(images_root, texts_root, output_file):
    """
    Create JSON to store the paths to the imgs, texts, and also the patient study as the corresponding labels
    """
    data_index = {'images': [], 'texts': [], 'labels': []}

    # go through all subdirectories in the images_root directory (downloaded_jpgs)
    for subdir in os.listdir(images_root):
        subdir_path = os.path.join(images_root, subdir)
        
        #check if it's a directory
        if os.path.isdir(subdir_path):
            # radiolofy report for this subdir
            text_file_path = os.path.join(texts_root, subdir + '.txt')
            
            # check the text file exists
            if os.path.exists(text_file_path):
                # read the text file
                with open(text_file_path, 'r') as file:
                    text_data = file.read()

                # iterate over all images in this subdirectory
                for file in os.listdir(subdir_path):
                    if file.endswith('.jpg'):
                        image_path = os.path.join(subdir_path, file)

                        #Append data to the index
                        data_index['images'].append(image_path)
                        data_index['texts'].append(text_data)
                        data_index['labels'].append(subdir)  #using subdir name as label


    # Write data to JSON file
    with open(output_file, 'w') as file:
        json.dump(data_index, file, indent=4)

# for example: 
images_root = '../data/downloaded_jpgs'
texts_root = '../data/downloaded_reports'
create_json(images_root, texts_root, 'all_data.json')


# expected json:
# {   
#  "images": ["../data/../image1.jpg", ...], 
# "texts": ["../data/../report1.txt", ...],
# "labels": ["p1000053", ...] 
#  }


# train - validation split 

# all data is initially in one JSON file so
with open(os.path.join('all_data.json'), 'r') as file:
    data = json.load(file)


# Split daya to train and val
train_data, val_data = train_test_split(data, test_size=0.2) # 80-20 split


# Save the splits into separate JSON files
with open(os.path.join('train.json'), 'w') as file:
    json.dump(train_data, file, indent=4)

with open(os.path.join('val.json'), 'w') as file:
    json.dump(val_data, file, indent=4)