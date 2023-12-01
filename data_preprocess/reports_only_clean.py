import pandas as pd
import os
import requests
from getpass import getpass
import glob
import pandas as pd
import json
import os
import re

def get_common_view(csv_file, json_file, output_file):
    """
    jso file is the train.json for example
    """
    with open(json_file, 'r') as file:
        texts = json.load(file) # paths
    print(f"number of reports before cleaning: {len(texts)}")

    # make a dict of texts_dict = {subject/study: "path.. subj/study..."}
    all_text_dict = {}
    for path in texts:
        # /w/331/yasamin/multimodal-xray/data/downloaded_reports/physionet.org/files/mimic-cxr/2.0.0/files/p10/p10995687/s56935682.txt

        splitted_path = path.split('/')

        study = splitted_path[-1].split('.')[0]
        patient = splitted_path[-2]

        short_path = patient + "/" + study

        #print("all_text_dict: ", short_path)

        all_text_dict[short_path] = path


    df = pd.read_csv(csv_file)
    most_common_view = df['ViewPosition'].mode()[0]
    filtered_df = df[df['ViewPosition'] == most_common_view]

    grouped = filtered_df.groupby(['study_id', 'subject_id'])
    filtered_ids = {}

    for (study_id, subject_id), group in grouped:
        # get the first dicom_id from each study_id, patientid pair
        dicom_id = group['dicom_id'].iloc[0]
        short_path = "p" + str(subject_id) + "/" + "s" + str(study_id)
        #print("filtered_ids: ", short_path)
        filtered_ids[short_path] = dicom_id

    
    #{dicom:text}
    dicom_text_dict = {}
    for key_short_path in all_text_dict.keys():
        if key_short_path in filtered_ids.keys():
            try:
                with open(all_text_dict[key_short_path], 'r') as text_file:
                    text_split_list = text_file.read().split("IMPRESSION:")
                    if len(text_split_list) > 1:
                        text = text_split_list[-1]
                        text = ''.join(re.sub('[^A-Za-z ]+', '', text).lower().strip())
                        dicom_text_dict[filtered_ids[key_short_path]] = text
                    
            except FileNotFoundError:
                print(f"File not found: {all_text_dict[key_short_path]}")
    print(f"number of reports after cleaning: {len(dicom_text_dict)}")
    with open(output_file, 'w') as file:
        json.dump(dicom_text_dict, file, indent=4)



if __name__ == "__main__":
    csv_file = '../data/mimic-cxr-2.0.0-metadata.csv'
    json_file = '../data/p10_text.json'  
    output_file = '../data/p10_train_text_clean.json'


    get_common_view(csv_file, json_file, output_file)
    # train: 
    #images before cutting 17926, texts before cuttings 17926
    # images after cutting 9661, texts after cuttings 9661

    # json_file = '../data/val.json'  
    # output_file = '../data/val_text_clean.json'

    # get_common_view(csv_file, json_file, output_file)

    # val:
    # images before cutting 4271, texts before cuttings 4271
    # images after cutting 2241, texts after cuttings 2241