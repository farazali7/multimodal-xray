import pandas as pd
import os
import requests
from getpass import getpass
import os
import glob
import json

# helper

def get_ap_pa_dicom_ids(csv_file):
    """Get the dicom ids of all AP and PA views in the data."""
    # Load the data from the CSV file
    df = pd.read_csv(csv_file)

    # Filter the DataFrame for rows where ViewPosition is either 'AP' or 'PA'
    filtered_df = df[df['ViewPosition'].isin(['AP', 'PA'])]

    # Array of dicom IDs for AP or PA view
    dicom_id_array = filtered_df['dicom_id'].to_numpy()

    return dicom_id_array



def clean_json_file(csv_file, json_file):
   dicom_id_array = get_ap_pa_dicom_ids(csv_file)

   with open(json_file, 'r') as file:
      data = json.load(file)
   images = data['images']

   print(f"images before cutting {len(images)}")

   # not include entries where image is not part of the common views
   cleaned_images = []
   for path_img in images:
    if os.path.splitext(os.path.basename(path_img))[0] in dicom_id_array:
        cleaned_images.append(path_img)
   

   with open(json_file, 'w') as file:
      json.dump({'images': cleaned_images}, file, indent=4)


   print(f"images after cutting {len(cleaned_images)}")

   print("JSON cleanup complete.")


if __name__ == "__main__":
   csv_file = '../data/mimic-cxr-2.0.0-metadata.csv'
   json_file = 'train.json'  
   clean_json_file(csv_file, json_file)


   json_file = 'val.json'  
   clean_json_file(csv_file, json_file)
   
   json_file = 'test.json'  
   clean_json_file(csv_file, json_file)

