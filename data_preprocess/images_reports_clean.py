import pandas as pd
import os
import requests
from getpass import getpass
import os
import glob

def get_common_view(csv_file):
    """Get the dicom ids of the most common view of the data"""
    # I will first grab the most common view from the metadata file
    # '../data/mimic-cxr-2.0.0-metadata.csv'
    df = pd.read_csv(csv_file)
        
    # Get the most common ViewPosition
    most_common_view = df['ViewPosition'].mode()[0]

    filtered_df = df[df['ViewPosition'] == most_common_view]

    # array of dicom IDs with the most common view
    dicom_id_array = filtered_df['dicom_id'].to_numpy()

    return dicom_id_array


# Clean up to only have most Common views: 

dicom_id_array = get_common_view('../data/mimic-cxr-2.0.0-metadata.csv')

# TODO: 
image_directory = '../data/downloaded_jpgs' 

# List all jpg files in the directory
jpg_files = glob.glob(os.path.join(image_directory, '*.jpg'))

# Loop all jpg files and delete those not in the common views of CXR imgs array
for jpg_file in jpg_files:
    dicom_id = os.path.splitext(os.path.basename(jpg_file))[0]
    
    # If the dicom_id is not in the array, delete the file
    if dicom_id not in dicom_id_array:
        os.remove(jpg_file)

print("Cleanup complete.")

# running this script will clean the jpgs of the view that are not common

    

