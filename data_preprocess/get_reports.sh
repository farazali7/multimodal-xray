#!/bin/bash

# get the username
read -p "Enter your username: " username

# Define the base URL
base_url="https://physionet.org/files/mimic-cxr/2.0.0/files/p10/p10000032"
output_dir="../data/downloaded_reports"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Change to the output directory
cd "$output_dir"

# Run wget 
# wget -r -N -c -np --user "$username" --ask-password -A .jpg -I /files/mimic-cxr/2.0.0/files/p10/p10000032 "$base_url"
#wget -r -N -c -np --user "$username" --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/
# All data get
#wget -r -N -c -np -A .txt --user "$username" --ask-password https://physionet.org/files/mimic-cxr/2.0.0/
#Subset data get
wget -r -N -c -np -A .txt --user "$username" --ask-password https://physionet.org/files/mimic-cxr/2.0.0/files/p10/p10000032/
# Return to the original directory
cd -

# Running:
# chmod +x get_reports.sh
# ./get_reports.sh

# if run on linux 
#  sed -i 's/\r$//' get_reports.sh


# EOS
