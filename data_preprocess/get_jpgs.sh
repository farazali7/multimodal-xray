#!/bin/bash

# get the username
read -p "Enter your username: " username

# Define the base URL
base_url="https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000032"
output_dir="../data/downloaded_jpgs"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Change to the output directory
cd "$output_dir"

# Run wget 
wget -r -N -c -np --user "$username" --ask-password -A .jpg -I /files/mimic-cxr-jpg/2.0.0/files/p10/p10000032 "$base_url"

# Return to the original directory
cd -

# Running:
# chmod +x get_jpgs.sh
# ./get_jpgs.sh

# EOS
