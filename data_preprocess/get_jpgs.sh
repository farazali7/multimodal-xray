#!/bin/bash

#read -p "Enter your username: " username

# base_url="https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000032"
output_dir="../data/downloaded_jpgs"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Change to the output directory
cd "$output_dir"

# Run wget 
#wget -r -N -c -np --user "$username" --ask-password -A .jpg -I /files/mimic-cxr-jpg/2.0.0/files/p10/p10000032 "$base_url"

#wget -r -N -c -np --user --ask-password -A .jpg -I /files/mimic-cxr-jpg/2.0.0/files/p10/p10000032 https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000032
# all data get
# wget -r -N -c -np --user "$username" --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/
# subset data get
#wget -r -N -c -np --user "$username" --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/

username=""

password=""

csv_file="../mimic-cxr-2.0.0-chexpert.csv"
ids=$(tail -n +2 "$csv_file" | cut -d, -f1 | sort | uniq | sed 's/^/p/' )

for id in $ids; do
    #wget -r -N -c -np --user "$username" --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000032/

    wget -r -N -c -np --user "$username" --password "$password" https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/${id}/
done


cd -

# Running:
# chmod +x get_jpgs.sh
# ./get_jpgs.sh

# if linux: 
# sed -i 's/\r$//' get_jpgs.sh

# EOS
