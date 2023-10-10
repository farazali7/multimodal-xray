## Data Retrieval

1. Navigate to [PhysioNet MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/).
2. Create an account and get approved for access to the data.
3. Use the following command to download the data:
   
   ```shell
   wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-cxr/2.0.0/
