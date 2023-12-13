## Abstract

We propose a novel approach to synthesize chest x-ray imaging data. The lack of large-scale, accessible and reliable labeled data in the medical imaging domain is a problem that plagues the research community. This can occur due to a multitude of reasons such as poor data collection and organization at medical centers, lack of domain knowledge while labeling, and patient anonymity or confidentiality concerns. There is also a colossal requirement for computational resources when dealing with huge volumes of medical data or building models with the same. Hence, our objective is to provide a simple yet effective pipeline to artificially synthesize chest X-ray images based on the knowledge gained from pre-existing data (such as the MIMIC-CXR dataset). This essentially involves training a model on the dataset along with text reports, merging them into one embedded feature vector, and sampling from that distribution to obtain our final output image. This is posed as a multimodal problem with 2 forms of inputs being taken into consideration. Text is obtained from text reports that correspond to each X-ray image. We obtain image embeddings using a vector quantized Generative Adversarial Network or VQ-GAN and the text embeddings from CXR Bert facilitated by Biovil-T. Results for this pipeline are evaluated using metrics such as Structural Similarity Index Measure (SSIM) and Peak signal noise ratio (PSNR) to determine the performance of our model.


## Data Retrieval

1. Navigate to [PhysioNet MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/).
2. Create an account and get approved for access to the data.
3. Use the following command to download the data:
   
   ```shell
   wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-cxr/2.0.0/
