# RayGen: A Vision-Language Masked Generative Transformer for Chest X-Ray Generation
A scarcity of high-quality, labeled datasets often hinders the development of accurate and reliable diagnostic models in medical imaging, particularly in chest X-ray analysis. Recognizing the importance of addressing these challenges, our research proposes RayGen, a novel model synthesizing chest X-ray images by integrating vision and language processing techniques. RayGen aims to mitigate data scarcity by generating synthetic, yet realistic, chest X-ray images.

We adopt a multimodal approach with two input forms being considered. Images are obtained as embeddings using a vector quantized Generative Adversarial Network or VQ-GAN. The text embeddings from CXR Bert facilitated by Biovil-T. Further, text is obtained from reports that correspond to each X-ray image.  Finally, they are sent through our transformer decoder layer to obtain our synthetic image.

<img width="741" alt="pic11" src="https://github.com/farazali7/multimodal-xray/assets/66198904/88418d69-2405-4536-b0da-9fc810c299a6">

Our model consists of several parts, some trained and others adapted from pre-trained models. The figure above provides a visual representation of the architecture. During training, the impression section of the radiology reports is inputted to the CXR Bert model. At a high level, a given CXR in the VQ-GAN encoder model was already pre-trained on the MIMIC-CXR dataset. The image embeddings are randomly masked according to a cosine schedule, flattened into a sequence, and then passed through an embedding layer before being added to learnable positional embeddings. The impressions section of a corresponding radiology report is also encoded via a pre-trained CXR BERT model and projected to the hidden dimensionality of the transformer decoder. The transformer decoder takes the masked image embeddings sequence as queries and the report embeddings as the context through several stacked layers of attention to eventually output a reconstruction of the latent representation of the image.

## Manuscript
Please see the following associated manuscript of this project for further details of 
the model and results. 

[![](https://img.shields.io/badge/PAPER-37a779?style=for-the-badge)](./RayGen_Paper.pdf)

## Model architectures used
- Image Encoder: VQ-GAN (pre-trained)
- Text Encoder: CXR-BERT (pre-trained)
- Transformer Decoder: Custom model (inspired by MUSE + MaskGIT) (trained from scratch)
- Downstream Classifier: DenseNet-121 (trained from scratch)

## Dataset
The MIMIC-CXR database is a large, publicly available dataset of chest radiographs paired with free-text radiology reports. It includes 377,110 images from 227,835 imaging studies of 65,379 patients treated at the Beth Israel Deaconess Medical Center Emergency Department between 2011-2016. In addition to image data, we extracted and utilized the interpretation section of the radiology reports associated with each X-ray image.

If you use it, please cite it as follows:
```
@article{article,
author = {Johnson, Alistair and Pollard, Tom and Berkowitz, Seth and Greenbaum, Nathaniel and Lungren, Matthew and Deng, Chih-ying and Mark, Roger and Horng, Steven},
year = {2019},
month = {12},
pages = {317},
title = {MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports},
volume = {6},
journal = {Scientific Data},
doi = {10.1038/s41597-019-0322-0}
}
```
## Data Retrieval

1. Navigate to [PhysioNet MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/).
@@ -6,3 +42,40 @@

   ```shell
   wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-cxr/2.0.0/

## Instructions to run
Training can be done once the associated paths to your data and other desired parameters are set in `config.py`.
```
!git clone https://github.com/farazali7/multimodal-xray.git
!pip install -r requirements.txt
python3 -m src.train
```
To run inference using our pipeline, run the following commands:
```
!git clone https://github.com/farazali7/multimodal-xray.git
!pip install -r requirements.txt
python3 -m src.predict
```
## Results
This figure is an illustration of the progressive unmasking of the fully masked figure using the prompt: "Pneumothorax" at different time steps from t=0 having the fully masked images and t=3 depicted with a fully unmasked chest x-ray image.
<img width="479" alt="pic2" src="https://github.com/farazali7/multimodal-xray/assets/66198904/f275ad30-d34e-4d9c-84f8-fd5d471c54a3">

The image grid below displays the synthetic images generated by our model for each class (mentioned above the image) to demonstrate our model performance.
<img width="554" alt="pic3" src="https://github.com/farazali7/multimodal-xray/assets/66198904/cfe562d9-c365-4d4c-b7a4-8317b4a691c8">
## Acknowledgements
We would like to thank the University of Toronto, Professor Dr. Rahul Krishnan, and other CSC2541 course staff who have guided and helped us immensely in bringing this project to life.