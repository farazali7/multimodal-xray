# RayGen
A scarcity of high-quality, labeled datasets often hinders the development of accurate and reliable diagnostic models in medical imaging, particularly in chest X-ray analysis. Recognizing the importance of addressing these challenges, our research proposes RayGen, a novel model synthesizing chest X-ray images by integrating vision and language processing techniques. RayGen aims to mitigate data scarcity by generating synthetic, yet realistic, chest X-ray images.

We adopt a multimodal approach with two input forms being considered. Images are obtained as embeddings using a vector quantized Generative Adversarial Network or VQ-GAN. The text embeddings from CXR Bert facilitated by Biovil-T. Further, text is obtained from reports that correspond to each X-ray image. They are projected using linear layers and concatenated using cross-attention. Finally, they are sent through our transformer decoder layer to obtain our synthetic image.

<img width="741" alt="pic11" src="https://github.com/farazali7/multimodal-xray/assets/66198904/88418d69-2405-4536-b0da-9fc810c299a6">

Our model consists of several parts, some trained and others adapted from pre-trained models. The figure above provides a visual representation of the architecture. During training, the impression section of the radiology reports is inputted to the CXR Bert model. These texts then go into a dense layer to generate a Tx$D_m$ embedding to be inputted to the cross attention of the Transformer Decoder. Concurrently chest x-ray images are input to the VQ-GAN. This creates a VxV embedding for each image which is then randomly masked and combined with the positional embeddings. This generated a V^2 x D_m embedding which is then passed to the self-attention layer of the Transformer decoder. The Transformer then unmasks the image tokens using the text embeddings and is trained using Cross Entropy Loss.

## Model architectures used
- Image Encoder: VQ-GAN (pre-trained)
- Text Encoder: CXR-BERT (pre-trained)
- Transformer Decoder: custom model (MUSE + MaskGIT) (trained from scratch)
- Downstream classifier: Densenet 121 (trained from scratch)

## Dataset used
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
2. Create an account and get approved for access to the data.
3. Use the following command to download the data:
   
   ```shell
   wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-cxr/2.0.0/


## Metrics used for evaluation
1. Area Under the Receiver Operating Characteristic Curve (AUROC)
2. Frechet Inception Distance (FID score)
3. Structural Similarity Index Measure (SSIM and Multi-scale SSIM)
4. F1 score

## Results

![image](https://github.com/farazali7/multimodal-xray/assets/66198904/17e5d73e-f6b9-48c3-b826-51e907834a4c)

![image](https://github.com/farazali7/multimodal-xray/assets/66198904/4da30cd6-02df-4a70-bc59-114452022c3f)

<img width="479" alt="pic2" src="https://github.com/farazali7/multimodal-xray/assets/66198904/f275ad30-d34e-4d9c-84f8-fd5d471c54a3">

<img width="554" alt="pic3" src="https://github.com/farazali7/multimodal-xray/assets/66198904/cfe562d9-c365-4d4c-b7a4-8317b4a691c8">





