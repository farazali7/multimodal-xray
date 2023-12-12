import datetime
import uuid

from src.models.text_encoders import get_text_embeddings, get_cxr_bert_tokenizer_and_encoder
from src.models.model import ModelV2
from src.models.image_encoders import VQGanVAE
from src.utils.metrics import calculate_fid
from config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchxrayvision as xrv
import os
import matplotlib.pyplot as plt
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import json
import cv2
import skimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FIDDataset(Dataset):
    def __init__(self, orig_data_path, syn_data_path, transform=None):
        """
        Another implementation of dataset class used for pytorch dataloader to handle paired images for FID.

        Args:
            orig_data_path: the path to the original data directory JSON file
            syn_data_path: the path to the data directory with synthetic images
            transform (callable, optional): functions for img pre-processing
        """
        print(f'Loading Dataset class for FID')
        self.transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                         xrv.datasets.XRayResizer(224,
                                                                                  engine='cv2')])

        self.syn_data_path = syn_data_path
        # choose to open train or val image and reports
        with open(orig_data_path, "r") as f:
            self.orig_data_dict = json.load(f)
        self.orig_data_names = list(self.orig_data_dict.keys())

    def __len__(self):
        return len(self.orig_data_names)

    def __getitem__(self, idx):
        # load image
        img_name = self.orig_data_names[idx]
        filename_png = img_name.split('/')[-1].split('.')[0] + '.png'

        # Point path to original image location
        orig_path = img_name.replace('..', '/w/331/yasamin/multimodal-xray')
        syn_path = os.path.join(self.syn_data_path, filename_png)

        # Load images & normalize for DenseNet model
        orig_image = skimage.io.imread(orig_path)
        orig_image = xrv.datasets.normalize(orig_image, 255)  # convert 8-bit image to [-1024, 1024] range
        if len(orig_image.shape) > 2:
            orig_image = orig_image.mean(2)
        orig_image = orig_image[None, ...]  # Make single color channel
        orig_image = self.transform(orig_image)
        orig_image = torch.from_numpy(orig_image)

        syn_image = skimage.io.imread(syn_path)
        syn_image = xrv.datasets.normalize(syn_image, 255)  # convert 8-bit image to [-1024, 1024] range
        if len(syn_image.shape) > 2:
            syn_image = syn_image.mean(2)
        syn_image = syn_image[None, ...]  # Make single color channel
        syn_image = self.transform(syn_image)
        syn_image = torch.from_numpy(syn_image)

        # syn_image = cv2.imread(syn_path, 0)
        # syn_image = cv2.resize(syn_image, [224, 224])
        # syn_image = torch.Tensor(syn_image)[None, ...].expand(3, -1, -1)
        # syn_image = syn_image/255.

        return orig_image, syn_image


def generate_synthetic_cxr(model, vae, txt_tok, txt_enc, prompt: List[str] = None, temperature: float = 1.0,
                           timesteps: int = 18, topk_filter_thresh=0.9, image_name: str = "image",
                           save: bool = True):
    """

    Args:
        model: Transformer model
        vae: VQGAN VAE model
        txt_tok: Text tokenizer model
        txt_enc: Encoder model
        prompt: Input prompt embeddings

    Returns:
        The decoded CXR message
    """
    prompt_emb = get_text_embeddings(prompt, txt_tok, txt_enc, device=device)

    synthetic = model.generate(prompt_emb, vae, temperature=temperature,
                               timesteps=timesteps, topk_filter_thresh=topk_filter_thresh)

    if save:
        save_image(synthetic[0], os.path.join('results/images', image_name) + ".png")

    return synthetic


def experiment_decoding_params(model, vae, txt_tok, txt_enc, prompt: List[str] = None, model_name=''):
    """ Experiment with different decoding parameter configurations to visually see what works best.

    Args:
        model: Transformer model
        vae: VQGAN VAE model
        txt_tok: Text tokenizer model
        txt_enc: Encoder model
        prompt: Input prompt embeddings
        model_name: Name of model being experimented with

    Returns:
        Saves a figure showing the different sets of results.
    """
    temperatures = np.arange(1.0, 2.1, 0.1)
    steps = np.arange(2, 5, 1)
    topk_thresholds = np.arange(0.5, 0.7, 0.05)

    # Generate results array for each topk thresh of size (temps, steps, *img.size)
    # results in (6, 8, 3, 512, 512)
    for topk_thresh in tqdm(topk_thresholds, total=len(topk_thresholds)):
        res = torch.zeros(len(temperatures), len(steps), 3, 512, 512)
        for i, temperature in enumerate(temperatures):
            for j, step in enumerate(steps):
                img = generate_synthetic_cxr(model, vae, txt_tok, txt_enc, prompt,
                                             temperature, step, topk_thresh, save=False)
                res[i, j, ...] = img.squeeze()

        # Plot and save
        fig, axes = plt.subplots(len(temperatures), len(steps), figsize=(12, 12))  # Create a 6x8 subplot grid
        for i in range(len(temperatures)):  # Iterate through the y-axis
            for j in range(len(steps)):  # Iterate through the x-axis
                img = res[i, j].detach().numpy().transpose(1, 2, 0)  # Extract the image data for the subplot
                axes[i, j].imshow(img, cmap='gray')  # Plot the image in the corresponding subplot
        plt.suptitle(f'TopK Threshold: {round(topk_thresh, 1)} \n Prompt: {prompt[0]}')
        fig.supxlabel("Steps")
        fig.supylabel("Temperature")

        # Add outer x-axis labels
        for j in range(len(steps)):
            axes[-1, j].set_xlabel(f'{steps[j]}')  # Assuming indexing starts from 0
        # Add outer y-axis labels
        for i in range(len(temperatures)):
            axes[i, 0].set_ylabel(f'{round(temperatures[i], 2)}')  # Assuming indexing starts from 0
        plt.setp(axes, xticks=[], yticks=[])
        plt.tight_layout()  # Adjust subplot parameters for better layout
        fig.savefig(f'results/images/{model_name}_topk{round(topk_thresh, 1)}_{prompt[0]}.png')
        fig.clf()
        plt.close()


def generate_batch(model, vae, txt_tok, txt_enc, class_proportions: Dict):
    """ Generate batch of synthetic data given optional class list and corresponding frequencies.

    Args:
        model: Transformer model
        vae: VQGAN VAE model
        txt_tok: Text tokenizer model
        txt_enc: Encoder model
        class_proportions: Dict of class name (prompt) as key and desired amount as value

    Returns:
        Saves a batch of synthetic images.
    """
    # Decoding parameters
    temperature = 1.3
    steps = 3
    topk_threshold = 0.5
    batch_size = 16

    curr_batch_dir = f'batch_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    dir_path = os.path.join('results/images', curr_batch_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    data_info = {'images': [],
                 'class': []}

    print(f'CLASS PROPORTIONS: {class_proportions}')
    for cls_name, cls_amount in class_proportions.items():
        print(f'GENERATING {cls_amount} SAMPLES FOR CLASS: {cls_name}')
        prompt = [cls_name]
        # Create batches of the prompt
        # Calculate the number of iterations needed to reach the total_elements
        num_iterations = cls_amount // batch_size + (cls_amount % batch_size > 0)

        for i in range(num_iterations):
            # Calculate the number of instances of the string for the current iteration
            instances = min(batch_size, cls_amount - i * batch_size)
            prompt_list = prompt * instances

            synthetic = generate_synthetic_cxr(model, vae, txt_tok, txt_enc, prompt_list,
                                               temperature, steps, topk_threshold, save=False)

            # Save and log id + class
            for j in range(instances):
                id = str(uuid.uuid4()) + f'-{str.lower(cls_name)}'
                data_info['images'].append(id)
                data_info['class'].append(str.lower(cls_name))

                save_image(synthetic[j], os.path.join(dir_path, id) + ".png")

    output_file = os.path.join(dir_path, 'data_info.json')
    # Save final data info as json
    with open(output_file, 'w') as file:
        json.dump(data_info, file, indent=4)


def generate_class_proportions(total, props, class_list):
    res = {}
    for cls, prop in zip(class_list, props):
        n = int(total * prop)
        res[cls] = n

    return res


def generate_p10_test_set(model, vae, txt_tok, txt_enc):
    # Decoding parameters
    temperature = 1.3
    steps = 3
    topk_threshold = 0.5
    batch_size = 16

    # Load p10 data as filenames and prompts
    with open('data/p10_test.json', 'r') as file:
        test_p10 = json.load(file)

    # Now find text prompts for each image name
    image_names = list(test_p10.keys())
    prompts = list(test_p10.values())

    print(f"Got all image names ({len(image_names)}) and prompts ({len(prompts)})")
    curr_batch_dir = f'p10_test_synthetic'
    dir_path = os.path.join('results/images', curr_batch_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        filename = image_name.split('/')[-1].split('.')[0]
        prompt = [prompts[i]]
        synthetic = generate_synthetic_cxr(model, vae, txt_tok, txt_enc, prompt,
                                           temperature, steps, topk_threshold, save=False)
        save_image(synthetic[0], os.path.join(dir_path, filename) + ".png")


class DenseNet121(nn.Module):
    """
    This model is from:
    https://github.com/arnoweng/CheXNet

    The architecture of the model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = xrv.models.DenseNet(weights="densenet121-res224-chex")
        # self.densenet121 = torchvision.models.densenet121(pretrained=True)
        # num_ftrs = self.densenet121.classifier.in_features
        # # defining the classification layer
        # self.densenet121.classifier = nn.Sequential(
        #     nn.Linear(num_ftrs, out_size)
        # )

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out


def compute_fid():
    '''Compute Frechet-Inception Distance (FID) metric between original and synthetic data distributions.

    Returns:
        FID score for a test set of data.
    '''
    batch_size = 32
    dataset = FIDDataset(orig_data_path='data/p10_test.json', syn_data_path='results/images/p10_test_synthetic')

    # Data loader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)

    # Pretrained model that gets embeddings
    model = DenseNet121(10)
    # print("=> loading checkpoint")
    # checkpoint = torch.load("results/model.pth.tar_epoch_3_1702271233.pth.tar")
    # model.load_state_dict(checkpoint["model_state_dict"])
    # print("=> loaded checkpoint")
    model = model.to(device)
    model.eval()

    # Step through data samples
    all_orig_latents = []
    all_syn_latents = []
    for orig, syn in tqdm(dl, total=len(dl)):
        orig, syn = orig.to(device), syn.to(device)

        with torch.no_grad():
            orig_latents = model(orig)
            syn_latents = model(syn)

            all_orig_latents.append(orig_latents)
            all_syn_latents.append(syn_latents)

    # Shapes: [N, 1024]
    orig_latents_vec = torch.concatenate(all_orig_latents, dim=0).cpu().numpy()
    syn_latents_vec = torch.concatenate(all_syn_latents, dim=0).cpu().numpy()

    print(f'Got all latents, computing FID between distributions...')
    print(f'Original data latents shape: {orig_latents_vec.shape}')
    print(f'Synthetic data latents shape: {syn_latents_vec.shape}')

    # Compute FID scores between two distributions
    fid = calculate_fid(orig_latents_vec, syn_latents_vec)

    print(f'FID SCORE: {fid}')


if __name__ == "__main__":
    model_args = cfg['MODEL']['ModelV2']
    encoder_args = model_args['ENCODER']
    projector_args = model_args['PROJECTOR']
    decoder_args = model_args['DECODER']

    txt_tok, txt_enc = get_cxr_bert_tokenizer_and_encoder()

    vae = VQGanVAE(**encoder_args, device=device)

    vae = vae.to(device)

    prompt = ['Pneumothorax']

    model = ModelV2(txt_enc, decoder_args, projector_args, device=device)

    ckpt_path = 'results/all_maskloss/epoch=98-step=740223.ckpt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    sd = {x.replace('model.', '') : v for x, v in checkpoint['state_dict'].items()}
    model.load_state_dict(sd)

    model = model.to(device)

    print(f"GENERATING WITH CKPT PATH: {ckpt_path}...")
    # Create and save pictures
    # experiment_decoding_params(model, vae, txt_tok, txt_enc, prompt, 'all_maskloss_moredetail')

    # GENERATE BATCHES OF DATA BY CLASS (50% and 100% of train set)
    # total = 4684
    #
    # # Given proportions for a multi-label problem
    # proportions_multi_label = [1307, 1312, 226, 855, 217, 260, 1148, 1537, 629, 326]
    # class_list = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Fracture", "Lung Lesion", "Lung Opacity",
    #               "Pleural Effusion", "Pneumonia", "Pneumothorax"]
    # # Sort them together
    # proportions_multi_label, class_list = (list(t) for t in zip(*sorted(zip(proportions_multi_label, class_list))))
    #
    # # Convert to proportions for a multi-class problem
    # proportions_sum = np.sum(proportions_multi_label)
    # proportions_multi_class = [prop / proportions_sum for prop in proportions_multi_label]
    # proportions_multi_class.reverse()  # Least frequency class gets highest proportion now
    #
    # class_proportions = generate_class_proportions(total=total, props=proportions_multi_class, class_list=class_list)
    # generate_batch(model, vae, txt_tok, txt_enc, class_proportions)

    # generate_p10_test_set(model, vae, txt_tok, txt_enc)

    compute_fid()

    print('Done')
