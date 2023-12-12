import datetime
import uuid

from src.models.text_encoders import get_text_embeddings, get_cxr_bert_tokenizer_and_encoder
from src.models.model import ModelV2
from src.models.image_encoders import VQGanVAE

import torch
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import json

from config import cfg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    curr_batch_dir = f'batch_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    dir_path = os.path.join('results/images', curr_batch_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    data_info = {'images': [],
                 'class': []}
    for cls_name, cls_amount in class_proportions.items():
        print(f'GENERATING {cls_amount} SAMPLES FOR CLASS: {cls_name}')
        prompt = [cls_name]
        for i in tqdm(range(cls_amount), total=cls_amount):
            synthetic = generate_synthetic_cxr(model, vae, txt_tok, txt_enc, prompt,
                                               temperature, steps, topk_threshold, save=False)

            # Save and log id + class
            id = str(uuid.uuid4()) + f'-{str.lower(cls_name)}'
            data_info['images'].append(id)
            data_info['class'].append(str.lower(cls_name))

            save_image(synthetic[0], os.path.join(dir_path, id) + ".png")

    output_file = os.path.join(dir_path, 'data_info.json')
    # Save final data info as json
    with open(output_file, 'w') as file:
        json.dump(data_info, file, indent=4)


def generate_class_proportions(total, props, class_list):
    res = {}
    for cls, prop in zip(class_list, props):
        n = int(total * (1.0 - prop))
        res[cls] = n

    return res


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
    # cxr1 = generate_synthetic_cxr(model, vae, txt_tok, txt_enc, prompt, 1.0, 18, 0.9, 'image1')
    # cxr2 = generate_synthetic_cxr(model, vae, txt_tok, txt_enc, prompt, 1.0, 10, 0.9, 'image2')

    # GENERATE BATCHES OF DATA BY CLASS (50% and 100% of train set)
    total = 4684
    half = 2342

    # Given proportions for a multi-label problem
    proportions_multi_label = [0.279, 0.280, 0.048, 0.183, 0.046, 0.056, 0.245, 0.328, 0.134, 0.070]

    # Convert to proportions for a multi-class problem
    proportions_sum = np.sum(proportions_multi_label)
    proportions_multi_class = [prop / proportions_sum for prop in proportions_multi_label]

    class_list = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Fracture", "Lung Lesion", "Lung Opacity",
                  "Pleural Effusion", "Pneumonia", "Pneumothorax"]
    class_proportions = generate_class_proportions(total=half, props=proportions_multi_class, class_list=class_list)
    generate_batch(model, vae, txt_tok, txt_enc, class_proportions)

    print('Done')
