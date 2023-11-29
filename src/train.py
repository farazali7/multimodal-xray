'''
TRAINING SCRIPT
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Tuple

import torch
import lightning as L
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS

from config import cfg
from src.models.model import FinalModelV1, FinalModelV2
from src.models.image_encoders import VQGanVAE
from src.models.text_encoders import get_cxr_bert_tokenizer_and_encoder, get_text_embeddings

from torch.utils.data import Dataset
from PIL import Image
import os
import json

from torch.utils.data import DataLoader
from torchvision import transforms

from lightning.pytorch.loggers import WandbLogger

class UpdatedDatasetClass(Dataset):
    def __init__(self, data_path, text_tokens_path, vae, transform=None):
        """
        Another implementation of dataset class used for pytorch dataloader to handle the imgs and text

        Args:
            data_path: the path to the data directory with The JSON file corresponding to the data (../data_preprocess)
            vae: VQGANVAE Model for image encoding (from dictionary)
            text_tokens_path: Path to file containing all text reports tokenized
            transform (callable, optional): functions for img pre-processing
        """
        print(f'Loading Dataset class for {data_path}')
        self.data_path = data_path
        self.vae = vae
        self.transform = transform
        # choose to open train or val image and reports
        with open(data_path, "rb") as f:
            self.data_index = pickle.load(f)

        with (open(text_tokens_path, "rb")) as txt_tokens:
            self.text_tokens = pickle.load(txt_tokens)

    def __len__(self):
        return len(self.data_index[:300])

    def __getitem__(self, idx):
        # load image
        img_name = self.data_index[idx]
        image = self.vae.get_codebook_indices(img_name).squeeze()

        # text data and label
        # [1, 256, 2]
        text = self.text_tokens[img_name].squeeze()

        return image, text


def plot_loss(loss_array):
    plt.scatter(range(len(loss_array)), loss_array, c="red", s=1)
    plt.title('Plot of the Loss function')
    plt.xlabel('epochs')
    plt.ylabel('Train Loss')
    plt.show()


def train(train_paths: Tuple[str, str], val_paths: Tuple[str, str], model_args: dict,
          log_args:dict, chkpt_args:dict, trainer_args: dict, batch_size:int):
    """Train a model.

    Args:
        data_path: Path to data files
        model_args: Dictionary of kwargs for model
        trainer_args: Dictionary of kwargs for Trainer

    Returns:
        Trained model instance.
    """
    # Create train and validation dataloaders via data_path: each dataloader must give batches of:
    # [x_img, x_txt, y] samples, where x_img and y are input images resized to 512x512, and x_txt is
    # corresponding report.

    # image preprocessing
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    encoder_args = model_args['model_args']['ENCODER']
    print(f'Loading VQGAN...')
    vae = VQGanVAE(**encoder_args)
    print(f'VQGAN loaded!')

    _, text_model = get_cxr_bert_tokenizer_and_encoder()
    model_args['model_args']['tokenizer'] = text_model

    # ---------------

    # WANDB Logger
    wandb_logger = WandbLogger(project="multimodal_xray")

    train_dataset = UpdatedDatasetClass(*train_paths, vae=vae, transform=transform)
    val_dataset = UpdatedDatasetClass(*val_paths, vae=vae, transform=transform)

    # data loader  
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # Instantiate the model
    model = FinalModelV2(**model_args)
    checkpoint = ModelCheckpoint(**chkpt_args)

    # log gradients and model topology
    wandb_logger.watch(model)
    logger = wandb_logger
    # logger = TensorBoardLogger(**log_args)

    # Instantiate the PyTorch Lightning Trainer
    trainer = L.Trainer(**trainer_args, callbacks=checkpoint, logger=logger)
    
    # Fit the model
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    wandb_logger.experiment.unwatch(model)


if __name__ == "__main__":
    # Organize arguments here
    train_args = cfg['TRAIN']
    model_def = train_args['model_def']
    model_instance_args = cfg['MODEL'][model_def]
    chkpt_args = cfg['CALLBACK']
    trainer_args = train_args['TRAINER']
    LR = train_args['LR']
    train_data_path = cfg['DATA']['TRAIN_PATH']
    train_text_tokens_path = cfg['DATA']['TRAIN_TEXT_TOKEN_PATH']
    val_data_path = cfg['DATA']['VAL_PATH']
    val_text_tokens_path = cfg['DATA']['VAL_TEXT_TOKEN_PATH']
    model_args = {'model_args': model_instance_args,
                  'lr': LR}
    log_args = cfg['LOGGER']
    batch_size = train_args['BATCH_SIZE']

    # Train the model
    train((train_data_path, train_text_tokens_path), (val_data_path, val_text_tokens_path),
          model_args, log_args, chkpt_args, trainer_args, batch_size=batch_size)
    