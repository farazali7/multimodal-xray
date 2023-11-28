'''
TRAINING SCRIPT
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

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

from torch.utils.data import Dataset
from PIL import Image
import os
import json

from torch.utils.data import DataLoader
from torchvision import transforms


class UpdatedDatasetClass(Dataset):
    def __init__(self, names, data_path, text_tokens_path, vae, transform=None, is_train=True):
        """
        Another implementation of dataset class used for pytorch dataloader to handle the imgs and text

        Args:
            data_path: the path to the data directory with The JSON file corresponding to the data (../data_preprocess)
            vae: VQGANVAE Model for image encoding (from dictionary)
            text_tokens_path: Path to file containing all text reports tokenized
            transform (callable, optional): functions for img pre-processing
            is_train (bool): to differentiate between train and validation dataset.
        """
        self.data_path = data_path
        self.vae = vae
        self.transform = transform
        self.is_train = is_train
        # # choose to open train or val image and reports
        # with open(os.path.join(data_path, 'train.json' if is_train else 'val.json')) as f:
        #     self.data_index = json.load(f)
        #
        # with (open(text_tokens_path, "rb")) as txt_tokens:
        #     self.text_tokens = pickle.load(txt_tokens)
        self.names = names
        self.text_tokens = text_tokens_path

    def __len__(self):
        return len(self.data_index['images'])

    def __getitem__(self, idx):
        # load image
        # img_path = os.path.join(self.data_path, self.data_index['images'][idx])
        # img_name = img_path.rsplit('/')[-1].split('.')[0]
        # image = self.vae.get_codebook_indices(img_name)
        #
        # # text data and label
        # text = self.data_index['texts'][idx]
        # text = self.text_tokens[img_name]

        img_name = self.names[idx]
        image = self.vae.get_codebook_indices(img_name)
        text = self.text_tokens(img_name)

        return image, text


def plot_loss(loss_array):
    plt.scatter(range(len(loss_array)), loss_array, c="red", s=1)
    plt.title('Plot of the Loss function')
    plt.xlabel('epochs')
    plt.ylabel('Train Loss')
    plt.show()


def train(data_path: str, text_tokens_path: str, model_args: dict, log_args:dict,
          chkpt_args:dict, trainer_args: dict, batch_size:int):
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

    # TODO: REMOVE BELOW CODE
    # Load the text tokenized file
    with (open(text_tokens_path, "rb")) as txt_tokens:
        text_tokens = pickle.load(txt_tokens)
    print(f'Text tokens file loaded!')
    all_keys = text_tokens.keys()
    # train and test
    train_keys = all_keys[:250]
    val_keys = all_keys[250:]
    text_tokens_path = text_tokens
    # ----------------------

    encoder_args = model_args['model_args']['ENCODER']
    print(f'Loading VQGAN...')
    vae = VQGanVAE(**encoder_args)
    print(f'VQGAN loaded!')

    train_dataset = UpdatedDatasetClass(train_keys, data_path, text_tokens_path, vae=vae, transform=transform, is_train=True)
    val_dataset = UpdatedDatasetClass(val_keys, data_path, text_tokens_path, vae=vae, transform=transform, is_train=False)

    # data loader  
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
    model = FinalModelV2(**model_args)
    checkpoint = ModelCheckpoint(**chkpt_args)
    logger = TensorBoardLogger(**log_args)
    # Instantiate the PyTorch Lightning Trainer
    trainer = L.Trainer(**trainer_args, callbacks=checkpoint, logger=logger)
    
    # Fit the model
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    # Organize arguments here
    train_args = cfg['TRAIN']
    model_def = train_args['model_def']
    model_instance_args = cfg['MODEL'][model_def]
    chkpt_args = cfg['CALLBACK']
    trainer_args = train_args['TRAINER']
    LR = train_args['LR']
    data_path = cfg['DATA']['PATH']
    text_tokens_path = cfg['DATA']['TEXT_TOKEN_PATH']
    model_args = {'model_def': model_def,
                  'model_args': model_instance_args,
                  'lr': LR}
    log_args = cfg['LOGGER']
    batch_size = train_args['BATCH_SIZE']

    # Train the model
    train(data_path, text_tokens_path, model_args, log_args, chkpt_args, trainer_args, batch_size=batch_size)
    