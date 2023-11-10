'''
TRAINING SCRIPT
'''

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import lightning as L
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DS

from config import cfg
from src.models.model import FinalModel

def plot_loss(loss_array):
    plt.scatter(range(len(loss_array)), loss_array, c="red", s=1)
    plt.title('Plot of the Loss function')
    plt.xlabel('epochs')
    plt.ylabel('Train Loss')
    plt.show()


def train(data_path: str, model_args: dict, chkpt_args:dict, trainer_args: dict):
    """Train a model.

    Args:
        data_path: Path to data files
        model_args: Dictionary of kwargs for model
        trainer_args: Dictionary of kwargs for Trainer

    Returns:
        Trained model instance.
    """
    # TODO: Create train and validation dataloaders via data_path: each dataloader must give batches of:
    # TODO: [x_img, x_txt, y] samples, where x_img and y are input images resized to 512x512, and x_txt is
    # TODO: corresponding report.
    train_dl = ...
    val_dl = ...
    
    # Instantiate the model
    model = FinalModel(**model_args)
    checkpoint = ModelCheckpoint(**chkpt_args)
    # Instantiate the PyTorch Lightning Trainer
    trainer = L.Trainer(**trainer_args,callback=checkpoint)
    
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
    model_args = {'model_def': model_def,
                  'model_args': model_instance_args,
                  'lr': LR}

    # Train the model
    train(data_path, model_args, chkpt_args, trainer_args)
    