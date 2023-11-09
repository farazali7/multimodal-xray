'''
TRAINING SCRIPT
'''

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from src.models.model import FinalModel
import lightning as L


from config import cfg


def plot_loss(loss_array):
    plt.scatter(range(len(loss_array)), loss_array, c="red", s=1)
    plt.title('Plot of the Loss function')
    plt.xlabel('epochs')
    plt.ylabel('Train Loss')
    plt.show()


def train(data_path: str, model_args: dict, trainer_args: dict):
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

    # Instantiate the PyTorch Lightning Trainer
    trainer = L.Trainer(**trainer_args)

    # Fit the model
    # TODO: Checkpointing, callbacks, & distributed training support
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    # Organize arguments here
    train_args = cfg['TRAIN']
    model_def = train_args['model_def']
    model_instance_args = cfg['MODEL'][model_def]
    trainer_args = train_args['TRAINER']
    LR = train_args['LR']

    data_path = cfg['DATA']['PATH']
    model_args = {'model_def': model_def,
                  'model_args': model_instance_args,
                  'lr': LR}

    # Train the model
    train(data_path, model_args, trainer_args)
