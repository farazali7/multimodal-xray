# encoding: utf-8

"""
Read images and corresponding labels.

"""



import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class ChestXrayDataSet(Dataset):
    def __init__(self, path_trainval_json, transform=None, is_train=0):

    
        """
        Args:
            path_trainval_json : ../data
            transform: optional transform to be applied on a sample.
            is_train: 0 for train, 1 for test, 2 for val
        """
        if is_train == 0:
            path = 'path_to_label_train.json'
        elif is_train == 1:
            path = 'path_to_label_test.json'
        elif is_train == 2:
            path = 'path_to_label_val.json'
        else:
            print("indicate if it is train, test, or val")
        with open(path) as f:
            data = json.load(f)
        self.image_names = list(data.keys())
        self.labels = list(data.values())
        self.transform = transform
        self.is_train = is_train


    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

if __name__ == "__main__":

    data_path = "../data"

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])


    train_dataset = ChestXrayDataSet(data_path, transform=transform, is_train=0)
    val_dataset = ChestXrayDataSet(data_path, transform=transform, is_train=2)

