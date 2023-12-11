import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
import time
from tqdm import tqdm
from torch import tensor
from torchmetrics.classification import MultilabelAUROC
CKPT_PATH = 'model.pth.tar' # start from pretrained chkpt
N_CLASSES = 10
CLASS_NAMES = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Fracture","Lung Lesion","Lung Opacity","Pleural Effusion","Pneumonia","Pneumothorax"]
DATA_DIR = '../../../multimodal-xray/data/downloaded_jpgs/physionet.org/files/mimic-cxr-jpg/2.0.0'
TEST_IMAGE_LIST = '../../../multimodal-xray/data/mimic-cxr-2.0.0-chexpert.csv'
BATCH_SIZE = 16
TRAIN_VAL_JSON = '../data'


def train(model, data_path, ckpt_path, device, resume=True):

    
    print(f'Training on device: {device}')

    best_val_auroc = 0.0
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()  # divide by max pixel value (255)
        # check correct shape for each image
    ])

    if resume:
        if os.path.isfile(ckpt_path):
            print(f"Loading checkpoint '{ckpt_path}'")
            checkpoint = torch.load(ckpt_path)

            # update the checkpoint from chexnet to be able to handle different number of classes in the last classification layer
            model_dict = model.state_dict()

            pretrained_dict = {k: v for k, v in checkpoint.items() if 'classifier' not in k}
            model_dict.update(pretrained_dict)
            #print(model_dict)
            model.load_state_dict(model_dict)
            #print(model)

            print("Checkpoint loaded")
        else:
            print(f"No checkpoint found at '{ckpt_path}', starting training from scratch")


    train_dataset = ChestXrayDataSet(data_path, transform=transform, is_train=0)
    val_dataset = ChestXrayDataSet(data_path, transform=transform, is_train=2)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)




    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        # Wrap the training loader with tqdm for a progress bar
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # prediction
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Update the progress bar with the current loss
            train_loop.set_postfix(loss=loss.item())


        val_auroc = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Validation AUROC: {val_auroc}')
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            # Save checkpoint with unique identifier (e.g., timestamp or epoch)
            timestamp = int(time.time())
            new_ckpt_path = f"{ckpt_path}_epoch_{epoch+1}_{timestamp}.pth.tar"
            save_checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auroc': best_val_auroc,
                'epoch': epoch
            }
            torch.save(save_checkpoint, new_ckpt_path)
            print(f"New checkpoint saved at {new_ckpt_path}")


def evaluate(model, val_loader, device):

    
    print(f'On device: {device}')

   

    auroc_metric = MultilabelAUROC(num_labels=N_CLASSES)


    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.int()
            outputs = model(images)
            predictions = torch.sigmoid(outputs)  # get probabilities
            auroc_metric.update(predictions, labels)

    aurocs = auroc_metric.compute() 
    avg_auroc = torch.mean(aurocs)
    print(f'Average AUROC: {avg_auroc.item()}')

    return avg_auroc





def test(data_path, new_ckpt, device):

    print(f'Testing on device: {device}')


    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES).to(device)

    if os.path.isfile(new_ckpt):
        print("=> loading checkpoint")
        checkpoint = torch.load(new_ckpt)
        #print(checkpoint['densenet121.features.conv0.weight'])
        model.load_state_dict(checkpoint["model_state_dict"])
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    test_dataset = ChestXrayDataSet(data_path, transform=transform, is_train=1)
    # 3,494 image label pairs
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_auroc = evaluate(model, test_loader, device)
    print(f'Average Test AUROC: {test_auroc}')

    

class DenseNet121(nn.Module):
    """
    This model is from: 
    https://github.com/arnoweng/CheXNet
    
    The architecture of the model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        # defining the classification layer
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == "__main__":

    data_path = "../data"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = DenseNet121(N_CLASSES)
    ckpt_path = CKPT_PATH

    new_ckpt = "model.pth.tar_epoch_3_1702271233.pth.tar"

    #train(model, data_path, ckpt_path, device, resume=True)

    test(data_path, new_ckpt, device)