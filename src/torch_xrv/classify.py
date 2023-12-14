#!git clone https://github.com/mlmed/torchxrayvision or 
#pip install torchxrayvision

import os
import cv2
import torch
import numpy as np
import torchxrayvision as xrv

def classify(imgp,model):
    
    store = []

    for i in os.listdir(imgp):
        img = cv2.imread(i)
        img = cv2.resize(img,(224,224))
        img = xrv.datasets.normalize(img,255)
        img = img.mean(2)[np.newaxis,np.newaxis,:]
        img = torch.from_numpy(img)
        #print(img.size())

        out = model(img)
        store.append(dict(zip(model.pathologies,out[0].detach().numpy())))
        
    return (store)

def main():
        
    imgp = "./path/to/images"

    #chexpert labels
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
    # or load xrv.models.DenseNet(weights="densenet121-res224-mimic_nb")

    print("Model loaded!")
    print("Classes:\n",model.pathologies)
    
    results = classify(imgp,model)
    
    print(results)
    
if __name__ == "__main__":
    main()
