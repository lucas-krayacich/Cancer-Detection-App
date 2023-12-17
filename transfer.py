import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import numpy as np 
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt 
import time 
import os 
import copy

from train import train_model

############# TRANSFER LEARNING RESNET-18 ####################
device = torch.device("cpu")

# Handle data

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    ]),
}

if __name__ == '__main__':
    data_path = './chest_cancer_CT_Scans/Data'

    # Create custom datasets for training and validation sets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x),
                                            data_transforms[x])
                    for x in ['train', 'valid']}

    # Add all datasets to loaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                    for x in ['train', 'valid']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}


    # fine tune the classification layers on ct scans
    model_conv = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')

    # uncomment below to perform only fine-tuned training
    # for param in model_conv.parameters():
    #     param.requires_grad = False

    # Only the new fc layers will be trained
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 4)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

    model_conv = train_model(model=model_conv, criterion=criterion, 
                            optimizer=optimizer_conv, scheduler=exp_lr_scheduler, 
                            num_epochs=25, dataloaders=dataloaders, 
                            dataset_sizes=dataset_sizes, device=device)

