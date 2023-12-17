import torch 
import torch.nn as nn
import numpy as np 
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from PIL import Image


def test_model(model, dataloader, dataset_sizes, device):


    model.eval()   # Set model to evaluate mode

    dataiter = iter(dataloader)
    inputs, labels = next(dataiter)

    with torch.no_grad(): 
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs,1)
        print(preds)
    
    
    correct_predictions = (preds == labels).sum().item()
    print(correct_predictions)
    total = len(labels)
    print(total)

    prediction_accuracy = correct_predictions/total

    print('########################### Prediction Accuracy: ', prediction_accuracy, ' #######################')

def run_model(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        # Move the image tensor to the device
        image_tensor = image_tensor.to(device)

        # Make predictions
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        probabilities = nn.functional.softmax(outputs, dim=1)[0].tolist()

    return preds.item(), probabilities



def make_prediction(image_path):
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    ])

    # Load and transform the image
    image = Image.open(image_path)
    image_tensor = data_transforms(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    model_conv = torchvision.models.resnet18(pretrained=False)
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 4)
    model_conv.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))

    device = torch.device("cpu")
    model_conv = model_conv.to(device)

    preds, probabilities = run_model(model=model_conv, image_tensor=image_tensor, device=device)
    return preds, probabilities

    

if __name__ == '__main__':

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    ])

    data_path = './test'

    image_dataset = datasets.ImageFolder(data_path, data_transforms)


    # Add all datasets to loaders
    # Change batch-size to test different numbers of images
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=10000, shuffle=False)

    dataset_size = len(image_dataset)

    model_conv = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
    for param in model_conv.parameters():
            param.requires_grad = False

        
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 4)
    model_conv.load_state_dict(torch.load('model_weights.pth'))

    device=torch.device("cpu")
    model_conv = model_conv.to(device)

    test_model(model=model_conv,dataloader=dataloader, dataset_sizes=dataset_size, device=device)


