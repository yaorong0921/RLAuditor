# Loading Basic Libraries 
import os
import gc   # No Version
from pathlib import Path

import random
import six
import sys
import warnings

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Union, List, Optional, Mapping
from tqdm import tqdm
from tqdm.notebook import tqdm

import yaml
import pickle
import pandas as pd
import random
from torchvision import transforms

# Loading Data Processing Libraries 
import numpy as np
import pandas as pd

from logging import getLogger

import pydicom as pdc
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2

# Loading Statistics Libraries 
import scipy as sp

# Loading Models Libraries 
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold
# import lightgbm as lgb
import xgboost as xgb
# import catboost as cb

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

class VinBigDataDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, num_samples=300):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Randomly sample the specified number of samples
#         self.annotations = self.annotations.sample(n=num_samples, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0] + ".dicom")
        dicom_image = pydicom.dcmread(img_path)
        
        # Convert to uint8 by scaling the pixel values
        img = dicom_image.pixel_array
        img = img.astype(np.uint8)  # Convert uint16 to uint8 by truncation

        # Optionally normalize the pixel values to the [0, 255] range
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        img = cv2.resize(img, (224, 224))  # Resize for VGG16

        if self.transform:
            img = self.transform(img)

        label = int(self.annotations.iloc[idx, 2])  # class_id
        return img, label
    
import torch
import torchvision.transforms as transforms
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from captum.attr import LayerGradCam
import numpy as np
import os

def save_gradcam_explanations_to_single_npy(model, test_loader, target_layer, output_file):
    """
    Generate and save Grad-CAM explanations for all images in the test set to a single .npy file.

    Args:
        model (torch.nn.Module): The PyTorch model to be explained.
        test_loader (torch.utils.data.DataLoader): The data loader for the test set.
        target_layer (torch.nn.Module): The layer in the model for which Grad-CAM is computed.
        output_file (str): File path to save the Grad-CAM explanations.

    Returns:
        None
    """
    # Ensure the model is in evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    explanations = {}
    image_count = 0

    # Iterate through the test set
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        for i in range(inputs.size(0)):
            input_tensor = inputs[i:i+1]  # Select a single image
            label = labels[i].item()

            # Initialize LayerGradCam with the model and target layer
            gradcam = LayerGradCam(model, target_layer)
            with torch.no_grad():
                # Compute the target class
                output = model(input_tensor)
                target_class = torch.argmax(output, dim=1).item()

            # Compute Grad-CAM attributions
            attributions = gradcam.attribute(input_tensor, target=target_class)

            # Upsample the attributions to match the input size
            attributions = attributions.squeeze().cpu().detach().numpy()

            # Normalize the attributions
            attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions))

            # Add the explanation to the dictionary
            explanations[image_count] = attributions

            image_count += 1

    # Save all explanations to a single .npy file
    np.save(output_file, explanations)
    
# Hyperparameters and configuration
num_classes = 15  # 14 abnormalities + "No finding"
batch_size = 16
learning_rate = 0.001
num_epochs = 2

# Transformations for DICOM images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset paths
train_csv = "../vu2/train.csv"
train_img_dir = "../vu2/test/"
test_img_dir = "../vu2/test/"

from torch.utils.data import random_split
full_dataset = VinBigDataDataset(
    csv_file='../vu2/train.csv',
    img_dir='../vu2/train',
    transform=transform,
    num_samples=300
)

with open("../vu2/CXR/train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)

with open("../vu2/CXR/val_dataset.pkl", "rb") as f:
    val_dataset = pickle.load(f)

with open("../vu2/CXR/test_dataset.pkl", "rb") as f:
    test_dataset = pickle.load(f)

# DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Display dataset sizes
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

train_annotations = pd.read_csv("../vu2/train.csv")
train_annotations.head(5)

# Specify output file
output_file = "../vu2/RL4Auditing/CXR/val_expls.npy"

# Generate and save Grad-CAM explanations
save_gradcam_explanations_to_single_npy(model, val_loader, target_layer, output_file)

