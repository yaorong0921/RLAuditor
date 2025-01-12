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

# Loading Data Processing Libraries 
import numpy as np
import pandas as pd

from logging import getLogger
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

# from ignite.engine import Events, Engine

# Loading Imaging Libraries 
import pydicom as pdc
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
# import albumentations as A

import pandas as pd
import random
from torchvision import transforms

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
    
    
    # Hyperparameters and configuration
num_classes = 15  # 14 abnormalities + "No finding"
batch_size = 32
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


import torch
import torchvision.models as models

# Model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the number of output classes

# Load the pretrained ResNet-50 model
model = models.resnet50(pretrained=True)

# Modify the final fully connected layer to match the number of classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("ResNet-50 model loaded with a custom number of output classes.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import pydicom
from tqdm import tqdm
       
for epoch in range(num_epochs):
    # Training loop
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 200 == 0:
            print(batch_idx, loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if batch_idx % 200 == 0:
                print(batch_idx, 100 * correct / total)
    val_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
# Save the trained model
torch.save(model.state_dict(), '../vu2/CXR-model.pth')