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

import pydicom
import torch
import torchvision.models as models

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

from tqdm import tqdm

def get_wrong_predictions(model, dataloader):
    """
    Get the indices of wrongly predicted samples in the test dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader with the evaluation dataset.
        device (torch.device): The device to perform the evaluation on (CPU or GPU).

    Returns:
        list: A list of indices corresponding to wrongly predicted samples.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    wrong_indices = []
    prediction_list = []
    label_list = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass to get predictions
            outputs = model(inputs)
            predictions = torch.sigmoid(outputs).float()  # Convert logits to binary predictions
            prediction_list.extend(predictions.cpu())
            label_list.extend(targets.cpu())
            # Get predicted class (index of the maximum logit)
            _, predicted = torch.max(outputs, dim=1)

            # Find indices of incorrect predictions
            wrong = (predicted != targets).nonzero(as_tuple=True)[0]

            # Map batch indices to dataset indices
            batch_start_idx = batch_idx * dataloader.batch_size
            wrong_indices_batch = wrong.cpu().numpy() + batch_start_idx

            # Append the wrong indices for this batch
            wrong_indices.extend(wrong_indices_batch)
            if batch_idx % 40 == 0:
                print(batch_idx)

    return wrong_indices, torch.stack(prediction_list, dim=0), torch.stack(label_list, dim=0)

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

# Split the dataset into train, validation, and test sets
random_seed = 42
annotations = pd.read_csv('../vu2/train.csv')
train_size = int(0.6 * len(annotations))  # 60% for training
val_size = int(0.2 * len(annotations))    # 20% for validation
test_size = len(annotations) - train_size - val_size  # Remaining 20% for testing
indices = list(range(len(annotations)))
train_dataset_csv, val_dataset_csv, test_dataset_csv = random_split(
    indices, 
    [train_size, val_size, test_size], 
    generator=torch.Generator().manual_seed(random_seed)  # Use the random seed
)

val_annotations = annotations.iloc[val_dataset_csv.indices]
val_annotations = val_annotations.reset_index()

test_annotations = annotations.iloc[test_dataset_csv.indices]
test_annotations = test_annotations.reset_index()

unique_val = val_annotations.drop_duplicates(subset='image_id', keep='first').index.tolist()
unique_test = test_annotations.drop_duplicates(subset='image_id', keep='first').index.tolist()

val_dataset = [val_dataset[i] for i in tqdm(unique_val)]
test_dataset = [test_dataset[i] for i in tqdm(unique_test)]

# DataLoaders for training and validation
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Display dataset sizes
# print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
# # DataLoaders for training and validation
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

with open("../vu2/CXR/val_dataset_unique.pkl", "wb") as f:
    pickle.dump(val_dataset, f)

with open("../vu2/CXR/test_dataset_unique.pkl", "wb") as f:
    pickle.dump(test_dataset, f)

import pydicom
import torch
import torchvision.models as models

# Load the pretrained ResNet-50 model
model = models.resnet50(pretrained=True)

num_classes = 15  # 14 abnormalities + "No finding"

# Modify the final fully connected layer to match the number of classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load("../vu2/CXR-model.pth"))

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# # Evaluate accuracy
# accuracy = evaluate_model_accuracy(model, val_loader)
# print(f"Model Accuracy Val: {accuracy:.2f}%")

# # Evaluate accuracy
# accuracy = evaluate_model_accuracy(model, test_loader)
# print(f"Model Accuracy Test: {accuracy:.2f}%")

target_list, y_proba, y_test = get_wrong_predictions(model, val_loader)

target4test_list, y4test_proba, y4test = get_wrong_predictions(model, test_loader)


# Save the list to a file
with open("../vu2/RL4Auditing/CXR/test_target_list2.pkl", "wb") as f:
    pickle.dump(target4test_list, f)
# Save the tensor to a file
torch.save(y4test_proba, "../vu2/RL4Auditing/CXR/test_y_proba2.pt")
torch.save(y4test, "../vu2/RL4Auditing/CXR/test_y_test2.pt")


