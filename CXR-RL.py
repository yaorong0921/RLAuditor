import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import pickle

import torch
import torchvision.transforms as transforms
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from captum.attr import LayerGradCam
import numpy as np
import os
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
# import gym
# from gym import spaces
import numpy as np
from typing import Optional
from copy import deepcopy

import torch.optim as optim

from dqn_model import DQN, StateActionQNetwork
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

BATCH_SIZE = 128
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 16
TARGER_UPDATE_FREQ = 1000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

# Custom CNN Model
class CNN5Layer(nn.Module):
    def __init__(self, num_classes=15):
        super(CNN5Layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Layer 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Layer 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Layer 3
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Layer 4
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # Layer 5
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling
        self.relu = nn.ReLU()  # Activation function
        
        self.fc = nn.Linear(512 * 7 * 7, num_classes)  # Fully Connected Layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, return_embedding=False):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        
        embeddings = x.view(x.size(0), -1)
        x = self.dropout(embeddings)
        x = self.fc(x)  # No activation (BCEWithLogitsLoss expects raw logits)
        if return_embedding:
            return x, embeddings
        return x

def evaluate_model_accuracy(model, dataloader):
    """
    Evaluate the accuracy of a model on a given dataloader.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader with the evaluation dataset.
        device (torch.device): The device to perform the evaluation on (CPU or GPU).

    Returns:
        float: The accuracy of the model as a percentage.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, targets in dataloader:
            # Move inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass to get predictions
            outputs = model(inputs)

            # Get predicted class (index of the maximum logit)
            _, predicted = torch.max(outputs, dim=1)

            # Update total and correct counts
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Calculate accuracy
    accuracy = (correct / total) * 100
    return accuracy

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
batch_size = 16
learning_rate = 0.001
num_epochs = 2

# Transformations for DICOM images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_annotations = pd.read_csv("../vu2/train.csv")

val_annotations = pd.read_csv("../vu2/CXR/val_unique_anno.pkl")
test_annotations = pd.read_csv("../vu2/CXR/test_unique_anno.pkl")

with open("../vu2/CXR/val_dataset_unique.pkl", "rb") as f:
    val_dataset = pickle.load(f)

with open("../vu2/CXR/test_dataset_unique.pkl", "rb") as f:
    test_dataset = pickle.load(f)


# val_dataset = [val_dataset[i] for i in unique_val]
# test_dataset = [test_dataset[i] for i in unique_test]

# DataLoaders for training and validation
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Display dataset sizes
# print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

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


with open("../vu2/RL4Auditing/CXR/val_target_list2.pkl", "rb") as f:
    target_list = pickle.load(f)
# Save the tensor to a file
y_proba = torch.load("../vu2/RL4Auditing/CXR/val_y_proba2.pt")
y_test = torch.load("../vu2/RL4Auditing/CXR/val_y_test2.pt")

# Save the list to a file
with open("../vu2/RL4Auditing/CXR/test_target_list2.pkl", "rb") as f:
    target4test_list = pickle.load(f)
# Save the tensor to a file
y4test_proba = torch.load("../vu2/RL4Auditing/CXR/test_y_proba2.pt")
y4test = torch.load("../vu2/RL4Auditing/CXR/test_y_test2.pt")

val_expl = np.load("../vu2/RL4Auditing/CXR/val_expls2.npy", allow_pickle=True)
test_expl = np.load("../vu2/RL4Auditing/CXR/test_expls2.npy", allow_pickle=True)

original_sizes = pd.read_csv("../vu2/CXR/train_meta.csv")

def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    Box format: [x_min, y_min, x_max, y_max]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Compute areas of the boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute union area
    union = box1_area + box2_area - intersection

    # Return IoU
    return intersection / union if union > 0 else 0

def check_positions_in_saved_explanations(explanations, annotations, original_sizes, all_annotations, top_k = 0.3):
    """
    Check if the values at specific positions in saved Grad-CAM explanations
    are larger than the top k% values, and return indices where any value fails the check.

    Args:
        explanations (dict): A dictionary where keys are indices and values are Grad-CAM arrays.
        positions_list (list of list of tuple): List of lists where each sublist contains (x, y) positions
                                                corresponding to an image in the original size (e.g., 300x300).
        k (float): Percentage (0-100) to define the top k% threshold.
        original_size (tuple): The original size (width, height) of the input images.
        target_size (tuple): The target size (width, height) of the Grad-CAM explanation.

    Returns:
        list: Indices of images where any position fails the check.
    """
    failing_indices = []
    threshold = 1e-5  # IoU threshold
    target_size = 224  # Target image size (224x224)
    K = int(top_k * explanations.shape[-1] * explanations.shape[-2])
    for index, explanation in enumerate(explanations):
        img_annotations = annotations.iloc[index]
        img_id = img_annotations['image_id']
        original_height = original_sizes.loc[original_sizes['image_id'] == img_id, 'dim0']
        original_width = original_sizes.loc[original_sizes['image_id'] == img_id, 'dim1']

        img_annotations = all_annotations.loc[all_annotations['image_id'] == img_id]

        boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].dropna().to_numpy().tolist()
        
        if len(boxes_viz) == 0: #No Findings
            continue
        
        # Rescale the explanation map to 224x224
#         explanation_resized = cv2.resize(explanation, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # Identify the top-k values in the 7x7 explanation map
        flat_indices = np.argsort(explanation, axis=None)[-K:]  # Flattened indices of top-k values
        top_k_indices = [np.unravel_index(i, explanation.shape) for i in flat_indices]

        # Calculate the corresponding areas in the 224x224 space for each top-k value
        cell_height = target_size // explanation.shape[0]
        cell_width = target_size // explanation.shape[1]

        top_k_areas = []
        for y, x in top_k_indices:
            top_k_areas.append([
                x * cell_width,
                y * cell_height,
                (x + 1) * cell_width - 1,
                (y + 1) * cell_height - 1
            ])

        # Rescale the bounding boxes to 224x224
        boxes_rescaled = []
        for box in boxes_viz:
            x_min_rescaled = int(box[0] / original_width * target_size)
            y_min_rescaled = int(box[1] / original_height * target_size)
            x_max_rescaled = int(box[2] / original_width * target_size)
            y_max_rescaled = int(box[3] / original_height * target_size)
            boxes_rescaled.append([x_min_rescaled, y_min_rescaled, x_max_rescaled, y_max_rescaled])

        # Check IoU of each top-k area with the bounding boxes
        max_iou = 0
        for area in top_k_areas:
            for box in boxes_rescaled:
                iou = compute_iou(area, box)
                max_iou = max(max_iou, iou)
        # If the max IoU is less than the threshold, record the index
        if max_iou < threshold:
            failing_indices.append(index)

    return failing_indices

index_list_expl = check_positions_in_saved_explanations(val_expl, val_annotations, original_sizes, train_annotations)
target_list += index_list_expl
target_list = list(set(target_list))

print(len(target_list))


import torch.nn.functional as F

def extract_embeddings_with_map(model, dataloader, maps):
    """
    Extracts embeddings from a model for images in a dataloader, using corresponding maps.

    Args:
        model (nn.Module): The PyTorch model modified to output embeddings.
        dataloader (DataLoader): PyTorch DataLoader providing batches of images.
        maps (torch.Tensor): Tensor of maps with shape (total_images, 1, map_height, map_width).
        device (str): Device to run the model on ("cpu" or "cuda").

    Returns:
        list[torch.Tensor]: List of embeddings for all images in the dataloader.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    intermediate_output = []
    # Define the hook function
    def hook_fn(module, input, output):
        intermediate_output.append(output.detach().cpu())
    # Register the hook
    hook = model.avgpool.register_forward_hook(hook_fn)
    # Process each batch of images from the dataloader
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # Extract corresponding maps for the batch
            start_idx = batch_idx * len(images)
            end_idx = start_idx + len(images)
            batch_maps = torch.stack([torch.tensor(maps[idx], dtype=torch.float32).unsqueeze(0).to(device) for idx in range(start_idx, end_idx)])

            # Resize maps to match the spatial dimensions of images
            batch_maps = F.interpolate(batch_maps, size=images.shape[2:], mode='bilinear', align_corners=False)

            # Ensure maps have the same number of channels as the images
            if batch_maps.shape[1] != images.shape[1]:
                batch_maps = batch_maps.expand(-1, images.shape[1], -1, -1)

            # Multiply images with maps
            modified_images = images * batch_maps

            outputs = model(modified_images)  # Forward pass

    # Concatenate all embeddings into a single tensor
    return torch.cat(intermediate_output, dim=0)


expl = extract_embeddings_with_map(model, val_loader, val_expl)
expl = expl.squeeze()
print("Embeddings Shape:", expl.shape)

index4test_list_expl = check_positions_in_saved_explanations(test_expl, test_annotations, original_sizes, train_annotations)
target4test_list += index4test_list_expl
target4test_list = list(set(target4test_list))

expl_test = extract_embeddings_with_map(model, test_loader, test_expl)
expl_test = expl_test.squeeze()
print("Embeddings Shape:", expl_test.shape)

print(len(target4test_list))


import gymnasium as gym
from gymnasium import spaces
# import gym
# from gym import spaces
import numpy as np
from typing import Optional
from copy import deepcopy

class AuditingEnv(gym.Env):
    """
    Custom Environment where the state is a matrix, and the action is selecting an index.
    """
    def __init__(self, state_shape=(25,), expl_shape=(50), action_list_length=50, target_list=[], dataset=[], explanation=[], dataset_label=[], max_steps=10000, test_env=False, index4test_list_expl=None):
        super(AuditingEnv, self).__init__()
        
        # Define the matrix as the observation space
        self.state_shape = state_shape
        self.expl_shape = expl_shape
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(state_shape[0] + expl_shape[0], ) #
        )
        self.target_list = deepcopy(target_list)
        self.dataset = np.float32(dataset)
        self.dataset_expl = np.float32(explanation)
        self.dataset_label = dataset_label

        # Define the action space as selecting an index from a list of length X
        self.action_list_length = action_list_length
        self.action_space = spaces.Discrete(action_list_length, start=0)

        # Initialize the state (the matrix)
        self.state = np.random.uniform(-0, 1, size=self.state_shape) #np.zeros(state_shape)
        self.current_step = 0
        self.max_steps = max_steps  # Maximum steps in an episode
        
        self.rewards = None
        self.episode_rewards = []
        
        self.target_unseen = []
        
        self.test_env = test_env
        
        self.diversity = np.ones(action_list_length)
        
        self.index4test_list_expl = index4test_list_expl

#     def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
#         """
#         Reset the environment to the initial state and return the initial observation.
#         """
# #         super().reset(seed=seed)
#         self.state = np.random.uniform(-0, 1, size=self.state_shape)
#         self.current_step = 0
#         info = self._get_info()
#         return self.state, info

    def reset(self, init_state=None):
        """Resets the environment to the initial state."""
        self.rewards = []
        self.rewards_expl = []
        if not init_state is None:
            self.state = init_state[:self.state_shape[0]]  #np.random.uniform(-0, 1, size=self.state_shape)
            self.expl = init_state[self.state_shape[0]:]
        else:
            self.state = np.zeros(self.state_shape) #np.zeros(state_shape)
            self.expl = np.zeros(self.expl_shape)
        self.current_step = 0
        self.target_unseen = deepcopy(self.target_list)
        self.dataset_seen = []
        self.done = False
#         self.episode_rewards = []
#         H = entropy(self.state.reshape(-1,5), base=base, axis=1)
        self.diversity = np.ones(self.action_list_length) #np.zeros(state_shape)
        state = np.concatenate((self.state, self.expl)) #self.state # 
        return state, self.diversity

    def step(self, action):
        """
        Execute the action and return the next state, reward, done, and info.
        Args:
            action (int): Index of the target list chosen by the agent.
        Returns:
            observation (array): The matrix (state).
            reward (float): Reward for the action.
            done (bool): Whether the episode is finished.
            info (dict): Additional debug information.
        """
        action = int(action)
        self.current_step += 1

        # Check if action is valid
        assert self.action_space.contains(action), f"Invalid action: {action}, Action space: {self.action_space}"

# #         # Get the reward for the action

        if not self.test_env:
            if action in self.target_unseen:
            # delete this sample
                self.target_unseen.remove(action)
                reward = 1 #+ entropy(self.diversity, base=base, axis=0)*0.1
            elif action in self.target_list:
                reward = 1 #+ entropy(self.diversity, base=base, axis=0)*0.1
            else:
                reward = 0
        else:
            if action in self.target_unseen:
                reward = 1
                self.target_unseen.remove(action)
                if not self.index4test_list_expl is None:
                    if action in self.index4test_list_expl:
                        reward_expl = 1
                    else:
                        reward_expl = 0
            else:
                reward = 0
                reward_expl = 0
#             self.rewards_expl.append(reward_expl)

        if not action in self.dataset_seen:
            self.dataset_seen.append(action)
#         self.rewards.append(reward)
        # Update the states
        if action in self.target_list:
            self.diversity[action] += 1

        pred = np.argmax(self.dataset[action,:])

        self.state[pred*15:(pred+1)*15] += self.dataset[action,:]
        self.expl[pred*2048:(pred+1)*2048] += self.dataset_expl[action,:]        
#         if action in self.dataset_seen:
#              self.state[pred*5:(pred+1)*5] += np.random.uniform(-0, 1, size=5) +1e-4
# #         else:
# #             self.dataset_seen.append(action)
#         state_reshape = self.state.reshape(5, -1)
#     #         state_reshape[self.dataset_label[action], pred] += 1
#     #         self.state = state_reshape.reshape(-1)
#         norm_rows = np.tile(np.linalg.norm(state_reshape, axis=1, keepdims=True), (1,5)).reshape(-1)
        norm_rows = np.linalg.norm(self.state)
        norm_diver = np.linalg.norm(self.diversity)
        norm_expl_rows = np.linalg.norm(self.expl)
        # Check if the episode is done
        self.done = (len(self.target_unseen) == 0) or (self.current_step >= self.max_steps)

        # Debugging information
        info = self._get_info()
        
#         truncated = False
        self.rewards.append(reward)

        if self.done:
            ep_rew = sum(self.rewards)
            self.episode_rewards.append(ep_rew)
#         arr = np.eye(5).reshape(-1)

        state = np.concatenate((self.state/norm_rows, self.expl/norm_expl_rows)) #self.state/norm_rows #

        return state, reward, self.done, self.dataset_seen, self.diversity/norm_diver  #abs(arr-self.state/norm_rows)

    def _get_info(self):
           return {
             "Step": self.current_step,
             "Matrix": self.state,
             "Target List": self.target_list,
           "Reward": self.rewards,
           "Unseen reward": self.target_unseen,
           "Diversity list": self.diversity,
           "Seen data": self.dataset_seen,
           "Reward Expl": self.rewards_expl}

    def close(self):
        """Clean up resources."""
        pass
    
import torch.optim as optim

from dqn_model import DQN, StateActionQNetwork
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule

BATCH_SIZE = 128
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 16
TARGER_UPDATE_FREQ = 1000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

def main(env, env_test, num_timesteps):

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return env.done #env.current_step >= num_timesteps 
    
    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1, initial_p=0.5)

    dqn_learing(
        env=env,
        env_test=env_test,
        q_func=StateActionQNetwork,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
        save_file_name="../vu2/RL4Auditing/CXR/expl_statistics_1000steps"
    )
    
n_classes = 15
n_features = 2048
env = AuditingEnv(state_shape=(n_classes*n_classes,), expl_shape=(n_classes*n_features,), action_list_length=len(y_proba), target_list=target_list, dataset=y_proba, explanation=expl, dataset_label=y_test, max_steps=1000)
env_test = AuditingEnv(state_shape=(n_classes*n_classes,), expl_shape=(n_classes*n_features,), action_list_length=len(y4test_proba), target_list=target4test_list, dataset=y4test_proba, explanation=expl_test, dataset_label=y4test, max_steps=1000, test_env=True, index4test_list_expl=index4test_list_expl)
env.reset()
env_test.reset()

main(env, env_test, 100000)
    