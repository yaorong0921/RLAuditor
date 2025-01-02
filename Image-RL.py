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

# Custom Dataset Class
class NPZImageDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.images = data['x']
        self.labels = data['y']  # Multi-labels as a numpy array
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Use float for CE loss
        
        if len(image.shape) == 2:  # Grayscale to 3 channels
            image = image.unsqueeze(0).repeat(3, 1, 1)
        else:
            image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        if self.transform:
            image = self.transform(image)
        return image, label


# Training Code
def train_cnn(train_loader, test_loader, num_classes=15, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN5Layer(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # For multi-label classification
#     criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting Training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}")

        # Evaluation
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        print(f"Test Loss: {total_loss/len(test_loader):.4f}\n")

    print("Training Completed!")
    return model

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

    return wrong_indices, torch.stack(prediction_list, dim=0), torch.stack(label_list, dim=0)


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

        self.state[pred*10:(pred+1)*10] += self.dataset[action,:]
        self.expl[pred*25088:(pred+1)*25088] += self.dataset_expl[action,:]        
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
        save_file_name="expl_statistics_1000steps"
    )    

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
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        for i in range(inputs.size(0)):
            input_tensor = inputs[i:i+1]  # Select a single image
            label = labels[i].item()

            # Initialize LayerGradCam with the model and target layer
            gradcam = LayerGradCam(model, target_layer)

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

def check_positions_in_saved_explanations(explanations, positions_list, k, original_size, target_size):
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
    counter = 0
    K = k * target_size[0] * target_size[1]
    for index, explanation in explanations.items():
        if not positions_list[index]:
            counter += 1
            continue
        # Flatten and compute the top k% threshold
        flat_explanation = explanation.flatten()
        sorted_values = np.sort(flat_explanation)[::-1]  # Sort in descending order
        threshold = sorted_values[K - 1] if K <= len(sorted_values) else sorted_values[-1]
        # Get rescaled positions for this image
        rescaled_positions = [
            (int(x * target_size[0] / original_size[0]), int(y * target_size[1] / original_size[1]))
            for x, y in positions_list[index]
        ]

        # Check each rescaled position
        fail = False
        for x, y in rescaled_positions:
            if explanation[x, y] <= threshold:
                fail = True
                break

        if fail:
            failing_indices.append(index)
    return failing_indices

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

    all_embeddings = []

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

            # Forward pass through the model
            _, embeddings = model(modified_images, return_embedding=True)

            # Collect embeddings
            all_embeddings.append(embeddings.cpu())

    # Concatenate all embeddings into a single tensor
    return torch.cat(all_embeddings, dim=0)

VAL_NPZ = "../vu2/RL4Auditing/split_data/new2/val.npz"
TEST_NPZ = "../vu2/RL4Auditing/split_data/new2/test.npz"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Data Loaders
# train_dataset = NPZImageDataset(TRAIN_NPZ, transform=transform)
val_dataset = NPZImageDataset(VAL_NPZ, transform=transform)
test_dataset = NPZImageDataset(TEST_NPZ, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the trained model
model = CNN5Layer(num_classes=10)
model.load_state_dict(torch.load("../vu2/RL4Auditing/split_data/new2/cnn5layer_multi_label.pth"))

# Evaluate accuracy
accuracy = evaluate_model_accuracy(model, val_loader)
print(f"Model Accuracy Val: {accuracy:.2f}%")

# Evaluate accuracy
accuracy = evaluate_model_accuracy(model, test_loader)
print(f"Model Accuracy Test: {accuracy:.2f}%")


test_expl = np.load("../vu2/RL4Auditing/split_data/new2/test_expls.npy", allow_pickle=True).item()
val_expl = np.load("../vu2/RL4Auditing/split_data/new2/val_expls.npy", allow_pickle=True).item()

with open("../vu2/RL4Auditing/split_data/new2/val_position.pkl", 'rb') as file:
     val_location = pickle.load(file)
with open("../vu2/RL4Auditing/split_data/new2/test_position.pkl", 'rb') as file:  # Open the file in binary read mode
     test_location = pickle.load(file)
        
        
model.eval()
target_list, y_proba, y_test = get_wrong_predictions(model, val_loader)
target_layer = model.conv5
original_size = (10, 10)
target_size = (14, 14)
# print(test_location)
index_list_expl = check_positions_in_saved_explanations(val_expl, val_location, k=30, original_size=original_size, target_size=target_size)
target_list += index_list_expl

target_list = list(set(target_list))

print(len(target_list))

# Extract embeddings
expl = extract_embeddings_with_map(model, val_loader, val_expl)
print("Embeddings Shape:", expl.shape)

target4test_list, y4test_proba, y4test = get_wrong_predictions(model, test_loader)
index4test_list_expl = check_positions_in_saved_explanations(test_expl, test_location, k=30, original_size=original_size, target_size=target_size)
target4test_list += index4test_list_expl
target4test_list = list(set(target4test_list))
# Specify the class of interest
# target_class = 3  # Change this to the desired class index

# Get indices of wrongly predicted samples for the target class
# target4test_list, y4test_proba, y4test = get_wrongly_predicted_indices_for_class(model, test_loader, target_class, threshold=0.5)
print(len(target4test_list), (y4test).shape)

expl_test = extract_embeddings_with_map(model, test_loader, test_expl)
print("Embeddings Shape:", expl_test.shape)

n_classes = 10
n_features = 25088
env = AuditingEnv(state_shape=(n_classes*n_classes,), expl_shape=(n_classes*n_features,), action_list_length=len(y_proba), target_list=target_list, dataset=y_proba, explanation=expl, dataset_label=y_test, max_steps=1000)
env_test = AuditingEnv(state_shape=(n_classes*n_classes,), expl_shape=(n_classes*n_features,), action_list_length=len(y4test_proba), target_list=target4test_list, dataset=y4test_proba, explanation=expl_test, dataset_label=y4test, max_steps=1000, test_env=True, index4test_list_expl=index4test_list_expl)
env.reset()
env_test.reset()

main(env, env_test, 100000)