import torch
import torch.nn as nn
import torch.nn.functional as F
# import ipdb

class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

class DQN_RAM(nn.Module):
    def __init__(self, in_features=4, num_actions=18):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN_RAM, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class StateActionQNetwork(nn.Module):
    def __init__(self, state_dim, action_shape, expl_shape, action_feature_dim=64):
        """
        Custom Q-network that incorporates state and action features.
        Args:
            state_dim (int): Dimension of the state space.
            action_dim: action list.
            action_feature_dim (int): Dimension of action feature space.
        """
        super(StateActionQNetwork, self).__init__()
        # State processing network
        self.base_feature_dim = 128
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, self.base_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.base_feature_dim * 2, self.base_feature_dim),
        )
#         self.action = action
        self.action_dim = action_shape[-1] # self.action.shape[-1]
        self.expl_dim = expl_shape[-1]
#         self.diversity_dim = action_shape[0]
        
        # Action feature embedding (for each action)
        self.action_feature_dim = 64 #self.base_feature_dim
        self.expl_feature_dim = 64
        self.action_embeddings = nn.Sequential(
            nn.Linear(self.action_dim, self.action_feature_dim),
            nn.LayerNorm(self.action_feature_dim),
            nn.ReLU())
        self.expl_embeddings = nn.Sequential(
            nn.Linear(self.expl_dim, self.expl_feature_dim),
            nn.LayerNorm(self.expl_feature_dim),
            nn.ReLU())
#         self.diversity_embeddings = nn.Linear(self.diversity_dim, self.diversity_dim)
        
#         self.diversity_embeddings = nn.Linear(self.diversity_dim, self.base_feature_dim)        
        
        # Combine state and action features
        self.q_net = nn.Sequential(
            nn.Linear(self.base_feature_dim + self.action_feature_dim, self.base_feature_dim*2),  #self.base_feature_dim + self.action_feature_dim, +self.expl_feature_dim, 
            nn.LayerNorm(self.base_feature_dim*2),
            nn.ReLU(),
            nn.Linear(self.base_feature_dim*2, self.base_feature_dim),
            nn.LayerNorm(self.base_feature_dim),
            nn.ReLU(),
            nn.Linear(self.base_feature_dim, 1),  # Output a single Q-value for state-action pair
        )
    def forward(self, state, action, expl):
        """
        Forward pass for the Q-network.
        Args:
            state (torch.Tensor): Batch of states (batch_size, state_dim).
            action (torch.Tensor): Batch of action indices (batch_size,).
        Returns:
            Q-values for the given state-action pairs (batch_size, 1).
        """
#         print("before")
#         ipdb.set_trace()
        state_features = self.state_net(state)  # Process state
#         print("after")
#         ipdb.set_trace()
        action_features = self.action_embeddings(action)  # Get action features
#         expl_features = self.expl_embeddings(expl)  # Get action features
        
        
#         all_action_features = torch.cat([action_features, expl_features], dim=-1)
#         q_diver = self.diversity_embeddings(diversity)

        obs_repeated = state_features.unsqueeze(1).expand(-1, action_features.shape[0], -1)
        all_actions = action_features.unsqueeze(0).expand(state_features.shape[0], -1, -1)
        
#         all_actions = action_features.unsqueeze(0).expand(state_features.shape[0], -1, -1)

#         all_diversity = q_diver.unsqueeze(1).expand(-1, action_features.shape[0], -1)
        combined = torch.cat([obs_repeated, all_actions], dim=-1)
#         obs_repeated = state.unsqueeze(1).expand(-1, action.shape[0], -1)
#         all_actions = action.unsqueeze(0).expand(state.shape[0], -1, -1)
#         combined = torch.cat([obs_repeated, all_actions], dim=-1)
        q_values = self.q_net(combined) # Compute Q-values
#         print(q_values)
        q = q_values.squeeze(-1)
        return q

# class StateActionQNetwork(nn.Module):
#     def __init__(self, observation_space, action_space):
#         super(StateActionQNetwork, self).__init__()
#         self.state_dim = observation_space.shape[0]
#         self.action_dim = action_space.n
#         action_feature_dim = 8  # Example: Embedding size for each action
#         self.q_net = QNetwork(self.state_dim, self.action_dim, action_feature_dim)

#     def forward(self, obs, deterministic=False):
#         """
#         Compute Q-values for all actions given a batch of states.
#         Args:
#             obs (torch.Tensor): Batch of states (batch_size, state_dim).
#         Returns:
#             Q-values for all actions (batch_size, action_dim).
#         """
#         batch_size = obs.shape[0]
#         all_actions = torch.arange(self.action_dim).expand(batch_size, -1).to(obs.device)
#         obs_repeated = obs.unsqueeze(1).expand(-1, self.action_dim, -1)
#         print(obs_repeated[:, 0, :].shape, all_actions[:, 0].shape)
# #         q_values = torch.stack([
# #             self.q_net(obs_repeated[:, i], all_actions[:, i])
# #             for i in range(self.action_dim)
# #         ], dim=1).squeeze(-1)
#         q_values = self.q_net(obs_repeated[:, 0, :], all_actions[:, 0])
#         print(q_values.shape)
#         return q_values
