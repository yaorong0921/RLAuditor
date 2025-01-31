"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
# import gym.spaces
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name
from scipy.stats import entropy


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": [],
    "test_reward": [],
    "test_reward_expl": []
}


def dqn_learing(
    env,
    env_test,
    q_func,
    optimizer_spec,
    exploration,
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    save_file_name="statistics"
    ):

    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    env_test: gym.Env
        gym environment to test.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channel of input.
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    stopping_criterion: (env) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

#     if len(env.observation_space.shape) == 1:
#         # This means we are running on low-dimensional observations (e.g. RAM)
#         input_arg = env.observation_space.shape[0]
#     else:
#         img_h, img_w, img_c = env.observation_space.shape
#         input_arg = frame_history_len * img_c
    num_actions = env.action_space.n

    def index_to_mask(indices, mask_length=200):
        mask = np.ones(mask_length)
        for i in indices:
            mask[i] -= 1
        return mask

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, actions, expls, diversity, t, eps_threshold=None, mask=None):
        sample = random.random()
        if eps_threshold is None:
            eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)
            diversity = torch.from_numpy(diversity).type(dtype).unsqueeze(0)
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            if mask is None:
                action = model(Variable(obs, volatile=True), actions, expls).data.max(1)[1].cpu() #, Variable(diversity, volatile=True)
            else:
                action = (model(Variable(obs, volatile=True), actions, expls).data.cpu()*mask[np.newaxis, :]).max(1)[1] #, Variable(diversity, volatile=True)
            return action
        else:
            return torch.IntTensor([random.randrange(actions.shape[0])])
  
    # Test the model in the environment
    def test_rl_model(env, model, init_state=None, episodes=1):
        """
        Test a trained RL model in a test environment.

        :param env: Gym environment
        :param model: Trained RL model
        :param episodes: Number of episodes to run
        """
        avg_reward = 0
        avg_reward_expl = 0
        all_test_actions = Variable(torch.tensor(env_test.dataset).cuda(), volatile=True) 
        all_test_expls = Variable(torch.tensor(env_test.dataset_expl).cuda(), volatile=True)

        for episode in range(episodes):
            obs, diversity = env.reset(init_state)
            done = False
            step = 0  
            seen = []
            while not done:
                # Predict the action using the trained model
#                 obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)
                # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
                mask = index_to_mask(seen, mask_length=env.action_list_length)
                action = select_epilson_greedy_action(model, obs, all_test_actions, all_test_expls, diversity, t, eps_threshold=0.0, mask=mask) #model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
                obs, reward, done, seen, diversity = env.step(action)
                step += 1
                
            total_reward = sum(env.rewards)
            avg_reward += total_reward
            avg_reward_expl += sum(env.rewards_expl)
        env.close()
        return avg_reward/episodes, avg_reward_expl/episodes

    # Initialize target q function and q function
    Q = q_func(env.observation_space.shape[-1], env.dataset.shape, env.dataset_expl.shape).cuda() #.type(dtype)
    target_Q = q_func(env.observation_space.shape[-1], env.dataset.shape, env.dataset_expl.shape).cuda() #.type(dtype)
    
    all_actions = Variable(torch.tensor(env.dataset).cuda(), volatile=True)
    all_expls = Variable(torch.tensor(env.dataset_expl).cuda(), volatile=True)
    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    best_test_reward = -float('inf')
    last_obs, last_div = env.reset()
    learned_obs = None
    LOG_EVERY_N_STEPS = 10000

    for t in count():
        ### Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env):
            break

        ### Step the env and store the transition
        # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
        last_idx = replay_buffer.store_frame(last_obs, last_div)
        # encode_recent_observation will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        recent_observations, recent_diversity = replay_buffer.encode_recent_observation()
        # Choose random action if not yet start learning
        if t > learning_starts:
            action = select_epilson_greedy_action(Q, recent_observations, all_actions, all_expls, recent_diversity, t)[0] #[0, 0]
        else:
            action = random.randrange(num_actions)
        # Advance one step
        obs, reward, done, _, diversity = env.step(action)
        # clip rewards between -1 and 1
        reward = max(-1.0, min(reward, 1.0))
        # Store other info in replay memory
        replay_buffer.store_effect(last_idx, action, reward, done, diversity)
        # Resets the environment when reaching an episode boundary.
        if done:
            learned_obs = obs
            obs, div = env.reset()
        last_obs = obs

        ### Perform experience replay and train the network.
        # Note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Use the replay buffer to sample a batch of transitions
            # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
            # in which case there is no Q-value at the next state; at the end of an
            # episode, only the current state reward contributes to the target
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, diver_batch, next_diver_batch = \
                                                                                            replay_buffer.sample(batch_size)
            # Convert numpy nd_array to torch variables for calculation
            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype))
            diver_batch = Variable(torch.from_numpy(diver_batch).type(dtype))
            act_batch = Variable(torch.from_numpy(act_batch).long())
            rew_batch = Variable(torch.from_numpy(rew_batch))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype))
            next_diver_batch = Variable(torch.from_numpy(next_diver_batch).type(dtype))
            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()
            # Compute current Q value, q_func takes only state and output value for every state-action pair
            # We choose Q based on action taken.
            current_Q_values = Q(obs_batch, all_actions, all_expls).gather(1, act_batch.unsqueeze(1)).squeeze(-1)
            # Compute next Q value based on which action gives max Q values
            # Detach variable from the current graph since we don't want gradients for next Q to propagated
            next_max_q = target_Q(next_obs_batch, all_actions, all_expls).detach().max(1)[0]
            next_Q_values = not_done_mask * next_max_q
            # Compute the target of the current Q values
            target_Q_values = rew_batch + (gamma * next_Q_values)
            # Compute Bellman error
            bellman_error = target_Q_values - current_Q_values
            # clip the bellman error between [-1 ,  1]
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            # Note: clipped_bellman_delta * -1 will be right gradient
            d_error = clipped_bellman_error * -1.0
            # Clear previous gradients before backward pass
            optimizer.zero_grad()
            # run backward pass
            current_Q_values.backward(d_error.data) #.unsqueeze(1)

            # Perfom the update
            optimizer.step()
            num_param_updates += 1

            # Periodically update the target network by Q network to target Q network
            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())

####################
#         ### 4. Log progress and keep track of statistics
        if done:
            episode_rewards =  env.episode_rewards   # get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
#             if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
#                 if episode_rewards[-1] > max(episode_rewards[:-1]):
                    # Run the test
                test_reward, test_reward_expl = test_rl_model(env_test, Q, init_state=learned_obs) #
#                 print(entropy(learned_obs.reshape(-1,5), base=2, axis=1))
                if best_test_reward < test_reward:
                    best_test_reward = test_reward
                Statistic["test_reward"].append(test_reward)
                Statistic["test_reward_expl"].append(test_reward_expl)

            Statistic["mean_episode_rewards"].append(mean_episode_reward)
            Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)
            

        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("best test reward %f" % best_test_reward)
            print("Expl reward %f" % Statistic["test_reward_expl"][-1])
            sys.stdout.flush()

            # Dump statistics to pickle
            with open('%s.pkl'%save_file_name, 'wb') as f:
                pickle.dump(Statistic, f)
                print("Saved to %s.pkl"%save_file_name)
