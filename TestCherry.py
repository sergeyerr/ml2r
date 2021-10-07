#!/usr/bin/env python3
# conda install -c conda-forge gym 
# conda install pytorch torchvision torchaudio cpuonly -c pytorch
# pip install cherry_rl
# pip install simple_rl

import gym
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
from simple_rl.run_experiments import parse_args

import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from Base.bpStateGenerators import random_state_generator
from BoxPlacementEnvironment import BoxPlacementEnvironment

env = gym.make('CartPole-v0').unwrapped

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gym.register(
    'BoxPlacementEnvironment-v0',
    entry_point = BoxPlacementEnvironment
)
test_state = random_state_generator((10, 10),5,1,3,2,8)
env = gym.make('BoxPlacementEnvironment-v0', bpState= test_state).unwrapped

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
        self.good = 0
        self.bad = 0

    def push(self, *args):
        """Save a transition"""
        reward = args[3]
        if reward > 0:
            if self.good < self.bad + 5:
                self.good += 1
                self.memory.append(Transition(*args))
        else:
            if self.bad < self.good + 5:
                self.bad += 1
                self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.linear1 = Linear(inputs, 16)
        self.linear2 = Linear(16, outputs)
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x))
        return x


env.reset()

ATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TARGET_UPDATE = 10

INPUTS = 200
ACTIONS = 101
policy_net = DQN(INPUTS, ACTIONS).to(device)
target_net = DQN(INPUTS, ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            if steps_done % 1000 == 0:
                print(policy_net(state)[0][34:38])
                print(policy_net(state).max(1))
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(ACTIONS)]], device=device, dtype=torch.long)


episode_durations = []

BATCH_SIZE = 64
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 50
for i_episode in range(num_episodes):
    print('episode', i_episode)
    # Initialize the environment and state
    env.reset(random_state_generator((10, 10),5,4,4,5,6))
    state = torch.from_numpy(env._next_observation().astype(np.float32)).unsqueeze(0)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        if reward == 1:
            print('count ', t, ' action ', action.item(), ' reward ', reward.item(), ' done ', done, ' ', env.nr_remaining_boxes, ' eps ', EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY))
        if not done:
            next_state = torch.from_numpy(env._next_observation().astype(np.float32)).unsqueeze(0)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()