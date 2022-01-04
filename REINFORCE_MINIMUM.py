import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98

LOAD_PERIOD = 20
MAX_EPISODE = 10000
MAX_BUFFER_LEN = 50000
EPSILON = 0.01
BATCH_SIZE = 32


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    policy = DQN()
    # print(policy.state_dict())
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate) # Try RMSprop
    for n_episode in range(MAX_EPISODE):
        score = 0.0
        state = env.reset()
        # print(state)

        done = False
        experience = []
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float)
            # state_tensor = torch.from_numpy(state).float()
            action_prob = policy(state_tensor)
            m = Categorical(action_prob)
            action = m.sample()
            # print(action.item())
            state_n, reward, done, _ = env.step(action.item())
            score += reward
            experience += [(state_tensor, action_prob[action], reward)]
            state = state_n
        # start training
        len_exp = len(experience)
        value_list = []
        value_temp = 0
        # calculate value function
        for i in range(len_exp):
            value_temp = gamma * value_temp + experience[::-1][i][2]
            value_list += [value_temp]
        value_list = value_list[::-1]
        # print(value_list[0],score)
        # print(value_list)
        print(score)
        for index, sampling in enumerate(experience):
            state = sampling[0]
            action_prob = sampling[1]
            reward = sampling[2]

            loss = -torch.log(action_prob) * value_list[index]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    env.close()