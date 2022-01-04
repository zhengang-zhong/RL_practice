import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98

LOAD_PERIOD = 20
MAX_EPISODE = 10000
MAX_BUFFER_LEN = 50000
EPSILON = 0.01
BATCH_SIZE = 32


class Replay_Buffer():
    def __init__(self):
        self.buffer = collections.deque([], maxlen = MAX_BUFFER_LEN)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size = BATCH_SIZE):
        return random.sample(self.buffer, batch_size)

    def len(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, state, epsilon):
        out = self.forward(state)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else :
            return out.argmax().item()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    rb = Replay_Buffer()
    q = DQN()
    q_target = DQN() # Target action NN
    q_target.load_state_dict(q.state_dict()) # Initialized with the same parameters
    optimizer = optim.Adam(q.parameters(), lr=learning_rate) # Try RMSprop

    for n_episode in range(MAX_EPISODE):
        epsilon = EPSILON # possibly decreasing
        epsilon = max(0.01, 0.08 - 0.01 * (n_episode / 200))  # Linear annealing from 8% to 1%
        state = env.reset()
        done = False

        score = 0.0
        while not done:
            action = q.sample_action(torch.from_numpy(state).float(), epsilon)
            state_n, reward, done, _ = env.step(action)
            score += reward
            experience = (state, action, reward, state_n, 1 - done) # 1 - done to flip the value of true and false. For mini batch learning

            rb.push(experience)
            state = state_n


            if done:
                break
        len_rb = rb.len()
        if len_rb >= BATCH_SIZE:
        # if len_rb >= 2000:
            # start to optimize NN

            # sample a mini batch from the reply buffer
            mini_batch = rb.sample()
            state_list, action_list, reward_list, state_n_list, done_list = [], [], [], [], []
            for experience in mini_batch:
                state_temp, action_temp, reward_temp, state_n_temp, done_temp = experience
                state_list += [state_temp]
                action_list += [[action_temp]]
                reward_list += [[reward_temp]]
                state_n_list += [state_n_temp]
                done_list += [[done_temp]]

            state_tensor = torch.tensor(state_list, dtype=torch.float)
            action_tensor = torch.tensor(action_list)
            reward_tensor = torch.tensor(reward_list)
            state_n_tensor = torch.tensor(state_n_list, dtype=torch.float)
            done_tensor = torch.tensor(done_list, dtype=torch.float)

            q_out = q(state_tensor)
            q_a = q_out.gather(1, action_tensor)  # gather the value of corresponding action
            # print(q_a)

            # print(q_target(state_n_tensor).max(1)) # q_target(state_n_tensor).max(1) to get maximum and the correponding index
            # print(q_target(state_n_tensor).max(1)[0].unsqueeze(1)) # unsqueeze along certain axis
            max_q_prime = q_target(state_n_tensor).max(1)[0].unsqueeze(1)
            target = reward_tensor + gamma * max_q_prime * done_tensor
            loss = F.smooth_l1_loss(q_a, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if n_episode % LOAD_PERIOD == 0 and n_episode != 0:
            q_target.load_state_dict(q.state_dict())
        print(score)
    env.close()