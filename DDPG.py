import gym
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

q_lr = 0.0005
policy_lr = 0.0005
gamma = 0.98
tau = 0.005
BATCH_SIZE = 32
MAX_BUFFER_LEN = 50000
MAX_EPISODE = 10000
EXPLORATION_NOISE = 0.1
class Replay_Buffer():
    def __init__(self):
        self.buffer = collections.deque([], maxlen = MAX_BUFFER_LEN)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size = BATCH_SIZE):
        return random.sample(self.buffer, batch_size)

    def len(self):
        return len(self.buffer)

class OU_noise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

class Q_func(nn.Module):

    def __init__(self, N_state, N_input):
        super(Q_func, self).__init__()

        self.fc1 = nn.Linear(N_state + N_input, 400)
        self.fc2 = nn.Linear(400 , 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, input):
        x = F.relu(self.fc1(torch.cat([state, input], 1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Policy_func(nn.Module):
    def __init__(self,N_state, N_input):
        super(Policy_func, self).__init__()
        self.fc1 = nn.Linear(N_state, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, N_input)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # env = gym.make('CartPole-v1')
    env = gym.make('Pendulum-v1')
    rb = Replay_Buffer()
    N_input = env.action_space.shape[0]
    N_state = env.observation_space.shape[0]
    q, q_target = Q_func(N_state, N_input), Q_func(N_state, N_input)
    q_target.load_state_dict(q.state_dict())
    # print(N_input,N_state)
    policy, policy_target = Policy_func(N_state, N_input),  Policy_func(N_state, N_input)
    policy_target.load_state_dict(policy.state_dict())
    ou_noise = OU_noise(mu=np.zeros(1))
    q_optimizer = optim.Adam(q.parameters(), lr=q_lr) # Try RMSprop
    policy_optimizer = optim.Adam(policy.parameters(), lr = policy_lr)

    score = 0.0

    for n_episode in range(MAX_EPISODE):
        state = env.reset()
        done = False

        while not done:
            # print(torch.from_numpy(state).float())
            action = policy(torch.from_numpy(state).float())
            # action = (action.data.numpy().flatten() + np.random.normal(0, EXPLORATION_NOISE, size=N_input)).clip(
                # env.action_space.low, env.action_space.high)
            action = (action.data.numpy().flatten() + ou_noise()[0]).clip(
                env.action_space.low, env.action_space.high).item()
            state_n, reward, done, _ = env.step([action])
            # print(state_n,reward,done)
            experience = (state, action, reward, state_n, 1 - done)
            rb.push(experience)
            state = state_n

            score += reward

        len_rb = rb.len()
        if len_rb >= BATCH_SIZE:
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

            target = reward_tensor + gamma * q_target(state_n_tensor, policy_target(state_n_tensor)) * done_tensor
            # print(state_tensor.shape,action_tensor.shape)
            q_loss = F.smooth_l1_loss(q(state_tensor, action_tensor), target.detach())
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            policy_loss = -q(state_tensor, policy(state_tensor)).mean()  # That's all for the policy loss.
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

        for param_target, param in zip(q_target.parameters(), q.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
        for param_target, param in zip(policy_target.parameters(), policy.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

        print(score)
        score = 0.0
    env.close()
if __name__ == '__main__':
    main()