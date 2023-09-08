import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import gymnasium as gym
import random
import random
import numpy as np

def analyze_rewards(rewards):
    average_reward = sum(rewards) / len(rewards)
    std_dev = (sum([(r - average_reward) ** 2 for r in rewards]) / len(rewards)) ** 0.5
    return average_reward, std_dev

def evaluate_policy(policy, env, polycy_type, episodes=100, N_step = None):
    state_trajectories = []
    input_trajectories = []
    rewards = []
    for _ in range(episodes):
        episode_reward = 0

        # observation, _ = env.reset(seed = 1)
        observation, _= env.reset()
        episode_states = []
        episode_inputs = []

        if N_step is not None:
            i = 0
            while i <= N_step:
                episode_states.append(observation)

                with torch.no_grad():
                    action_dist = policy(torch.tensor(observation, dtype=torch.float32))
                    if polycy_type == "DDPG":
                        # action = action_dist.numpy().reshape(1)
                        action = action_dist
                    elif polycy_type == "PPO":
                        # action = torch.argmax(action_dist).numpy().reshape(1)
                        action = torch.argmax(action_dist).numpy().reshape(1).item()  # for cart pole
                    else:
                        print("Policy type error")
                if type(action) != np.ndarray:
                    episode_inputs.append(np.array(action).reshape(-1))
                else:
                    episode_inputs.append(action)
                observation, reward, done, _, _= env.step(action)
                episode_reward += reward

                i += 1
                if done:
                    break
        else:
            while True:
                episode_states.append(observation)

                with torch.no_grad():
                    action_dist = policy(torch.tensor(observation, dtype=torch.float32))
                    if polycy_type == "DDPG":
                        action = action_dist
                        # action = action_dist.numpy().reshape(1)
                    elif polycy_type == "PPO":
                        # action = torch.argmax(action_dist).numpy().reshape(1)
                        action = torch.argmax(action_dist).numpy().reshape(1).item() # for cart pole
                    else:
                        print("Policy type error")
                if type(action) != np.ndarray:
                    episode_inputs.append(np.array(action).reshape(1))
                else:
                    episode_inputs.append(action)
                observation, reward, done, _, _= env.step(action)
                episode_reward += reward
                if done:
                    break

        state_trajectories.append(episode_states)
        input_trajectories.append(episode_inputs)
        rewards.append(episode_reward)

    return state_trajectories, input_trajectories, rewards

class ReplayBuffer:
    def __init__(self, N_sample_max):
        self.N_sample_max = N_sample_max
        self.buffer = []
        # self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) == self.N_sample_max:
            self.buffer.pop(0)
        self.buffer.append([state, action, reward, next_state, done])
        # self.position = self.position + 1

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, Ns, Na, a_max):
        super(Actor, self).__init__()
        self.a_max = a_max

        self.l1 = nn.Linear(Ns, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, Na)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.a_max
        return x

class Critic(nn.Module):
    def __init__(self, Ns, Na):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(Ns, 400)
        self.l2 = nn.Linear(400 + Na, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, s, a):
        x = F.relu(self.l1(s))
        x = torch.cat([x, a], 1)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG():
    def __init__(self, env, parameter=None):
        self.env = env

        Ns = env.observation_space.shape[0]
        Na = env.action_space.shape[0]
        a_max = env.action_space.high.item()

        self.Ns = Ns
        self.Na = Na

        self.a_max = a_max

        self.actor = Actor(Ns, Na, a_max)
        self.actor_target = Actor(Ns, Na, a_max)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(Ns, Na)
        self.critic_target = Critic(Ns, Na)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.lr = parameter["lr"]

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)



    def replay_buffer_init(self, replay_buffer, N_size):
        state, _ = env.reset()
        actor = self.actor
        # score = 0.0
        for j in range(N_size):
            action = actor(torch.from_numpy(state))
            next_state, reward, done, _, _ = env.step(action.detach())
            # score += reward

            replay_buffer.push(state, action.detach(), reward, next_state, done)

            if done:
                state, _ = env.reset()
            else:
                state = next_state

    def train(self, replay_buffer, T, episode = 10, batch_size=64, gamma=0.99, tau=0.005):
        env = self.env
        Na = self.Na
        critic = self.critic
        critic_target = self.critic_target

        actor = self.actor
        actor_target = self.actor_target

        actor_optimizer = self.actor_optimizer
        critic_optimizer = self.critic_optimizer
        loss = torch.nn.MSELoss(reduction='mean')
        for i in range(episode):
            state, _ = env.reset()
            score = 0.0
            for j in range(T):

                action = actor(torch.from_numpy(state)) + torch.normal(0, 0.001, size=(Na,))
                next_state, reward, done, _, _ = env.step(action.detach())
                score += reward


                replay_buffer.push(state, action.detach(), reward, next_state, done)



                state_rb, action_rb, reward_rb, next_state_rb, done_rb = replay_buffer.sample(batch_size)
                state_tensor = torch.tensor(state_rb, dtype=torch.float)
                action_tensor = torch.tensor(action_rb)
                reward_tensor = torch.tensor(reward_rb)
                next_state_tensor = torch.tensor(next_state_rb, dtype=torch.float)
                done_tensor = torch.tensor(done_rb, dtype=torch.float)

                y = reward_tensor + gamma * critic_target(next_state_tensor, action_tensor).view(-1)
                L = loss(y, critic(state_tensor, action_tensor).view(-1))
                critic_optimizer.zero_grad()
                L.backward()
                critic_optimizer.step()

                actor_loss = -self.critic(state_tensor, self.actor(state_tensor)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                for param_target, param in zip(critic_target.parameters(), critic.parameters()):
                    param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
                for param_target, param in zip(actor_target.parameters(), actor.parameters()):
                    param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
                if done:
                    state, _ = env.reset()
                else:
                    state = next_state

            print(score)
            print(i, "-th iteration is done \n")
            state_trajectories, input_trajectories, rewards = evaluate_policy(actor, env, "DDPG", episodes=10, N_step=200)
            average_reward, std_dev = analyze_rewards(rewards)
            print(f"Average Reward: {average_reward:.2f}")
            print(f"Standard Deviation of Reward: {std_dev:.2f}")
        env.close()








if __name__ == '__main__':
    # env = gym.make('BipedalWalker-v3')
    parameter = dict()
    parameter["lr"] = 5e-4
    env = gym.make('Pendulum-v1')
    ddpg = DDPG(env, parameter)
    batch_size = 64
    MAX_BUFFER_LEN = 50000
    rb = ReplayBuffer(MAX_BUFFER_LEN)
    T = 200
    N_episode = 1000

    ddpg.replay_buffer_init(rb, batch_size)



    ddpg.train(rb, T, episode = N_episode, batch_size=batch_size, gamma=0.99, tau=0.001)

