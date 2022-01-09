import gym
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

learning_rate = 0.0005
LOAD_PERIOD = 20
MAX_EPISODE = 50
# MAX_EPISODE = 10000
MAX_BUFFER_LEN = 50000
EPSILON = 0.01
BATCH_SIZE = 32
GAMMA = 0.99


env = gym.make('CartPole-v1')
N_INPUT = env.action_space.n
N_STATE = env.observation_space.shape[0]
N_ATOM = 51
V_MIN = -10
V_MAX = 10

class Replay_Buffer():
    def __init__(self):
        self.buffer = collections.deque([], maxlen = MAX_BUFFER_LEN)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size = BATCH_SIZE):
        return random.sample(self.buffer, batch_size)

    def len(self):
        return len(self.buffer)

class Q_func(nn.Module):

    def __init__(self, N_state, N_input, N_atom = 51):
        super(Q_func, self).__init__()

        self.N_state = N_state
        self.N_input = N_input
        self.N_atom = N_atom

        self.fc1 = nn.Linear(N_state, 400)
        self.fc2 = nn.Linear(400 , 300)
        self.fc3 = nn.Linear(300, N_input * N_atom)


    def forward(self, state):
        dim_state = state.dim()
        x = F.relu(self.fc1(state))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if dim_state == 1:
            x = F.softmax(x.view(-1, N_INPUT), dim = 0) # (N_ATOM, N_INPUT)
        elif dim_state == 2:
            x = F.softmax(x.view(-1, N_ATOM, N_INPUT), dim = 1) #(N_batch, N_ATOM, N_INPUT)
            # print(x.shape)
        return x



def categorical(state_tensor, input_tensor, reward, Q_func):
    N_input = Q_func.N_input
    N_state = Q_func.N_state
    N_atom = Q_func.N_atom

    q_out = Q_func(state_tensor)

    Q_list = []
    # for i in range(N_input):


class C51:
    def __init__(self):
        self.rb = Replay_Buffer()


        # print(N_state, N_input)
        self.q = Q_func(N_STATE, N_INPUT, N_ATOM)
        self.q_target = Q_func(N_STATE, N_INPUT, N_ATOM)
        self.q_target.load_state_dict(self.q.state_dict())
        self.Z_range_np = np.linspace(V_MIN, V_MAX, N_ATOM)
        self.Z_range_ts = torch.from_numpy(self.Z_range_np).float()

        self.Delta_z = (V_MAX - V_MIN) / (N_ATOM - 1)
        self.optimizer = optim.Adam(self.q.parameters(), lr=learning_rate) # Try RMSprop

    def sample_action(self, state, epsilon):
        # out = self.q.forward(state)
        coin = random.random()
        # print("coin", coin)
        if coin < epsilon:
            return random.randint(0,1) # TODO: modify this for more general learning settings
        else:
            q_out = self.q(state)
            # print(q_out.shape)
            # print(self.Z_range_ts.view(-1, 1) * q_out)
            q_sum = torch.sum(self.Z_range_ts.view(-1, 1) * q_out, dim=0)
            # print(q_sum)
            return q_sum.argmax().item()

    def learn(self):
        Delta_z = self.Delta_z


        mini_batch = self.rb.sample()
        state_list, action_list, reward_list, state_n_list, done_list = [], [], [], [], []
        for experience in mini_batch:
            state_temp, action_temp, reward_temp, state_n_temp, done_temp = experience
            state_list += [state_temp]
            action_list += [[action_temp]]
            reward_list += [[reward_temp]]
            state_n_list += [state_n_temp]
            done_list += [[done_temp]]

        state_tensor = torch.tensor(state_list, dtype=torch.float) #(N_batch, N_state)
        action_tensor = torch.tensor(action_list)
        reward_tensor = torch.tensor(reward_list) #(N_batch)
        state_n_tensor = torch.tensor(state_n_list, dtype=torch.float)
        done_tensor = torch.tensor(done_list, dtype=torch.float)

        # state_tensor_reshape = state_tensor.reshape(N_STATE,-1)
        # state_N_tensor_reshape = state_n_tensor.reshape(N_STATE, -1)
        # print(state_tensor_reshape.shape)
        p_out = self.q(state_tensor) #(N_batch, N_ATOM, N_INPUT)
        p_eval = torch.stack([p_out[i].index_select(1,action_tensor[i]) for i in range(BATCH_SIZE)]).squeeze(2) #(N_bacth, N_ATOM)
        # print("p_eval",p_eval.shape)
        p_out_n = self.q(state_n_tensor) #(N_batch, N_ATOM, N_INPUT)
        # print(q_out)

        # print(self.Z_range_ts.view(-1, 1).shape)
        # print((self.Z_range_ts.view(-1, 1) * p_out).shape)
        # p_sum = torch.sum(self.Z_range_ts.view(-1, 1) * p_out, dim=1) # (N_batch, N_INPUT)
        p_sum_n = torch.sum(self.Z_range_ts.view(-1, 1) * p_out_n, dim=1) # (N_batch, N_INPUT)
        # print(q_sum.shape)
        # a_max = torch.argmax(p_sum, dim=1) #(N_batch)
        a_max_n = torch.argmax(p_sum_n, dim=1) #(N_batch)
        # print(a_max.shape)
        m_torch = torch.zeros(BATCH_SIZE, N_ATOM)

        next_z_range = self.Z_range_ts.expand(BATCH_SIZE, -1) #(N_batch, N_ATOM)
        # print(next_z_range.shape)
        Tz = reward_tensor + GAMMA * next_z_range
        Tz = torch.clamp(Tz, min =V_MIN, max = V_MAX)
        # eps = torch.finfo(torch.float32).eps
        b = (Tz - V_MIN) / Delta_z

        l = torch.floor(b)
        u = torch.ceil(b)

        l_int = torch.floor(b).int()
        u_int = torch.ceil(b).int() #(N_batch, N_ATOM)
        # print(l,u)

        # p_out_max_n
        p_out_max_n, p_out_max_n_idxs = torch.max(p_out_n,dim=2) #p_out_max_n shape:(N_batch, N_ATOM)
        # print(p_out_max_n.shape)

        # p_add_u = p_out_max_n * (u - b)
        # p_add_l = p_out_max_n * (b - l)
        for i in range(BATCH_SIZE):
            for j in range(N_ATOM):
                if l_int[i,j] == u_int[i,j]:
                   m_torch[i, l_int[i,j]] += p_out_max_n[i,j]
                else:
                    m_torch[i,l_int[i,j]] +=  (p_out_max_n * (u - b))[i,j]
                    m_torch[i,l_int[i,j]] +=  (p_out_max_n * (b - l))[i,j]
        #m_torch shape: (N_batch, N_ATOM)
        # print(torch.sum(m_torch,dim=1))
        loss = - m_torch * torch.log(p_eval) #(N_batch, N_ATOM)
        # print("loss shape", loss.shape)
        loss = torch.mean(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def data_generation(self):
        for i in range(MAX_EPISODE):
            epsilon = EPSILON  # possibly decreasing
            # epsilon = max(0.01, 0.08 - 0.01 * (n_episode / 200))  # Linear annealing from 8% to 1%
            state = env.reset()
            done = False

            while not done:
                action = self.sample_action(torch.from_numpy(state).float(), epsilon)
                # print(action)
                state_n, reward, done, _ = env.step(action)
                # score += reward
                experience = (state, action, reward, state_n,
                              1 - done)  # 1 - done to flip the value of true and false. For mini batch learning

                self.rb.push(experience)
                state = state_n

                if done:
                    break
            len_rb = self.rb.len()

if __name__ == '__main__':
    C51 = C51()
    C51.data_generation()
    C51.learn()















