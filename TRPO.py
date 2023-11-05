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
import math
from torch.autograd import Variable

def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
    x = torch.zeros(b.size(), device=b.device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
    return x


def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).item()

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).item()
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


# class ReplayBuffer:
#     def __init__(self, N_sample_max):
#         self.N_sample_max = N_sample_max
#         self.buffer = []
#         # self.position = 0
#
#     def push(self, state, action, reward, next_state, done):
#         if len(self.buffer) == self.N_sample_max:
#             self.buffer.pop(0)
#         self.buffer.append([state, action, reward, next_state, done])
#         # self.position = self.position + 1
#
#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         state, action, reward, next_state, done = map(np.stack, zip(*batch))
#         return state, action, reward, next_state, done
#
#     def __len__(self):
#         return len(self.buffer)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

class Actor(nn.Module):
    def __init__(self, Ns, Na, a_max, log_std=0):
        super(Actor, self).__init__()
        self.Na = Na
        self.a_max = a_max
        # self.r = r
        self.l1 = nn.Linear(Ns, 128)
        # self.l2 = nn.Linear(64, Na, bias =False) # map to mean value
        self.l2 = nn.Linear(128, 128)  # map to mean value
        self.l3 = nn.Linear(128, Na)  # map to mean value
        self.l3.weight.data.mul_(0.1)
        self.l3.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, Na) * log_std)

        # self.l1 = nn.Linear(Ns, 400)
        # self.l2 = nn.Linear(400, 300)
        # self.l3 = nn.Linear(300, Na, bias =False) # map to mean value



    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.a_max
        if x.shape[0] > 1:
            action_log_std = self.action_log_std.expand_as(x)
        else:
            action_log_std = torch.reshape(self.action_log_std, x.shape)
        action_std = torch.exp(action_log_std)


        return x, action_log_std, action_std


    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    # def get_std(self):
    #     return self.r

class Critic(nn.Module):
    def __init__(self, Ns):
    # def __init__(self, Ns, Na):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(Ns, 128)
        # self.l2 = nn.Linear(400 + Na, 300)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, s):
        x = F.relu(self.l1(s))
        # x = torch.cat([x, a], 1)
        x = F.relu(self.l2(x))
        # x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class TRPO():
    def __init__(self, env, parameter=None):
        Ns = env.observation_space.shape[0]
        Na = env.action_space.shape[0]
        a_max = torch.tensor(env.action_space.high, dtype=torch.float32)

        self.env = env

        self.Ns = Ns
        self.Na = Na

        self.a_max = a_max

        self.actor = Actor(Ns, Na, a_max)

        # self.critic = Critic(Ns, Na)
        self.critic = Critic(Ns)

        self.lr = parameter["lr"]
        self.N_rollouts = parameter["N_rollouts"]
        self.episode = parameter["episode"]
        self.T = parameter["T"]
        self.gamma = parameter["gamma"]

        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-3)

    def sample(self):
        N_rollouts = self.N_rollouts
        T = self.T
        env =  self.env
        actor = self.actor

        rollouts_list = []

        for i in range(N_rollouts):
            sample = []
            state, _ = env.reset()
            done = False
            score = 0.0
            for j in range(T):
            # while not done:
                action = actor.select_action(torch.tensor(state, dtype=torch.float32))
                next_state, reward, done, _, _ = env.step(action.detach())
                score += reward
                # print(done)
                sample.append([state, action.detach(), reward, next_state, done])
                if done:
                    state, _ = env.reset()
                else:
                    state = next_state
            print(i,"th sampling is done")

            state, action, reward, next_state, done = map(np.stack, zip(*sample))
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_tensor = torch.tensor(action, dtype=torch.float32)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            done_tensor = torch.tensor(done, dtype=torch.float32)
            sample_dict = dict()
            sample_dict["state"] = state_tensor
            sample_dict["action"] = action_tensor
            sample_dict["reward"] = reward_tensor
            sample_dict["next_state"] = next_state_tensor
            sample_dict["done"] = done_tensor
            rollouts_list += [sample_dict]

        return rollouts_list

    def estimate_advantages(self, states, rewards):
        critic = self.critic
        gamma = self.gamma

        last_state = states[-1, :]
        values = critic(states)
        last_value = critic(last_state.unsqueeze(0))
        Q_values = torch.zeros_like(rewards)
        for i in reversed(range(rewards.shape[0])):
            last_value = Q_values[i] = rewards[i] + gamma * last_value
        advantages = Q_values.view(-1,1) - values

        return advantages

    # def obj_func(self, rollouts_list):
    #     # 1. Calculate advantages
    #     N_rollouts = self.N_rollouts
    #     # gamma = self.gamma
    #
    #     critic = self.critic
    #     actor = self.actor
    #
    #     # std = actor.get_std()
    #
    #     advantage_list = []
    #
    #     critic_optimizer = self.critic_optimizer
    #     actor_optimizer = self.actor_optimizer
    #
    #     for i in range(N_rollouts):
    #         sample_dict = rollouts_list[i]
    #         state_tensor  = sample_dict["state"]
    #         action_tensor  = sample_dict["action"]
    #         reward_tensor = sample_dict["reward"]
    #         next_state_tensor  = sample_dict["next_state"]
    #         done_tensor = sample_dict["done"]
    #
    #         # for j in (state_tensor).shape[0]:
    #         advantages = self.estimate_advantages(state_tensor, reward_tensor)
    #         advantage_list += [advantages]
    #
    #     advantages = torch.cat(advantage_list, dim=0)
    #
    #     # 2. Policy gradient
    #     #2.1 get the density funtion
    #     # mean_list = []
    #     state_list = []
    #     action_list = []
    #     for i in range(N_rollouts):
    #         sample_dict = rollouts_list[i]
    #         state_tensor = sample_dict["state"]
    #         action_tensor = sample_dict["action"]
    #         mean_tensor = actor(state_tensor)
    #         # mean_list += [mean_tensor]
    #         state_list += [state_tensor]
    #         action_list += [action_tensor]
    #
    #     # mean = torch.cat(mean_list, dim=0).flatten()
    #     state = torch.cat(state_list, dim=0)
    #     action = torch.cat(action_list, dim=0)
    #
    #
    #     with torch.no_grad():
    #         fixed_log_probs = actor.get_log_prob(state, action)
    #
    #     log_probs = actor.get_log_prob(state, action)
    #     loss = - advantages.detach() * torch.exp(log_probs - fixed_log_probs)
    #
    #
    #
    #     # density_val = 1/(std * torch.sqrt(torch.tensor(2 * torch.pi, requires_grad=False))) * torch.exp(-(action - mean)**2/(2 * std**2))  # This is a 1d version
    #
    #     # loss_obj = torch.mean(density_val * advantages.detach().clone())
    #
    #
    #     # kl = (std**2 + (mean.detach().clone() - mean) ** 2) / (2.0 * std ** 2) - 0.5 # Only mean funtion
    #     # kl_mean = torch.mean(kl)
    #
    #
    #     return loss.mean(), advantages
        # return loss_obj, kl, advantages


        # actor_optimizer.zero_grad()
        # grads = torch.autograd.grad(loss_obj, actor.parameters())





        # Compute the Hessian of KL divergence


        # kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1)**2) / (2.0 * std1**2) - 0.5
        # kl = (std**2 + (mean.detach().clone() - mean) ** 2) / (2.0 * std ** 2) - 0.5 # Only mean funtion
        # kl_mean = torch.mean(kl)
        #
        # grads = torch.autograd.grad(kl_mean, actor.parameters(), create_graph=True)
        #
        # hessian = []
        # for grad in grads:
        #     grad_grad = []
        #     for g in grad.view(-1):
        #         # Compute the gradient of each gradient w.r.t. parameters
        #         g_grad = torch.autograd.grad(g, actor.parameters(), retain_graph=True)
        #         grad_grad.append(g_grad)
        #     hessian.append(grad_grad)
        # hessian = torch.autograd.functional.hessian(loss_obj, actor.parameters())
        # print(hessian)
        # actor_optimizer.zero_grad()
        # loss_obj.backward()
        #
        # for name, param in actor.named_parameters():
        #     print(name)
        #     print(param.grad)
        #     print('---' * 10)






        # loss = .5 * (advantages ** 2).mean()  # MSE
        # for _ in range(5):
        #     critic_optimizer.zero_grad()
        #     loss.backward()
        #     critic_optimizer.step()



        # 2. Calculate densities


        # 3. Get objective function
        print("test")



    # def calc_new_para(self, loss, kl, old_para, damping = 1e-1):
    #     actor = self.actor
    #
    #     grads = torch.autograd.grad(loss, actor.parameters(), retain_graph=True)
    #     loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
    #
    #     kl_clone = kl.clone()
    #     def Fvp(v):
    #         kl_mean = torch.mean(kl_clone)
    #         grads = torch.autograd.grad(kl_mean, actor.parameters(), create_graph=True)
    #         flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
    #
    #         kl_v = (flat_grad_kl * v).sum()
    #         grads = torch.autograd.grad(kl_v, actor.parameters(), retain_graph = True)
    #         flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()
    #
    #         return flat_grad_grad_kl + v * damping
    #
    #     stepdir = conjugate_gradients(Fvp, loss_grad.detach(), 10)
    #     # shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
    #     # lm = torch.sqrt(shs)
    #     # fullstep = stepdir / lm[0]
    #     # neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
    #     # success, new_params = linesearch(actor, loss, old_para, fullstep,
    #     #                                  neggdotstepdir / lm[0])
    #
    #     new_params = stepdir
    #     return new_params
        # neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        #
        # success, new_params = linesearch(model, get_loss, prev_params, fullstep,
        #                                  neggdotstepdir / lm[0])

    def train(self, N_iter = 1000):
        actor = self.actor
        critic = self.critic
        critic_optimizer = self.critic_optimizer
        actor_optimizer = self.actor_optimizer
        for i in range(N_iter):
            # 1. Use the single path or vine procedures to collect a set of state-action pairs along with
            # Monte Carlo estimates of their $Q$-values.
            rollouts_list = self.sample()



            # 2. By averaging over samples, construct the estimated objective and constraint in Equation (14).
            # loss, kl, advantages = self.obj_func(rollouts_list)
            # loss, advantages = self.obj_func(rollouts_list)
            # 1. Calculate advantages
            N_rollouts = self.N_rollouts
            # gamma = self.gamma

            # critic = self.critic
            # actor = self.actor

            # std = actor.get_std()

            advantage_list = []
            state_list = []
            action_list = []
            # critic_optimizer = self.critic_optimizer
            # actor_optimizer = self.actor_optimizer

            for i in range(N_rollouts):
                sample_dict = rollouts_list[i]
                state_tensor = sample_dict["state"]
                action_tensor = sample_dict["action"]
                reward_tensor = sample_dict["reward"]
                next_state_tensor = sample_dict["next_state"]
                done_tensor = sample_dict["done"]
                state_list += [state_tensor]
                action_list += [action_tensor]

                # for j in (state_tensor).shape[0]:
                advantages = self.estimate_advantages(state_tensor, reward_tensor)
                advantage_list += [advantages]

            advantages = torch.cat(advantage_list, dim=0)

            # 2. Policy gradient
            # 2.1 get the density funtion
            # mean_list = []

            # for i in range(N_rollouts):
            #     sample_dict = rollouts_list[i]
            #     state_tensor = sample_dict["state"]
            #     action_tensor = sample_dict["action"]
            #     mean_tensor = actor(state_tensor)
            #     # mean_list += [mean_tensor]
            #     state_list += [state_tensor]
            #     action_list += [action_tensor]

            # mean = torch.cat(mean_list, dim=0).flatten()
            state = torch.cat(state_list, dim=0)
            action = torch.cat(action_list, dim=0)

            with torch.no_grad():
                fixed_log_probs = actor.get_log_prob(state, action)

            # log_probs = actor.get_log_prob(state, action)
            # loss = - advantages.detach() * torch.exp(log_probs - fixed_log_probs)
            # loss = loss.mean()


            def get_loss(volatile=False):
                with torch.set_grad_enabled(not volatile):
                    log_probs = actor.get_log_prob(state, action)
                    action_loss = -advantages.detach() * torch.exp(log_probs - fixed_log_probs)
                return action_loss.mean()

            def Fvp_direct(v):
                damping = 1e-2
                # state_list = []
                # N_rollouts = len(rollouts_list)
                # for i in range(N_rollouts):
                #     sample_dict = rollouts_list[i]
                #     state_tensor = sample_dict["state"]
                #
                #     state_list += [state_tensor]
                # state = torch.cat(state_list, dim=0)
                kl = actor.get_kl(state)
                kl = kl.mean()

                grads = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
                flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

                kl_v = (flat_grad_kl * v).sum()
                grads = torch.autograd.grad(kl_v, actor.parameters())
                flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

                return flat_grad_grad_kl + v * damping

            # state_list = []
            # N_rollouts = len(rollouts_list)
            # for i in range(N_rollouts):
            #     sample_dict = rollouts_list[i]
            #     state_tensor = sample_dict["state"]
            #
            #     state_list += [state_tensor]
            # state = torch.cat(state_list, dim=0)
            # kl = actor.get_kl(state)
            # max_kl =torch.max(kl.detach())

            loss = get_loss()
            grads = torch.autograd.grad(loss, actor.parameters())
            loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
            stepdir = conjugate_gradients(Fvp_direct, -loss_grad, 10)

            # shs = 0.5 * (stepdir.dot(Fvp_direct(stepdir)))

            # lm = math.sqrt(max_kl / shs)
            # fullstep = stepdir * lm
            fullstep = stepdir
            expected_improve = -loss_grad.dot(fullstep)

            prev_params = get_flat_params_from(actor)
            new_params = 0.01 * fullstep + prev_params
            # success, new_params = line_search(actor, get_loss, prev_params, fullstep, expected_improve)
            set_flat_params_to(actor, new_params)

            # actor_optimizer.zero_grad()
            # loss.backward()
            # actor_optimizer.step()

            # print("success", success)
            print("obj", loss)



            # 3. Update critic NN
            # 3. Approximately solve this constrained optimization problem to update the policy's parameter vector $\theta$.
            # Conjugate gradient and line search
            loss_critic = torch.mean(advantages**2)
            critic_optimizer.zero_grad()
            loss_critic.backward()
            critic_optimizer.step()
            print("advantage", loss_critic)

            # 3. Update actor NN
            # flat_para = get_flat_params_from(actor)
            # new_para = self.calc_new_para(loss, kl, flat_para)
            # set_flat_params_to(actor, new_para)




def evaluate_policy(policy, env, episodes=100, N_step = 200):
    state_trajectories = []
    rewards = []
    for _ in range(episodes):
        episode_reward = 0

        observation,_ = env.reset()
        episode_states = []

        i = 0
        while i <= N_step:
            episode_states.append(observation)

            with torch.no_grad():
                # action_dist = policy(torch.tensor(observation, dtype=torch.float32))
                action = policy.select_action(torch.tensor(observation, dtype=torch.float32).detach())
                # action = torch.argmax(action_dist).item()
                # action = torch.distributions.Categorical(action_dist).sample().item()

            observation, reward, done, _, _ = env.step(action)
            episode_reward += reward
            if done:
                break

            i += 1
        state_trajectories.append(episode_states)
        rewards.append(episode_reward)

    return state_trajectories, rewards
def plot_state_trajectories(trajectories):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    state_names = ["x1", "x2", "x3"]

    for i, state_name in enumerate(state_names):
        for trajectory in trajectories:
            states = [x[i] for x in trajectory]
            axes[i].plot(states)
        axes[i].set_title(state_name)
        axes[i].set_xlabel('Timesteps')
        axes[i].set_ylabel('Value')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # env = gym.make('BipedalWalker-v3')
    parameter = dict()
    parameter["lr"] = 1e-2
    parameter["N_rollouts"] = 20
    parameter["episode"] = 20
    parameter["T"] = 100
    parameter["gamma"] = 0.99

    env = gym.make('Pendulum-v1')
    ddpg = TRPO(env, parameter)
    batch_size = 64
    # T = 200
    N_iter = 1000

    ddpg.train(N_iter = N_iter)

    trajectories, rewards = evaluate_policy(ddpg.actor, env, episodes=1, N_step = 200)
    plot_state_trajectories(trajectories)
    print(rewards)