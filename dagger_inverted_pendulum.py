import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import gym
import random

import numpy as np
import casadi as ca

import scipy
import scipy.optimize
import scipy.integrate
from scipy.linalg import solve_discrete_are

from stable_baselines3.common.env_checker import check_env


# In[3]:


def integrator_rk4(f, x, u, delta_t):
    '''
    This function calculates the integration of stage cost with RK4.
    '''

    k1 = f(x, u)
    k2 = f(x + delta_t / 2 * k1, u)
    k3 = f(x + delta_t / 2 * k2, u)
    k4 = f(x + delta_t * k3, u)


    x_next = x + delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


# ref: Imitation Learning from Nonlinear MPC via the Exact Q-Loss and its Gauss-Newton Approximation

# In[4]:


def inverted_pendulum(x, u):
    M = 1
    m = 0.1
    l = 0.8
    g = 9.81

    dx1_dt = x[1]
    dx2_dt = (- m * l * ca.sin(x[2]) * x[3] ** 2 - m * g * ca.cos(x[2]) * ca.sin(x[2]) + u[0]) / (M + m - m * ca.cos(x[2]) ** 2)
    dx3_dt = x[3]
    dx4_dt =  ( - m * l * ca.cos(x[2]) * ca.sin(x[2]) * x[3] ** 2 + (M + m) * g * ca.sin(x[2]) +  u[0] * ca.cos(x[2])) / ((M + m) * l - m * l * ca.cos(x[2]) ** 2)
    

    rhs = [dx1_dt,
           dx2_dt,
           dx3_dt,
           dx4_dt
           ]

    return ca.vertcat(*rhs)


# In[5]:


def dlqr(A,B,Q,R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # first, solve the ricatti equation
    P = np.matrix(solve_discrete_are(A, B, Q, R))
    print("P is:",P)
    # compute the LQR gain
    K = np.matrix(np.linalg.inv(B.T*P*B+R)*(B.T*P*A))
    return -K, P


def rk4(func, y0, t, args=()):
    
    control = args[0].item()
    
    h = t[1] - t[0]
    
#     print(h, control, type(func(y0, t[0], control)))
    
    k1 = h * np.array(func(y0, t[0], control))
    k2 = h * np.array(func(y0 + 0.5 * k1, t[0] + 0.5 * h, control))
    k3 = h * np.array(func(y0 + 0.5 * k2, t[0] + 0.5 * h, control))
    k4 = h * np.array(func(y0 + k3, t[0] + h, control))
    
    value = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return value.reshape(1,-1)


# In[225]:


class Inverted_pendulum_env(gym.Env):
    """

    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    def __init__(self, *args, **kwargs):
        y_setpoint = [0,0,0,0]
        self.y_target = np.array(y_setpoint)
        
        self.Q = np.diag([0.25, 0.025, 0.25, 0.025])
        
        x1_start = 0.
        x2_start = 0.
        x3_start = np.pi/6
        x4_start = 0.       
        
        self.x0 = np.array([x1_start, x2_start, x3_start, x4_start]).reshape(1, -1)
        # define hard bounds on the (continuous) controls
        self.u_lb = [-25]
        self.u_ub = [25]
        
        x1_ub = 2
        x2_ub = 4
        x3_ub = np.pi/3
        x4_ub = 2

        x1_lb = -x1_ub
        x2_lb = -x2_ub
        x3_lb = -x3_ub
        x4_lb = -x4_ub
        
        self.lbx = [x1_lb, x2_lb, x3_lb, x4_lb]
        self.ubx = [x1_ub, x2_ub, x3_ub, x4_ub]          
        

        # dimensions
        self.nx = self.x0.shape[1]
        self.nu = 1

        # time discretisation
        self.t0 = 0
        self.tf = 2.5
        self.tspan = [self.t0, self.tf]
        self.steps = 50
        self.dt = (self.tf - self.t0)/self.steps
        self.reset()
        
        # define control space 
        self.action_space = gym.spaces.Box(
            low = np.array(self.u_lb), high=np.array(self.u_ub), dtype=np.float64)
        # define observation space (needs modification)
        # self.observation_space = gym.spaces.Box(
        #     low = np.zeros(self.nx),
        #     high = np.ones(self.nx)*np.array([1., 300.]), dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low = np.ones(self.nx)* np.array(self.lbx),
            high = np.ones(self.nx)* np.array(self.ubx), dtype=np.float64)
#         print("obs shape",np.shape(self.observation_space))

    def _model(self, state, t, control):    
        M = 1
        m = 0.1
        l = 0.8
        g = 9.81
        
        x1, x2, x3, x4 = state.tolist()
        u = control

        dx1_dt = x2
        dx2_dt = (- m * l * ca.sin(x3) * x4 ** 2 - m * g * ca.cos(x3) * ca.sin(x3) + u) / (M + m - m * ca.cos(x3) ** 2)
        dx3_dt = x4
        dx4_dt =  ( - m * l * ca.cos(x3) * ca.sin(x3) * x4 ** 2 + (M + m) * g * ca.sin(x3) +  u * ca.cos(x3)) / ((M + m) * l - m * l * ca.cos(x3) ** 2)

        return dx1_dt, dx2_dt, dx3_dt, dx4_dt 
    

    
  
    def _integration(self, t_step, state, control):
        nx = self.nx
        dt = self.dt
        t_current = t_step*dt
        
        ts = [t_current,t_current + dt]
#         x_next = scipy.integrate.RK45(self._model, state, ts, 1, args = (control,))        
#         x_next = scipy.integrate.odeint(self._model, state, ts, args = (control,))
        x_next = rk4(self._model, state, ts, args=(control,))
#         print(x_next, "111nextshape", np.shape(x_next))

#         print("x_next",np.array(x_next)[-1,:].reshape(1,nx))
#         return np.array(x_next)[-1,:].reshape(1,nx)
        return x_next


    def _RESET(self):
        '''
        Create and initialize all variables and containers.
        Nomenclature:
            
        '''
        N, nx, nu = self.steps, self.nx, self.nu
        
        # simulation result lists
        self.state_traj = np.zeros((N+1, nx))
        self.control_traj = np.zeros((N, nu))
        self.current_state = self.x0.squeeze()
#         print("current state", self.current_state)
        y_obs  = np.array(self._state_obs(self.current_state)).reshape(-1)
        self.state_traj[0,:] = self.current_state
        self.period = 0
#         print(y_obs, "shape", np.shape(y_obs))
        return y_obs     
    
    def _state_obs(self, x):
        """ Obserse system output"""
#         x = x.reshape(1,-1)
#         nx = self.nx
#         y  = x[0,0] # Get the density of cA
        return x


    def _reward_allocation(self, x, u):
        Q = self.Q
        state_diff = x - self.y_target
        state_diff = state_diff.reshape(-1,1)
#         print(state_diff)
#         return - state_diff.T @ Q @ state_diff

        return state_diff.T @ Q @ state_diff

    def _STEP(self, control):
        '''
        Take a step in time in the multiperiod single stage parallel production scheduling problem.
        action = [integer; dimension |units|] order to process in each unit as scheduled by the control function
        '''     
        # define state of the system at current time period
        nu  = self.nu
        t = self.period
        x = self.current_state.copy()
        u = control.copy()
        
        # ---- integrate system ---- #
#         print("sys state and input",x,u)
        x_new   = self._integration(t, x, u).flatten() 
#         print(x_new.shape)
        self.state = x_new      
        self.current_state = x_new.copy()
#         print("current state", self.current_state.reshape(1,-1))
        
        # --- observation mechanism --- ##
        x_obs  = self._state_obs(x_new)

        # ---- step time index ---- # 
        t += 1
        self.period = t 
        
        # ---- update self containers to track history ---- #
        self.state_traj[t,:]     = self.current_state.reshape(1,-1)
        self.control_traj[t-1,:] = u.reshape(1,-1)
        
        # ---- allocate reward ---- # 
        reward = self._reward_allocation(x_new, u)
        # if self.penalise_v: reward -= self.pf *g.squeeze()
        
        # ---- determine if simulation should terminate ---- #
        if (self.period >= self.steps):
            done = True
        else:
            done = False
        
            self.prev_u = u.reshape(nu,1)
#         print(type(x_obs))
        return x_obs.reshape(-1), float(reward), done, {"info":None}   # state, reward, done, info
    

    def step(self, control):
        return self._STEP(control)

    def reset(self):
        return self._RESET()

# Nx = 4
# Nu = 1

# cstr = Inverted_pendulum_env()
# cstr.reset()
# print(cstr.reset())
# for i in range(u_plot_real.shape[0]):
    # u = u_plot_real[i].reshape(1,-1)
#     cstr.step(u)
#     print(cstr.step(u))

# u = np.array(0.1).reshape(1,-1)
# cstr.step(u)
# cstr.step(u)


# In[231]:


class NN(nn.Module):
    def __init__(self, Nx, Nu):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(Nx, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, Nu)

    def forward(self, x):
        self.tanh = nn.Tanh()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def sample_action(self, state, epsilon):
        out = self.forward(state)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else :
            return out.argmax().item()


# cstr = Inverted_pendulum_env()
# # cstr.reset()
# print(cstr.reset())
# for i in range(u_plot_real.shape[0]):
#     u = u_plot_real[i].reshape(1,-1)
# #     cstr.step(u)
#     print(cstr.step(u))





class dagger():
    def __init__(self, env, demonstrator_opt, opt_setting, opt_u_ind,learner, horizons, episodes, p = 0.5):
        self.p = p
        self.env = env
        self.demonstrator_opt = demonstrator_opt
        self.opt_setting = opt_setting
        self.learner = learner
        self.N_hor = horizons
        self.N_epi = episodes
        self.u_ind = opt_u_ind

        self.u_norm = 25

        # self.dataset_state = []
        # self.dataset_action = []
        
    def data_collection(self, N_epi_i):
        demonstrator_opt = self.demonstrator_opt
        opt_setting = self.opt_setting

        u_norm = self.u_norm

        learner = self.learner
        N_hor = self.N_hor
        beta = self.p ** N_epi_i
        
        env = self.env
        state = env.reset()
        state_mixed_list, action_list, reward_list, done_list = [], [], [], []
        state_mixed_list += [state.tolist()]
        u_ind = self.u_ind
        for i in range(N_hor):
            control_policy = np.random.choice(a = ["demonstrator", "learner"], size = 1, p=[beta, 1.0 - beta])

            # Apply demonstrator policy
            if control_policy == "demonstrator":
                p = np.array(state).reshape(-1,1)
                opt_setting['p'] = p
                sol = demonstrator_opt(**opt_setting)
                a = sol['x'].full()[u_ind,0]

            # Apply learner policy
            else:
                state_tensor = torch.tensor(state, dtype=torch.float)
                a = np.float64(learner(state_tensor).item())
                a = a * u_norm
            
            
            state, reward, done, _ = env.step(a)

            a_normalized = a / u_norm
            if i < N_hor - 1:
                state_mixed_list += [state.tolist()]
            action_list += [[a_normalized]]
            #             reward_list += [reward]
#             state_n_list += [state_n_temp]
#             done_list += [done]
            
        action_list_expert = []
        
        for i in range(N_hor):
            p = np.array(state_mixed_list[i]).reshape(-1,1)
            nl['p'] = p
            sol = solver_real(**nl)
            a = sol['x'].full()[u_ind,0]

            a_normalized = a / u_norm
            action_list_expert += [[a_normalized]]
            
        return state_mixed_list, action_list, action_list_expert
        
    def train(self):
        N_epi = self.N_epi
        # N_hor = self.N_hor
        learner = self.learner

        learning_rate = 1e-2
        # learning_rate = 0.05
        # lambda_l2 = 1e-5
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(learner.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(learner.parameters(), lr=learning_rate, weight_decay=lambda_l2) # built-in L2
        
        state_list = []
        action_list = []
        action_list_expert = []
        for i in range(N_epi):
            state_mixed_list_temp, action_list_temp, action_list_expert_temp = self.data_collection(i)
            
            state_list += state_mixed_list_temp
            action_list += action_list_temp
            action_list_expert += action_list_temp
            
            
            state_tensor = torch.tensor(state_list, dtype=torch.float)
            # action_tensor = torch.tensor(action_list)
            action_expert_tensor = torch.tensor(action_list_expert, dtype=torch.float)

            for j in range(80):
                preds = self.learner(state_tensor)
                loss   = criterion(preds, action_expert_tensor)
                optimizer.zero_grad()
                loss.backward()
                # for j in range(10):
                optimizer.step()

            print(i,"-th iteration is done \n")
            
    def save_policy(self,path):
        self.learner.save(path)


if __name__ == '__main__':
    Nx = 4
    Nu = 1

    x_SX = ca.SX.sym("x_SX", Nx)
    u_SX = ca.SX.sym("u_SX", Nu)

    delta_t = 0.05

    ode_func = ca.Function("ode_func", [x_SX, u_SX], [inverted_pendulum(x_SX, u_SX)])

    sys_int = integrator_rk4(ode_func, x_SX, u_SX, delta_t)

    sys_int_fn = ca.Function("sys_int_fn", [x_SX, u_SX], [sys_int])

    A_lin_SX = ca.jacobian(sys_int, x_SX)
    B_lin_SX = ca.jacobian(sys_int, u_SX)

    A_lin_func = ca.Function("A_lin_func", [x_SX, u_SX], [A_lin_SX])
    B_lin_func = ca.Function("B_lin_func", [x_SX, u_SX], [B_lin_SX])

    # In[9]:

    A = A_lin_func([0, 0, 0, 0], [0]).full()
    B = B_lin_func([0, 0, 0, 0], [0]).full()

    Q = np.diag([0.25, 0.025, 0.25, 0.025])
    R = 0.0025
    # Qf = 10 * np.eye(Nx)

    K, P = np.array(dlqr(A, B, Q, R))

    obj = 0

    N_pred = 20

    xi_var = ca.SX.sym('x', Nx, N_pred + 1)
    x0_para = ca.SX.sym('x0', Nx, 1)
    ui_var = ca.SX.sym('u', Nu, N_pred)

    g = []

    for i in range(N_pred):
        obj += xi_var[:, i].T @ Q @ xi_var[:, i] + ui_var[:, i].T @ R @ ui_var[:, i]
        if i == 0:
            g += [xi_var[:, i] - x0_para]
        else:
            g += [xi_var[:, i] - sys_int_fn(xi_var[:, i - 1], ui_var[:, i - 1])]

    obj += xi_var[:, -1].T @ P @ xi_var[:, -1]
    g += [xi_var[:, -1] - sys_int_fn(xi_var[:, -2], ui_var[:, -1])]  # Terminal state
    # g += [terminal_state - integrator_rk4(ode, xi_var[:,-2], ui_var[:,-1], delta_t)] # Terminal state
    dec_list = [ca.reshape(xi_var, -1, 1), ca.reshape(ui_var, -1, 1)]
    para_list = [x0_para]
    nlp_prob = {
        'f': obj,
        'x': ca.vertcat(*dec_list),
        'g': ca.vertcat(*g),
        'p': ca.vertcat(*para_list)
    }

    solver_opt = {}
    solver_opt['print_time'] = False
    solver_opt['ipopt'] = {
        'max_iter': 500,
        'print_level': 1,
        'acceptable_tol': 1e-6,
        'acceptable_obj_change_tol': 1e-6}

    solver_real = ca.nlpsol("solver", "ipopt", nlp_prob, solver_opt)

    x1_ub = 2
    x2_ub = 4
    x3_ub = ca.pi / 3
    x4_ub = 2

    x1_lb = -x1_ub
    x2_lb = -x2_ub
    x3_lb = -x3_ub
    x4_lb = -x4_ub

    u_ub = 25
    u_lb = -u_ub

    lbg = [0] * Nx * (N_pred + 1)
    ubg = [0] * Nx * (N_pred + 1)
    lbx = [x1_lb, x2_lb, x3_lb, x4_lb] * (N_pred + 1) + [u_lb] * (N_pred)
    ubx = [x1_ub, x2_ub, x3_ub, x4_ub] * (N_pred + 1) + [u_ub] * (N_pred)

    nl = {}
    nl['lbg'] = lbg
    nl['ubg'] = ubg
    nl['lbx'] = lbx
    nl['ubx'] = ubx

    env = Inverted_pendulum_env()
    horizons = 50
    episodes = 50
    learner_policy = NN(Nx, Nu)
    opt_u_ind = Nx * (N_pred+1)
    dagger_instance = dagger(env, solver_real, nl, opt_u_ind, learner_policy, horizons, episodes)

    # dagger_instance.data_collection(1)
    dagger_instance.train()

    path = "./learner_policy.pth"
    dagger_instance.save_policy(path)