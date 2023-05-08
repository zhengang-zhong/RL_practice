#!/usr/bin/env python
# coding: utf-8

# In[1]:


import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
# import gym
from scipy.linalg import solve_discrete_are


# In[2]:






def mass_string_ode(x, u):
    
    A = np.array([[0.9, 0.35], [0, 1.1]])  # state-space matrix A
    B = np.array([[0.0813], [0.2]])  # state-space matrix B
        
        
    x_next = A @ x + B @ u

    return x_next


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


# In[6]:


# Step 1. Data collection via MPC/RMPC
Nx = 2
Nu = 1

x_SX = ca.SX.sym("x_SX", Nx)
u_SX = ca.SX.sym("u_SX", Nu)

next_state_SX = mass_string_ode(x_SX, u_SX)

sys_next_fn = ca.Function("ode_func", [x_SX, u_SX], [mass_string_ode(x_SX, u_SX)])


# In[7]:


A = np.array([[0.9, 0.35], [0, 1.1]])  # state-space matrix A
B = np.array([[0.0813], [0.2]])  # state-space matrix B


# In[8]:


# Q = 0.5 * np.eye(Nx)
# R = 0.5 * 0.5 * np.eye(Nu)
Q = np.eye(Nx)
R = 0.5 * np.eye(Nu)

# In[9]:


K, P = dlqr(A, B, Q, R)
P = Q

# In[10]:

#
# N_sim = 500
# x1_start = 0
# x2_start = 0.15
#
#
#
u_lb = [-1]
# # u_lb_LSH = [1]
u_ub = [1]
#
# x1_ub = 1
# x2_ub = 1
#
# x1_lb = 0
# x2_lb = -1
#
# lbx = [x1_lb, x2_lb]
# # lbx_RHS = [-x1_lb, -x2_lb]
# ubx = [x1_ub, x2_ub]
#
#
# # In[11]:
#
#
# obj = 0
#
# N_pred = 4
#
# xi_var = ca.SX.sym('x', Nx, N_pred+1)
# x0_para = ca.SX.sym('x0', Nx, 1)
# ui_var = ca.SX.sym('u', Nu, N_pred)
#
# g = []
#
# g += [ca.reshape(xi_var,-1,1)] # lower bound
# g += [ca.reshape(xi_var,-1,1)] # Upper bound
# g += [ca.reshape(ui_var,-1,1)] # lower bound
# g += [ca.reshape(ui_var,-1,1)] # Upper bound
#
#
# for i in range(N_pred):
#     obj +=  xi_var[:,i].T @ Q @  xi_var[:,i] + ui_var[:,i].T @ R @ ui_var[:,i]
#     if i == 0:
#         g += [xi_var[:,i] - x0_para]
#     else:
#         g += [xi_var[:,i] - sys_next_fn(xi_var[:,i-1], ui_var[:,i-1])]
#
# obj += xi_var[:, -1].T @ P @ xi_var[:, -1]
# g += [xi_var[:, -1] - sys_next_fn(xi_var[:,-2], ui_var[:,-1])] # Terminal state
# # g += [terminal_state - integrator_rk4(ode, xi_var[:,-2], ui_var[:,-1], delta_t)] # Terminal state
# dec_list = [ca.reshape(xi_var,-1,1), ca.reshape(ui_var,-1,1)]
# para_list = [x0_para]
# nlp_prob = {
#     'f': obj,
#     'x': ca.vertcat(*dec_list),
#     'g': ca.vertcat(*g),
#     'p': ca.vertcat(*para_list)
# }
#
#
# solver_opt = {}
# solver_opt['bound_consistency'] = True
# solver_opt['print_time'] = False
# solver_opt['ipopt'] = {
#     'max_iter': 500,
#     'print_level': 1,
#     'acceptable_tol': 1e-6,
#     'acceptable_obj_change_tol': 1e-6}
#
#
# solver_real = ca.nlpsol("solver", "ipopt", nlp_prob, solver_opt)
#
# lbg =  lbx * (N_pred+1) + [-ca.inf, -ca.inf] * (N_pred+1) + u_lb * (N_pred) + [-ca.inf] * (N_pred) +   [0] * 2 * (N_pred + 1)
# ubg =   [ca.inf, ca.inf] * (N_pred+1) + ubx * (N_pred+1) +  [ca.inf] * (N_pred) + u_ub * (N_pred) +   [0] * 2 * (N_pred + 1)
# lbx = [-ca.inf, -ca.inf] * (N_pred+1) + [-ca.inf] * (N_pred)
# ubx = [ca.inf, ca.inf] * (N_pred+1) + [ca.inf] * (N_pred)
# nl = {}
# nl['lbg'] = lbg
# nl['ubg'] = ubg
# nl['lbx'] = lbx
# nl['ubx'] = ubx
#
#
# # In[12]:
#
#
# x0 = np.array([[x1_start],[x2_start]])
# u0 = np.array([[0]])
#
# x_real_list = []
# x_real_pred_list = []
#
# u_real_list = []
# u_real_pred_list = []
#
# xk = x0
# uk = u0
#
#
# x_real_list.append(xk.flatten().tolist())
#
# u_ind = ca.reshape(xi_var,-1,1).shape[0]
#
#
# # In[13]:
#
#
# # Simulation for 20 seconds
# N_sim = 500
#
# for i in range(N_sim):
#     p = np.vstack([xk])
#     nl['p'] = p
#
#     sol = solver_real(**nl)
#     uk = sol['x'].full()[u_ind,0]
# #     print(uk)
#     x_real_pred_list += [sol['x'].full()[:u_ind]]
#     u_real_pred_list += [sol['x'].full()[u_ind:]]
#
#     x_next = sys_next_fn(xk,uk)
#     print(uk,xk)
#     x_real_list.append(x_next.full().flatten().tolist())
#     u_real_list.append(uk.flatten().tolist())
#
#     x_lin = xk
#     xk = x_next
#
# x_plot_real = np.array(x_real_list)
# u_plot_real = np.array(u_real_list)
# tgrid = [1 * k for k in range(N_sim + 1)]
#
# plt.figure(1, figsize=(6, 10))
# plt.clf()
# for i in range (2):
#     plt.subplot(3,1,i+1)
#     plt.grid()
#
# #     if i == 1:
# #         constr_plot_temp = [0.65] * (N_sim + 1)
# #         plt.plot(tgrid, constr_plot_temp, color="k")
#
#     x_opt_real = x_plot_real[:,i]
#     plt.plot(tgrid, x_opt_real, 'r')
#     plt.ylabel('x' + str(i + 1))
#
#
#
# plt.subplot(3,1,3)
# plt.grid()
#
# u_opt_real = u_plot_real[:,0]
# plt.plot(tgrid[:-1], u_opt_real, 'r')
# plt.ylabel('u1')
#
# plt.show()


N_pred = 4

# N_step = 20

# Initial Guess
A_init = np.array([[1, 0.25], [0, 1]])  # state-space matrix A
B_init = np.array([[0.0312], [0.25]])  # state-space matrix B

# Accurate model
# A_init = np.array([[0.9, 0.35], [0, 1.1]])  # state-space matrix A
# B_init = np.array([[0.0813], [0.2]])  # state-space matrix B


w = np.array([[1e2], [1e2]])
# w = np.array([[1e6], [1e6]])

V0 = 0
x_low = np.array([[0],[0]])
# x_upper = np.array([[1],[0]])  # Try x_upper = np.array([[0],[0]])
x_upper = np.array([[0],[0]])  # Try x_upper = np.array([[0],[0]])
b = np.array([[0],[0]])
f = np.array([[0],[0], [0]])

gamma_discount = 0.9

#     learnable_pars_init = {
#         "V0": np.asarray(0.0),
#         "x_lb": np.asarray([0, 0]),
#         "x_ub": np.asarray([1, 0]),
#         "b": np.zeros(LtiSystem.nx),
#         "f": np.zeros(LtiSystem.nx + LtiSystem.nu),
#         "A": np.asarray([[1, 0.25], [0, 1]]),
#         "B": np.asarray([[0.0312], [0.25]]),
#     }


# Set up MPC

# V
obj = 0

xi_var = ca.SX.sym('x', Nx, N_pred+1)
ui_var = ca.SX.sym('u', Nu, N_pred)
si_var = ca.SX.sym('s', Nx, N_pred+1)


x0_para = ca.SX.sym('x0', Nx, 1)
V0_para = ca.SX.sym('V0_par', 1)
x_low_para = ca.SX.sym('x_low_par', Nx)
x_upper_para = ca.SX.sym('x_upper_par', Nx)
b_para = ca.SX.sym('b_par', Nx)
f_para = ca.SX.sym('f_par', Nx + Nu)
A_para = ca.SX.sym('b_par', Nx, Nx)
B_para = ca.SX.sym('f_par', Nx)


g = []

obj += V0_para
for i in range(N_pred):
    obj +=  1/2 * gamma_discount ** i * (xi_var[:,i].T @ Q @  xi_var[:,i] + ui_var[:,i].T @ R @ ui_var[:,i] + w.T @  si_var[:,i])
    obj +=  f_para.T @ ca.vertcat(xi_var[:,i], ui_var[:,i])
    
    if i == 0:
        g += [xi_var[:,i] - x0_para]
    else:
        g += [xi_var[:,i] - (A_para @ xi_var[:,i-1] + B_para @ ui_var[:,i-1] + b_para)]

obj += 1/2 * gamma_discount ** N_pred * (xi_var[:, -1].T @ P @ xi_var[:, -1] + w.T @  si_var[:,-1])
g += [xi_var[:, -1] - (A_para @ xi_var[:,-2] + B_para @ ui_var[:,-1] + b_para)] # Terminal state
              
for i in range(N_pred + 1):
    g += [np.array([[0],[-1]]) + x_low_para - si_var[:,i] - xi_var[:,i]] # <= 0
for i in range(N_pred + 1):
    g += [xi_var[:,i] - np.array([[1],[1]]) - x_upper_para - si_var[:,i] ] # <= 0

# g += [np.array([[0],[-1]]) + x_low_para - si_var[:, -1] - xi_var[:, -1]] # <= 0
# g += [xi_var[:, -1] - np.array([[1],[1]]) - x_upper_para - si_var[:, -1] ] # <= 0
              

              
# g += [terminal_state - integrator_rk4(ode, xi_var[:,-2], ui_var[:,-1], delta_t)] # Terminal state         
dec_list = [ca.reshape(xi_var,-1,1), ca.reshape(ui_var,-1,1), ca.reshape(si_var,-1,1)]
para_list = [x0_para, V0_para, x_low_para, x_upper_para, b_para, f_para, ca.reshape(A_para, -1, 1), B_para]
nlp_prob = {
    'f': obj,
    'x': ca.vertcat(*dec_list),
    'g': ca.vertcat(*g),
    'p': ca.vertcat(*para_list)
}
    

solver_opt = {}
solver_opt['bound_consistency'] = True
solver_opt['print_time'] = False
solver_opt['ipopt'] = {
    'max_iter': 500,
    'print_level': 1,
    'acceptable_tol': 1e-6,
    'acceptable_obj_change_tol': 1e-6}


solver_V = ca.nlpsol("solver", "ipopt", nlp_prob, solver_opt)  


# In[20]:


lbg_V =  [0] * Nx * (N_pred + 1) + [-ca.inf] * 2 * Nx * (N_pred + 1) # constraints for dynamics, x lower bound, x upper bound
ubg_V =  [0] * Nx * (N_pred + 1) + [0] * 2 * Nx * (N_pred + 1)

lbx_V = [-ca.inf, -ca.inf] * (N_pred+1)  + u_lb * (N_pred) + [0] * Nx * (N_pred + 1)
# lbx_V = [-ca.inf, -ca.inf] * (N_pred+1)  + u_lb * (N_pred) + [-ca.inf] * Nx * (N_pred + 1)# Constraints for x, u, s
ubx_V = [ca.inf, ca.inf] * (N_pred+1) + u_ub * (N_pred) + [ca.inf] * Nx * (N_pred + 1)

nl_V = {}
nl_V['lbg'] = lbg_V
nl_V['ubg'] = ubg_V
nl_V['lbx'] = lbx_V
nl_V['ubx'] = ubx_V

x1_start = 0
# x1_start = -0.01
x2_start = 0.15
x0 = np.array([[x1_start],[x2_start]])

x_real_list = []
x_real_pred_list = []

u_real_list = []
u_real_pred_list = []

xk = x0
# uk = u0


x_real_list.append(xk.flatten().tolist())

u_ind = ca.reshape(xi_var,-1,1).shape[0]


# In[21]:


# N_sim = 1
# for i in range(N_sim):
#     p = np.vstack([xk])
#     nl_V['p'] = ca.vertcat(xk, V0, x_low, x_upper, b, f, ca.reshape(A_init, -1, 1), B_init)
    
#     if i == 0:
#         nl_V['x0'] =  np.random.normal(0,1,np.shape(ca.vertcat(*dec_list))[0])
#     else:
#         nl_V['x0'] = sol['x']
    
#     sol = solver_V(**nl_V)
#     uk = sol['x'].full()[u_ind,0]
# #     print(uk)
#     x_real_pred_list += [sol['x'].full()[:u_ind]]
#     u_real_pred_list += [sol['x'].full()[u_ind:]]
    
#     x_next = sys_next_fn(xk,uk) 
#     print(uk,xk)
#     x_real_list.append(x_next.full().flatten().tolist())
#     u_real_list.append(uk.flatten().tolist())

#     x_lin = xk
#     xk = x_next


# In[22]:


# sol


# In[23]:


# lambda_dual_ind = Nx * (N_pred + 1)

# lambda_dual_eq_val = sol['lam_g'][:lambda_dual_ind]
# lambda_dual_ineq_lower_val = sol['lam_g'][lambda_dual_ind:lambda_dual_ind + Nx * (N_pred + 1)]
# lambda_dual_ineq_upper_val = sol['lam_g'][lambda_dual_ind + Nx * (N_pred + 1) : ]


# In[24]:


# lambda_dual_eq_val


# In[25]:


# lambda_dual_ineq_lower_val


# In[26]:


# lambda_dual_ineq_upper_val


# In[27]:


# lambda_dual_ind = Nx * (N_pred + 1)

# lambda_dual_eq_val = sol_Q['lam_g'][:lambda_dual_ind]
# lambda_dual_ineq_lower_val = sol_Q['lam_g'][lambda_dual_ind:lambda_dual_ind + Nx * (N_pred + 1)]
# lambda_dual_ineq_upper_val = sol_Q['lam_g'][lambda_dual_ind + Nx * (N_pred + 1) : ]

# lbg_Q =  [0] * Nx * (N_pred + 1) + [-ca.inf] * 2 * Nx * (N_pred + 1) # constraints for dynamics, x lower bound, x upper bound
# ubg_Q =  [0] * Nx * (N_pred + 1) + [0] * 2 * Nx * (N_pred + 1) 


# In[28]:


# Define Q function
obj = 0

xi_var = ca.SX.sym('x', Nx, N_pred+1)
ui_var = ca.SX.sym('u', Nu, N_pred)
si_var = ca.SX.sym('s', Nx, N_pred+1)

u0_para = ca.SX.sym('u0', Nu, 1)
x0_para = ca.SX.sym('x0', Nx, 1)
V0_para = ca.SX.sym('V0_par', 1)
x_low_para = ca.SX.sym('x_low_par', Nx)
x_upper_para = ca.SX.sym('x_upper_par', Nx)
b_para = ca.SX.sym('b_par', Nx)
f_para = ca.SX.sym('f_par', Nx + Nu)
A_para = ca.SX.sym('b_par', Nx, Nx)
B_para = ca.SX.sym('f_par', Nx)



g = []

g += [ui_var[:, 0] - u0_para]
obj += V0_para
for i in range(N_pred):    
    if i == 0:
        g += [xi_var[:, i] - x0_para]
        obj +=  1/2 * gamma_discount ** i * (xi_var[:,i].T @ Q @  xi_var[:,i] + u0_para.T @ R @ u0_para +  w.T @  si_var[:,i])
        obj +=  f_para.T @ ca.vertcat(xi_var[:,i], u0_para)
    # elif i == 1:
    #     g += [xi_var[:,i] - (A_para @ xi_var[:,i-1] + B_para @ u0_para + b_para)]
    #     obj +=  1/2 * gamma_discount ** i * (xi_var[:,i].T @ Q @  xi_var[:,i] + ui_var[:,i-1].T @ R @ ui_var[:,i-1] + w.T @  si_var[:,i])
    #     obj +=  f_para.T @ ca.vertcat(xi_var[:,i], ui_var[:,i-1])
    
    else:
        g += [xi_var[:,i] - (A_para @ xi_var[:,i-1] + B_para @ ui_var[:,i-1] + b_para)]
        obj +=  1/2 * gamma_discount ** i * (xi_var[:,i].T @ Q @  xi_var[:,i] + ui_var[:,i].T @ R @ ui_var[:,i] + w.T @  si_var[:,i])
        obj +=  f_para.T @ ca.vertcat(xi_var[:,i], ui_var[:,i])

obj += 1/2 * gamma_discount ** N_pred * (xi_var[:, -1].T @ P @ xi_var[:, -1] + w.T @  si_var[:,-1])
g += [xi_var[:, -1] - (A_para @ xi_var[:,-2] + B_para @ ui_var[:,-1] + b_para)] # Terminal state
              
for i in range(N_pred + 1):
    g += [np.array([[0],[-1]]) + x_low_para - si_var[:,i] - xi_var[:,i]] # <= 0
for i in range(N_pred + 1):
    g += [xi_var[:,i] - np.array([[1],[1]]) - x_upper_para - si_var[:,i] ] # <= 0


# g += [np.array([[0],[-1]]) + x_low_para - si_var[:, -1] - xi_var[:, -1]] # <= 0
# g += [xi_var[:, -1] - np.array([[1],[1]]) - x_upper_para - si_var[:, -1] ] # <= 0
              

              
# g += [terminal_state - integrator_rk4(ode, xi_var[:,-2], ui_var[:,-1], delta_t)] # Terminal state         
dec_list = [ca.reshape(xi_var,-1,1), ca.reshape(ui_var,-1,1), ca.reshape(si_var,-1,1)]
para_list = [u0_para, x0_para, V0_para, x_low_para, x_upper_para, b_para, f_para, ca.reshape(A_para, -1, 1), B_para]
nlp_prob = {
    'f': obj,
    'x': ca.vertcat(*dec_list),
    'g': ca.vertcat(*g),
    'p': ca.vertcat(*para_list)
}
    

solver_opt = {}
solver_opt['bound_consistency'] = True
solver_opt['print_time'] = False
solver_opt['ipopt'] = {
    'max_iter': 500,
    'print_level': 1,
    'acceptable_tol': 1e-6,
    'acceptable_obj_change_tol': 1e-6}


solver_Q = ca.nlpsol("solver", "ipopt", nlp_prob, solver_opt)  


# In[29]:


lbg_Q =  [0] * (Nu + Nx * (N_pred + 1)) + [-ca.inf] * 2 * Nx * (N_pred + 1) # constraints for dynamics, x lower bound, x upper bound
ubg_Q =  [0] * (Nu + Nx * (N_pred + 1)) + [0] * 2 * Nx * (N_pred + 1)

lbx_Q = [-ca.inf, -ca.inf] * (N_pred+1)  + u_lb * (N_pred) + [0] * Nx * (N_pred + 1)
# lbx_Q = [-ca.inf, -ca.inf] * (N_pred+1)  + u_lb * (N_pred) + [-ca.inf] * Nx * (N_pred + 1)# Constraints for x, u, s
ubx_Q = [ca.inf, ca.inf] * (N_pred+1) + u_ub * (N_pred) + [ca.inf] * Nx * (N_pred + 1)

nl_Q = {}
nl_Q['lbg'] = lbg_Q
nl_Q['ubg'] = ubg_Q
nl_Q['lbx'] = lbx_Q
nl_Q['ubx'] = ubx_Q


# In[30]:


# Get Q_grad

# x0_para = ca.SX.sym('x0', Nx, 1)
# V0_para = ca.SX.sym('V0_par', 1)
# x_low_para = ca.SX.sym('x_low_par', Nx)
# x_upper_para = ca.SX.sym('x_upper_par', Nx)
# b_para = ca.SX.sym('b_par', Nx)
# f_para = ca.SX.sym('f_par', Nx + Nu)
# A_para = ca.SX.sym('b_par', Nx, Nx)
# B_para = ca.SX.sym('f_par', Nx)

lambda_dual_eq_list = []
lambda_dual_ineq_lower_list = []
lambda_dual_ineq_upper_list = []


# Q_SX = V0_para
# for i in range(N_pred):
#     if i == 0:
#         Q_SX +=  f_para.T @ ca.vertcat(xi_var[:,i], u0_para)
#     else:
#         Q_SX += f_para.T @ ca.vertcat(xi_var[:,i], ui_var[:,i - 1])

Q_SX = obj

lambda_dual_eq_u_SX = ca.SX.sym("lambda_dual_eq_u_SX" , Nu)
lambda_dual_eq_list += [lambda_dual_eq_u_SX]
Q_SX += lambda_dual_eq_u_SX.T @ (ui_var[:,0] - u0_para)

for i in range(N_pred):
    lambda_dual_eq_SX = ca.SX.sym("lambda_dual_eq_SX_" + str(i), Nx)
    lambda_dual_eq_list += [lambda_dual_eq_SX]
    if i == 0:
        Q_SX += lambda_dual_eq_SX.T @ (xi_var[:,i] - x0_para)
    # elif i == 1:
        # Q_SX += lambda_dual_eq_SX.T @ (xi_var[:,i] - (A_para @ xi_var[:,i-1] + B_para @ u0_para + b_para))
    else:
        Q_SX += lambda_dual_eq_SX.T @ (xi_var[:,i] - (A_para @ xi_var[:,i-1] + B_para @ ui_var[:,i-1] + b_para))

lambda_dual_eq_SX = ca.SX.sym("lambda_dual_eq_SX_" + str(N_pred), Nx) 
lambda_dual_eq_list += [lambda_dual_eq_SX]
Q_SX += lambda_dual_eq_SX.T @ (xi_var[:, -1] - (A_para @ xi_var[:,-2] + B_para @ ui_var[:,-1] + b_para))
    
for i in range(N_pred + 1):    
    lambda_dual_ineq_SX_lower = ca.SX.sym("lambda_dual_ineq_SX_lower_" + str(i), Nx)
    lambda_dual_ineq_lower_list += [lambda_dual_ineq_SX_lower]
    Q_SX += lambda_dual_ineq_SX_lower.T @ (np.array([[0], [-1]]) + x_low_para - si_var[:, i] - xi_var[:, i])
for i in range(N_pred + 1):
    lambda_dual_ineq_SX_upper = ca.SX.sym("lambda_dual_ineq_SX_upper_" + str(i), Nx)
    lambda_dual_ineq_upper_list += [lambda_dual_ineq_SX_upper]    
    Q_SX += lambda_dual_ineq_SX_upper.T @ (xi_var[:,i] - np.array([[1],[1]]) - x_upper_para - si_var[:,i])


# In[33]:


V0_para_grad = ca.gradient(Q_SX, V0_para)
x_low_para_grad = ca.gradient(Q_SX, x_low_para)
x_upper_para_grad = ca.gradient(Q_SX, x_upper_para)
b_para_grad = ca.gradient(Q_SX, b_para)
f_para_grad = ca.gradient(Q_SX, f_para)
A_para_grad = ca.reshape(ca.gradient(Q_SX, ca.reshape(A_para, -1, 1)), Nx,Nx)
B_para_grad = ca.gradient(Q_SX, B_para)


# grad_func_var_list = [ca.vertcat(*lambda_dual_eq_list), ca.vertcat(*lambda_dual_ineq_lower_list), ca.vertcat(*lambda_dual_ineq_upper_list),
#                       ca.reshape(xi_var,-1,1), ca.reshape(ui_var,-1,1), ca.reshape(si_var,-1,1), u0_para, x0_para,
#                       V0_para, x_low_para, x_upper_para, b_para, f_para, ca.reshape(A_para, -1, 1), B_para]
grad_func_var_list = [ca.vertcat(*lambda_dual_eq_list), ca.vertcat(*lambda_dual_ineq_lower_list), ca.vertcat(*lambda_dual_ineq_upper_list)] + dec_list + para_list
                      # ca.reshape(xi_var,-1,1), ca.reshape(ui_var,-1,1), ca.reshape(si_var,-1,1), u0_para, x0_para,
                      # V0_para, x_low_para, x_upper_para, b_para, f_para, ca.reshape(A_para, -1, 1), B_para]
# dec_list = [ca.reshape(xi_var,-1,1), ca.reshape(ui_var,-1,1), ca.reshape(si_var,-1,1)]
# para_list = [u0_para, x0_para, V0_para, x_low_para, x_upper_para, b_para, f_para, ca.reshape(A_para, -1, 1), B_para]
grad_func_var = ca.vertcat(*grad_func_var_list)


V0_para_grad_fn = ca.Function("V0_para_grad_fn", [grad_func_var], [V0_para_grad])
x_low_para_grad_fn = ca.Function("x_low_para_grad_fn", [grad_func_var], [x_low_para_grad])
x_upper_para_grad_fn = ca.Function("x_upper_para_grad_fn", [grad_func_var], [x_upper_para_grad])
b_para_grad_fn = ca.Function("b_para_grad_fn", [grad_func_var], [b_para_grad])
f_para_grad_fn = ca.Function("f_para_grad_fn", [grad_func_var], [f_para_grad])
A_para_grad_fn = ca.Function("A_para_grad_fn", [grad_func_var], [A_para_grad])
B_para_grad_fn = ca.Function("B_para_grad_fn", [grad_func_var], [B_para_grad])


# In[34]:


# solver_V.size_in(1)


# In[35]:


# solver_V


# In[36]:


x0 = np.array([[x1_start],[x2_start]])
u0 = np.array([[0]])

x_real_list = []
u_real_list = []


xk = x0
uk = u0


x_real_list.append(xk.flatten().tolist())

u_ind = ca.reshape(xi_var,-1,1).shape[0]


def addtive_dist_func_gene(scale_additive_dist1, scale_additive_dist2 , a, b):
    def addtive_dist_func():
        additive_dist =  np.vstack([scale_additive_dist1 * (np.random.uniform(a,b,1)), scale_additive_dist2 * (np.random.uniform(a,b,1))]).reshape(-1, 1)
        return additive_dist

    return addtive_dist_func


# In[41]:


scale_additive_dist1 = 1
scale_additive_dist2 = 0
dis_low = -1e-1
dis_up = 0
addtive_disturb_func = addtive_dist_func_gene(scale_additive_dist1, scale_additive_dist2, dis_low, dis_up)


# In[ ]:


N_sim = 10000
lambda_dual_ind = Nx * (N_pred + 1) + Nu

# alpha = 1e-2
alpha = 1e-4


A_init = np.array([[1, 0.25], [0, 1]])  # state-space matrix A
B_init = np.array([[0.0312], [0.25]])  # state-space matrix B

# # Accurate model
# A_init = np.array([[0.9, 0.35], [0, 1.1]])  # state-space matrix A
# B_init = np.array([[0.0813], [0.2]])  # state-space matrix B


w = np.array([[1e2], [1e2]])
# w = np.array([[1e6], [1e6]])

V0 = 0
x_low = np.array([[0],[0]])
# x_upper = np.array([[1],[0]])  # Try x_upper = np.array([[0],[0]])
x_upper = np.array([[0],[0]])  # Try x_upper = np.array([[0],[0]])
b = np.array([[0],[0]])
f = np.array([[0],[0], [0]])

Ak = A_init
Bk = B_init



nl_V['p'] = ca.vertcat(xk, V0, x_low, x_upper, b, f, ca.reshape(Ak, -1, 1), Bk)

# if i == 0:
# nl_V['x0'] =  np.random.normal(0,1,solver_V.size_in(0)[0])
# nl_V['x0'] =  np.zeros(solver_V.size_in(0))
# else:
#     nl_V['x0'] = sol_V['x']

sol_V = solver_V(**nl_V)
uk = sol_V['x'][u_ind:u_ind+Nu].full()

for i in range(N_sim):
    Lk = xk.T @ Q @ xk + uk.T @ R @ uk + w.T @ np.maximum(0, np.array([[0],[-1]]) - xk) + w.T @ np.maximum(0, xk - np.array([[1],[1]]))
    nl_Q['p'] =  ca.vertcat(uk, xk, V0, x_low, x_upper, b, f, ca.reshape(Ak, -1, 1), Bk)

    # print(Lk)
#     if i == 0:
#         nl_Q['x0'] =  np.random.normal(0,1,solver_Q.size_in(0)[0])
#     else:
#         nl_Q['x0'] = sol_Q['x']
        
    sol_Q = solver_Q(**nl_Q)

    lambda_dual_eq_val = sol_Q['lam_g'][:lambda_dual_ind].full()
    lambda_dual_ineq_lower_val = sol_Q['lam_g'][lambda_dual_ind:lambda_dual_ind + Nx * (N_pred + 1)].full()
    lambda_dual_ineq_upper_val = sol_Q['lam_g'][lambda_dual_ind + Nx * (N_pred + 1) : ].full()
    
    
    x_next = sys_next_fn(xk,uk).full() + addtive_disturb_func()
    x_real_list.append(x_next.flatten().tolist())
    u_real_list.append(uk.flatten().tolist())    
    
    nl_V['p'] = ca.vertcat(x_next, V0, x_low, x_upper, b, f, ca.reshape(Ak, -1, 1), Bk)
    # nl_V['x0'] = sol_V['x']

    sol_V = solver_V(**nl_V)
    uk = sol_V['x'][u_ind:u_ind+Nu].full()
    
    
    tau_k = Lk + gamma_discount * sol_V['f'] - sol_Q['f']
    # tau_k = - tau_k

    print(tau_k)

    grad_func_val_list = [sol_Q['lam_g'].full(),
        # lambda_dual_eq_val, lambda_dual_ineq_lower_val, lambda_dual_ineq_upper_val,
                          sol_Q['x'].full(),
                      # sol_Q['x'][:u_ind].full(), sol_Q['x'][u_ind:u_ind + Nu * (N_pred - 1)].full(), sol_Q['x'][u_ind + Nu * (N_pred - 1):].full(),
                           uk, xk, V0, x_low, x_upper, b, f, ca.reshape(Ak, -1, 1), Bk]
    grad_func_val = ca.vertcat(*grad_func_val_list)
    
    V0 = V0 + (alpha * tau_k * V0_para_grad_fn(grad_func_val)).full()
    x_low = x_low + (alpha * tau_k * x_low_para_grad_fn(grad_func_val)).full()
    x_upper = x_upper + (alpha * tau_k * x_upper_para_grad_fn(grad_func_val)).full()
    b = b + (alpha * tau_k * b_para_grad_fn(grad_func_val)).full()
    f = f + (alpha * tau_k * f_para_grad_fn(grad_func_val)).full()
    Ak = Ak + (alpha * tau_k * A_para_grad_fn(grad_func_val)).full()
    Bk = Bk + (alpha * tau_k * B_para_grad_fn(grad_func_val)).full()
#     print(uk)
#     x_real_pred_list += [sol['x'].full()[:u_ind]]
#     u_real_pred_list += [sol['x'].full()[u_ind:]]
    

    # print(uk,xk)
    xk = x_next


# In[ ]:


x_plot_real = np.array(x_real_list)
u_plot_real = np.array(u_real_list)
tgrid = [1 * k for k in range(N_sim + 1)]

plt.figure(1, figsize=(6, 10))
plt.clf()
for i in range (2):
    plt.subplot(3,1,i+1)
    plt.grid()
    
#     if i == 1:
#         constr_plot_temp = [0.65] * (N_sim + 1)
#         plt.plot(tgrid, constr_plot_temp, color="k")
    
    x_opt_real = x_plot_real[:,i]
    plt.plot(tgrid, x_opt_real, 'r')
    plt.ylabel('x' + str(i + 1))



plt.subplot(3,1,3)
plt.grid()
    
u_opt_real = u_plot_real[:,0]
plt.plot(tgrid[:-1], u_opt_real, 'r')
plt.ylabel('u1')
plt.show()

# In[ ]:


Ak

