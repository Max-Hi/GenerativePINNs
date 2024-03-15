import numpy as np
import scipy.io as sio
from pyDOE import lhs

# Create the training data for the Burgers equation

""" 
equation configuration: 
x \in [-1, 1]
t \in [0, 1]
u0 = -sin(pi*x)
u(t, -1) = u(t, 1) =0
"""

nu = 0.01/np.pi
N_x = 256 # number of spatial points
N_t = 100 # number of time points
x = np.linspace(-1, 1, N_x, endpoint = True)[:, None]
# convert to column vector
t = np.linspace(0, 1, N_t, endpoint= False)[:, None]
usol = np.zeros((N_x, N_t))

# Create the initial condition
usol[:,0] = -np.sin(np.pi*x).flatten()

# Create the solution at each time point
# iterative solution: overflow problem
# for n in range(1, N_t):
#     # # Add the boundary conditions
#     # usol[0, n] = 0
#     # usol[-1, n] = 0
#     # Create the solution in the interior of the domain
#     for i in range(1, N_x):
#         usol[i, n] = usol[i, n-1] - usol[i, n-1]*(usol[i, n-1] - usol[i-1, n-1])/(x[i] - x[i-1]) + \
#            nu*(usol[i-1, n-1] - 2*usol[i, n-1] + usol[i+1, n-1])/(x[i] - x[i-1])**2

# Exact solution
usol = -np.sin(np.pi*x).flatten() * np.exp(-np.pi**2 * nu * t)[:, None]

# reshape the data to 2D
usol = usol.T.squeeze()

# save the data to mat file
sio.savemat('Data/burgers.mat', {'x': x, 't': t, 'usol': usol})

# # Test to load the data
# N0 = 50 # number of data for initial samples
# N_b = 50 # number of data for boundary samples
# N_f = 20000 # number of data for collocation points

# # Load data from simulated dataset
# data = sio.loadmat('Data/burgers.mat')
# """ 
# equation configuration: 
# x \in [-1, 1]
# t \in [0, 1]
# u0 = -sin(pi*x)
# u(t, -1) = u(t, 1) =0
# """
# # input 
# t = data['t'].flatten()[:,None]
# x = data['x'].flatten()[:,None]
# lb = np.array([x.min(), t.min()]) # lower bound for [x, t]
# ub = np.array([x.max(), t.max()]) # upper bound for [x, t]
# Exact = data['usol'].flatten()[:,None]

# X, T = np.meshgrid(x,t)
# X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
# u_star = Exact.T.flatten()[:,None]

# # Initial and boundary data
# tb = t[np.random.choice(t.shape[0], N_b, replace=False),:] # random time samples for boundary points
# ti = np.zeros(N_b)

# # exact observations
# id_x = np.random.choice(x.shape[0], N0, replace=False)
# x_exact = x[id_x,:]
# id_t = np.random.choice(t.shape[0], N0, replace=False)
# t_exact = x[id_t,:]
# u_exact = data['usol'][id_x, id_t]

# X_exact = np.concatenate((x_exact, t_exact), axis=1)

# # Collocation points
# X_f = lb + (ub-lb)*lhs(2, N_f)

# # initial points
# x0 = np.linspace(-1, 1, N0, endpoint = True)
# X0 = np.vstack((x0, np.zeros_like(x0, dtype=np.float32))).transpose() # (x, 0)
# # boundary points
# boundary = np.vstack((lb, ub))
# X_lb = np.concatenate((lb[0]*np.ones_like(tb, dtype=np.float32), tb), axis=1)
# X_ub = np.concatenate((ub[0]*np.ones_like(tb, dtype=np.float32), tb), axis=1)
