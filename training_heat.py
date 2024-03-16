import torch
import numpy as np
import scipy.io
from pyDOE import lhs
import time
import matplotlib.pyplot as plt
from PINN_burgers import PINN_GAN_burgers, Discriminator, Generator
from utils.plot import plot_with_ground_truth, plot_loss
import pickle
torch.set_default_dtype(torch.float32)
# random seed for reproduceability
np.random.seed(42)

# Hyperparameters
noise = 0.0 
nu = 1e-2/np.pi       

N0 = 50 # number of data for initial samples
N_b = 50 # number of data for boundary samples
N_f = 20000 # number of data for collocation points

# Define the physics-informed neural network
layers_G = [2, 20, 20, 20, 20, 20, 20, 20, 1]
layers_D = [3, 20, 20, 20, 20, 20, 20, 2]

# Load data from simulated dataset
data = scipy.io.loadmat('Data/heat.mat')
"""
equation configuration:
x, y \in [0, 1]
t \in [0, 10]
u(x, y, 0) = x - y
"""
mat = torch.load("heat_pred.pt")


# input 
y = data['y'].flatten()[:,None]
x = data['x'].flatten()[:,None]
lb = np.array([x.min(), y.min()]) # lower bound for [x, y]
ub = np.array([x.max(), y.max()]) # upper bound for [x, y]
Exact = data['usol'].T


X, Y = np.meshgrid(x,y)
X_star = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
u_star = Exact.flatten()[:,None]

#plot_with_ground_truth(mat, X_star, X , T, u_star , ground_truth_ref=False, ground_truth_refpts=[], filename = "ground_truth_comparison.png")

# Initial and boundary data
tb = t[np.random.choice(t.shape[0], N_b, replace=False),:] # random time samples for boundary points
ti = np.zeros(N_b)

# exact observations
idx = np.random.choice(X_star.shape[0], N0, replace=False)
X_exact = X_star[idx,:]
x_exact = X_exact[:,0]
t_exact = X_exact[:,1]

u_exact = u_star[idx, :]


# Collocation points
X_f = lb + (ub-lb)*lhs(2, N_f)

# initial points
x0 = np.linspace(-1, 1, N0, endpoint = True)
X0 = np.vstack((x0, np.zeros_like(x0, dtype=np.float32))).transpose() # (x, 0)
# boundary points
boundary = np.vstack((lb, ub))
X_lb = np.concatenate((lb[0]*np.ones_like(tb, dtype=np.float32), tb), axis=1)
X_ub = np.concatenate((ub[0]*np.ones_like(tb, dtype=np.float32), tb), axis=1)


num_boundary_conditions = 4
# Train the model
# model = PINN_GAN_burgers(X_exact, u_exact, nu, X_f, X0, X_lb, X_ub, boundary, num_boundary_conditions, layers_G, layers_D)
# start_time = time.time()                
# model.train(X_star, u_star, epochs=500)
# print('Training time: %.4f' % (time.time() - start_time))


# Predictions
# u_pred, _ = model.predict(X_star)

# torch.save(u_pred, "burgers_pred.pt")

# Errors
# errors = {'u': np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)}
# print('Errors: ', errors)

# plot result
mat = torch.load("burgers_pred.pt")
plot_with_ground_truth(mat, X_star, X , T, u_star , ground_truth_ref=False, ground_truth_refpts=[], filename = "ground_truth_comparison.png")
# plot errors
with open('loss_history_burgers.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
plot_loss(loaded_dict,'loss_history_burgers.png')