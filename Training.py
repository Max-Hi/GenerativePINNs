import torch
import numpy as np
import scipy.io
from pyDOE import lhs
import time
import matplotlib.pyplot as plt

from PINN_minimal_example import PINN_GAN, Discriminator, Generator

# random seed for reproduceability
np.random.seed(42)

# Hyperparameters
noise = 0.0        
lb = np.array([-5.0, 0.0]) # lower bound for [x, t]
ub = np.array([5.0, np.pi/2]) # upper bound for [x, t]
N0 = 50 # number of data for initial samples
N_b = 50 # number of data for boundary samples
N_f = 20000 # number of data for collocation points

# Define the physics-informed neural network
layers_G = [2, 100, 100, 100, 100, 2]
layers_D = [4, 100, 100, 2]

# Load data from simulated dataset
data = scipy.io.loadmat('./Data/NLS.mat')


t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['uu']
Exact_h = np.sqrt(np.real(Exact)**2 + np.imag(Exact)**2)
X, T = np.meshgrid(x,t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
h_star = Exact_h.T.flatten()[:,None]
v_star = np.imag(Exact).T.flatten()[:,None]
u_star = np.real(Exact).T.flatten()[:,None]

# Initial and boundary data
idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x,:]
u0 = np.real(Exact[idx_x,0:1]) # or computed using h(0, x) = 2*sech(x)
v0 = np.imag(Exact[idx_x,0:1])
Y0 = np.hstack((u0, v0))
tb = t[np.random.choice(t.shape[0], N_b, replace=False),:] # random time samples for boundary points

# Collocation points
X_f = lb + (ub-lb)*lhs(2, N_f)

# initial points
X0 = np.concatenate((x0, np.zeros_like(x0, dtype=np.float32)), 1) # (x, 0)

# boundary points
boundary = np.vstack((lb, ub))
X_lb = np.concatenate((lb[0]*np.ones_like(tb, dtype=np.float32), tb), axis=1)
X_ub = np.concatenate((ub[0]*np.ones_like(tb, dtype=np.float32), tb), axis=1)

# Train the model
model = PINN_GAN(X0, Y0, X_f, X_lb, X_ub, boundary, layers_G, layers_D)
start_time = time.time()                
model.train(2000)
print('Training time: %.4f' % (time.time() - start_time))


# Predictions
u_pred, v_pred, _, _ = model.predict(torch.tensor(X_star, requires_grad=True))
h_pred = np.sqrt(u_pred**2 + v_pred**2)
        
# Errors
errors = {'u': np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2),
          'v': np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2),
          'h': np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)}
print('Errors: ', errors)
