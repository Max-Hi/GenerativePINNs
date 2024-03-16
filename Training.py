import torch
import numpy as np
import scipy.io
from pyDOE import lhs
import time
import os
import matplotlib.pyplot as plt

from PDE_PINNs import Schroedinger_PINN_GAN, Heat_PINN_GAN, Helmholtz_PINN_GAN, Poisson_PINN_GAN, PoissonHD_PINN_GAN, Burgers_PINN_GAN

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
layers_D = [4, 100, 100, 1]

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

# Training Samples with Y values
n, m = len(x), len(t)
k1, k2 = 20, 20  # Number of samples we want to draw

idx_x = np.random.choice(n, k1, replace=False) 
idx_t = np.random.choice(m, k2, replace=False) 
# sample X
sampled_x = x[idx_x]
sampled_t = t[idx_t]
mesh_x, mesh_t = np.meshgrid(sampled_x, sampled_t, indexing='ij')
X_t = np.hstack((mesh_x.flatten()[:,None], mesh_t.flatten()[:,None]))
# sample Y
mesh_idx_x, mesh_idx_t = np.meshgrid(idx_x, idx_t, indexing='ij')
# Use the mesh to index into u
Y_t = np.hstack((np.real(Exact[mesh_idx_x, mesh_idx_t]).flatten()[:,None],np.imag(Exact[mesh_idx_x, mesh_idx_t]).flatten()[:,None]))

# get name for saving
model_name = input("Give your model a name to be saved under. Press Enter without name to not save. ")
if model_name != "":
    taken_names = os.listdir("Saves")
    taken_names.remove(".gitignore")
    while model_name in list(map(lambda x: x.split("_")[0], taken_names))+["list"] or "_" in model_name:
        model_name = input("The Name is taken or you included '_' in your name. Give your model a different name. To list existing names, enter 'list'. ")
        if model_name == "list":
            for name in list(map(lambda x: x[:-4], os.listdir("Saves"))):
                print(name)

# Train the model

model = Schroedinger_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, \
                    enable_GAN = True, enable_PW = True, dynamic_lr = False, model_name = model_name)
start_time = time.time()         
model.train(6500, X_star, u_star, v_star, h_star)
print('Training time: %.4f' % (time.time() - start_time))


# Predictions
y_pred, f_pred = model.predict(torch.tensor(X_star, requires_grad=True))
u_pred, v_pred = y_pred[:,0:1], y_pred[:,1:2]
h_pred = np.sqrt(u_pred**2 + v_pred**2)


plt.plot(np.linspace(0,len(u_star),len(u_star)),u_star, label="true")
plt.plot(np.linspace(0,len(u_pred),len(u_pred)),u_pred, label="predicted")
plt.legend()
plt.savefig("Plots/"+model_name)

# Errors
errors = {
        'u': np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2),
        'v': np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2),
        'h': np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
          }
print('Errors: ')
for key in errors:
    print(key+": ", errors[key])
    
print("value of f: ",np.sum(f_pred**2))
