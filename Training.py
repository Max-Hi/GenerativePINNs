import torch
import numpy as np
import scipy.io
from pyDOE import lhs
import time
import os
import matplotlib.pyplot as plt

from PINN import PINN_GAN, Discriminator, Generator

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

X_t = None # TODO for now just default
Y_t = None

# get name for saving
model_name = input("Give your model a name to be saved under. Press Enter without name to not save. ")
if model_name != "":
    taken_names = os.listdir("Saves")
    taken_names.remove(".gitignore")
    print(taken_names)
    while model_name in list(map(lambda x: x.split("_")[0], taken_names))+["list"] or "_" in model_name:
        model_name = input("The Name is taken or you included '_' in your name. Give your model a different name. To list existing names, enter 'list'. ")
        if model_name == "list":
            for name in list(map(lambda x: x[:-4], os.listdir("Saves"))):
                print(name)

# Train the model
model = PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G, layers_D, model_name)
start_time = time.time()         
model.train(1000)
print('Training time: %.4f' % (time.time() - start_time))


# Predictions
y_pred, f_pred = model.predict(torch.tensor(X_star, requires_grad=True))
u_pred, v_pred = y_pred[:,0:1], y_pred[:,1:2]
h_pred = np.sqrt(u_pred**2 + v_pred**2)

print(u_pred.shape, x.shape)

plt.plot(np.linspace(0,len(u_pred),len(u_pred)),u_pred, label="predicted")
plt.plot(np.linspace(0,len(u_star),len(u_star)),u_star, label="true")
plt.legend()
plt.savefig(model_name+".png")

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
