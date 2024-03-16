import numpy as np
import scipy.io as sio
from pyDOE import lhs

# Create the training data for the Helmholtz equation

""" 
equation configuration: 
x \in [0, 1]
y \in [0, 1]
u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = sin(kx)
"""

N_x = 256 # number of spatial points
N_y = 256 # number of spatial points
k = np.pi
x = np.linspace(0, 1, N_x, endpoint = True)[:, None]
# convert to column vector
# y = np.linspace(0, 1, N_y, endpoint= True)[:, None]
usol = np.zeros((N_x, N_y))

# Create the initial condition
usol[:,0] = -np.sin(np.pi*x).flatten()

# Exact solution
usol = -np.sin(np.pi*x).repeat(N_y, 1)

print(usol.shape)
# reshape the data to 2D
usol = usol.T.squeeze()

# save the data to mat file
# sio.savemat('Data/burgers.mat', {'x': x, 't': t, 'usol': usol})
