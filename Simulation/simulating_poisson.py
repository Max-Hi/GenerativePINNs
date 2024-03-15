import numpy as np
import scipy.io as sio
from pyDOE import lhs

# Create the training data for the Poisson equation

""" 
equation configuration: 
x \in [0, 1]
y \in [0, 1]
u(x, -1) = u(x, 1) = u(-1, y) = u(1, y) = 0
"""

N_x = 256 # number of spatial points
N_y = 256 # number of spatial points
x = np.linspace(-1, 1, N_x, endpoint = True)[:, None]
# convert to column vector
y = np.linspace(0, 1, N_y, endpoint= False)[:, None]
usol = np.zeros((N_x, N_y))

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
