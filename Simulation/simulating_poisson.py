import numpy as np
import scipy.io as sio
from pyDOE import lhs

# Create the training data for the Poisson equation

""" 
equation configuration: 
x1 \in [0, 1]
x2 \in [0, 1]
u(x1, -1) = u(x1, 1) = u(-1, x2) = u(1, x2) = 0

Delta u = -sin(pi*x1)*sin(pi*x2)

Solve that we can get the ex1act solution:
u(x1, x2) = 1/(2*pi^2) * sin(pi*x1) * sin(pi*x2)
"""

N_x1 = 2000 # number of spatial points
N_x2 = 2000 # number of spatial points
x1 = np.linspace(0, 1, N_x1, endpoint = True)[None, :]
x2 = np.linspace(0, 1, N_x2, endpoint= True)[None, :]
usol = np.zeros((N_x1, N_x2))

# Create the initial condition
usol[:, 0] = 0
usol[0, :] = 0

# Ex1act solution
usol = 1/(2*np.pi**2) * np.dot(np.sin(np.pi*x1.T), np.sin(np.pi*x2))

# save the data to mat file
sio.savemat('Data/poisson.mat', {'x1': x1, 'x2': x2, 'usol': usol})

# plot the solution for the Poisson equation
import matplotlib.pyplot as plt
x1, x2 = np.meshgrid(x1, x2)
fig = plt.contourf(x1, x2, usol, 100, cmap='jet')
plt.colorbar(fig)
plt.title('Poisson equation solution')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
