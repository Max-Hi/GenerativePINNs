import numpy as np
import scipy.io as sio
from pyDOE import lhs

# Create the training data for the Poisson equation

""" 
equation configuration: 
x \in [0, 1]
y \in [0, 1]
u(x, -1) = u(x, 1) = u(-1, y) = u(1, y) = 0

Delta u = -sin(pi*x)*sin(pi*y)

Solve that we can get the exact solution:
u(x, y) = 1/(2*pi^2) * sin(pi*x) * sin(pi*y)
"""

N_x = 2000 # number of spatial points
N_y = 2000 # number of spatial points
x = np.linspace(0, 1, N_x, endpoint = True)[:, None]
# convert to row vector
y = np.linspace(0, 1, N_y, endpoint= True)[:, None]
usol = np.zeros((N_x, N_y))

# Create the initial condition
usol[:, 0] = 0
usol[0, :] = 0

# Exact solution
usol = 1/(2*np.pi**2) * np.dot(np.sin(np.pi*x), np.sin(np.pi*y.T))

# save the data to mat file
sio.savemat('Data/poisson.mat', {'x': x, 'y': y, 'usol': usol})

# plot the solution for the Poisson equation
import matplotlib.pyplot as plt
X, Y = np.meshgrid(x, y)
fig = plt.contourf(X, Y, usol, 100, cmap='jet')
plt.colorbar(fig)
plt.title('Poisson equation solution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
