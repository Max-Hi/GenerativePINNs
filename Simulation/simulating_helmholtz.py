import numpy as np
import scipy.io as sio
from pyDOE import lhs

# Create the training data for the Helmholtz equation

"""
Configuration:
x, y \in (0, 1)
u(0, y) = sin(k*y)
u(1, y) = sin(k*y)
u(x, 0) = sin(k*x)
u(x, 1) = sin(k*x) 

Delta u + k^2 u = 0

Solve that we can get the exact solution:
u(x, y) = sin(k*x)
"""

N_x = 2000 # number of spatial points
N_y = 2000 # number of spatial points

x = np.linspace(0, 1, N_x, endpoint = True)[:, None]
y = np.linspace(0, 1, N_y, endpoint = True)[:, None]
usol = np.zeros((N_x, N_y))

k = 2*np.pi # wavenumber

# Exact solution
usol = np.sin(k*x) + np.zeros_like(y.T)

# Training data
sio.savemat('Data/helmholtz.mat', {'x': x, 'y': y, 'usol': usol})

# plot the solution for the Helmholtz equation
import matplotlib.pyplot as plt
X, Y = np.meshgrid(y, x)
fig = plt.contourf(X, Y, usol, 100, cmap='jet')
plt.colorbar(fig)
plt.title('Helmholtz equation solution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
