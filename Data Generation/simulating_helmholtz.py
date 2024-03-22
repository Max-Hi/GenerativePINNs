import numpy as np
import scipy.io as sio
from pyDOE import lhs

# Create the training data for the Helmholtz equation

"""
Configuration:
x1, x2 \in (0, 1)
u(0, x2) = sin(k*x1)
u(1, x2) = sin(k*x1)
u(x1, 0) = sin(k*x1)
u(x1, 1) = sin(k*x1) 

Delta u + k^2 u = 0

Solve that we can get the exact solution:
u(x, x2) = sin(k*x)
"""

N_x1 = 200 # number of spatial points
N_x2 = 200 # number of spatial points

x1 = np.linspace(0, 1, N_x1, endpoint = True)[None, :]
x2 = np.linspace(0, 1, N_x2, endpoint = True)[None, :]
usol = np.zeros((N_x1, N_x2))

k = 2*np.pi # wavenumber

# Exact solution
usol = np.sin(k*x1) + np.zeros_like(x2.T)

# Training data
sio.savemat('Data/helmholtz.mat', {'x1': x1, 'x2': x2, 'usol': usol})

# plot the solution for the Helmholtz equation
import matplotlib.pyplot as plt
X1, X2 = np.meshgrid(x1, x2)
fig = plt.contourf(X1, X2, usol, 100, cmap='jet')
plt.colorbar(fig)
plt.title('Helmholtz equation solution')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
