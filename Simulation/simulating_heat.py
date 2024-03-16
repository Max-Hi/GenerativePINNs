import numpy as np
import scipy.io as sio
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Create the training data for the heat equation

"""
equation configuration:
x, y \in [0, 1]
t \in [0, 10]
u(x, y, 0) = x - y
"""

N_x = 256 # number of spatial points
N_y = 256 # number of spatial points
N_t = 100 # number of time points


x = np.linspace(0, 1, N_x, endpoint = True)[:, None]
y = np.linspace(0, 1, N_y, endpoint = True)[:, None]
t = np.linspace(0, 100, N_t, endpoint = False)[:, None]
usol = np.zeros((N_x, N_y, N_t))

# initial condition
usol[:, :, 0] = x - y.T

nu = (t[1] - t[0])/(x[1] - x[0])*(y[1] - y[0]) # diffusion coefficient

# Create the matrix A for the Crank-Nicolson method
diagonals = [-nu, 1 + 2*nu, -nu]
A = diags(diagonals, [-1, 0, 1], shape=(N_x-2, N_x-2))

# Create the solution at each time point
for n in range(1, N_t):
    # Create the right-hand side for the Crank-Nicolson method
    b = np.zeros(N_x-2)
    for j in range(1, N_y-1):
        b[:] = usol[1:-1, j, n-1] + nu/2 * (usol[:-2, j, n-1] - 2*usol[1:-1, j, n-1] + usol[2:, j, n-1]) + \
                nu/2 * (usol[1:-1, j-1, n-1] - 2*usol[1:-1, j, n-1] + usol[1:-1, j+1, n-1])
        # Add the boundary conditions to the right-hand side
        b[0] += nu/2 * usol[0, j, n]
        b[-1] += nu/2 * usol[-1, j, n]
        # Solve the system of linear equations
        usol[1:-1, j, n] = spsolve(A, b)

# plot the solution for the heat equation for each 10 time point
import matplotlib.pyplot as plt
X, Y = np.meshgrid(x, y)

for i in range(0, N_t, 10):
    plt.contourf(X, Y, usol[:, :, i], 100, cmap='jet')
    plt.colorbar()
    plt.title('t = {}'.format(t[i, 0]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Save the data to a mat file
sio.savemat('Data/heat.mat', {'x': x, 'y': y, 't': t, 'usol': usol})