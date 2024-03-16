import numpy as np
import scipy.io as sio
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Create the training data for the heat equation

"""
equation configuration:
x1, x2 \in [0, 1]
t \in [0, 10]
u(x1, x2, 0) = x1 - x2
"""

N_x1 = 256 # number of spatial points
N_x2 = 256 # number of spatial points
N_t = 100 # number of time points


x1 = np.linspace(0, 1, N_x1, endpoint = True)[None, :]
x2 = np.linspace(0, 1, N_x2, endpoint = True)[None, :]
t = np.linspace(0, 100, N_t, endpoint = False)[None, :]
usol = np.zeros((N_x1, N_x2, N_t))

# initial condition
usol[:, :, 0] = x1 - x2.T

nu = (t[0,1] - t[0,0])/(x1[0,1] - x1[0,0])*(x2[0,1] - x2[0,0]) # diffusion coefficient

# Create the matrix1 A for the Crank-Nicolson method
diagonals = [-nu, 1 + 2*nu, -nu]
A = diags(diagonals, [-1, 0, 1], shape=(N_x1-2, N_x1-2))

# Create the solution at each time point
for n in range(1, N_t):
    # Create the right-hand side for the Crank-Nicolson method
    b = np.zeros(N_x1-2)
    for j in range(1, N_x2-1):
        b[:] = usol[1:-1, j, n-1] + nu/2 * (usol[:-2, j, n-1] - 2*usol[1:-1, j, n-1] + usol[2:, j, n-1]) + \
                nu/2 * (usol[1:-1, j-1, n-1] - 2*usol[1:-1, j, n-1] + usol[1:-1, j+1, n-1])
        # Add the boundarx2 conditions to the right-hand side
        b[0] += nu/2 * usol[0, j, n]
        b[-1] += nu/2 * usol[-1, j, n]
        # Solve the sx2stem of linear equations
        usol[1:-1, j, n] = spsolve(A, b)

# plot the solution for the heat equation for each 10 time point
import matplotlib.pyplot as plt
x1, x2 = np.meshgrid(x1, x2)

for i in range(0, N_t, 10):
    plt.contourf(x1, x2, usol[:, :, i], 100, cmap='jet')
    plt.colorbar()
    plt.title('t = {}'.format(t[0, i]))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Save the data to a mat file
sio.savemat('Data/heat.mat', {'x1': x1, 'x2': x2, 't': t, 'usol': usol})