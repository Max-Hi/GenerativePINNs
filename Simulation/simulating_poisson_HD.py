import numpy as np
import scipy.io as sio

# Create the training data for the HD Poisson equation

"""
Configuration:

x_1, x_2, ..., x_10 \in (0, 1)

u(x) = x_1^2 - x_2^2 + x_3^2 - x_4^2 + x_5x_6 + x_7x_8x_9x_10
"""

N_d = 4 # number of spatial points

x = np.linspace(0, 1, N_d, endpoint = True)[:, None]
usol = np.zeros((N_d, N_d, N_d, N_d, N_d, N_d, N_d, N_d, N_d, N_d))

# Create a N_d-dimensional grid of indices
indices = np.indices([N_d]*10)

# Compute usol using vectorized operations
usol = x[indices[0]]**2 - x[indices[1]]**2 + x[indices[2]]**2 - x[indices[3]]**2 + x[indices[4]]*x[indices[5]] + x[indices[6]]*x[indices[7]]*x[indices[8]]*x[indices[9]]

sio.savemat('poisson_HD_train_data.mat', {'x': x, 'usol': usol})