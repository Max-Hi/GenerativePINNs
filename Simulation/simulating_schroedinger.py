import numpy as np
import scipy.io as sio

# Create the training data for the Schroedinger equation

"""
Configuration:
x \in [-5, 5]
t \in [0, pi/2]
h(0, x) = 2 sech(x)
h(t, -5) = h(t, 5)
h_x(t, -5) = h_x(t, 5)

i h_t + 0.5 h_xx + |h|^2 h = 0, -5 <= x <= 5, 0 <= t <= pi/2

Solve that we can get the exact solution:
h(x, t) = sqrt(2) * exp(i*t)*sech(sqrt(2)*x)

We can also get the derivative of the solution:
h_x(x, t) = -2*i*exp(i*t)*sech(sqrt(2)*x)*tanh(sqrt(2)*x)
h_t(x, t) = i*sqrt(2)*exp(i*t)*sech(sqrt(2)*x)^2

We can also get the second derivative of the solution:
h_xx(x, t) = -2sqrt(2)*exp(i*t)*sech(sqrt(2)*x)*(2*sech(sqrt(2)*x)^2 - 1)

|h|^2 h = 2sqrt(2)*exp(i*t)*sech(sqrt(2)*x)^2*sech(sqrt(2)*x)
"""

N_x = 256 # number of spatial points
N_t = 100 # number of time points

x = np.linspace(-5, 5, N_x, endpoint = True)[:, None]
t = np.linspace(0, np.pi/2, N_t, endpoint = False)[:, None]
usol = np.zeros((N_x, N_t), dtype = complex)
usol[:, :] = np.sqrt(2) * np.exp(1j*t.T) * 1/np.cosh(np.sqrt(2)*x)

# save the data to mat file
sio.savemat('Data/schroedinger.mat', {'x': x, 't': t, 'usol': usol})

# plot the solution for the Schroedinger equation
import matplotlib.pyplot as plt
X, T = np.meshgrid(t, x)
print(usol.shape)
print(X.shape)
fig = plt.contourf(X, T, np.abs(usol), 100, cmap='jet')
plt.colorbar(fig)
plt.title('Schroedinger equation solution')
plt.xlabel('x')
plt.ylabel('t')
plt.show()
