import scipy
import torch
from PDE_PINNs import Heat_PINN_GAN
from data_structuring import structure_data
from utils.plot import plot_with_ground_truth, plot_loss

# Hyperparameters
noise = 0.0        
N0 = 50 # number of data for initial samples
N_b = 50 # number of data for boundary samples
N_f = 20000 # number of data for collocation points
N_exact = 40 # number of data points that are passed with their exact solutions

# Define the physics-informed neural network
layers_G = [2, 100, 100, 100, 100, 2] # first entry should be X.shape[0], last entry should be Y.shape[0]
layers_D = [4, 100, 100, 100, 100, 1] # input should be X.shape[0]+Y.shape[0], output 1.

pde = 'heat'

# Load data from simulated dataset
data = scipy.io.loadmat('./Data/'+pde+'.mat')

# structure data
grid, X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, X_star, Y_star = structure_data(pde, data, noise, N0, N_b, N_f, N_exact)

model_name ='1'

layers_G[0] = 3
layers_G[-1] = 1
layers_D[0] = 4
model = Heat_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
          layers_G= layers_G, layers_D = layers_D, \
            enable_GAN = True, enable_PW = False, dynamic_lr = False, model_name = model_name, \
                lambdas = [1,1], lr = (1e-3, 1e-3, 5e-3), e = [5e-4]+[5e-6], q = [10e-4]+[5e-5])
model.load("Trained model/heat_3615.pth") # load
model.plot_loss()
y_pred, _ = model.predict(torch.tensor(X_star, requires_grad=True))
model.plot_with_ground_truth(X_star, Y_star, y_pred, grid)

