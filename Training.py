import torch
import numpy as np
import scipy.io
import time
import os
import matplotlib.pyplot as plt
import pickle

import questionary

from PDE_PINNs import Schroedinger_PINN_GAN, Heat_PINN_GAN, Helmholtz_PINN_GAN, Poisson_PINN_GAN, PoissonHD_PINN_GAN, Burgers_PINN_GAN
from data_structuring import structure_data
from utils.plot import plot_with_ground_truth, plot_loss


# random seed for reproduceability
np.random.seed(42)

# Hyperparameters
noise = 0.0        
N0 = 50 # number of data for initial samples
N_b = 50 # number of data for boundary samples
N_f = 20000 # number of data for collocation points

# Define the physics-informed neural network
layers_G = [2, 100, 100, 100, 100, 1] # first entry should be X.shape[0], last entry should be Y.shape[0]
layers_D = [3, 100, 100, 1] # input should be X.shape[0]+Y.shape[0], output 1.

pde = questionary.select("Which pde do you want to choose?", choices=["burgers", "heat", "schroedinger", "poisson", "poissonHD", "helmholtz"]).ask()


# Load data from simulated dataset
data = scipy.io.loadmat('./Data/'+pde+'.mat')

# structure data
grid, X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, X_star, Y_star = structure_data(pde, data, noise, N0, N_b, N_f)
for thing in [grid, X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, X_star, Y_star]:
    try:
        print(thing.shape)
    except Exception as e:
        print(e)

# get name for saving
model_name = input("Give your model a name to be saved under. Press Enter without name to not save. ")
if model_name != "":
    taken_names = os.listdir("Saves")
    taken_names.remove(".gitignore")
    while model_name in list(map(lambda x: x.split("_")[0], taken_names))+["list"] or "_" in model_name:
        model_name = input("The Name is taken or you included '_' in your name. Give your model a different name. To list existing names, enter 'list'. ")
        if model_name == "list":
            for name in list(map(lambda x: x[:-4], os.listdir("Saves"))):
                print(name)

# Train the model

match pde:
    case "schroedinger":
        model = Schroedinger_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, \
                    enable_GAN = True, enable_PW = True, dynamic_lr = False, model_name = model_name, \
                        e = [5e-4, 1e-4], q = [5e-3, 5e-3])
    case "burgers":
        nu = 1e-2/np.pi 
        # NOTE: added extra X, T for plotting
        model = Burgers_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, \
                    enable_GAN = True, enable_PW = True, dynamic_lr = False, model_name = model_name, nu=nu, \
                        e = [2e-2, 5e-4], q = [10e-4, 10e-4])
    case "heat":
        model = Heat_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, \
                    enable_GAN = True, enable_PW = True, dynamic_lr = False, model_name = model_name, \
                        e = [5e-6, 5e-6], q = [5e-5, 6e-5])
    case "poisson":
        model = Poisson_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, \
                    enable_GAN = True, enable_PW = True, dynamic_lr = False, model_name = model_name, \
                        e = [5e-6, 5e-6], q = [5e-5, 5e-5])
    case "poissonHD":
        pass
    case "helmholtz":
        k = 1
        model = Helmholtz_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, \
                    enable_GAN = True, enable_PW = True, dynamic_lr = False, model_name = model_name, k=k, \
                        e = [5e-4, 5e-4], q = [6e-5, 6e-5])
    case _:
        print("pde not recognised")
start_time = time.time()         
model.train(6500, grid, X_star, Y_star)
print('Training time: %.4f' % (time.time() - start_time))


match pde:
    case "schroedinger":
        u_star, v_star, h_star = Y_star[0], Y_star[1], Y_star[2]
        
        # Predictions
        y_pred, f_pred = model.predict(torch.tensor(X_star, requires_grad=True))
        u_pred, v_pred = y_pred[:,0:1], y_pred[:,1:2]
        h_pred = np.sqrt(u_pred**2 + v_pred**2)

        plt.plot(np.linspace(0,len(u_star),len(u_star)),u_star, label="true")
        plt.plot(np.linspace(0,len(u_pred),len(u_pred)),u_pred, label="predicted")
        plt.legend()
        plt.savefig("Plots/"+model_name)
        
        # Errors
        errors = {
                'u': np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2),
                'v': np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2),
                'h': np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
                }
        print('Errors: ')
        for key in errors:
            print(key+": ", errors[key])

    case "burgers":
        y_pred, f_pred = model.predict(torch.tensor(X_star, requires_grad=True))
        
        mat = torch.load("burgers_pred.pt")
        
        X, T = grid # TODO if grid has more than two entries ???

        plot_with_ground_truth(mat, X_star, X, T, Y_star, ground_truth_ref=False, ground_truth_refpts=[], filename = "ground_truth_comparison.png")
        # plot errors
        with open('loss_history_burgers.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        plot_loss(loaded_dict,'loss_history_burgers.png')
        # NOTE: formerly I used this: plt.savefig("Plots/"+model_name) Can we implement it like that again?
        print("Error y: ", np.linalg.norm(Y_star-y_pred,2)/np.linalg.norm(Y_star,2))

    case "heat":
        pass
    case "poisson":
        pass
    case "poissonHD":
        pass
    case "helmholtz":
        pass
    case _:
        print("pde not recognised")
    
print("value of f: ",np.sum(f_pred**2))


