import torch
import numpy as np
import scipy.io
import time
import os
import matplotlib.pyplot as plt
import pickle
import argparse
import questionary

from PDE_PINNs import Schroedinger_PINN_GAN, Heat_PINN_GAN, Helmholtz_PINN_GAN, Poisson_PINN_GAN, PoissonHD_PINN_GAN, Burgers_PINN_GAN
from data_structuring import structure_data
from utils.plot import plot_with_ground_truth, plot_loss


# random seed for reproduceability
np.random.seed(42)



# Argparser for automation:
def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse command-line options for a PDE-based machine learning model training.")

    # Adding the arguments
    parser.add_argument("-p", "--pde", required=False, type=str, help="The PDE that the model is trained for.")
    parser.add_argument("-n", "--name", required=False, type=str, help="The name under which the model will be saved.")
    parser.add_argument("-e", "--epochs", required=False, type=int, help="The number of epochs the model is trained for.")
    parser.add_argument("--lambda1", required=False, type=float, help="Lambda1 for the model.")
    parser.add_argument("--lambda2", required=False, type=float, help="Lambda2 for the model.")
    parser.add_argument("-g", "--gan", type=lambda x: (str(x).lower() == 'y'), help="Enable or disable the GAN use.")
    parser.add_argument("-w", "--pointweighting", type=lambda x: (str(x).lower() == 'y'), help="Enable or disable the Point Weighting (PW) use.")
    parser.add_argument("-a", "--architecture", type=str, help="choose architecture. Should be one of: standard, deep, wide, convolution")
    parser.add_argument("--lr1", required=False, type=float, help="Learning rate 1.")
    parser.add_argument("--lr2", required=False, type=float, help="Learning rate 2.")
    parser.add_argument("--lr3", required=False, type=float, help="Learning rate 3.")
    parser.add_argument("--e-value", dest="e_value", required=False, type=float, help="The boundary between easy and hard to learn points.")
    parser.add_argument("--noise", dest="noise", required=False, type=float, help="The amount of noise added.")
    parser.add_argument("--N_exact", dest="N_exact", required=False, type=int, help="The amount of exact solutions used.")
    parser.add_argument("--N_b", dest="N_b", required=False, type=int, help="The amount of boundary points used.")

    # Parsing the arguments
    args = parser.parse_args()

    # Storing values in variables
    pde = args.pde if args.pde is not None else ""
    model_name = args.name if args.name is not None else ""
    epochs = args.epochs if args.epochs is not None else 1000
    lambdas = [args.lambda1 if args.lambda1 is not None else 1, args.lambda2 if args.lambda2 is not None else 1]
    enable_GAN = args.gan if args.gan is not None else True
    enable_PW = args.pointweighting if args.pointweighting is not None else True
    architecture = args.architecture if args.architecture is not None else "standard"
    lr = (args.lr1 if args.lr1 is not None else 1e-3, args.lr2 if args.lr2 is not None else 1e-3, args.lr3 if args.lr3 is not None else 5e-3)
    e = args.e_value if args.e_value is not None else 5e-4
    noise = args.noise if args.noise is not None else 0.0
    N_exact = args.N_exact if args.N_exact is not None else 40
    N_b = args.N_b if args.N_b is not None else 50


    # Returning parsed values
    return pde, model_name, epochs, lambdas, enable_GAN, enable_PW, architecture, lr, e, noise, N_exact, N_b

if __name__ == "__main__": # only execute when running not when possibly importing from here
    pde, model_name, epochs, lambdas, enable_GAN, enable_PW, architecture, lr, e, noise, N_exact, N_b = parse_arguments()

print("-------------------------------------------")
print(f"training {model_name} to learn {pde} "+f"with {architecture} architecture "+"with GAN " if enable_GAN else "" + "with PW " if enable_PW else "")
print("-------------------------------------------")




# Hyperparameters
N0 = 50 # number of data for initial samples
# N_b = 50 # number of data for boundary samples
N_f = 20000 # number of data for collocation points
# N_exact = 40 # number of data points that are passed with their exact solutions

# Define the physics-informed neural network
match architecture:
    case "standard":
        layers_G = [2, 100, 100, 100, 100, 2] # first entry should be X.shape[0], last entry should be Y.shape[0]
        layers_D = [4, 100, 100, 100, 100, 1] # input should be X.shape[0]+Y.shape[0], output 1.
    case "deep":
        layers_G = [2, 100, 100, 100, 100, 2] # first entry should be X.shape[0], last entry should be Y.shape[0]
        layers_D = [4, 100, 100, 100, 100, 1] # input should be X.shape[0]+Y.shape[0], output 1.
    case "wide":
        layers_G = [2, 100, 100, 100, 100, 2] # first entry should be X.shape[0], last entry should be Y.shape[0]
        layers_D = [4, 100, 100, 100, 100, 1] # input should be X.shape[0]+Y.shape[0], output 1.
    case "convolution":
        layers_G, layers_D = [2, "conv", 100, 100, 100, 100, 100, 2], [4, "conv", 100, 100, 100, 100, 100, 1]

if pde == "":
    pde = questionary.select("Which pde do you want to choose?", choices=["burgers", "heat", "schroedinger", "poisson", "poissonHD", "helmholtz"]).ask()
    intermediary_pictures = True # clearly not running in automated mode
else:
    intermediary_pictures = False  

# Load data from simulated dataset
data = scipy.io.loadmat('./Data/'+pde+'.mat')

# structure data
grid, X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, X_star, Y_star = structure_data(pde, data, noise, N0, N_b, N_f, N_exact)

# get name for saving
if model_name == "":
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
        layers_G[0] = 2
        layers_G[-1] = 2
        layers_D[0] = 4
        model = Schroedinger_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, \
                    intermediary_pictures=intermediary_pictures, enable_GAN = enable_GAN, enable_PW = enable_PW, dynamic_lr = False, model_name = model_name, \
                        lambdas = lambdas, lr = lr, e = [e]+[5e-4, 1e-4, 1e-4], q = [10e-4]+[5e-3, 5e-3, 5e-3])
    case "burgers":
        layers_G[0] = 2
        layers_G[-1] = 1
        layers_D[0] = 3
        nu = 1e-2/np.pi 
        # NOTE: added extra X, T for plotting
        model = Burgers_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, \
                    intermediary_pictures=intermediary_pictures, enable_GAN = enable_GAN, enable_PW = enable_PW, dynamic_lr = False, model_name = model_name, nu=nu, \
                        lambdas = [1,1], lr = (1e-3, 1e-3, 5e-3), e = [5e-4]+[2e-2, 5e-4, 5e-4], q = [10e-4]+[10e-4, 10e-4, 10e-4])
    case "heat":
        layers_G[0] = 3
        layers_G[-1] = 1
        layers_D[0] = 4
        model = Heat_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, \
                    intermediary_pictures=intermediary_pictures, enable_GAN = enable_GAN, enable_PW = enable_PW, dynamic_lr = False, model_name = model_name, \
                        lambdas = lambdas, lr = lr, e = [e]+[5e-6], q = [10e-4]+[5e-5])
    case "poisson":
        layers_G[0] = 2
        layers_G[-1] = 1
        layers_D[0] = 3
        model = Poisson_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, \
                    intermediary_pictures=intermediary_pictures, enable_GAN = enable_GAN, enable_PW = enable_PW, dynamic_lr = False, model_name = model_name, \
                        lambdas = lambdas, lr = lr, e = [e]+[5e-6, 5e-6, 5e-6, 5e-6], q = [10e-4]+[5e-5, 5e-5, 5e-5, 5e-5])
    case "poissonHD":
        pass
    case "helmholtz":
        layers_G[0] = 2
        layers_G[-1] = 1
        layers_D[0] = 3
        model = Helmholtz_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, \
                    intermediary_pictures=intermediary_pictures, enable_GAN = enable_GAN, enable_PW = enable_PW, dynamic_lr = False, model_name = model_name, k=2*np.pi, \
                        lambdas = lambdas, lr = lr, e = [e]+[5e-4, 5e-4, 5e-4, 5e-4], q = [10e-4]+[6e-5, 6e-5, 6e-5, 6e-5])
    case _:
        print("pde not recognised")
start_time = time.time()         
model.train(epochs, grid, X_star, Y_star, visualize=intermediary_pictures)
print("done training")
print('Training time: %.4f' % (time.time() - start_time))


match pde:
    case "schroedinger":
        u_star, v_star, h_star = Y_star[0], Y_star[1], Y_star[2]
        
        # Predictions
        y_pred, f_pred = model.predict(torch.tensor(X_star, requires_grad=True))
        u_pred, v_pred = y_pred[:,0:1], y_pred[:,1:2]
        h_pred = np.sqrt(u_pred**2 + v_pred**2)

        if intermediary_pictures:
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

    case "burgers" | "poisson" | "helmholtz":
        y_pred, f_pred = model.predict(torch.tensor(X_star, requires_grad=True))
        
        if model_name != "":
            model_name = pde+"_"+model_name
        
        if intermediary_pictures:
            with open("Saves/last_output_"+model_name+".pkl", "rb") as f:
                mat = pickle.load(f)
        
            X, T = grid # TODO if grid has more than two entries ???

            
            plot_with_ground_truth(mat, X_star, X, T, Y_star, ground_truth_ref=False, ground_truth_refpts=[], filename = "Plots/"+model_name+"ground_truth_comparison.png")
            # plot errors
            with open('Saves/loss_history_'+model_name+'.pkl', 'rb') as f:
                loaded_dict = pickle.load(f)
            plot_loss(loaded_dict,'Plots/'+model_name+'loss_history.png')
        # NOTE: formerly I used this: plt.savefig("Plots/"+model_name) Can we implement it like that again?
        print("Error y: ", np.linalg.norm(Y_star-y_pred,2)/np.linalg.norm(Y_star,2))

    case "heat":
        # Predictions
        y_pred, f_pred = model.predict(torch.tensor(X_star, requires_grad=True))
        
        print("Error y: ", np.linalg.norm(Y_star-y_pred,2)/np.linalg.norm(Y_star,2))
        
    case "poissonHD":
        pass
    
print("value of f: ",np.sum(f_pred**2))


