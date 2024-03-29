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
    parser.add_argument("-a", "--architecture", type=str, help="choose architecture. Should be one of: standard, deep, wide, convolution, lstm")
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
    match pde:
        case "schroedinger" | "burgers" | "heat":
            lr_default = (1e-3, 1e-3, 5e-3)
        case "poisson":
            lr_default = (1e-3, 1e-6, 5e-6)
        case "poissonHD":
            pass
        case "helmholtz":
            lr_default = (1e-3, 1e-5, 5e-5)
        case _:
            lr_default = (1e-3, 1e-3, 5e-3)
            print("learning rate is chosen to be "+str(lr_default)+" because pde is not yet specified. Please use argparser to change.")
    lr = (args.lr1 if args.lr1 is not None else lr_default[0], args.lr2 if args.lr2 is not None else lr_default[1], args.lr3 if args.lr3 is not None else lr_default[2])
    e = args.e_value if args.e_value is not None else 5e-4
    noise = args.noise if args.noise is not None else 0.0
    N_exact = args.N_exact if args.N_exact is not None else 40
    N_b = args.N_b if args.N_b is not None else 50


    # Returning parsed values
    return pde, model_name, epochs, lambdas, enable_GAN, enable_PW, architecture, lr, e, noise, N_exact, N_b

if __name__ == "__main__": # only execute when running not when possibly importing from here
    pde, model_name, epochs, lambdas, enable_GAN, enable_PW, architecture, lr, e, noise, N_exact, N_b = parse_arguments()

print("-------------------------------------------")
info_string = f"training {model_name} to learn {pde} "+f"with {architecture} architecture "
if enable_GAN:
    info_string += "with GAN "
if enable_PW:
    info_string += "with PW "
print(info_string)
print("-------------------------------------------")




# Hyperparameters
match pde:
    case "schroedinger":
        layers_G = [2, 100, 100, 100, 100, 2] # first entry should be X.shape[0], last entry should be Y.shape[0]
        layers_D = [4, 100, 100, 100, 1] # input should be X.shape[0]+Y.shape[0], output 1.
    case "burgers":
        layers_G = [2, 20, 20, 20, 20, 20, 20, 20, 1] # first entry should be X.shape[0], last entry should be Y.shape[0]
        layers_D = [3, 20, 20, 20, 20, 20, 20, 1] # input should be X.shape[0]+Y.shape[0], output 1.
    case "heat" | "poisson" | "helmholtz":
        layers_G = [2, 100, 100, 100, 100, 1] # first entry should be X.shape[0], last entry should be Y.shape[0]
        layers_D = [3, 100, 1] # input should be X.shape[0]+Y.shape[0], output 1.
    case "poissonHD":
        pass
    case _:
        print("pde not recognised")
N0 = 50 # number of data for initial samples
# N_b = 50 # number of data for boundary samples
N_f = 20000 # number of data for collocation points
# N_exact = 40 # number of data points that are passed with their exact solutions


if pde == "":
    pde = questionary.select("Which pde do you want to choose?", choices=["burgers", "heat", "schroedinger", "poisson", "poissonHD", "helmholtz"]).ask()
    intermediary_pictures = True # clearly not running in automated mode
    match pde:
        case "schroedinger" | "burgers" | "heat":
            lr = (1e-3, 1e-3, 5e-3)
        case "poisson":
            lr = (1e-3, 1e-6, 5e-6)
        case "poissonHD":
            pass
        case "helmholtz":
            lr = (1e-3, 1e-5, 5e-5)
        case _:
            print("pde not recognized")
else:
    intermediary_pictures = False  

# Define the physics-informed neural network
lstm = False
match architecture:
    case "standard":
        match pde:
            case "schroedinger":
                layers_G = [2, 100, 100, 100, 100, 2] # first entry should be X.shape[0], last entry should be Y.shape[0]
                layers_D = [4, 100, 100, 100, 1] # input should be X.shape[0]+Y.shape[0], output 1.
            case "burgers":
                layers_G = [2, 20, 20, 20, 20, 20, 20, 20, 1] # first entry should be X.shape[0], last entry should be Y.shape[0]
                layers_D = [3, 20, 20, 20, 20, 20, 20, 1] # input should be X.shape[0]+Y.shape[0], output 1.
            case "heat" | "poisson" | "helmholtz":
                layers_G = [2, 100, 100, 100, 100, 1] # first entry should be X.shape[0], last entry should be Y.shape[0]
                layers_D = [3, 100, 1] # input should be X.shape[0]+Y.shape[0], output 1.
            case "poissonHD":
                pass
            case _:
                print("pde not recognised")
    case "deep":
        match pde:
            case "schroedinger":
                layers_G = [2, 100, 100, 100, 100, 100, 100, 100, 100, 2] # first entry should be X.shape[0], last entry should be Y.shape[0]
                layers_D = [4, 100, 100, 100, 100, 100, 100, 1] # input should be X.shape[0]+Y.shape[0], output 1.
            case "burgers":
                layers_G = [2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1] # first entry should be X.shape[0], last entry should be Y.shape[0]
                layers_D = [3, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1] # input should be X.shape[0]+Y.shape[0], output 1.
            case "heat" | "poisson" | "helmholtz":
                layers_G = [2, 100, 100, 100, 100, 100, 100, 100, 100, 1] # first entry should be X.shape[0], last entry should be Y.shape[0]
                layers_D = [3, 100, 100, 1] # input should be X.shape[0]+Y.shape[0], output 1.
            case "poissonHD":
                pass
            case _:
                print("pde not recognised")
    case "wide":
        match pde:
            case "schroedinger":
                layers_G = [2, 200, 200, 200, 2] # first entry should be X.shape[0], last entry should be Y.shape[0]
                layers_D = [4, 200, 200, 1] # input should be X.shape[0]+Y.shape[0], output 1.
            case "burgers":
                layers_G = [2, 80, 80, 80, 80, 80, 80, 1] # first entry should be X.shape[0], last entry should be Y.shape[0]
                layers_D = [3, 80, 80, 80, 80, 80, 1] # input should be X.shape[0]+Y.shape[0], output 1.
            case "heat" | "poisson" | "helmholtz":
                layers_G = [2, 200, 200, 200, 1] # first entry should be X.shape[0], last entry should be Y.shape[0]
                layers_D = [3, 200, 1] # input should be X.shape[0]+Y.shape[0], output 1.
            case "poissonHD":
                pass
            case _:
                print("pde not recognised")
    case "convolution":
        layers_G, layers_D = [2, "conv", 80, 80, 80, 80, 80, 80, 80, 80, 80, 2], [4, "conv", 80, 80, 80, 80, 80, 80, 80, 80, 80, 1]
    case "lstm":
        lstm = True
        match pde:
            case "schroedinger":
                layers_G = [2, 100, 100, 100, 100, 2] # first entry should be X.shape[0], last entry should be Y.shape[0]
                layers_D = [4, 100, 100, 100, 1] # input should be X.shape[0]+Y.shape[0], output 1.
            case "burgers":
                layers_G = [2, 20, 20, 20, 20, 20, 20, 20, 1] # first entry should be X.shape[0], last entry should be Y.shape[0]
                layers_D = [3, 20, 20, 20, 20, 20, 20, 1] # input should be X.shape[0]+Y.shape[0], output 1.
            case "heat" | "poisson" | "helmholtz":
                layers_G = [2, 100, 100, 100, 100, 1] # first entry should be X.shape[0], last entry should be Y.shape[0]
                layers_D = [3, 100, 1] # input should be X.shape[0]+Y.shape[0], output 1.
            case "poissonHD":
                pass
            case _:
                print("pde not recognised")

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
                 layers_G= layers_G, layers_D = layers_D, enable_lstm=lstm,\
                    intermediary_pictures=intermediary_pictures, enable_GAN = enable_GAN, enable_PW = enable_PW, dynamic_lr = False, model_name = model_name, \
                        lambdas = lambdas, lr = lr, e = [e]+[5e-4, 1e-4, 1e-4], q = [10e-4]+[5e-3, 5e-3, 5e-3])
    case "burgers":
        layers_G[0] = 2
        layers_G[-1] = 1
        layers_D[0] = 3
        nu = 1e-2/np.pi 
        # NOTE: added extra X, T for plotting
        model = Burgers_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, enable_lstm=lstm,\
                    intermediary_pictures=intermediary_pictures, enable_GAN = enable_GAN, enable_PW = enable_PW, dynamic_lr = False, model_name = model_name, nu=nu, \
                        lambdas = [1,1], lr = (1e-3, 1e-3, 5e-3), e = [5e-4]+[2e-2, 5e-4, 5e-4], q = [10e-4]+[10e-4, 10e-4, 10e-4])
    case "heat":
        layers_G[0] = 3
        layers_G[-1] = 1
        layers_D[0] = 4
        model = Heat_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, enable_lstm=lstm,\
                    intermediary_pictures=intermediary_pictures, enable_GAN = enable_GAN, enable_PW = enable_PW, dynamic_lr = False, model_name = model_name, \
                        lambdas = lambdas, lr = lr, e = [e]+[5e-6], q = [10e-4]+[5e-5])
    case "poisson":
        layers_G[0] = 2
        layers_G[-1] = 1
        layers_D[0] = 3
        model = Poisson_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, enable_lstm=lstm,\
                    intermediary_pictures=intermediary_pictures, enable_GAN = enable_GAN, enable_PW = enable_PW, dynamic_lr = False, model_name = model_name, \
                        lambdas = lambdas, lr = lr, e = [e]+[5e-6, 5e-6, 5e-6, 5e-6], q = [10e-4]+[5e-5, 5e-5, 5e-5, 5e-5])
    case "poissonHD":
        pass
    case "helmholtz":
        layers_G[0] = 2
        layers_G[-1] = 1
        layers_D[0] = 3
        model = Helmholtz_PINN_GAN(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, \
                 layers_G= layers_G, layers_D = layers_D, enable_lstm=lstm,\
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

        if model_name != "":
            model_name = pde+"_"+model_name
        
        if not intermediary_pictures:
            X, T = grid 
            plot_with_ground_truth(h_pred, X_star, X, T, h_star, ground_truth_ref=False, ground_truth_refpts=[], filename = "Plots/"+model_name+"ground_truth_comparison.png", show_figure = False)
            # plot errors
            with open('Saves/loss_history_'+model_name+'.pkl', 'rb') as f:
                loaded_dict = pickle.load(f)
            plot_loss(loaded_dict,'Plots/'+model_name+'loss_history.png', show_figure = False)
        
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
        
        if not intermediary_pictures:
            with open("Saves/last_output_"+model_name+".pkl", "rb") as f:
                mat = pickle.load(f)
        
            X, T = grid
            
            plot_with_ground_truth(mat, X_star, X, T, Y_star, ground_truth_ref=False, ground_truth_refpts=[], filename = "Plots/"+model_name+"ground_truth_comparison.png", show_figure = False)
            # plot errors
            with open('Saves/loss_history_'+model_name+'.pkl', 'rb') as f:
                loaded_dict = pickle.load(f)
            plot_loss(loaded_dict,'Plots/'+model_name+'loss_history.png', show_figure = False)
        # NOTE: formerly I used this: plt.savefig("Plots/"+model_name) Can we implement it like that again?
        print("Error y: ", np.linalg.norm(Y_star-y_pred,2)/np.linalg.norm(Y_star,2))

    case "heat":
        # Predictions
        y_pred, f_pred = model.predict(torch.tensor(X_star, requires_grad=True))
        
        if model_name != "":
            model_name = pde+"_"+model_name
        
        if not intermediary_pictures:
            with open("Saves/last_output_"+model_name+".pkl", "rb") as f:
                mat = pickle.load(f)
            
            X1, X2, _ = grid #NOTE: visualisation will be less meaningfull
            
            nX1, nX2, nT = X1.shape
            # nPixels = X1.shape[0]*X1.shape[1]
            for t in range(0, nT, 10):
                x1 = np.linspace(boundary[0, 0], boundary[1, 0], nX1, endpoint = True)[None, :]
                x2 = np.linspace(boundary[0, 1], boundary[1, 1], nX2, endpoint = True)[None, :]

                x1, x2 =np.meshgrid(x1, x2)
                x_star_m = np.hstack((x1.flatten()[:, None], x2.flatten()[:, None]))
                y_pred_m = y_pred.reshape(X1.shape)[:,:,t]
                y_star_m = Y_star.reshape(X1.shape)[:,:,t]
                plot_with_ground_truth(y_pred_m, x_star_m, x1, x2, y_star_m, ground_truth_ref=False, 
                                       ground_truth_refpts=[], filename = "Plots/heat_map_" + model_name + "_" + str(t) +".png") # TODO y_star dimensionality
                # plot errors
            with open('Saves/loss_history_'+model_name+'.pkl', 'rb') as f:
                loaded_dict = pickle.load(f)
            plot_loss(loaded_dict,'Plots/'+model_name+'loss_history.png', show_figure = False)
        
        print("Error y: ", np.linalg.norm(Y_star-y_pred,2)/np.linalg.norm(Y_star,2))
        
    case "poissonHD":
        pass
    
print("value of f: ",np.sum(f_pred**2))


