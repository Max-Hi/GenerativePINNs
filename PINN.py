# import modules
import numpy as np
import sys
from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from scipy.integrate import odeint

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import adam

from utils.plot import plot_with_ground_truth

# set random seeds for reproducability
np.random.seed(42)
torch.manual_seed(42)

# set default dtype
torch.set_default_dtype(torch.float64)

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
print(f"Using {device} device")

torch.autograd.set_detect_anomaly(True)
# Initialize NNs
class Generator(nn.Module):
    def __init__(self, layers_G, info_for_error=(None, None)):
        super(Generator, self).__init__()
        self.layers = layers_G
        self.info_for_error = info_for_error
        self.model = nn.Sequential()
        # TODO: baseline structure to be altered: More than fully connected
        for l in range(0, len(self.layers) - 2):
            self.model.add_module("linear" + str(l), nn.Linear(self.layers[l], self.layers[l+1]))
            self.model.add_module("tanh" + str(l), nn.Tanh())
        self.model.add_module("linear" + str(len(self.layers) - 2), nn.Linear(self.layers[-2], self.layers[-1]))
    
    def forward(self, x):
        # print(x.shape)
        try:
            return self.model(x)
        
        except RuntimeError as e:
            print(f"Caught RuntimeError: {e}")
            
            if self.info_for_error is not None:
                print(f"It could be that the values in layers_G do not match with the dimensions of the data. The first entry of layers_D should be {self.info_for_error[0]} and is {self.layers[0]}.")
            else:
                print("It could be that the values in layers_G do not match with the dimensions of the data.")
            if self.info_for_error is not None:
                print(f"The last entry of layers_G should be {self.info_for_error[1]} and is {self.layers[-1]}.")
            sys.exit(1)

class Generator_LSTM(nn.Module):
    def __init__(self, layers_G):
        super(Generator_LSTM, self).__init__()
        self.layers = layers_G
        self.num_layers = 2
        # input_size, hidden_size, num_layers, output_size
        self.lstm = nn.LSTM(self.layers[0],self.layers[1], self.num_layers, batch_first= True)
        self.linear = nn.Linear(self.layers[1], self.layers[-1])
    def forward(self, x):
        # print(x.unsqueeze(1).shape)
        out, _ = self.lstm(x.unsqueeze(1))#, (h0, c0))  # Forward pass through LSTM layer
        out = self.linear(out[:, -1, :]) 
        return out

class Discriminator(nn.Module):
    def __init__(self, layers_D, info_for_error=(None, None)):
        super(Discriminator, self).__init__()
        self.layers = layers_D
        self.info_for_error = info_for_error
        self.model = nn.Sequential()
        # TODO: baseline structure to be altered 
        for l in range(0, len(self.layers) - 1):
            self.model.add_module("linear" + str(l), nn.Linear(self.layers[l], self.layers[l+1]))
            self.model.add_module("tanh" + str(l), nn.Tanh())
        self.model.add_module("sigmoid" + str(len(self.layers) - 1), nn.Sigmoid()) 
        self.model = self.model.double()
    def forward(self, x):
        # print("D", x.shape)
        try:
            return self.model(x)
        except RuntimeError as e:
            print(f"Caught RuntimeError: {e}")
            if self.info_for_error is not None:
                print(f"It could be that the values in layers_D do not match with the dimensions of the data. The first entry of layers_D should be {self.info_for_error[0]} and is {self.layers[0]}.")
            else:
                print("It could be that the values in layers_D do not match with the dimensions of the data.")
            if self.info_for_error is not None:
                print(f"The last entry of layers_D should be {self.info_for_error[1]} and is {self.layers[-1]}.")
            sys.exit(1)
    
class weighted_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets,weights):
        return torch.dot(weights.flatten(),\
                          torch.sum(torch.square(inputs-targets), axis = 1).flatten())
        
class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets):
        return -(1/(inputs.shape[0]))*torch.sum(torch.log(inputs-targets+torch.ones_like(inputs)*1e-3))
    
    
class PINN_GAN(nn.Module):
    def __init__(self, X0, Y0, X_f, X_t, Y_t, X_lb, u_lb, X_ub, u_ub, boundary, layers_G : list=[], layers_D: list=[], enable_GAN = True, enable_PW = True, dynamic_lr = False, enable_augmentation = True, enable_PID = True, model_name: str="", lr: tuple=(1e-3, 1e-3, 5e-3), lambdas: tuple = (1,1), e: list=[2e-2, 5e-4], q: list=[1e-4, 1e-4],):
        """
        for any initial / boundary conditions: pass None if they are not needed.
        X0: T=0, initial condition, randomly drawn from the domain
        Y0: T=0, initial condition, given (u0, v0)
        X_f: the collocation points with time, size (Nf, dim(X)+1)
        X_t: x values that have exact y values. These will be used to calculate the L_t loss. 
        Y_t: y values to the exact x values
        X_lb: the lower boundary, size (N_b, 2)
        X_ub: the upper boundary, size (N_b, 2)
        boundary: the lower and upper boundary, size (2, 2) : [(x_min, t_min), (x_max, t_max)]
        layers: the number of neurons in each layer (_D for discriminator, _G for generator)
        enable_GAN, enable_PW: 
        model_name: the name the model is saved under. If the name is an empty string, it is not saved. This is the default.
        lr: the learning rate as a 3-tupel in the order lr_G, lr_PW, lr_D
        """
        super().__init__()

       # Hyperparameters
        self.q = q
        self.lambdas = lambdas
        self.e = e  # Hyperparameter for PW update
        
        # parameters for saving
        self.create_saves = model_name!=""
        self.name = model_name
        
        # Arrays for interesting values
        self.rho_values = []
        self.loss_values = {"Generator": [], "Discriminator": [], "Pointwise": []}
        

        # training points that have values
        self.x_t = torch.tensor(X_t)
        self.y_t = torch.tensor(Y_t)

        if enable_PID:
            self.D_input_ref = torch.concat((self.x_t, self.y_t, torch.ones(self.x_t.shape[0], 1)), 1)
        else:
            self.D_input_ref = torch.concat((self.x_t, self.y_t), 1)
        # print(self.D_input_ref.shape)
        self.G_input_ref = self.x_t
        # Initial Data
        if X0 is not None:
            self.x0 = torch.tensor(X0, requires_grad=True)
        if Y0 is not None:
            self.y0 = torch.tensor(Y0)
            # print(torch.cat((self.x0, self.y0, torch.ones(self.x0.shape[0], 1)), 1).shape)
            if enable_PID:
                self.D_input_ref = torch.vstack((self.D_input_ref, 
                                                torch.cat((self.x0, self.y0, torch.ones(self.x0.shape[0], 1)), 1)))
            else:
                self.D_input_ref = torch.vstack((self.D_input_ref, 
                                                torch.cat((self.x0, self.y0), 1)))
            self.G_input_ref = torch.vstack((self.G_input_ref, self.x0))
        # Boundary Data
        if X_lb is not None:
            self.x_lb = torch.tensor(X_lb, requires_grad=True)
        if u_lb is not None:
            self.u_lb = torch.tensor(u_lb, requires_grad = True)
            # print(torch.cat((self.x_lb, self.u_lb, torch.ones(self.x_lb.shape[0], 1)), 1).shape)
            # print(self.x_lb-self.u_lb)
            if enable_PID:
                self.D_input_ref = torch.vstack((self.D_input_ref, 
                                                torch.cat((self.x_lb, self.u_lb, torch.ones(self.x_lb.shape[0], 1)), 1)))
            else:
                self.D_input_ref = torch.vstack((self.D_input_ref, 
                                                torch.cat((self.x_lb, self.u_lb), 1)))
            self.G_input_ref = torch.vstack((self.G_input_ref, self.x_lb))
        if X_ub is not None:
            self.x_ub = torch.tensor(X_ub, requires_grad=True)
        if u_ub is not None:
            self.u_ub = torch.tensor(u_ub, requires_grad = True)
            self.D_input_ref = torch.vstack((self.D_input_ref, 
                                            torch.cat((self.x_ub, self.u_ub, torch.ones(self.x_ub.shape[0], 1)), 1)))
            self.G_input_ref = torch.vstack((self.G_input_ref, self.x_ub))
        # exact reference points for D/G input
        
        
        
        # Collocation Points
        self.x_f = torch.tensor(X_f, requires_grad=True)
        
        # Bounds
        self.lb = torch.tensor(boundary[0:1, :].T)
        self.ub = torch.tensor(boundary[1:2, :].T)
        
        # Sizes
        if enable_GAN:
            self.layers_D = layers_D
        self.layers_G = layers_G
        
        self.generator = Generator(self.layers_G, info_for_error=(self.x_f.shape[1],self.y_t.shape[1]))
        if enable_GAN:
            self.discriminator = Discriminator(self.layers_D, info_for_error=(self.y_t.shape[1]+self.x_f.shape[1]+1,1))
        
        # options
        self.enable_GAN = enable_GAN
        self.enable_PW = enable_PW
        self.dynamic_lr = dynamic_lr
        self.enable_augmentation = enable_augmentation
        self.enable_PID = enable_PID
        
        # Optimizer
        if self.enable_GAN:
            self.optimizer_G = adam.Adam(self.generator.parameters(), lr=lr[0])
            self.optimizer_D = adam.Adam(self.discriminator.parameters(), lr=lr[2])
        if self.enable_PW:
            self.optimizer_PW = adam.Adam(self.generator.parameters(), lr=lr[1])
        if not self.enable_GAN and not self.enable_PW:
            self.optimizer = adam.Adam(self.generator.parameters(), lr=lr[0])


    def init_D_ref(self):
        # clear D_ref data for exery N epochs
        self.D_input_ref = torch.concat((self.x_t, self.y_t, torch.ones(self.x_t.shape[0], 1)), 1)

        # Initial Data
        if hasattr(self, "x0") and hasattr(self, "y0"):
            # print(torch.cat((self.x0, self.y0, torch.ones(self.x0.shape[0], 1)), 1).shape)
            self.D_input_ref = torch.vstack((self.D_input_ref, 
                                            torch.cat((self.x0, self.y0, torch.ones(self.x0.shape[0], 1)), 1)))
        # Boundary Data
        if hasattr(self, "x_lb") and hasattr(self, "u_lb"):
            self.D_input_ref = torch.vstack((self.D_input_ref, 
                                            torch.cat((self.x_lb, self.u_lb, torch.ones(self.x_lb.shape[0], 1)), 1)))
        if hasattr(self, "x_ub") and hasattr(self, "u_ub"):
            self.D_input_ref = torch.vstack((self.D_input_ref, 
                                            torch.cat((self.x_ub, self.u_ub, torch.ones(self.x_ub.shape[0], 1)), 1)))

    def save(self, epoch, n_critic):
        checkpoint = {
            "generator_model_state_dict": self.generator.model.state_dict(),
            "discriminator_model_state_dict": self.discriminator.model.state_dict(),
            "weights": [self.domain_weights]+self.boundary_weights,
            "generator_optimizer_state_dict": self.optimizer_G.state_dict(),
            "discriminator_optimizer_state_dict": self.optimizer_D.state_dict(),
            "pointwise_optimizer_state_dict": self.optimizer_PW.state_dict(),
            "regular_optimizer_state_dict": self.optimizer.state_dict(),
            "activations": {"GAN": self.enable_GAN, "PW": self.enable_PW},
            "epoch": epoch,
            "rho_values": self.rho_values,
            "loss_values": self.loss_values,
            "n_critic": n_critic,
        }
        torch.save(checkpoint, "Saves/"+self.name+"_"+str(epoch)+".pth")
    
    def load(self):
        checkpoint = torch.load("model_checkpoint.pth")
        self.enable_GAN = checkpoint["activations"]["GAN"]
        self.enable_PW = checkpoint["activations"]["PW"]
        self.generator.model.load_state_dict(checkpoint["generator_model_state_dict"])
        if self.enable_GAN:
            self.discriminator.model.load_state_dict(checkpoint["discriminator_model_state_dict"])
            self.optimizer_D.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["generator_optimizer_state_dict"])
        if self.enable_PW:
            self.optimizer_PW.load_state_dict(checkpoint["pointwise_optimizer_state_dict"])
            self.domain_weights = checkpoint["weights"][0]
            self.boundary_weights = checkpoint["weights"][1:]
        if not self.enable_GAN and not self.enable_PW:
            self.optimizer.load_state_dict(checkpoint["regular_optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        n_critic = checkpoint["n_critic"]
        
        epoch_stop = int(input("currently at epoch {epoch}. Train till epoch: "))
        self.train(epoch_stop, epoch, n_critic)
    
    # calculate the function h(x, t) using neural nets
    # NOTE: regard net_uv as baseline  
    def net_y(self, x):
        print(x.shape)
        X = x.transpose(0,1)
        H = (X - self.lb) / (self.ub - self.lb) * 2.0 - 1.0 # normalize to [-1, 1]
        self.H = H.transpose(0,1)
        # NOTE: ????
        y = self.generator.forward(self.H)
        self.y = y
        return y

    # compute the Schrodinger function on the collocation points
    # TODO: adjust according to different equation
    # TODO: pass function as parameter in init configs
    def net_f(self, X):
        try:
            return self._net_f(X)
        except IndexError as e:
            print(f"Caught IndexError: {e}")
            print("This was caught in net_f, so it is likely that your implementation for calculating f is wrong, probably due to a missunderstanding concerning dimensionality.")
            sys.exit(1)

    def forward(self, x):
        y = self.net_y(x)
        f = self.net_f(x)
        return y, f
    
    def beta(self, f_pred, e):
        '''
        A function that differentiates between easy to learn and hard to learn points.
        It is used for the weightupdate. 
        f_u_pred and f_v_pred are the values of the DGL function which should be close to zero
        e: a hyperparameter that is a measure for the exactness to which N should approximate u
        '''
        beta = (torch.sum(f_pred**2, dim=1) <= e).to(torch.float32) - (torch.sum(f_pred**2, dim=1) > e).to(torch.float32)
        beta.requires_grad_(False)
        return beta
    
    def weight_update(self, f_pred):
        # f_pred 
        '''
        This function changes the weights used for loss calculation according to the papers formular. 
        It should be called after each iteration. 
        '''
        boundaries = self.boundary()
        u_pred = self.generator(self.x_f)
        rho_values = []
        e = self.e[0]
        w = self.domain_weights
        rho = torch.sum(w*(self.beta(f_pred, e)==-1.0))
        epsilon = 1e-6 # this is added to rho because rho has a likelyhood (that empirically takes place often) to be 0 or 1, both of which break the algorithm
        # NOTE: it is probably ok, but think about it that this makes it possible that for rho close to 0 the interior of the log below is greater than one, giving a positive alpha which would otherwise be impossible. 
        # NOTE: we think it is ok because this sign is then given into an exponential where a slight negative instead of 0 should not make a difference (?) 
        alpha = self.q[0] * torch.log((1-rho+epsilon)/(rho+epsilon))
        w_new = w*torch.exp(-alpha*self.beta(f_pred, e).to(torch.float32)) / \
            torch.sum(w*torch.exp(-alpha*self.beta(f_pred, e).to(torch.float32))) # the sum sums along the values of w
        w_new.requires_grad_(False)
        w_new.to(torch.float32)
        self.domain_weights = w_new
        # print("____________________")
        # print(torch.sum(f_pred**2, dim=1))
        # print(self.domain_weights)
        # print(self.boundary_weights)
        # print("____________________")
        rho_values.append(rho)
        signifier = (self.beta(f_pred, e)==1)
        if torch.sum(signifier.to(torch.int))>0 and self.enable_augmentation:
                    # print("debug")
            #print("self.D_input_ref.shape", self.D_input_ref.shape)
            #print("self.x_f[signifier].shape", self.x_f[signifier].shape)
                    # print(torch.sum((self.beta(f_update, e)==1.0).to(torch.int)))        
            self.D_input_ref = torch.vstack(\
                (self.D_input_ref, torch.cat((self.x_f[signifier].detach(), \
                                        u_pred[signifier].detach(),\
                                            torch.ones(self.x_f[signifier].shape[0], 1)), 1)))
        for index, w in enumerate(self.boundary_weights): 
            e = self.e[index+1]
            rho = torch.sum(w*(self.beta(boundaries[index], e)==-1.0))
            epsilon = 1e-6 # this is added to rho because rho has a likelyhood (that empirically takes place often) to be 0 or 1, both of which break the algorithm
            # NOTE: it is probably ok, but think about it that this makes it possible that for rho close to 0 the interior of the log below is greater than one, giving a positive alpha which would otherwise be impossible. 
            # NOTE: we think it is ok because this sign is then given into an exponential where a slight negative instead of 0 should not make a difference (?) 
            alpha = self.q[index+1] * torch.log((1-rho+epsilon)/(rho+epsilon))
            w_new = w*torch.exp(-alpha*self.beta(boundaries[index], e).to(torch.float32)) / \
                torch.sum(w*torch.exp(-alpha*self.beta(boundaries[index], e).to(torch.float32))) # the sum sums along the values of w
            w_new.requires_grad_(False)
            w_new.to(torch.float32)
            self.boundary_weights[index] = w_new
            
            rho_values.append(rho)
        
        return torch.tensor(rho_values)

    def loss_T(self):
        '''
        returns the mse loss of samples that have x and y values. These should be saved in x_t, y_t
        '''
        loss = nn.MSELoss()
        
        self.y_t_pred = self.net_y(self.x_t)
        
        return loss(self.y_t_pred, self.y_t)
    
    def loss_plain(self):
        loss = nn.MSELoss()
        
        L_T = self.loss_T()
        
        y_f = self.net_f(self.x_f)
        L = loss(y_f, torch.zeros_like(y_f))
        
        for index, boundary in enumerate(self.boundary()):
            L += self.lambdas[0]*loss(boundary, torch.zeros_like(boundary))
        
        return L # self.lambdas[1] * L_T + L
    
    def loss_G(self):
        ''' 
        input dim for G: 
        (x, t, u, v)
        possible error of dimensionality noted.
        '''
        # TODO: call util.py for point loss
        loss_l1 = nn.L1Loss()
        
        self.y_f_pred = self.net_y(self.x_f)
        
        input_D = torch.concat((self.x_f, self.y_f_pred), 1)
        D_output = self.discriminator.forward(input_D)
        L_D = loss_l1(torch.ones_like(D_output), 
                    D_output)
        # NOTE: 
        L_T = self.loss_T()

        return self.lambdas[1]*L_T + L_D

        # TODO : implement boundary data and boundary condition for GAN: ? Should be in pointwise loss where it is, right?
        # TODO: normalize the loss/dynamic ratio of importance between 2 loss components
        # NOTE: Q: does it differ if optimizer not take step for loss(GAN) and loss(eq) separately?
    
    def PCS(self, r_k):
        "physics consistency scores."
        lambd_val = 10 # hyperparam for tuning
        return torch.exp(-lambd_val*torch.sum(torch.abs(r_k), dim = 1)).unsqueeze(1)
    

    def loss_PW(self):
        
        f_loss = weighted_MSELoss()
        L_PW = f_loss(self.f_pred, torch.zeros_like(self.f_pred), self.domain_weights.to(torch.float32))
        # print("!!!!!! L_PW -> ", L_PW)
        for index, boundary in enumerate(self.boundary()):
            L = self.lambdas[0]*f_loss(boundary, torch.zeros_like(boundary), self.boundary_weights[index].to(torch.float32))
            # print("!!!!!! L_B -> ", L)
            L_PW += L
        return L_PW 
    
    def loss_D(self):
        '''
        input dim for D: 
        (x, y)
        possible error of dimensionality noted -> transpose
        '''
        #TODO 
        loss = nn.L1Loss()
        self.y_t_pred = self.net_y(self.x_t)
        discriminator_T = self.discriminator.forward(
            torch.concat((self.x_t, self.y_t), 1)
            )
        
        discriminator_L = self.discriminator.forward(
            torch.concat((self.x_t, self.y_t_pred), 1)
            )
        loss_D = loss(discriminator_L, torch.zeros_like(discriminator_L)) + \
                loss(discriminator_T, torch.ones_like(discriminator_T))
        return loss_D

    def loss_G_PI(self):
        ''' 
        input dim for G: 
        (x, t, u, v)
        possible error of dimensionality noted.
        '''
        # TODO: call util.py for point loss

        # NOTE: 
        # L_T = self.loss_T()
        loss_l1 = nn.L1Loss()
        
        self.y_f_pred = self.net_y(self.x_f)
        #print(self.y_f_pred.shape)
        self.f_pred = self.net_f(self.x_f)
        #print(self.f_pred.shape)
        input_D = torch.concat((self.x_f, self.y_f_pred, self.PCS(self.f_pred)), 1)
        D_output = self.discriminator.forward(input_D)
        L_D = loss_l1(torch.ones_like(D_output), 
                    D_output)
        # NOTE: 
        L_T = self.loss_T()

        return self.lambdas[1]*L_T + L_D
        # return self.lambdas[1]*L_T + L_D

        # TODO : implement boundary data and boundary condition for GAN: ? Should be in pointwise loss where it is, right?
        # TODO: normalize the loss/dynamic ratio of importance between 2 loss components
        # NOTE: Q: does it differ if optimizer not take step for loss(GAN) and loss(eq) separately?

    def loss_D_PI(self):
        """
        incorporating physics consistency scores.
        \cite{PID-GAN}
        """
        loss = NLLLoss() # NOTE to be fixed as -log
        self.y_f_pred = self.net_y(self.x_f)
        self.ref_sol = self.net_y(self.G_input_ref)
        self.ref_sol_f = self.net_f(self.G_input_ref)
        # print(self.PCS(self.f_pred))
        self.input_D_domain = torch.concat((self.x_f, self.y_f_pred, self.PCS(self.f_pred)), 1)
        self.input_D_boundary = torch.concat((self.G_input_ref, self.ref_sol, self.PCS(self.ref_sol_f)), 1)
        self.D_output_domain = self.discriminator.forward(self.input_D_domain)
        # print(self.D_output_domain.shape)
        # print(torch.sum(self.D_output_domain<0))
        self.D_output_boundary = self.discriminator.forward(self.input_D_boundary)
        self.D_output_ref = self.discriminator.forward(self.D_input_ref)
        loss_input = torch.vstack((self.D_output_domain,self.D_output_boundary))
        # print(torch.vstack((self.D_output_domain,self.D_output_boundary)).shape)
        loss_D_PI = loss(loss_input, torch.zeros_like(loss_input)) +\
                loss(torch.ones_like(self.D_output_ref), self.D_output_ref)#
        loss_temp = nn.MSELoss()
        
        #print(loss_temp(loss_input, torch.zeros_like(loss_input)))
        #print(loss_temp(torch.ones_like(self.D_output_ref), self.D_output_ref))

        return loss_D_PI


    def train(self, epochs, grid, X_star, y_star, start_epoch=0, n_critic = 1):
        """X, T: extra grid data for ground truth solution. passed for plotting. """
        self.no_enable = not self.enable_GAN and not self.enable_PW
        if len(grid) == 2:
            
            X, T = grid
        elif len(grid) == 3:
            X, T, _ = grid #NOTE: visualisation will be less meaningfull
        else:
            print(f"grid has unexpected length {len(grid)}. Expect errors")
        if type(y_star)==list:
            print("Using first component of y_star for error")
            y_star = y_star[0]
        
        # Training
        self.init_D_ref()
        for epoch in tqdm(range(start_epoch, epochs)):
            # TODO done?
            # self.y0_pred = self.net_y(self.x0)
            self.f_pred = self.net_f(self.x_f)
            if self.enable_GAN:
                self.optimizer_D.zero_grad()
                if self.enable_PID:
                    loss_Discr = self.loss_D_PI()
                else:
                    loss_Discr = self.loss_D()
                loss_Discr.backward(retain_graph=True) # retain_graph: tp release tensor for future use
            if epoch % n_critic == 0:
                if self.no_enable:
                    self.optimizer.zero_grad()
                    loss_G = self.loss_plain()
                if self.enable_GAN:
                    self.optimizer_G.zero_grad()
                    if not self.enable_PID:
                        loss_G = self.loss_G() # + self.loss_plain()
                    else:
                        loss_G = self.loss_G_PI()
                    # loss_plain = self.loss_plain()
                if self.no_enable or self.enable_GAN:
                    loss_G.backward(retain_graph=True)
                    # loss_plain.backward(retain_graph=True)
                if self.enable_PW:
                    self.optimizer_PW.zero_grad()
                    loss_PW = self.loss_PW()
                    # print("------------------->>>")
                    # print("loss diff", loss_PW - self.loss_plain())
                    # print("loss", loss_PW)
                    # print("loss", self.loss_plain())
                    # print("test")
                    loss_PW.backward(retain_graph=True)
                if self.no_enable:
                    self.optimizer.step()
                if self.enable_GAN:
                    # self.optimizer.step()
                    self.optimizer_G.step()
                # weight updates
                if self.enable_PW:
                    self.optimizer_PW.step()
                    rho = self.weight_update(self.f_pred)
                    self.rho_values.append(rho)
                    self.loss_values["Pointwise"].append(loss_PW.detach().numpy())
                if self.no_enable or self.enable_GAN:
                    self.loss_values["Generator"].append(loss_G.detach().numpy())   
            if self.enable_GAN:  
                self.optimizer_D.step()
                self.loss_values["Discriminator"].append(loss_Discr.detach().numpy())
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}')
                if self.no_enable or self.enable_GAN:
                    print(f'Loss_G: {loss_G.item()}')
                if self.enable_GAN and self.enable_augmentation:
                    self.init_D_ref()
                    print(f'Loss_D: {loss_Discr.item()}')
                if self.enable_PW:
                    print(f"rho: {rho}")
                    print(f"PW loss: {loss_PW}")
                print(f"Exact training loss: {self.loss_T()}, boundary loss: {list(map(lambda x: float(torch.sum(x**2).detach().numpy()),self.boundary()))}")
                y_pred, f_pred = self.predict(torch.tensor(X_star, requires_grad=True))
                y_pred = y_pred[:,0:1] # in case of multidim y
                print("normalized solution Error: ", np.linalg.norm(y_star-y_pred,2)/np.linalg.norm(y_star,2))
            if epoch % 100 == 0:
                if X_star.shape[1] == 2:#TODO dimensionality
                    plot_with_ground_truth(y_pred, X_star, X, T, y_star , ground_truth_ref=False, ground_truth_refpts=[], filename = self.name+".png") # TODO y_star dimensionality
                    
                # Error
                # print("y Error: ", np.linalg.norm(y_star-y_pred,2)/np.linalg.norm(y_star,2))
                print("value of f: ",np.sum(f_pred**2))
                if self.create_saves:
                    self.save(epoch, n_critic)
            if self.enable_PW:
                if torch.sum(rho)<1e-4 and epoch>10: # summ because there are multiple rho for domain and boundary condition.
                    print("early stopping")
                    if self.create_saves:
                        self.save(epoch, n_critic)
                    break
                
    def predict(self, X_star):
        '''
        y_star = self.generator.forward(X_star)
        f_star = self.net_f(X_star) '''
        y_star, f_star = self.forward(X_star)
        return y_star.detach().numpy(), f_star.detach().numpy()



    
    
    