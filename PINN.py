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
        return self.model(x)
        try:
            print(8.41)
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
    
class PINN_GAN(nn.Module):
    def __init__(self, X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G : list=[], layers_D: list=[], enable_GAN = True, enable_PW = True, dynamic_lr = False, model_name: str="", lr: tuple=(1e-3, 5e-3), lambdas: tuple = (1,1)):
        """
        
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
        lr: the learning rate as a tupel
        """
        super().__init__()

        # Hyperparameters
        self.q = [
            1e-4,
            1e-4     
        ]
        self.lambdas = lambdas
        self.e = [2e-2, 5e-4]  # Hyperparameter for PW update
        
        # parameters for saving
        self.create_saves = model_name!=""
        self.name = model_name
        
        # Arrays for interesting values
        self.rho_values = []
        self.loss_values = {"Generator": [], "Discriminator": [], "Pointwise": []}
        
        # Initial Data
        if X0 is not None:
            self.x0 = torch.tensor(X0, requires_grad=True)
        if Y0 is not None:
            self.y0 = torch.tensor(Y0)
        
        # Boundary Data
        print(X_lb, X_ub)
        if X_lb is not None:
            self.x_lb = torch.tensor(X_lb, requires_grad=True)
        if X_ub is not None:
            self.x_ub = torch.tensor(X_ub, requires_grad=True)
        
        # Collocation Points
        self.x_f = torch.tensor(X_f, requires_grad=True)
        
        # training points that have values
        self.x_t = torch.tensor(X_t)
        self.y_t = torch.tensor(Y_t)
        
        # Bounds
        boundary = torch.tensor(boundary)
        boundary = boundary.transpose(0,1)
        self.lb = torch.tensor(boundary[:, 0:1])
        self.ub = torch.tensor(boundary[:, 1:2])
        
        # Sizes
        self.layers_D = layers_D
        self.layers_G = layers_G
        
        self.generator = Generator(self.layers_G, info_for_error=(self.x_f.shape[1],self.y_t.shape[1]))
        self.discriminator = Discriminator(self.layers_D, info_for_error=(self.y_t.shape[1]+self.x_f.shape[1],1))
        
        # Optimizer
        self.optimizer_G = adam.Adam(self.generator.parameters(), lr=lr[0])
        self.optimizer_D = adam.Adam(self.discriminator.parameters(), lr=lr[1])
        self.optimizer_PW = adam.Adam(self.generator.parameters(), lr=lr[0])
        
        # options
        self.enable_GAN = enable_GAN
        self.enable_PW = enable_PW
        self.dynamic_lr = dynamic_lr
    
    def save(self, epoch, n_critic):
        checkpoint = {
            "generator_model_state_dict": self.generator.model.state_dict(),
            "discriminator_model_state_dict": self.discriminator.model.state_dict(),
            "weights": [self.domain_weights]+self.boundary_weights,
            "generator_optimizer_state_dict": self.optimizer_G.state_dict(),
            "discriminator_optimizer_state_dict": self.optimizer_D.state_dict(),
            "pointwise_optimizer_state_dict": self.optimizer_PW.state_dict(),
            "epoch": epoch,
            "rho_values": self.rho_values,
            "loss_values": self.loss_values,
            "n_critic": n_critic,
        }
        torch.save(checkpoint, "Saves/"+self.name+"_"+str(epoch)+".pth")
    
    def load(self):
        checkpoint = torch.load("model_checkpoint.pth")
        self.generator.model.load_state_dict(checkpoint["generator_model_state_dict"])
        self.discriminator.model.load_state_dict(checkpoint["discriminator_model_state_dict"])
        self.optimizer_D.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["generator_optimizer_state_dict"])
        self.optimizer_PW.load_state_dict(checkpoint["pointwise_optimizer_state_dict"])
        self.domain_weights = checkpoint["weights"][0]
        self.boundary_weights = checkpoint["weights"][1:]
        epoch = checkpoint["epoch"]
        n_critic = checkpoint["n_critic"]
        
        epoch_stop = int(input("currently at epoch {epoch}. Train till epoch: "))
        self.train(epoch_stop, epoch, n_critic)
    
    # calculate the function h(x, t) using neural nets
    # NOTE: regard net_uv as baseline  
    def net_y(self, x):
        print(8.1)
        X = x.transpose(0,1)
        print(8.2)
        H = (X - self.lb) / (self.ub - self.lb) * 2.0 - 1.0 # normalize to [-1, 1]
        print(8.3)
        self.H = H.transpose(0,1)
        print(8.4)
        # NOTE: ????
        y = self.generator.forward(self.H)
        print(8.5)
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
        print(7.111)
        y = self.net_y(x)
        print(7.112)
        f = self.net_f(x)
        print(7.113)
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
        
        rho_values.append(rho)
        
    
        for index, w in enumerate(self.boundary_weights):
            e = self.e[-1]
            rho = torch.sum(w*(self.beta(boundaries[index], e)==-1.0))
            epsilon = 1e-6 # this is added to rho because rho has a likelyhood (that empirically takes place often) to be 0 or 1, both of which break the algorithm
            # NOTE: it is probably ok, but think about it that this makes it possible that for rho close to 0 the interior of the log below is greater than one, giving a positive alpha which would otherwise be impossible. 
            # NOTE: we think it is ok because this sign is then given into an exponential where a slight negative instead of 0 should not make a difference (?) 
            alpha = self.q[-1] * torch.log((1-rho+epsilon)/(rho+epsilon))
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
        D_input = self.discriminator.forward(input_D)
        L_D = loss_l1(torch.zeros_like(D_input), 
                    D_input)
        # NOTE: 
        L_T = self.loss_T()

        return self.lambdas[1]*L_T + L_D

        # TODO : implement boundary data and boundary condition for GAN: ? Should be in pointwise loss where it is, right?
        # TODO: normalize the loss/dynamic ratio of importance between 2 loss components
        # NOTE: Q: does it differ if optimizer not take step for loss(GAN) and loss(eq) separately?

    def loss_PW(self):
        
        f_loss = weighted_MSELoss()
        L_PW = f_loss(self.f_pred, torch.zeros_like(self.f_pred), self.domain_weights.to(torch.float32))
        for index, boundary in enumerate(self.boundary()):
            L_PW += self.lambdas[0]*f_loss(boundary, torch.zeros_like(boundary), self.boundary_weights[index].to(torch.float32))
        # b_loss = torch.inner(self.boundary_weights, 
        # NOTE: leaving boundary conditions blank
        # TODO: boundary conditions&implement
        return L_PW # + b_loss
    
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

    def train(self, epochs, grid, X_star, y_star, start_epoch=0, n_critic = 2):
        """X, T: extra grid data for ground truth solution. passed for plotting. """
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

        for epoch in tqdm(range(start_epoch, epochs)):
            # TODO done?
            # self.y0_pred = self.net_y(self.x0)
            self.f_pred = self.net_f(self.x_f)
            if self.enable_GAN == True:
                print(1)
                self.optimizer_D.zero_grad()
                loss_Discr = self.loss_D()
                loss_Discr.backward(retain_graph=True) # retain_graph: tp release tensor for future use
            if epoch % n_critic == 0:
                self.optimizer_G.zero_grad()
                print(2)
                
                loss_G = self.loss_G()
                loss_G.backward(retain_graph=True)
                if self.enable_PW == True:
                    print(3)
                    self.optimizer_PW.zero_grad()
                    loss_PW = self.loss_PW()
                    loss_PW.backward(retain_graph=True)
                print(4)
                self.optimizer_G.step()
                # weight updates
                if self.enable_PW == True:
                    print(5)
                    self.optimizer_PW.step()
                    rho = self.weight_update(self.f_pred)
                    self.rho_values.append(rho)
                    self.loss_values["Pointwise"].append(loss_PW.detach().numpy())
                self.loss_values["Generator"].append(loss_G.detach().numpy())   
            if self.enable_GAN == True:  
                print(6)  
                self.optimizer_D.step()
                self.loss_values["Discriminator"].append(loss_Discr.detach().numpy())
                print(7)
  
            if epoch % 100 == 0:
                print('Epoch: %d, Loss_G: %.3e, Loss_D: %.3e' % (epoch, loss_G.item(), loss_Discr.item()))
                print(7.1)
                y_pred, f_pred = self.predict(torch.tensor(X_star, requires_grad=True))
                print(7.2)
                y_pred = y_pred[:,0:1] # in case of multidim y
                print(7.3)
                
                 #TODO dimensionality
                print(7.4)
                plot_with_ground_truth(y_pred, X_star, X, T, y_star , ground_truth_ref=False, ground_truth_refpts=[], filename = self.name+".png") # TODO y_star dimensionality
                print(7.5)
                # Error
                print("y Error: ", np.linalg.norm(y_star-y_pred,2)/np.linalg.norm(y_star,2))
                    
                print("value of f: ",np.sum(f_pred**2))
                if self.create_saves:
                    self.save(epoch, n_critic)
            print(8)
            if torch.sum(rho)<1e-4 and epoch>10: # summ because there are multiple rho for domain and boundary condition.
                if self.create_saves:
                    self.save(epoch, n_critic)
                break
            print(9)
                
    def predict(self, X_star):
        '''
        y_star = self.generator.forward(X_star)
        f_star = self.net_f(X_star) '''
        print(7.11)
        y_star, f_star = self.forward(X_star)
        print(7.12)
        return y_star.detach().numpy(), f_star.detach().numpy()



    
    
    