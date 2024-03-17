# import modules
import numpy as np

from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from scipy.integrate import odeint

import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from torch.optim import adam
# from utils.plot.py import 
torch.set_default_dtype(torch.float32)
# set random seeds for reproducability
np.random.seed(42)
torch.manual_seed(42)
torch._C._set_mkldnn_enabled(False)
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
    def __init__(self, layers_G):
        super(Generator, self).__init__()
        self.layers = layers_G
        self.model = nn.Sequential()
        # TODO: baseline structure to be altered 
        for l in range(0, len(self.layers) - 2):
            self.model.add_module("linear" + str(l), nn.Linear(self.layers[l], self.layers[l+1]))
            self.model.add_module("tanh" + str(l), nn.Tanh())
        self.model.add_module("linear" + str(len(self.layers) - 2), nn.Linear(self.layers[-2], self.layers[-1]))
        # self.model = self.model.double()
        
    def forward(self, x):
        return self.model(x)
# TODO: try a different variant of GAN 
    #link: https://github.com/Mohanned-Elkholy/ResNet-GAN  
class Discriminator(nn.Module):
    def __init__(self, layers_D):
        super(Discriminator, self).__init__()
        self.layers = layers_D
        # NOTE: discriminator input dim = dim(x) * dim(G(x))
        self.model = nn.Sequential()
        # TODO: baseline structure to be altered 
        for l in range(0, len(self.layers) - 1):
            self.model.add_module("linear" + str(l), nn.Linear(self.layers[l], self.layers[l+1]))
            self.model.add_module("tanh" + str(l), nn.Tanh())
        self.model.add_module("sigmoid" + str(len(self.layers) - 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)
    
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
        return 
    
class Generator_RNN(nn.Module):
    def __init__(self, layers_G):
        super(Generator_RNN, self).__init__()
        self.layers = layers_G
        self.num_layers = 5
        # input_size, hidden_size, num_layers, output_size
        self.rnn = nn.RNN(self.layers[0],self.layers[1], self.num_layers, batch_first= True)
        self.linear = nn.Linear(self.layers[1], self.layers[-1])
    def forward(self, x):
        # print(x.unsqueeze(1).shape)
        out, _ = self.rnn(x.unsqueeze(1))#, (h0, c0))  # Forward pass through LSTM layer
        out = self.linear(out[:, -1, :]) 
        return out

class Generator_ResNet(nn.Module):
    def __init__(self, layers_G):
        return

class weighted_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets,weights):
        return torch.dot(weights.flatten(),\
                          torch.square(inputs-targets).flatten())
    
class PINN_GAN_burgers(nn.Module):
    def __init__(self, X_exact, u_exact, nu, X_f, X0, X_lb, X_ub, boundary, num_boundary_conditions, layers_G, layers_D):
        """
        X0: T=0, initial condition, randomly drawn from the domain
        Y0: T=0, initial condition, given (u0, v0)
        X_f: the collocation points with time, size (Nf, dim(X)+1)
        X_lb: the lower boundary, size (N_b, 2)
        X_ub: the upper boundary, size (N_b, 2)
        boundary: the lower and upper boundary, size (2, 2) : [(x_min, t_min), (x_max, t_max)]
        layers: the number of neurons in each layer (_D for discriminator, _G for generator)
        """
        super(PINN_GAN_burgers, self).__init__()

        # Hyperparameters
        self.q = 1e-4
        self.e_boundary = 2e-2  # Hyperparameter for PW update
        self.e_interior = 5e-4
        
        self.nu = nu # PDE parameter
        
        # exact solution 
        self.X_exact = torch.tensor(X_exact, requires_grad=True, dtype = torch.float32)
        self.x_exact = torch.tensor(X_exact[:, :-1], requires_grad=True, dtype = torch.float32)
        self.t_exact = torch.tensor(X_exact[:, -1:], requires_grad=True, dtype = torch.float32)
        self.u_exact = torch.tensor(u_exact).to(torch.float32)
        # self.u_exact = torch.unsqueeze(self.u_exact, 1)
        self.u_exact.requires_grad_(True)

        # initial data
        self.X0 = torch.tensor(X0, requires_grad=True, dtype = torch.float32)
        self.x0 = torch.tensor(X0[:, :-1], requires_grad=True, dtype = torch.float32)
        self.t0 = torch.tensor(X0[:, -1:], requires_grad=True, dtype = torch.float32) 
        self.u0 = -torch.sin(torch.pi*self.x0)
        
        # self.u0 = torch.tensor()
        # formulate u0 only in loss function
        
        # Boundary Data
        self.X_lb = torch.tensor(X_lb, requires_grad=True, dtype = torch.float32)
        self.x_lb = torch.tensor(X_lb[:, :-1], requires_grad=True, dtype = torch.float32)
        self.t_lb = torch.tensor(X_lb[:, -1:], requires_grad=True, dtype = torch.float32)
        self.u_lb = torch.zeros_like(self.x_lb)

        self.X_ub = torch.tensor(X_ub, requires_grad=True, dtype = torch.float32)
        self.x_ub = torch.tensor(X_ub[:, :-1], requires_grad=True, dtype = torch.float32)
        self.t_ub = torch.tensor(X_ub[:, -1:], requires_grad=True, dtype = torch.float32)
        self.u_ub = torch.zeros_like(self.x_ub)
        
        # Collocation Points
        self.X_f = torch.tensor(X_f, requires_grad=True, dtype = torch.float32)
        self.x_f = torch.tensor(X_f[:, :-1], requires_grad=True, dtype = torch.float32)
        self.t_f = torch.tensor(X_f[:, -1:], requires_grad=True, dtype = torch.float32)
        #
        # basic formulation: X = [x t]

        # Bounds
        self.lb = torch.tensor(boundary[:, 0:1])
        self.ub = torch.tensor(boundary[:, 1:2])

        # print(self.u_exact.shape)
        # print(self.X_exact.shape)
        # print(self.X_lb.shape)
        # print(self.u_lb.shape)
        # print(self.X0.shape)
        # print(self.u0.shape)
        # input for discriminator
        self.input_D = torch.vstack((torch.cat((self.X_exact, self.u_exact), 1), \
                               torch.cat((self.X_lb, self.u_lb), 1),\
                               torch.cat((self.X_ub, self.u_ub), 1),
                               torch.cat((self.X0, self.u0), 1)))
        
        # weights for the point weighting algorithm
        self.n_boundary_conditions = num_boundary_conditions # NOTE: how to generalize?
        self.number_collocation_points = self.x_f.shape[0]
        self.number_boundary_points = self.x_lb.shape[0]
        self.domain_weights = torch.full((self.number_collocation_points,), 1/self.number_collocation_points, dtype = torch.float32, requires_grad=False)
        self.boundary_weights = [torch.full((self.number_boundary_points,), 1/self.number_boundary_points, requires_grad=False)]*self.n_boundary_conditions
        
        
        # Sizes
        self.layers_D = layers_D
        self.layers_G = layers_G
        
        self.generator = Generator_RNN(self.layers_G)
        self.discriminator = Discriminator(self.layers_D)
        

    # calculate the function h(x, t) using neurl nets
    # NOTE: regard net_uv as baseline  
    def net_uv(self, x, t):
        X = torch.cat([x, t], dim=1)
        # H = (X - self.lb) / (self.ub - self.lb) * 2.0 - 1.0 # normalize to [-1, 1]
        
        # self.X = X.transpose(0,1)
        # NOTE: ????
        u = self.generator.forward(X)

        self.u = torch.unsqueeze(u, 1)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0] # create_graph=True
        return u, u_x

    # compute the Schrodinger function on the collocation points
    # TODO: adjust according to different equation
    # TODO: pass function as parameter in init configs
    def net_f_uv(self, x, t):
        u, u_x = self.net_uv(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        f_u = u_t + u*u_x - self.nu*u_xx
        return f_u.to(torch.float32)

    def forward(self, x, t):
        u, _ = self.net_uv(x, t)
        f_u = self.net_f_uv(x, t)
        return u, f_u
    
    def beta(self, pred_val, e):
        '''
        A function that differentiates between easy to learn and hard to learn points.
        It is used for the weightupdate. 
        f_u_pred: PDE equation function which should be close to zero
        e: a hyperparameter that is a measure for the exactness to which N should approximate u
        '''
        
        beta = (torch.norm(pred_val, dim=1) <= e).to(torch.float32) - (torch.norm(pred_val, dim=1) > e).to(torch.float32)
        beta.requires_grad_(False)
        # print(beta.shape)
        return torch.unsqueeze(beta, 1)
    
    def weight_update(self):
        # f_pred 
        '''
        This function changes the weights used for loss calculation according to the papers formular. 
        It should be called after each iteration. 
        NOT independent of the PDE given.
        '''
        # NOTE: the error wrt observation is also here
        
        boundary_loss_list = [
            self.u_exact_pred-self.u_exact,
            self.u0_pred-self.u0,
            self.u_lb_pred-self.u_lb,
            self.u_ub_pred-self.u_ub
        ]
    
        for index, w in enumerate([self.domain_weights] + self.boundary_weights): # concatenate lists with domain and boundary weights
            # print("e: ", e)
            if index == 0: # inside domain
                f_update = self.f_u_pred
                e = self.e_interior
            else:
                f_update = boundary_loss_list[index-1]
                e = self.e_boundary
            
            rho = torch.sum(w*((self.beta(f_update, e)==-1.0).transpose(0,1)))

            epsilon = 1e-8 # this is added to rho because rho has a likelihood (that empirically takes place often) to be 0 or 1, both of which break the algorithm
            # NOTE: it is probably ok, but think about it that this makes it possible that for rho close to 0 the interior of the log below is greater than one, giving a positive alpha which would otherwise be impossible. 
            # NOTE: we think it is ok because this sign is then given into an exponential where a slight negative instead of 0 should not make a difference (?) 
            alpha = self.q * torch.log((1-rho+epsilon)/(rho+epsilon))
            # print(self.beta(f_update, e).shape)
            # print("alpha: ", alpha)
            w_new = w*torch.exp(-alpha*self.beta(f_update, e).to(torch.float32)).transpose(0,1) / \
                torch.sum(w*torch.exp(-alpha*self.beta(f_update, e).to(torch.float32)).transpose(0,1)) # the sum sums along the values of w
            w_new.requires_grad_(False)
            w_new.to(torch.float32)
            # print(w_new.shape)
            if index == 0:
                self.domain_weights = w_new
            else:
                self.boundary_weights[index-1] = w_new

    def loss_G(self):
        ''' 
        input dim for G: 
        (x, t, u, v)
        possible error of dimensionality noted.
        '''
        # TODO: call util.py for point loss
        loss = nn.MSELoss()
        loss_l1 = nn.L1Loss()
        self.u_lb_pred, self.u_x_lb_pred = self.net_uv(self.x_lb, self.t_lb)
        self.u_ub_pred, self.u_x_ub_pred = self.net_uv(self.x_ub, self.t_ub)
        self.u_exact_pred, _ = self.net_uv(self.x_exact, self.t_exact)
        self.f_u_pred = self.net_f_uv(self.x_f, self.t_f)
        self.u0_pred, _ = self.net_uv(self.x0, self.t0)
        
        # print("loss stuff")
        # print(self.f_u_pred)
        # print("____")
        # print(self.beta(self.f_u_pred, self.f_v_pred, 0.001))
        # self.weight_update(self.f_u_pred, self.f_v_pred, 0.001)
        # print("weights")
        # print(self.domain_weights)
        
        # instead of taking only initial observation from exact data, 
        # take observations that are scattered across the whole solution frame
        # since most initial/boundary data are formulated in an accessible way
        
        # initial condition + boundary condition

        # write L_PW outside
        # MSE = loss(self.u_exact_pred, self.u_exact)
        f_loss = nn.MSELoss()
        L_T = f_loss(self.u_exact_pred, self.u_exact) + \
            f_loss(self.u0_pred, self.u0) + \
            f_loss(self.u_lb_pred, self.u_lb) + \
            f_loss(self.u_ub_pred, self.u_ub)
        #f_loss(self.u0_pred, self.u0) + \
        input_D_G = torch.concat((self.X_exact, self.u_exact_pred), 1)
        D_output = self.discriminator(input_D_G)
        L_D = loss_l1(torch.zeros_like(D_output), 
                     D_output)
        # D_output = self.discriminator.model(self.input_D)
        # L_D = loss_l1(torch.ones_like(D_output), 
        #             D_output)
        # NOTE: dimensionality
        
        return L_T + L_D

        # TODO : implement boundary data and boundary condition for GAN
        # TODO: normalize the loss/dynamic ratio of importance between 2 loss components

    
    def loss_PW(self):
        self.u_lb_pred, self.u_x_lb_pred = self.net_uv(self.x_lb, self.t_lb)
        self.u_ub_pred, self.u_x_ub_pred = self.net_uv(self.x_ub, self.t_ub)
        self.u_exact_pred, _ = self.net_uv(self.x_exact, self.t_exact)
        self.f_u_pred = self.net_f_uv(self.x_f, self.t_f)
        self.u0_pred, _ = self.net_uv(self.x0, self.t0)

        f_loss = nn.MSELoss()

        L_PW = f_loss(self.f_u_pred, torch.zeros_like(self.f_u_pred)) + \
        f_loss(self.u_exact_pred, self.u_exact) + \
        f_loss(self.u0_pred, self.u0) + \
        f_loss(self.u_lb_pred, self.u_lb) + \
        f_loss(self.u_ub_pred, self.u_ub)
        # f_loss = weighted_MSELoss()
        # L_PW = f_loss(self.f_u_pred, torch.zeros_like(self.f_u_pred), self.domain_weights.to(torch.float32)) + \
        #         f_loss(self.u_exact_pred, self.u_exact, self.boundary_weights[0].to(torch.float32)) + \
        #         f_loss(self.u0_pred,self.u0, self.boundary_weights[1].to(torch.float32)) + \
        #         f_loss(self.u_lb_pred, self.u_lb, self.boundary_weights[0].to(torch.float32)) + \
        #         f_loss(self.u_ub_pred, self.u_ub, self.boundary_weights[1].to(torch.float32))

        # b_loss = torch.inner(self.boundary_weights, 
        # NOTE: leaving boundary conditions blank
        # TODO: boundary conditions&implement
        return L_PW # + b_loss
    
    def loss_D(self):
        '''
        input dim for D: 
        (x, t, u, v)
        possible error of dimensionality noted.
        '''
            # TODO: hstack boundary/initial with exact
        loss = nn.L1Loss()
        discriminator_T = self.discriminator.model(
            self.input_D
            )

        discriminator_L = self.discriminator.model(
            torch.concat((self.X_exact, self.u_exact_pred), 1)
            )
        loss_D = loss(discriminator_L, torch.zeros_like(discriminator_L)) + \
                 loss(discriminator_T, torch.ones_like(discriminator_T))
        return loss_D


    def train(self, X_star, u_star, epochs = 1e-4, lr_G = 1e-2, lr_D = 5e-3, lr_decay = 0.1, n_critic = 1):
        # Optimizer

        # LSTM dynamic training rate: starting 5e-2 lr_decay = 0.7
        optimizer_G = adam.Adam(self.generator.parameters(), lr=lr_G)
        optimizer_D = adam.Adam(self.discriminator.parameters(), lr=lr_D)
        optimizer_PW = adam.Adam(self.generator.parameters(), lr=lr_G)
        # Training
        loss_history = {"epoch":[], "loss_G":[], "loss_D": [], "loss_PW":[], "loss_u": []}
        
        for epoch in tqdm(range(epochs)):
            # TODO done?
            self.u0_pred, _  = self.net_uv(self.x0, self.t0)
            self.f_u_pred = self.net_f_uv(self.x_f, self.t_f)
            self.u_exact_pred, _ = self.net_uv(self.x_exact, self.t_exact)
            optimizer_D.zero_grad()
            loss_Discr = self.loss_D()
            # loss_Discr.backward(retain_graph=True) # retain_graph: tp release tensor for future use
            #if epoch % n_critic == 0:
            optimizer_G.zero_grad()
            optimizer_PW.zero_grad()
            loss_G = self.loss_G()
            loss_G.backward(retain_graph=True)
            # Update PW loss
            self.weight_update()
            
            loss_PW = self.loss_PW()
            loss_PW.backward(retain_graph=True)
            optimizer_D.step()
            optimizer_G.step()
            optimizer_PW.step()
            # TODO: dynamic learning rate:
            # record the last 2 updates of the u loss. If no significant improvement happened, 
            # decrease the learning rate by a parameter lambda. 

            # weight updates
            # record the loss and plot the loss

            loss_history["epoch"].append(epoch)
            loss_history["loss_G"].append(loss_G.item())
            loss_history["loss_D"].append(loss_Discr.item())
            loss_history["loss_PW"].append(loss_PW.item())
            if epoch % 10 == 0:
                print('Epoch: %d, Loss_G: %.3e, Loss_D: %.3e, Loss_PW: %.3e' % (epoch, loss_G.item(), loss_Discr.item(), loss_PW.item()))
                u_pred, _ = self.predict(X_star)
                loss_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
                errors = {'u': loss_u}
                print('Errors: ', errors)
                if epoch > 10:
                    if loss_u-loss_history["loss_u"][-1]>1e-3:
                        optimizer_G = adam.Adam(self.generator.parameters(), lr=lr_G*lr_decay)
                        optimizer_PW = adam.Adam(self.generator.parameters(), lr=lr_G*lr_decay)
            loss_history["loss_u"].append(loss_u)
        with open('loss_history_burgers.pkl', 'wb') as f:
                pickle.dump(loss_history, f)


    def predict(self, X_star):
        X_star = torch.tensor(X_star, dtype=torch.float32, requires_grad=True)
        u_star = self.generator.forward(X_star)
        f_u_star = self.net_f_uv(X_star[:,0:1], X_star[:,1:2])
        return u_star.detach().numpy(),  f_u_star.detach().numpy()


    