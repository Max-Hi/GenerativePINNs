# import modules
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import adam


device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
print(f"Using {device} device")


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
        self.model = self.model.double()

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
        self.model = self.model.double() 


class PINN_GAN(nn.Module):
    def __init__(self, X0, Y0, X_f, X_lb, X_ub, boundary, layers_G, layers_D):
        """
        X0: T=0, initial condition, randomly drawn from the domain
        Y0: T=0, initial condition, given (u0, v0)
        X_f: the collocation points with time, size (Nf, 2)
        X_lb: the lower boundary, size (N_b, 2)
        X_ub: the upper boundary, size (N_b, 2)
        boundary: the lower and upper boundary, size (2, 2) : [(x_min, t_min), (x_max, t_max)]
        layers: the number of neurons in each layer (_D for discriminator, _G for generator)
        """
        super(PINN_GAN, self).__init__()

        print("x0")
        print(X0, X0.shape)
        print("y0")
        print(Y0, Y0.shape)
        
        # Initial Data
        self.x0 = torch.tensor(X0[:, 0:1], requires_grad=True)
        self.t0 = torch.tensor(X0[:, 1:2], requires_grad=True)
        
        self.u0 = torch.tensor(Y0[:, 0:1])
        self.v0 = torch.tensor(Y0[:, 1:2])
        
        # Boundary Data
        self.x_lb = torch.tensor(X_lb[:, 0:1], requires_grad=True)
        self.t_lb = torch.tensor(X_lb[:, 1:2], requires_grad=True)
        self.x_ub = torch.tensor(X_ub[:, 0:1], requires_grad=True)
        self.t_ub = torch.tensor(X_ub[:, 1:2], requires_grad=True)
        
        # Collocation Points
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True)
        
        # Bounds
        self.lb = torch.tensor(boundary[:, 0:1])
        self.ub = torch.tensor(boundary[:, 1:2])

        # weights for the point weigthing algorithm
        n_boundary_conditions = 2 # EDIT manually
        M = 50 # EDIT manually the numnber of collocation points
        self.domain_weights = torch.full((M,), 1/M)
        self.boundary_weights = [torch.full((M,), 1/M)]*n_boundary_conditions
        
        # Sizes
        self.layers_D = layers_D
        self.layers_G = layers_G
        
        self.generator = Generator(self.layers_G)
        self.discriminator = Discriminator(self.layers_D)
                  

    # calculate the function h(x, t) using neural nets
    # NOTE: regard net_uv as baseline  
    def net_uv(self, x, t):
        X = torch.cat([x, t], dim=1)
        H = (X - self.lb) / (self.ub - self.lb) * 2.0 - 1.0 # normalize to [-1, 1]
        self.H = H
        # NOTE: ????
        uv = self.generator.model(H)
        self.uv = uv
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        return u, v, u_x, v_x

    # compute the Schrodinger function on the collocation points
    # TODO: adjust according to different equation
    # TODO: pass function as parameter in init config
    def net_f_uv(self, x, t):
        u, v, u_x, v_x = self.net_uv(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
        f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u
        return f_u, f_v

    def forward(self, x, t):
        u, v, _, _ = self.net_uv(x, t)
        f_u, f_v = self.net_f_uv(x, t)
        return u, v, f_u, f_v
    
    def beta(self, x, t, u, e):
        '''
        A function that differentiates between easy to learn and hard to learn points.
        It is used for the weightupdate. 
        x: x value of collocation points
        t: time of collocation points
        u: function value at collocation points (If N is the network we are trying to achieve N=u and we have certain values u(x_i) given that are in this variable)
        e: a hyperparameter that is a meassure for the exactness to which N should approximate u
        '''
        if abs(self.forward(x,t)[0]-u)**2 <= e:
            return 1
        else:
            return -1

    def loss_G(self, loss_d):
        ''' 
        input dim for G: 
        (x, t, u, v)
        possible error of dimensionality noted.
        '''
        # TODO: call util.py for point loss
        loss = nn.MSELoss()
        loss_l1 = nn.L1Loss()
        self.u0_pred, self.v0_pred, _, _ = self.net_uv(self.x0, self.t0)
        
        self.u_lb_pred, self.v_lb_pred, self.u_x_lb_pred, self.v_x_lb_pred = self.net_uv(self.x_lb, self.t_lb)
        self.u_ub_pred, self.v_ub_pred, self.u_x_ub_pred, self.v_x_ub_pred = self.net_uv(self.x_ub, self.t_ub)
        
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f, self.t_f)
        
        # initial condition + boundary condition + PDE constraint

        MSE = loss(self.u0_pred, self.u0) + loss(self.v0_pred, self.v0) + \
            loss(self.u_lb_pred, self.u_ub_pred) + loss(self.v_lb_pred, self.v_ub_pred) + \
            loss(self.u_x_lb_pred, self.u_x_ub_pred) + loss(self.v_x_lb_pred, self.v_x_ub_pred) + \
            loss(self.f_u_pred, torch.zeros_like(self.f_u_pred)) + loss(self.f_v_pred, torch.zeros_like(self.f_v_pred)) # NOTE what is lb pred, ub pred etc?
        
        input_D = torch.concat((self.x0, self.t0, self.u0_pred, self.v0_pred), 1)
        D_input = self.Discriminator.model(input_D)
        L_D = loss_l1(torch.ones_like(D_input), 
                      D_input)
        # NOTE: dimensionality
        return MSE + L_D

        # TODO : implement boundary data and boundary condition for GAN
        # TODO: normalize the loss/dynamic ratio of importance between 2 loss components

    def loss_D(self):
        '''
        input dim for D: 
        (x, t, u, v)
        possible error of dimensionality noted.
        '''
        #TODO 
        loss = nn.L1Loss()
        discriminator_T = self.Discriminator.model(
            torch.concat((self.x0, self.t0, self.u0, self.v0), 1)
            )

        discriminator_L = self.Discriminator.model(
            torch.concat((self.x0, self.t0, self.u0_pred, self.v0_pred), 1)
            )
        loss_D = loss(discriminator_L, torch.zeros_like(discriminator_L)) + \
                loss(discriminator_T, torch.ones_like(discriminator_T))
        return loss_D

    def train(self, epochs = 1e+4, lr = 1e-3):
        # Optimizer
        optimizer_G = adam.Adam(self.generator.parameters(), lr=lr)
        optimizer_D = adam.Adam(self.discriminator.parameters(), lr=lr)
        
        # Training
        for epoch in tqdm(range(epochs)):
            # TODO
            optimizer_G.zero_grad()
            loss = self.loss_G(1)
            loss.backward()
            optimizer_G.step()
            if epoch % 100 == 0:
                print('Epoch: %d, Loss: %.3e' % (epoch, loss.item()))

    def predict(self, X_star):
        u_star, v_star, f_u_star, f_v_star = self.forward(X_star[:, 0:1], X_star[:, 1:2])
        return u_star.detach().numpy(), v_star.detach().numpy(), f_u_star.detach().numpy(), f_v_star.detach().numpy()