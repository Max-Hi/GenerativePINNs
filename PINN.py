# import modules
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import adam


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



class PhysicsInformedNN(nn.Module):
    def __init__(self, X0, Y0, X_f, X_lb, X_ub, bounary, layers):
        """
        X0: T=0, initial condition, randomly drawn from the domain
        Y0: T=0, initial condition, given (u0, v0)
        X_f: the collocation points with time, size (Nf, 2)
        X_lb: the lower boundary, size (N_b, 2)
        X_ub: the upper boundary, size (N_b, 2)
        boundary: the lower and upper boundary, size (2, 2) : [(x_min, t_min), (x_max, t_max)]
        layers: the number of neurons in each layer
        """
        super(PhysicsInformedNN, self).__init__()

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
        self.lb = torch.tensor(bounary[:, 0:1])
        self.ub = torch.tensor(bounary[:, 1:2])
        
        # Initialize NNs
        self.layers = layers
        self.model = nn.Sequential()
        for l in range(0, len(layers) - 2):
            self.model.add_module("linear" + str(l), nn.Linear(layers[l], layers[l+1]))
            self.model.add_module("tanh" + str(l), nn.Tanh())
        self.model.add_module("linear" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1]))
        self.model = self.model.double()

    # calculate the function h(x, t) using neural nets
    def net_uv(self, x, t):
        X = torch.cat([x, t], dim=1)
        H = (X - self.lb) / (self.ub - self.lb) * 2.0 - 1.0 # normalize to [-1, 1]
        
        uv = self.model(H)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        return u, v, u_x, v_x

    # compute the Schrodinger function on the collacation points
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

    def loss_function(self):
        loss = nn.MSELoss()
        self.u0_pred, self.v0_pred, _, _ = self.net_uv(self.x0, self.t0)
        
        self.u_lb_pred, self.v_lb_pred, self.u_x_lb_pred, self.v_x_lb_pred = self.net_uv(self.x_lb, self.t_lb)
        self.u_ub_pred, self.v_ub_pred, self.u_x_ub_pred, self.v_x_ub_pred = self.net_uv(self.x_ub, self.t_ub)
        
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f, self.t_f)
        
        # initial condition + boundary condition + PDE constraint
        MSE = loss(self.u0_pred, self.u0) + loss(self.v0_pred, self.v0) + \
            loss(self.u_lb_pred, self.u_ub_pred) + loss(self.v_lb_pred, self.v_ub_pred) + \
            loss(self.u_x_lb_pred, self.u_x_ub_pred) + loss(self.v_x_lb_pred, self.v_x_ub_pred) + \
            loss(self.f_u_pred, torch.zeros_like(self.f_u_pred)) + loss(self.f_v_pred, torch.zeros_like(self.f_v_pred))

        return MSE
    
    def train(self, epochs = 1e+4, lr = 1e-3):
        # Optimizer
        optimizer = adam.Adam(self.model.parameters(), lr=lr)
        
        # Training
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            loss = self.loss_function()
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print('Epoch: %d, Loss: %.3e' % (epoch, loss.item()))

    def predict(self, X_star):
        u_star, v_star, f_u_star, f_v_star = self.forward(X_star[:, 0:1], X_star[:, 1:2])
        return u_star.detach().numpy(), v_star.detach().numpy(), f_u_star.detach().numpy(), f_v_star.detach().numpy()