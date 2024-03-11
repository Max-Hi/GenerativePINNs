import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import adam

# set random seeds for reproducability
np.random.seed(42)
torch.manual_seed(42)

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
        self.model = self.model.double()
        
    def forward(self, x):
        return self.model(x)

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

    def forward(self, x):
        return self.model(x)
class PINN_GAN(nn.Module):
    def __init__(self, X0, Y0, X_f, X_lb, X_ub, boundary, layers_G, layers_D):
        """
        X0: T=0, initial condition, randomly drawn from the domain
        Y0: T=0, initial condition, given (u0, v0)
        X_f: the collocation points with time, size (Nf, dim(X)+1)
        X_lb: the lower boundary, size (N_b, 2)
        X_ub: the upper boundary, size (N_b, 2)
        boundary: the lower and upper boundary, size (2, 2) : [(x_min, t_min), (x_max, t_max)]
        layers: the number of neurons in each layer (_D for discriminator, _G for generator)
        """
        super(PINN_GAN, self).__init__()

        # Hyperparameters
        self.q = [0.1,
                  ]
        
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
        self.n_boundary_conditions = 0 # NOTE: EDIT manually (why?)
        self.number_collocation_points = self.x_f.shape[0]
        self.domain_weights = torch.full((self.number_collocation_points,), 1/self.number_collocation_points, dtype = torch.float16, requires_grad=False)
        self.boundary_weights = [torch.full((self.number_collocation_points,), 1/self.number_collocation_points, requires_grad=False)]*self.n_boundary_conditions
        
        print("weights")
        print(self.domain_weights)
        
        # Sizes
        self.layers_D = layers_D
        self.layers_G = layers_G
        
        self.generator = Generator(self.layers_G)
        self.discriminator = Discriminator(self.layers_D)
        
        self.e = 1e-3  # Hyperparameter for PW update

    # calculate the function h(x, t) using neural nets
    # NOTE: regard net_uv as baseline  
    def net_uv(self, x, t):
        X = torch.cat([x, t], dim=1).transpose(0,1)
        H = (X - self.lb) / (self.ub - self.lb) * 2.0 - 1.0 # normalize to [-1, 1]
        self.H = H.transpose(0,1)
        uv = self.generator.forward(self.H)
        self.uv = uv
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0] # create_graph=True
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        return u, v, u_x, v_x

    # compute the Schrodinger function on the collocation points
    # TODO: adjust according to different equation
    # TODO: pass function as parameter in init configs
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

    def loss_G(self):
        
        loss = nn.MSELoss()
        
        MSE = loss(self.f_u_pred, torch.zeros_like(self.f_u_pred)) + loss(self.f_v_pred, torch.zeros_like(self.f_v_pred)) 
        
        return MSE
    
    def loss_PW(self):
        
        loss = nn.MSELoss()
        
        f_loss = loss(self.f_u_pred, torch.zeros_like(self.f_u_pred)) + loss(self.f_v_pred, torch.zeros_like(self.f_v_pred)) 
        return f_loss


    def train(self, epochs = 1e+4, lr_G = 1e-3, n_critic = 5):
        # Optimizer
        optimizer_G = adam.Adam(self.generator.parameters(), lr=lr_G)
        optimizer_PW = adam.Adam(self.discriminator.parameters(), lr=lr_G)
        # Training
        for epoch in tqdm(range(epochs)):
            # TODO done?
            self.u0_pred, self.v0_pred, _, _ = self.net_uv(self.x0, self.t0)
            self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f, self.t_f)

            if epoch % n_critic == 0:
                optimizer_G.zero_grad()
                loss_G = self.loss_G()
                loss_G.backward(retain_graph=True)
                # former step position
                
                optimizer_PW.zero_grad()
                loss_PW = self.loss_PW()
                loss_PW.backward(retain_graph=True)
                optimizer_PW.step()
                
                # new position
                optimizer_G.step()
            # weight updates
  
            if epoch % 100 == 0:
                print('Epoch: %d, Loss_G: %.3e' % (epoch, loss_G.item()))
                


    def predict(self, X_star):
        u_star, v_star, f_u_star, f_v_star = self.forward(X_star[:, 0:1], X_star[:, 1:2])
        return u_star.detach().numpy(), v_star.detach().numpy(), f_u_star.detach().numpy(), f_v_star.detach().numpy()