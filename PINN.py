# import modules
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
        n_boundary_conditions = 0 # EDIT manually
        self.number_collocation_points = self.x_f.shape[0]
        self.domain_weights = torch.full((self.number_collocation_points,), 1/self.number_collocation_points)
        self.boundary_weights = [torch.full((self.number_collocation_points,), 1/self.number_collocation_points)]*n_boundary_conditions
        
        print("weights")
        print(self.domain_weights)
        
        # Sizes
        self.layers_D = layers_D
        self.layers_G = layers_G
        
        self.generator = Generator(self.layers_G)
        self.discriminator = Discriminator(self.layers_D)
                  

    # calculate the function h(x, t) using neural nets
    # NOTE: regard net_uv as baseline  
    def net_uv(self, x, t):
        X = torch.cat([x, t], dim=1).transpose(0,1)
        H = (X - self.lb) / (self.ub - self.lb) * 2.0 - 1.0 # normalize to [-1, 1]
        self.H = H.transpose(0,1)
        # NOTE: ????
        uv = self.generator.forward(self.H)
        self.uv = uv
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
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
    
    def beta(self, f_u_pred, f_v_pred, e):
        '''
        A function that differentiates between easy to learn and hard to learn points.
        It is used for the weightupdate. 
        f_u_pred and f_v_pred are the values of the DGL function which should be close to zero
        e: a hyperparameter that is a meassure for the exactness to which N should approximate u
        '''
        return (f_u_pred**2 + f_v_pred**2 <= e).to(torch.float16) - (f_u_pred**2 + f_v_pred**2 > e).to(torch.float16)
        
    def weight_update(self, f_u_pred, f_v_pred, e):
        '''
        This function changes the weights used for loss calculation according to the papers formular. It should be called after each iteration. 
        '''
        for index, w in enumerate([self.domain_weights] + self.boundary_weights): # concatenate lists with domain and boundary weights
            # print("e: ", e)
            rho = torch.sum(w*(self.beta(f_u_pred, f_v_pred, e)==-1.0).transpose(0,1))
            '''print("rho alpha stuff")
            print(self.beta(f_u_pred, f_v_pred, e))
            print("rho: ", rho)'''
            epsilon = 10e-4 # this is added to rho because rho has a likelyhood (that empirically takes place often) to be 0 or 1, both of which break the algorithm
            # NOTE: it is probably ok, but think about it that this makes it possible that for rho close to 0 the interior of the log below is greater than one, giving a positive alpha which would otherwise be impossible. 
            # NOTE: we think it is ok because this sign is then given into an exponential where a slight negative instead of 0 should not make a difference (?) 
            alpha = self.q[index] * torch.log((1-rho+epsilon)/(rho+epsilon))
            # print("alpha: ", alpha)
            w_new = w*torch.exp(-alpha*self.beta(f_u_pred, f_v_pred, e)).transpose(0,1) / torch.sum(w*torch.exp(-alpha*self.beta(f_u_pred, f_v_pred, e)).transpose(0,1)) # the sum sums along the values of w
            '''print("w_new partly 1: ", w*torch.exp(-alpha*self.beta(f_u_pred, f_v_pred, e)).transpose(0,1))
            print("w_new partly 2: ", torch.sum(w*torch.exp(-alpha*self.beta(f_u_pred, f_v_pred, e)).transpose(0,1)))
            print("w_new: ", w_new )'''
            
            if index == 0:
                self.domain_weights = w_new
            else:
                self.boundary_weights[index-1] = w_new

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
        
        '''
        print("loss stuff")
        print(self.f_u_pred)
        print("____")
        print(self.beta(self.f_u_pred, self.f_v_pred, 0.001))
        self.weight_update(self.f_u_pred, self.f_v_pred, 0.001)
        print("weights")
        print(self.domain_weights)
        '''
        
        # initial condition + boundary condition + PDE constraint
        # TODO: incorporate weights into loss claculation
        MSE = loss(self.u0_pred, self.u0) + loss(self.v0_pred, self.v0) + \
            loss(self.u_lb_pred, self.u_ub_pred) + loss(self.v_lb_pred, self.v_ub_pred) + \
            loss(self.u_x_lb_pred, self.u_x_ub_pred) + loss(self.v_x_lb_pred, self.v_x_ub_pred) + \
            loss(self.f_u_pred, torch.zeros_like(self.f_u_pred)) + loss(self.f_v_pred, torch.zeros_like(self.f_v_pred)) # NOTE what is lb pred, ub pred etc?
        
        # weight updates
        self.weight_update(self.f_u_pred, self.f_v_pred, 0.001)
        
        input_D = torch.concat((self.x0, self.t0, self.u0_pred, self.v0_pred), 1)
        D_input = self.discriminator.model(input_D)
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

    def train(self, epochs = 1e+4, lr_G = 1e-3, lr_D = 2e-4, n_critic = 5):
        # Optimizer
        optimizer_G = adam.Adam(self.generator.parameters(), lr=lr_G)
        optimizer_D = adam.Adam(self.discriminator.parameters(), lr=lr_D)
        
        # Training
        for epoch in tqdm(range(epochs)):
            # TODO
            optimizer_D.zero_grad()
            loss = self.loss_D()
            loss.backward()
            if epoch % n_critic = 0:
                optimizer_G.zero_grad()
                loss = self.loss_G(1)
                loss.backward()
                optimizer_G.step()

            # TODO: point loss
            
            if epoch % 100 == 0:
                print('Epoch: %d, Loss: %.3e' % (epoch, loss.item()))
        

    def predict(self, X_star):
        u_star, v_star, f_u_star, f_v_star = self.forward(X_star[:, 0:1], X_star[:, 1:2])
        return u_star.detach().numpy(), v_star.detach().numpy(), f_u_star.detach().numpy(), f_v_star.detach().numpy()