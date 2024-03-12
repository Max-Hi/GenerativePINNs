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
    def __init__(self, layers_G, info_for_error=(None, None)):
        super(Generator, self).__init__()
        self.layers = layers_G
        self.info_for_error = info_for_error
        self.model = nn.Sequential()
        # TODO: baseline structure to be altered 
        for l in range(0, len(self.layers) - 2):
            self.model.add_module("linear" + str(l), nn.Linear(self.layers[l], self.layers[l+1]))
            self.model.add_module("tanh" + str(l), nn.Tanh())
        self.model.add_module("linear" + str(len(self.layers) - 2), nn.Linear(self.layers[-2], self.layers[-1]))
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

class Discriminator(nn.Module):
    def __init__(self, layers_D, info_for_error=(None, None)):
        super(Discriminator, self).__init__()
        self.layers = layers_D
        self.info_for_error = info_for_error
        # NOTE: discriminator input dim = dim(x) * dim(G(x))
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
    def __init__(self, X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G, layers_D):
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
        self.x0 = torch.tensor(X0, requires_grad=True)
        self.y0 = torch.tensor(Y0)
        
        # Boundary Data
        self.x_lb = torch.tensor(X_lb, requires_grad=True)
        self.x_ub = torch.tensor(X_ub, requires_grad=True)
        
        # Collocation Points
        self.x_f = torch.tensor(X_f, requires_grad=True)
        
        # training points that have values
        if X_t != None:
            self.x_t = torch.tensor(X_t)
            self.y_t = torch.tensor(Y_t)
        
        # Bounds
        self.lb = torch.tensor(boundary[:, 0:1])
        self.ub = torch.tensor(boundary[:, 1:2])

        # weights for the point weigthing algorithm
        self.n_boundary_conditions = 0 # NOTE: EDIT manually (why?)
        self.number_collocation_points = self.x_f.shape[0]
        self.domain_weights = torch.full((self.number_collocation_points,), 1/self.number_collocation_points, dtype = torch.float32, requires_grad=False)
        self.boundary_weights = [torch.full((self.number_collocation_points,), 1/self.number_collocation_points, requires_grad=False)]*self.n_boundary_conditions
        
        # Sizes
        self.layers_D = layers_D
        self.layers_G = layers_G
        
        self.generator = Generator(self.layers_G, info_for_error=(self.x0.shape[1],self.y0.shape[1]))
        self.discriminator = Discriminator(self.layers_D, info_for_error=(self.y0.shape[1]+self.x0.shape[1],1))
        
        self.e = 1e-3  # Hyperparameter for PW update

    # calculate the function h(x, t) using neural nets
    # NOTE: regard net_uv as baseline  
    def net_y(self, x):
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
        y = self.net_y(X)
        u = y[:,0:1]
        v = y[:,1:2]
        
        X.requires_grad_(True)
        Jacobian = torch.zeros(X.shape[0], y.shape[1], X.shape[1])
        for i in range(y.shape[1]):  # Loop over all outputs
            for j in range(X.shape[1]):  # Loop over all inputs
                if X.grad is not None:
                    X.grad.data.zero_()  # Zero out previous gradients; crucial for accurate computation
                grad_outputs = torch.zeros_like(y[:, i])
                grad_outputs[:] = 1  # Setting up a vector for element-wise gradient computation
                gradients = torch.autograd.grad(outputs=y[:, i], inputs=X, grad_outputs=grad_outputs,
                                                create_graph=True, retain_graph=True, allow_unused=True)
                if gradients[0] is not None:
                    Jacobian[:, i, j] = gradients[0][:, j]
                else:
                    # Handle the case where the gradient is None (if allow_unused=True)
                    Jacobian[:, i, j] = torch.zeros(X.shape[0])
        
        d2y_dx1_2 = torch.zeros(X.shape[0], y.shape[1])
        for i in range(y.shape[1]):  # Loop over each output component of y
            # Compute the first derivative of y[i] with respect to x1
            dy_dx1 = torch.autograd.grad(y[:, i], X, grad_outputs=torch.ones(X.shape[0], device=X.device), create_graph=True)[0][:, 0]
            
            # Compute the second derivative of y[i] with respect to x1
            # This is the gradient of the first derivative dy_dx1 with respect to x1 again
            d2y_dx1_2_i = torch.autograd.grad(dy_dx1, X, grad_outputs=torch.ones_like(dy_dx1), create_graph=True)[0][:, 0]
            
            # Store the computed second derivative in the placeholder tensor
            d2y_dx1_2[:, i] = d2y_dx1_2_i      

        f_u = Jacobian[:,0,0:1] + 0.5*d2y_dx1_2[:,0:1] + (u**2 + v**2)*v
        f_v = Jacobian[:,1,0:1] - 0.5*d2y_dx1_2[:,1:2] - (u**2 + v**2)*u
        return torch.concat((f_u, f_v),1).to(torch.float32)

    def boundary(self):
        # TODO implement
        return 0

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
    
    def weight_update(self, f_pred, e):
        # f_pred 
        '''
        This function changes the weights used for loss calculation according to the papers formular. 
        It should be called after each iteration. 
        '''
        # TODO: ???????????????????? boundary weight update?

        for index, w in enumerate([self.domain_weights] + self.boundary_weights): # concatenate lists with domain and boundary weights
            # print("e: ", e)
            rho = torch.sum(w*(self.beta(f_pred, e)==-1.0)) # .transpose(0,1))
            '''print("rho alpha stuff")
            print(self.beta(f_u_pred, f_v_pred, e))
            print("rho: ", rho)'''
            epsilon = 10e-4 # this is added to rho because rho has a likelyhood (that empirically takes place often) to be 0 or 1, both of which break the algorithm
            # NOTE: it is probably ok, but think about it that this makes it possible that for rho close to 0 the interior of the log below is greater than one, giving a positive alpha which would otherwise be impossible. 
            # NOTE: we think it is ok because this sign is then given into an exponential where a slight negative instead of 0 should not make a difference (?) 
            alpha = self.q[index] * torch.log((1-rho+epsilon)/(rho+epsilon))
            # print("alpha: ", alpha)
            w_new = w*torch.exp(-alpha*self.beta(f_pred, e).to(torch.float32)) / \
                torch.sum(w*torch.exp(-alpha*self.beta(f_pred, e).to(torch.float32))) # the sum sums along the values of w
            '''print("w_new partly 1: ", w*torch.exp(-alpha*self.beta(f_u_pred, f_v_pred, e)).transpose(0,1))
            print("w_new partly 2: ", torch.sum(w*torch.exp(-alpha*self.beta(f_u_pred, f_v_pred, e)).transpose(0,1)))
            print("w_new: ", w_new )'''
            w_new.requires_grad_(False)
            w_new.to(torch.float32)
            if index == 0:
                self.domain_weights = w_new

            else:
                self.boundary_weights[index-1] = w_new

    def loss_T(self):
        '''
        returns the mse loss of samples that have x and y values. These should be saved in x_t, y_t
        '''
        loss = nn.MSELoss()
        
        self.y_pred = self.net_y(self.x)
        
        return loss(self.y_pred, self.y_t)
    
    def loss_G(self):
        ''' 
        input dim for G: 
        (x, t, u, v)
        possible error of dimensionality noted.
        '''
        # TODO: call util.py for point loss
        loss_l1 = nn.L1Loss()
        
        # TODO: this calculates the boundary loss. This needs to go elsewhere:
        '''
        self.u_lb_pred, self.v_lb_pred, self.u_x_lb_pred, self.v_x_lb_pred = self.net_uv(self.x_lb, self.t_lb) # TODO get as array
        self.u_ub_pred, self.v_ub_pred, self.u_x_ub_pred, self.v_x_ub_pred = self.net_uv(self.x_ub, self.t_ub)# TODO get as array
        
        # initial condition + boundary condition + PDE constraint
        # TODO: incorporate weights into loss calculation
        MSE = loss(self.u0_pred, self.u0) + loss(self.v0_pred, self.v0) + \
            loss(self.u_lb_pred, self.u_ub_pred) + loss(self.v_lb_pred, self.v_ub_pred) + \
            loss(self.u_x_lb_pred, self.u_x_ub_pred) + loss(self.v_x_lb_pred, self.v_x_ub_pred)
        # NOTE what is lb pred, ub pred etc?
        # NOTE: write L_PW outside
        '''
        
        input_D = torch.concat((self.x0, self.y0_pred), 1)
        D_input = self.discriminator.forward(input_D)
        L_D = loss_l1(torch.ones_like(D_input), 
                    D_input)
        
        L_T = 0 # TODO call L_T for now default value

        # NOTE: dimensionality

        return L_T + L_D

        # TODO : implement boundary data and boundary condition for GAN
        # TODO: normalize the loss/dynamic ratio of importance between 2 loss components
        # NOTE: Q: does it differ if optimizer not take step for loss(GAN) and loss(eq) separately?

    
    def loss_PW(self):
        
        f_loss = weighted_MSELoss()
        L_PW = f_loss(self.f_pred, torch.zeros_like(self.f_pred), self.domain_weights.to(torch.float32))
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
        discriminator_T = self.discriminator.forward(
            torch.concat((self.x0, self.y0), 1)
            )

        discriminator_L = self.discriminator.forward(
            torch.concat((self.x0, self.y0_pred), 1)
            )
        loss_D = loss(discriminator_L, torch.zeros_like(discriminator_L)) + \
                loss(discriminator_T, torch.ones_like(discriminator_T))
        return loss_D


    def train(self, epochs = 1e+4, lr_G = 1e-3, lr_D = 2e-4, n_critic = 2):
        # Optimizer
        optimizer_G = adam.Adam(self.generator.parameters(), lr=lr_G)
        optimizer_D = adam.Adam(self.discriminator.parameters(), lr=lr_D)
        optimizer_PW = adam.Adam(self.discriminator.parameters(), lr=lr_G)
        # Training
        for epoch in tqdm(range(epochs)):
            # TODO done?
            self.y0_pred = self.net_y(self.x0) # TODO get as array
            self.f_pred = self.net_f(self.x_f)

            optimizer_D.zero_grad()
            loss_Discr = self.loss_D()
            loss_Discr.backward(retain_graph=True) # retain_graph: tp release tensor for future use
            if epoch % n_critic == 0:
                optimizer_G.zero_grad()
                optimizer_PW.zero_grad()
                loss_G = self.loss_G()
                loss_G.backward(retain_graph=True)
                # Update PW loss
                self.weight_update(self.f_pred, self.e)
               
                loss_PW = self.loss_PW()
                loss_PW.backward(retain_graph=True)
                optimizer_PW.step()
                optimizer_G.step()
            optimizer_D.step()
            # weight updates
  
            if epoch % 100 == 0:
                print('Epoch: %d, Loss_G: %.3e, Loss_D: %.3e' % (epoch, loss_G.item(), loss_Discr.item()))
                


    def predict(self, X_star):
        y_star = self.generator.forward(X_star)
        f_star = self.net_f(X_star) #TODO implement
        return y_star.detach().numpy(), f_star.detach().numpy()