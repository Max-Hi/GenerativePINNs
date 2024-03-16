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

from PINN import PINN_GAN

# set random seeds for reproducability
np.random.seed(42)
torch.manual_seed(42)



class Schroedinger_PINN_GAN(PINN_GAN):
    def __init__(self, X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G : list=[], layers_D: list=[], enable_GAN = True, enable_PW = True, dynamic_lr = False, model_name: str="", lr: tuple=(1e-3, 2e-4), lambdas: tuple = (1,1)):
        
        if model_name!="":
            model_name = "schroedinger"+model_name
        super().__init__(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G, layers_D, enable_GAN, enable_PW, dynamic_lr, model_name, lr, lambdas)
        
        n_boundaries = [X0.shape[0]]+[X_lb.shape[0]]*2
        self.number_collocation_points = self.x_f.shape[0]
        self.domain_weights = torch.full((self.number_collocation_points,), 1/self.number_collocation_points, dtype = torch.float32, requires_grad=False)
        self.boundary_weights = []
        for number_boundary_points in n_boundaries:
            self.boundary_weights.append(torch.full((number_boundary_points,), 1/number_boundary_points, requires_grad=False))
    
    def _net_f(self, X):
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
        
        X = self.x0
        y = self.net_y(X)
        X_lb = self.x_lb
        X_ub = self.x_lb
        y_lb = self.net_y(X_lb)
        y_ub = self.net_y(X_ub)
        
        X_lb.requires_grad_(True)
        Jacobian_lb = torch.zeros(X_lb.shape[0], y_lb.shape[1], X_lb.shape[1])
        for i in range(y_lb.shape[1]):  # Loop over all outputs
            for j in range(X_lb.shape[1]):  # Loop over all inputs
                if X_lb.grad is not None:
                    X_lb.grad.data.zero_()  # Zero out previous gradients; crucial for accurate computation
                grad_outputs = torch.zeros_like(y[:, i])
                grad_outputs[:] = 1  # Setting up a vector for element-wise gradient computation
                gradients = torch.autograd.grad(outputs=y_lb[:, i], inputs=X_lb, grad_outputs=grad_outputs,
                                                create_graph=True, retain_graph=True, allow_unused=True)
                if gradients[0] is not None:
                    Jacobian_lb[:, i, j] = gradients[0][:, j]
                else:
                    # Handle the case where the gradient is None (if allow_unused=True)
                    Jacobian_lb[:, i, j] = torch.zeros(X_lb.shape[0])
        
        X_ub.requires_grad_(True)
        Jacobian_ub = torch.zeros(X_ub.shape[0], y_ub.shape[1], X_ub.shape[1])
        for i in range(y_ub.shape[1]):  # Loop over all outputs
            for j in range(X_ub.shape[1]):  # Loop over all inputs
                if X_ub.grad is not None:
                    X_ub.grad.data.zero_()  # Zero out previous gradients; crucial for accurate computation
                grad_outputs = torch.zeros_like(y[:, i])
                grad_outputs[:] = 1  # Setting up a vector for element-wise gradient computation
                gradients = torch.autograd.grad(outputs=y_ub[:, i], inputs=X_ub, grad_outputs=grad_outputs,
                                                create_graph=True, retain_graph=True, allow_unused=True)
                if gradients[0] is not None:
                    Jacobian_ub[:, i, j] = gradients[0][:, j]
                else:
                    # Handle the case where the gradient is None (if allow_unused=True)
                    Jacobian_ub[:, i, j] = torch.zeros(X_ub.shape[0])
        
        boundaries = [y-2/torch.cosh(X), y_lb-y_ub, Jacobian_lb[:,0,:]-Jacobian_ub[:,0,:]]
        boundaries = list(map(lambda x: x.to(torch.float32),boundaries))
        return boundaries
    
    
class Heat_PINN_GAN(PINN_GAN):
    def __init__(self, X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G : list=[], layers_D: list=[], enable_GAN = True, enable_PW = True, dynamic_lr = False, model_name: str="", lr: tuple=(1e-3, 2e-4), lambdas: tuple = (1,1)):
    
        if model_name!="":
            model_name = "heat"+model_name
        super(Heat_PINN_GAN, self).__init__(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G, layers_D, enable_GAN, enable_PW, dynamic_lr, model_name, lr, lambdas)   

        n_boundaries = [X0.shape[0]]
        self.number_collocation_points = self.x_f.shape[0]
        self.domain_weights = torch.full((self.number_collocation_points,), 1/self.number_collocation_points, dtype = torch.float32, requires_grad=False)
        self.boundary_weights = []
        for number_boundary_points in n_boundaries:
            self.boundary_weights.append(torch.full((number_boundary_points,), 1/number_boundary_points, requires_grad=False))
            
    def _net_f(self, X):
        y = self.net_y(X)
        
        X.requires_grad_(True)
        Jacobian = torch.zeros(X.shape[0], y.shape[1], X.shape[1]) # Jacobian[:,i,j:j+1] will create a (:,1) shaped gradient of the ith y entry with regard to the jth x entry.
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
        
        d2y_dx2_2 = torch.zeros(X.shape[0], y.shape[1])
        for i in range(y.shape[1]):  # Loop over each output component of y
            # Compute the first derivative of y[i] with respect to x1
            dy_dx2 = torch.autograd.grad(y[:, i], X, grad_outputs=torch.ones(X.shape[0], device=X.device), create_graph=True)[0][:, 1]
            
            # Compute the second derivative of y[i] with respect to x1
            # This is the gradient of the first derivative dy_dx1 with respect to x1 again
            d2y_dx2_2_i = torch.autograd.grad(dy_dx2, X, grad_outputs=torch.ones_like(dy_dx2), create_graph=True)[0][:, 1]
            
            # Store the computed second derivative in the placeholder tensor
            d2y_dx2_2[:, i] = d2y_dx2_2_i      

        f = Jacobian[:,0,2:3] - d2y_dx1_2 - d2y_dx2_2
        return f.to(torch.float32)

    def boundary(self):
        X = self.x0
        y = self.net_y(X)
        
        boundaries = [y-(X[:,0:1]-X[:,1:2])]
        boundaries = list(map(lambda x: x.to(torch.float32),boundaries))
        return boundaries

class Poisson_PINN_GAN(PINN_GAN):
    def __init__(self, X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G : list=[], layers_D: list=[], enable_GAN = True, enable_PW = True, dynamic_lr = False, model_name: str="", lr: tuple=(1e-3, 2e-4), lambdas: tuple = (1,1)):
    
        if model_name!="":
            model_name = "poisson"+model_name
        super(Poisson_PINN_GAN, self).__init__(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G, layers_D, enable_GAN, enable_PW, dynamic_lr, model_name, lr, lambdas)  

        n_boundaries = [self.x_lb[0].shape[0]]*2+[self.x_lb[1].shape[0]]*2
        self.number_collocation_points = self.x_f.shape[0]
        self.domain_weights = torch.full((self.number_collocation_points,), 1/self.number_collocation_points, dtype = torch.float32, requires_grad=False)
        self.boundary_weights = []
        for number_boundary_points in n_boundaries:
            self.boundary_weights.append(torch.full((number_boundary_points,), 1/number_boundary_points, requires_grad=False))
            
    def _net_f(self, X):
        y = self.net_y(X)
        
        X.requires_grad_(True)
        d2y_dx1_2 = torch.zeros(X.shape[0], y.shape[1])
        for i in range(y.shape[1]):  # Loop over each output component of y
            # Compute the first derivative of y[i] with respect to x1
            dy_dx1 = torch.autograd.grad(y[:, i], X, grad_outputs=torch.ones(X.shape[0], device=X.device), create_graph=True)[0][:, 0]
            
            # Compute the second derivative of y[i] with respect to x1
            # This is the gradient of the first derivative dy_dx1 with respect to x1 again
            d2y_dx1_2_i = torch.autograd.grad(dy_dx1, X, grad_outputs=torch.ones_like(dy_dx1), create_graph=True)[0][:, 0]
            
            # Store the computed second derivative in the placeholder tensor
            d2y_dx1_2[:, i] = d2y_dx1_2_i      

        d2y_dx2_2 = torch.zeros(X.shape[0], y.shape[1])
        for i in range(y.shape[1]):  # Loop over each output component of y
            # Compute the first derivative of y[i] with respect to x1
            dy_dx2 = torch.autograd.grad(y[:, i], X, grad_outputs=torch.ones(X.shape[0], device=X.device), create_graph=True)[0][:, 1]
            
            # Compute the second derivative of y[i] with respect to x1
            # This is the gradient of the first derivative dy_dx1 with respect to x1 again
            d2y_dx2_2_i = torch.autograd.grad(dy_dx2, X, grad_outputs=torch.ones_like(dy_dx2), create_graph=True)[0][:, 1]
            
            # Store the computed second derivative in the placeholder tensor
            d2y_dx2_2[:, i] = d2y_dx2_2_i     

        f = d2y_dx2_2 + d2y_dx1_2 + torch.sin(torch.pi*X)*torch.sin(torch.pi*y)
        return f.to(torch.float32)

    def boundary(self):
        X1_lb = self.x_lb[0]
        X1_ub = self.x_lb[0]
        y1_lb = self.net_y(X1_lb)
        y1_ub = self.net_y(X1_ub)
        X2_lb = self.x_lb[1]
        X2_ub = self.x_lb[1]
        y2_lb = self.net_y(X2_lb)
        y2_ub = self.net_y(X2_ub)
        
        boundaries = [y1_lb, y1_ub, y2_lb, y2_ub]
        boundaries = list(map(lambda x: x.to(torch.float32),boundaries))
        return boundaries


class PoissonHD_PINN_GAN(PINN_GAN):
    def __init__(self, X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G : list=[], layers_D: list=[], enable_GAN = True, enable_PW = True, dynamic_lr = False, model_name: str="", lr: tuple=(1e-3, 2e-4), lambdas: tuple = (1,1)):
    
        if model_name!="":
            model_name = "poissonhd"+model_name
        super(PoissonHD_PINN_GAN, self).__init__(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G, layers_D, enable_GAN, enable_PW, dynamic_lr, model_name, lr, lambdas) 
        
        n_boundaries = []
        for x in self.x_lb:
            n_boundaries.append(x.shape[0])
        self.number_collocation_points = self.x_f.shape[0]
        self.domain_weights = torch.full((self.number_collocation_points,), 1/self.number_collocation_points, dtype = torch.float32, requires_grad=False)
        self.boundary_weights = []
        for number_boundary_points in n_boundaries:
            self.boundary_weights.append(torch.full((number_boundary_points,), 1/number_boundary_points, requires_grad=False))
             
    def _net_f(self, X):
        y = self.net_y(X)
        
        X.requires_grad_(True)
        # First derivatives
        first_derivatives = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

        # Compute second derivatives
        second_derivatives = []
        for i in range(10):
            # Compute the derivative of the ith component of the first derivative with respect to x
            second_derivative = torch.autograd.grad(outputs=first_derivatives[:, i], inputs=X, grad_outputs=torch.ones_like(first_derivatives[:, i]), create_graph=True)[0][:, i]
            second_derivatives.append(second_derivative)

        # Stack to get a tensor of shape (k, 10) for second derivatives
        second_derivatives_tensor = torch.stack(second_derivatives, dim=1)

        f = torch.sum(second_derivatives_tensor,1)
        return f.to(torch.float32)

    def boundary(self):
        Y_lb = []
        Y_ub = []
        X_lb = self.x_lb
        X_ub = self.x_lb
        for x_lb in X_lb:
            Y_lb.append(self.net_y(x_lb))
        for x_ub in X_ub:
            Y_ub.append(self.net_y(x_ub))
        
        boundaries = []
        for idx in range(len(Y_lb)):
            y_lb = Y_lb[idx]
            y_ub = Y_ub[idx]
            x_lb = X_lb[idx]
            x_ub = X_ub[idx]
            boundaries.append(x_lb[:,0:1]**2 - x_lb[:,1:2]**2 + x_lb[:,2:3]**2 - x_lb[:,3:4]**2 + x_lb[:,4:5]*x_lb[:,5:6] + x_lb[:,7:8]*x_lb[:,8:9]*x_lb[:,9:10]*x_lb[:,10:11])
        boundaries = list(map(lambda x: x.to(torch.float32),boundaries))
        return boundaries


class Helmholtz_PINN_GAN(PINN_GAN):
    def __init__(self, X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, k, layers_G : list=[], layers_D: list=[], enable_GAN = True, enable_PW = True, dynamic_lr = False, model_name: str="", lr: tuple=(1e-3, 2e-4), lambdas: tuple = (1,1)):
    
        if model_name!="":
            model_name = "holmholtz"+model_name
        super(Helmholtz_PINN_GAN, self).__init__(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G, layers_D, enable_GAN, enable_PW, dynamic_lr, model_name, lr, lambdas)  

        self.k = k
        
        n_boundaries = [self.x_lb[0].shape[0]]*2+[self.x_lb[1].shape[0]]*2
        self.number_collocation_points = self.x_f.shape[0]
        self.domain_weights = torch.full((self.number_collocation_points,), 1/self.number_collocation_points, dtype = torch.float32, requires_grad=False)
        self.boundary_weights = []
        for number_boundary_points in n_boundaries:
            self.boundary_weights.append(torch.full((number_boundary_points,), 1/number_boundary_points, requires_grad=False))
        
    def _net_f(self, X):
        y = self.net_y(X)
        
        X.requires_grad_(True)
        d2y_dx1_2 = torch.zeros(X.shape[0], y.shape[1])
        for i in range(y.shape[1]):  # Loop over each output component of y
            # Compute the first derivative of y[i] with respect to x1
            dy_dx1 = torch.autograd.grad(y[:, i], X, grad_outputs=torch.ones(X.shape[0], device=X.device), create_graph=True)[0][:, 0]
            
            # Compute the second derivative of y[i] with respect to x1
            # This is the gradient of the first derivative dy_dx1 with respect to x1 again
            d2y_dx1_2_i = torch.autograd.grad(dy_dx1, X, grad_outputs=torch.ones_like(dy_dx1), create_graph=True)[0][:, 0]
            
            # Store the computed second derivative in the placeholder tensor
            d2y_dx1_2[:, i] = d2y_dx1_2_i      

        d2y_dx2_2 = torch.zeros(X.shape[0], y.shape[1])
        for i in range(y.shape[1]):  # Loop over each output component of y
            # Compute the first derivative of y[i] with respect to x1
            dy_dx2 = torch.autograd.grad(y[:, i], X, grad_outputs=torch.ones(X.shape[0], device=X.device), create_graph=True)[0][:, 1]
            
            # Compute the second derivative of y[i] with respect to x1
            # This is the gradient of the first derivative dy_dx1 with respect to x1 again
            d2y_dx2_2_i = torch.autograd.grad(dy_dx2, X, grad_outputs=torch.ones_like(dy_dx2), create_graph=True)[0][:, 1]
            
            # Store the computed second derivative in the placeholder tensor
            d2y_dx2_2[:, i] = d2y_dx2_2_i 
            
        f = d2y_dx1_2 + d2y_dx2_2 + self.k*y**2
        return f.to(torch.float32)

    def boundary(self):
        X1_lb = self.x_lb[0]
        X1_ub = self.x_lb[0]
        y1_lb = self.net_y(X1_lb)
        y1_ub = self.net_y(X1_ub)
        X2_lb = self.x_lb[1]
        X2_ub = self.x_lb[1]
        y2_lb = self.net_y(X2_lb)
        y2_ub = self.net_y(X2_ub)
        
        boundaries = [torch.sin(self.k*X1_lb)-y1_lb,torch.sin(self.k*X1_ub)-y1_ub,torch.sin(self.k*X2_lb)-y2_lb,torch.sin(self.k*X2_ub)-y2_ub]
        boundaries = list(map(lambda x: x.to(torch.float32),boundaries))
        return boundaries
    
class Burgers_PINN_GAN(PINN_GAN):
    def __init__(self, X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G : list=[], layers_D: list=[], enable_GAN = True, enable_PW = True, dynamic_lr = False, model_name: str="", lr: tuple=(1e-3, 2e-4), lambdas: tuple = (1,1), nu: float=0.01/np.pi):
    
        if model_name!="":
            model_name = "burgers"+model_name
        super().__init__(X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, layers_G, layers_D, enable_GAN, enable_PW, dynamic_lr, model_name, lr, lambdas)  
        
        self.nu = nu
        
        n_boundaries = [self.y0.shape[0], self.x_lb.shape[0], self.x_ub.shape[0]] # TODO
        self.number_collocation_points = self.x_f.shape[0]
        self.domain_weights = torch.full((self.number_collocation_points,), 1/self.number_collocation_points, dtype = torch.float32, requires_grad=False)
        self.boundary_weights = []
        for number_boundary_points in n_boundaries:
            self.boundary_weights.append(torch.full((number_boundary_points,), 1/number_boundary_points, requires_grad=False))
        
    def _net_f(self, X):
        y = self.net_y(X)
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

        f = Jacobian[:,0,1:2] + y*Jacobian[:,0,0:1] - self.nu*d2y_dx1_2
        return f.to(torch.float32)
    
    def boundary(self):
        self.u_lb_pred = self.net_y(self.x_lb)
        self.u_ub_pred = self.net_y(self.x_ub)
        self.u_exact_pred = self.net_y(self.x_t)
        self.f_u_pred = self.net_f(self.x_f)
        self.u0_pred = self.net_y(self.x0)
        
        boundaries = [self.u0_pred - self.y0, self.u_lb_pred, self.u_ub_pred]
        boundaries = list(map(lambda x: x.to(torch.float32),boundaries))
        return boundaries
    


