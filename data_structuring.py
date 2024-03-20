import numpy as np
from pyDOE import lhs


def structure_data(pde, data, noise, N0, N_b, N_f, N_exact = 50):
    match pde:
        case "burgers":
            return structure_data_burgers(data, noise, N0, N_b, N_f, N_exact)
        case "heat":
            return structure_data_heat(data, noise, N0, N_f, N_exact)
        case "schroedinger":
            return structure_data_schroedinger(data, noise, N0, N_b, N_f, N_exact)
        case "poisson":
            return structure_data_poisson(data, noise, N_b, N_f, N_exact)
        case "poissonHD":
            return structure_data_poissonHD(noise, N_b, N_f, N_exact)
        case "helmholtz":
            return structure_data_helmholtz(data, noise, N_b, N_f, N_exact)
        case _:
            print("pde not recognised")


def add_noise(data, noise):
    base_sigma = np.abs(data)  
    
    # Scale the base noise by the noise level parameter
    noise = noise * np.random.normal(0, 1, data.shape) * base_sigma
    
    # Add the noise to the original data
    noisy_data = data + noise
    
    return noisy_data



####################### Data structuring Functions #########################

def structure_data_schroedinger(data, noise, N0, N_b, N_f, N_exact):
    #bounds of data
    lb = np.array([-5.0, 0.0]) # lower bound for [x, t]
    ub = np.array([5.0, np.pi/2]) # upper bound for [x, t]

    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = add_noise(data['uu'], noise)
    Exact_h = np.sqrt(np.real(Exact)**2 + np.imag(Exact)**2)
    X, T = np.meshgrid(x,t)
    grid = [X,T]
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    h_star = Exact_h.T.flatten()[:,None]
    v_star = np.imag(Exact).T.flatten()[:,None]
    u_star = np.real(Exact).T.flatten()[:,None]

    # Initial and boundary data
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    u0 = np.real(Exact[idx_x,0:1]) # or computed using h(0, x) = 2*sech(x)
    v0 = np.imag(Exact[idx_x,0:1])
    Y0 = np.hstack((u0, v0))
    tb = t[np.random.choice(t.shape[0], N_b, replace=False),:] # random time samples for boundary points

    # Collocation points
    X_f = lb + (ub-lb)*lhs(2, N_f)

    # initial points
    X0 = np.concatenate((x0, np.zeros_like(x0, dtype=np.float32)), 1) # (x, 0)

    # boundary points
    boundary = np.vstack((lb, ub))
    X_lb = np.concatenate((lb[0]*np.ones_like(tb, dtype=np.float32), tb), axis=1)
    X_ub = np.concatenate((ub[0]*np.ones_like(tb, dtype=np.float32), tb), axis=1)

    # Training Samples with exact values

    idx = np.random.choice(X_star.shape[0], N_exact, replace=False)
    X_exact = X_star[idx, :] # random samples for time points
    U_exact = u_star[idx, :] # exact observations at the time points
    V_exact = v_star[idx, :] # exact observations at the time points
    Y_exact = np.hstack((U_exact, V_exact)) #NOTE: is it the same as the previous Y_t?
    # Use the mesh to index into u
    # Y_t = np.hstack((np.real(Exact[mesh_idx_x, mesh_idx_t]).flatten()[:,None],np.imag(Exact[mesh_idx_x, mesh_idx_t]).flatten()[:,None]))
    
    return grid, X0, Y0, X_f, X_exact, Y_exact, X_lb, X_ub, boundary, X_star, [u_star, v_star, h_star]

def structure_data_heat(data, noise, N0, N_f, N_exact):
    # bounds of data
    lb = np.array([0, 0, 0]) # lower bound for [x1, x2, t]
    ub = np.array([1, 1, 10]) # upper bound for [x1, x2, t]
    boundary = np.vstack((lb, ub))
    
    t = data['t'].flatten()[:,None]
    x1 = data['x1'].flatten()[:,None]
    x2 = data['x2'].flatten()[:,None]
    Exact = add_noise(data['usol'], noise)
    X1, X2, T = np.meshgrid(x1, x2, t)
    grid = [X1, X2, T]
    X_star = np.hstack((X1.flatten()[:,None], X2.flatten()[:,None], T.flatten()[:,None])) # for prediction
    Y_star = Exact.flatten()[:,None] # for prediction as ground truth
    

    # Initial data
    X0 = lb[:2] + (ub[:2]-lb[:2])*lhs(2, N0) # random samples for initial points
    X0 = np.insert(X0, 2, 0, axis=1) # (x, 0)
    Y0 = (X0[:,0] - X0[:,1])[:,None] # u(x, 0) = x1 - x2
    
    # no boundary data so turn to the collocation points
    X_f = lb + (ub-lb)*lhs(3, N_f)

    # sample some points as known values
    idx = np.random.choice(X_star.shape[0], N_exact, replace=False)
    X_exact = X_star[idx, :]
    Y_exact = Y_star[idx, :]
    
    # default values just for compatibility
    X_lb, X_ub = None, None
    
    return grid, X0, Y0, X_f, X_exact, Y_exact, X_lb, X_ub, boundary, X_star, Y_star

def structure_data_helmholtz(data, noise, N_b, N_f, N_exact):
    # bounds of data
    lb = np.array([0, 0]) # lower bound for [x1, x2]
    ub = np.array([1, 1]) # upper bound for [x1, x2]
    boundary = np.vstack((lb, ub))
    
    x1 = data['x1'].flatten()[:,None]
    x2 = data['x2'].flatten()[:,None]
    Exact = add_noise(data['usol'], noise)
    X1, X2 = np.meshgrid(x1, x2)
    grid = [X1, X2]
    X_star = np.hstack((X1.flatten()[:,None], X2.flatten()[:,None])) # for prediction
    Y_star = Exact.flatten()[:,None] # for prediction as ground truth
    
    
    # No initial data so turn to boundary data
    idx1 = np.random.choice(np.where(X_star[:,0] == lb[0])[0], N_b//4, replace=False)
    idx2 = np.random.choice(np.where(X_star[:,0] == ub[0])[0], N_b//4, replace=False)
    idx3 = np.random.choice(np.where(X_star[:,1] == lb[1])[0], N_b//4, replace=False)
    idx4 = np.random.choice(np.where(X_star[:,1] == ub[1])[0], N_b//4, replace=False)
    X_lb = [X_star[idx1,:],X_star[idx3,:]]
    X_ub = [X_star[idx2,:],X_star[idx4,:]]
    
    # collocation points
    X_f = lb + (ub-lb)*lhs(2, N_f)

    # exact observations
    idx = np.random.choice(X_star.shape[0], N_exact, replace=False)
    X_exact = X_star[idx, :]
    Y_exact = Y_star[idx, :]
    
    # just for correct formating:
    X0, Y0 = None, None
    
    return grid, X0, Y0, X_f, X_exact, Y_exact, X_lb, X_ub, boundary, X_star, Y_star

def structure_data_poisson(data, noise, N_b, N_f, N_exact):
    # bounds of data
    lb = np.array([0, 0]) # lower bound for [x1, x2]
    ub = np.array([np.pi, np.pi]) # upper bound for [x1, x2]
    boundary = np.vstack((lb, ub))
    
    x1 = data['x1'].flatten()[:,None]
    x2 = data['x2'].flatten()[:,None]
    Exact = add_noise(data['usol'], noise)
    X1, X2 = np.meshgrid(x1, x2)
    grid = [X1, X2]
    X_star = np.hstack((X1.flatten()[:,None], X2.flatten()[:,None])) # for prediction
    Y_star = Exact.flatten()[:,None] # for prediction as ground truth
    
    # No initial data so turn to boundary data
    # select the points on the boundary from X_star: X_star[0, :] = lb[0] or lb[1] or ub[0] or ub[1]
    idx1 = np.random.choice(np.where(X_star[:,0] == lb[0])[0], N_b//4, replace=False)
    idx2 = np.random.choice(np.where(X_star[:,0] == ub[0])[0], N_b//4, replace=False)
    idx3 = np.random.choice(np.where(X_star[:,1] == lb[1])[0], N_b//4, replace=False)
    idx4 = np.random.choice(np.where(X_star[:,1] == ub[1])[0], N_b//4, replace=False)
    X_lb = [X_star[idx1,:],X_star[idx3,:]] # all 0 at the boundary points so no Y_b recorded
    X_ub = [X_star[idx2,:],X_star[idx4,:]]
    
    # collocation points
    X_f = lb + (ub-lb)*lhs(2, N_f)
    
    # exact observations
    idx = np.random.choice(X_star.shape[0], N_exact, replace=False)
    X_exact = X_star[idx,:]
    Y_exact = Y_star[idx,:]
    
    # just for correct formating:
    X0, Y0 = None, None

    return grid, X0, Y0, X_f, X_exact, Y_exact, X_lb, X_ub, boundary, X_star, Y_star

def structure_data_poissonHD(noise, N_b, N_f, N_exact):
    # bounds of data
    lb = np.zeros(10) # lower bound for [x1, x2, x3, ..., x10]
    ub = np.ones(10) # upper bound for [x1, x2, x3, ..., x10]
    boundary = np.vstack((lb, ub))
    
    def HD_poisson(X):
        return X[:,0]**2 - X[:,1]**2 + X[:,2]**2 - X[:,3]**2 + X[:,4]*X[:,5] + X[:,6]*X[:,7]*X[:,8]*X[:,9]
    
    for i in range(10):
        X_temp = lb[-i] + (ub[-i]-lb[-i])*lhs(9, N_b//10)
        X_lb = np.insert(X_temp, i, lb[i], axis=1)
        X_ub = np.insert(X_temp, i, ub[i], axis=1)
        if i == 0:
            X_b = np.vstack((X_lb, X_ub))
        else:
            X_b = np.vstack((X_b, X_lb, X_ub))
    
    Y_b = HD_poisson(X_b)[:,None]
    
    # collocation points
    X_f = lb + (ub-lb)*lhs(10, N_f)
    
    # exact observations
    X_exact = lb + (ub-lb)*lhs(10, N_exact)
    Y_exact = HD_poisson(X_exact)[:,None]
    
    # for prediction
    X_star = lb + (ub-lb)*lhs(10, 20000)
    Y_star = HD_poisson(X_star)[:,None]
    
    return X_b, Y_b, X_f, X_exact, Y_exact, boundary, X_star, Y_star

def structure_data_burgers(data, noise, N0, N_b, N_f, N_exact):
    # input 
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    lb = np.array([x.min(), t.min()]) # lower bound for [x, t]
    ub = np.array([x.max(), t.max()]) # upper bound for [x, t]
    Exact = add_noise(data['usol'], noise).T


    X, T = np.meshgrid(x,t)
    grid = [X, T]
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    Y_star = Exact.flatten()[:,None]

    #plot_with_ground_truth(mat, X_star, X, T, u_star, ground_truth_ref=False, ground_truth_refpts=[], filename = "ground_truth_comparison.png")

    # Initial and boundary data
    tb = t[np.random.choice(t.shape[0], N_b, replace=False),:] # random time samples for boundary points
    ti = np.zeros(N_b)

    # exact observations
    idx = np.random.choice(X_star.shape[0], N_exact, replace=False)
    X_exact = X_star[idx,:]
    Y_exact = Y_star[idx, :]


    # Collocation points
    X_f = lb + (ub-lb)*lhs(2, N_f)

    # initial points
    x0 = np.linspace(-1, 1, N0, endpoint = True)
    X0 = np.vstack((x0, np.zeros_like(x0, dtype=np.float32))).transpose() # (x, 0)
    Y0 = -np.sin(np.pi*X0)[:,0:1]
    # n = Y0.shape[0]
    # Y0.reshape(n,1)
    # boundary points
    boundary = np.vstack((lb, ub))
    X_lb = np.concatenate((lb[0]*np.ones_like(tb, dtype=np.float32), tb), axis=1)
    X_ub = np.concatenate((ub[0]*np.ones_like(tb, dtype=np.float32), tb), axis=1)
    # NOTE: added extra X, T for plotting
    return grid, X0, Y0, X_f, X_exact, Y_exact, X_lb, X_ub, boundary, X_star, Y_star