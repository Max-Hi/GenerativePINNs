import numpy as np
from pyDOE import lhs


def structure_data(pde, data, noise, N0, N_b, N_f):
    match pde:
        case "burgers":
            return structure_data_burgers(data, noise, N0, N_b, N_f)
        case "heat":
            return structure_data_heat(data, noise, N0, N_b, N_f)
        case "schroedinger":
            return structure_data_schroedinger(data, noise, N0, N_b, N_f)
        case "poisson":
            return structure_data_poisson(data, noise, N0, N_b, N_f)
        case "poissonHD":
            return structure_data_poissonHD(data, noise, N0, N_b, N_f)
        case "helmholtz":
            return structure_data_helmholtz(data, noise, N0, N_b, N_f)
        case _:
            print("pde not recognised")


####################### Data structuring Functions #########################

def structure_data_schroedinger(data, noise, N0, N_b, N_f):
    #bounds of data
    lb = np.array([-5.0, 0.0]) # lower bound for [x, t]
    ub = np.array([5.0, np.pi/2]) # upper bound for [x, t]

    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = data['uu']
    Exact_h = np.sqrt(np.real(Exact)**2 + np.imag(Exact)**2)
    X, T = np.meshgrid(x,t)
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

    # Training Samples with Y values
    n, m = len(x), len(t)
    k1, k2 = 20, 20  # Number of samples we want to draw

    idx_x = np.random.choice(n, k1, replace=False) 
    idx_t = np.random.choice(m, k2, replace=False) 
    # sample X
    sampled_x = x[idx_x]
    sampled_t = t[idx_t]
    mesh_x, mesh_t = np.meshgrid(sampled_x, sampled_t, indexing='ij')
    X_t = np.hstack((mesh_x.flatten()[:,None], mesh_t.flatten()[:,None]))
    # sample Y
    mesh_idx_x, mesh_idx_t = np.meshgrid(idx_x, idx_t, indexing='ij')
    # Use the mesh to index into u
    Y_t = np.hstack((np.real(Exact[mesh_idx_x, mesh_idx_t]).flatten()[:,None],np.imag(Exact[mesh_idx_x, mesh_idx_t]).flatten()[:,None]))
    
    return X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, X_star, [u_star, v_star, h_star]

def structure_data_heat(data, noise, N0, N_b, N_f):
    pass

def structure_data_helmholtz(data, noise, N0, N_b, N_f):
    pass

def structure_data_poisson(data, noise, N0, N_b, N_f):
    pass

def structure_data_poissonHD(data, noise, N0, N_b, N_f):
    pass

def structure_data_burgers(data, noise, N0, N_b, N_f):
    # input 
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    lb = np.array([x.min(), t.min()]) # lower bound for [x, t]
    ub = np.array([x.max(), t.max()]) # upper bound for [x, t]
    Exact = data['usol'].T


    X, T = np.meshgrid(x,t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    Y_star = Exact.flatten()[:,None]

    #plot_with_ground_truth(mat, X_star, X, T, u_star, ground_truth_ref=False, ground_truth_refpts=[], filename = "ground_truth_comparison.png")

    # Initial and boundary data
    tb = t[np.random.choice(t.shape[0], N_b, replace=False),:] # random time samples for boundary points
    ti = np.zeros(N_b)

    # exact observations
    idx = np.random.choice(X_star.shape[0], N0, replace=False)
    X_t = X_star[idx,:]
    
    Y_t = Y_star[idx, :]


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
    return X, T, X0, Y0, X_f, X_t, Y_t, X_lb, X_ub, boundary, X_star, Y_star