# GenerativePINNs

## General

This is an implementation for the Paper "Revisiting PINNs: Generative Adversarial Physics-informed Neural Networks and Point-weighting Method" by Li et al. from 2022. 

## How to use the PINN.py file

The file should work independent of dimensionality if you replace the equation in the ``_net_f()`` and ``boundary()`` functions. 

For Inputs, the asssumption is that the data is always passed as tensors with two axes. The first axis holds the number of samples as dimension, the second holds the dimensionality of the physical unit. Complex numbers are broken down into two real numbers and time is always the last entry in the x tensor. 

Thereby data would have the following dimensions if one has $100$ datapoints in each dimension including time:
- Burgers: ``y.shape=(100,1)`` and ``x.shape=(100,2)`` where ``t=x[:,1:2]``
- Poisson 2D: ``y.shape=(100,1)`` and ``x.shape=(100,2)``
- Helmholz 2D: ``y.shape=(100,1)`` and ``x.shape=(100,2)``
- Schodinger 1D: ``y.shape=(100,2)`` and ``x.shape=(100,2)`` where ``t=x[:,1:2]`` and ``y.imag=y[:,1:2]``
- Poisson 10D: ``y.shape=(10**10,1)`` and ``x.shape=(10**10,10)``
- Heat 2D: ``y.shape=(1000,1)`` and ``x.shape=(1000,3)`` where ``t=x[:,1:2]`` 

## Installation

This project runns in python 3.11.6

```
git clone https://github.com/Max-Hi/GenerativePINNs.git
cd GenerativePINNs; pip install -r requirements.txt
```

## Directory Structure

```
GenerativePINNs/
├── LICENSE
├── README.md
├── Data Generation
│   ├── simulating_heat.py
│   ├── simulating_poisson.py
│   └── simulating_helmholtz.py
├── Data
│   ├── burgers.mat
│   ├── poisson.mat
│   ├── helmholtz.mat
│   ├── schodinger.mat
│   └── heat.mat
├── Models
│   ├── Burgers_PINN.pth
│   ├── Poisson_PINN.pth
│   ├── Helmholtz_PINN.pth
│   ├── Schodinger_PINN.pth
│   └── Heat_PINN.pth
├── Plots
│   └── ...
├── Saves
│   └── ...
├── Data_structuring.py
├── PINN.py
├── PDE_PINN.py
├── Training.py
└── requirements.txt
```

## Model Training

**Manual training** can be done by running the ``Training.py`` file. The training can be done for different PDEs by changing the ``pde`` variable in the ``Training.py`` file. The trained models are saved in the ``Saves`` directory.

```
python Training.py

<!-- learning rate is chosen to be (0.001, 0.001, 0.005) because pde is not yet specified. Please use argparser to change.
-------------------------------------------
training  to learn  with standard architecture with GAN with PW 
-------------------------------------------
pde not recognised
? Which pde do you want to choose? (Use arrow keys)
 » burgers
   heat
   schroedinger
   poisson
   poissonHD
   helmholtz -->
```

**Automatic training** can be done by running the ``Training.py`` file with the parameters:

```
usage: Training.py [-h] [-p PDE] [-n NAME] [-e EPOCHS] [--lambda1 LAMBDA1] [--lambda2 LAMBDA2] [-g GAN] [-w POINTWEIGHTING] [-a ARCHITECTURE] [--lr1 LR1] [--lr2 LR2] [--lr3 LR3] [--e-value E_VALUE] [--noise NOISE] [--N_exact N_EXACT] [--N_b N_B]
```
Parameters:
- ``-p PDE``: The PDE to be learned. default ""
- ``-n NAME``: The name of the model to be saved. default ""
- ``-e EPOCHS``: The number of epochs to train the model. default 1000
- ``--lambda1 LAMBDA1``: The weight of the boundary loss. default 1.0
- ``--lambda2 LAMBDA2``: The weight of the exact point loss. default 1.0
- ``-g GAN``: Whether to use GAN or not. default True
- ``-w POINTWEIGHTING``: Whether to use point weighting or not. default True
- ``-a ARCHITECTURE``: The architecture of the model. default "standard"
- ``--lr1 LR1``: The learning rate of the first part of the model. default parameters from paper
- ``--lr2 LR2``: The learning rate of the second part of the model. default parameters from paper 
- ``--lr3 LR3``: The learning rate of the third part of the model. default parameters from paper
- ``--e-value E_VALUE``: The value of epsilon for the GAN. default 5e-4
- ``--noise NOISE``: The noise level for the GAN. default 0
- ``--N_exact N_EXACT``: The number of exact points to be used for training. default 40
- ``--N_b N_B``: The number of boundary points to be used for training. default 50
