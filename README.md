# GenerativePINNs

## General

This is an implementation for the Paper "Revisiting PINNs: Generative Adversarial Physics-informed Neural Networks and Point-weighting Method" by Li et al. from 2022. 

## How to use the PINN.py file

The file should work independent of dimensionality if you replace the equation in the ``_net_f()`` and ``boundary()`` functions. 

For Inputs, the sssumption is that the data is always passed as tensors with two axes. The first axis holds the number of samples as dimension, the second holds the dimensionality of the physical unit. Complex numbers are broken down into two real numbers and time is always the last entry in the x tensor. 

Thereby data would have the following dimensions if one has $100$ datapoints in each dimension including time:
- Burgers: ``y.shape=(100,1)`` and ``x.shape=(100,2)`` where ``t=x[:,1:2]``
- Poisson 2D: ``y.shape=(100,1)`` and ``x.shape=(100,2)``
- Helmholz 2D: ``y.shape=(100,1)`` and ``x.shape=(100,2)``
- Schodinger 1D: ``y.shape=(100,2)`` and ``x.shape=(100,2)`` where ``t=x[:,1:2]`` and ``y.imag=y[:,1:2]``
- Poisson 10D: ``y.shape=(10**10,1)`` and ``x.shape=(10**10,10)``
- Heat 2D: ``y.shape=(1000,1)`` and ``x.shape=(1000,3)`` where ``t=x[:,1:2]`` 

## Requirenments

This project runns in python 3.11.6
