# -*- coding: utf-8 -*-

"""
settings.py

Includes the initialising of global variables.

"""
import numpy as np
import pandas as pd

def init():
    global niter
    niter = 3000
    
    global RHO_z, RHO_beta, RHO_tau
    RHO_z = 0.01
    RHO_beta = 0.003
    RHO_tau = 0.01
    
    global N
    Y = np.loadtxt("Y.txt")
    N = Y.shape[0]
    X = np.loadtxt("X.txt").reshape((N, 3, 2))