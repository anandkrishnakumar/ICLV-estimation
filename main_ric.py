import settings; settings.init()
import drawer_ric
from drawer_ric import draw_z_ric, loglike, draw_z_mh, draw_beta
from drawer_ric import draw_b_fast, draw_Theta, draw_tau
from drawer_ric import draw_Lambda, draw_Psi
from helper_ric import b_to_B, B_to_b, w_to_W, lambda_to_Lambda, Lambda_to_lambda

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

#--------------------------------CODE FOR ALL LATENT VARIABLES----------------
def main(z_update):
    settings.init()
    niter = settings.niter # iterations
    # scroll down to find the actual loop in which simulation is done
    # following lines create empty arrays in which the draws will be stored
    
    I = np.loadtxt("I.txt")
    N = I.shape[0]
    Y = np.loadtxt("Y.txt")
    X = np.loadtxt("X.txt").reshape((N, 3, 2))
    x = np.zeros((N, 3, 4))
    aa = np.array([[0, 0], [1, 0], [0, 1]])
    for n in range(N):
        x[n] = np.concatenate([aa, X[n]], 1)
    ASC = np.loadtxt("ASC.txt")
    drawer_ric.N = N
    drawer_ric.Y = Y
    drawer_ric.X = X
    drawer_ric.x = x
    drawer_ric.ASC = ASC
    
    Theta = np.loadtxt("Theta.txt")
    Psi = np.loadtxt("Psi.txt")
    # Psi = np.eye(11)
    beta = np.loadtxt("beta.txt")
    tau = np.loadtxt("tau.txt")
    Lambda = np.loadtxt("Lambda.txt")
    B = np.loadtxt("B.txt")
    w = np.loadtxt("w.txt")
    W = w_to_W(w)
    
    mean_lambda = np.zeros((2))
    V_lambda = 100*np.eye(2)
    Lambda = np.zeros((niter+1, 9, 6))
    Lambda[0] = lambda_to_Lambda(np.random.rand(8))
    Lambdatrue = np.loadtxt("Lambda.txt")
    # Lambda[0, 0] = Lambdatrue[0]
    lambdatrue = Lambda_to_lambda(Lambdatrue)
    # Lambda = np.tile(Lambdatrue, (niter+1, 1, 1))
    
    alpha_T = 0.001
    theta_T = 0.001
    Theta = np.zeros((niter+1, 9, 9))
    Theta[0] = np.diag(np.random.uniform(0, 1, 9))
    Thetatrue = np.loadtxt("Theta.txt") 
    Theta = np.tile(Thetatrue, (niter+1, 1, 1))
    
    # mean_b = np.zeros((7))
    # V_b = 100*np.eye(7)
    # b = np.zeros((niter+1, 7))
    # b[0] = np.random.rand(7)
    # Btrue = np.loadtxt("B.txt")
    # btrue = B_to_b(Btrue)
    # # b = np.tile(btrue, (niter+1, 1))
    
    mean_b = np.zeros((11))
    V_b = 100*np.eye(11)
    b = np.zeros((niter+1, 11))
    b[0] = np.random.rand(11)
    Btrue = np.loadtxt("B.txt")
    btrue = B_to_b(Btrue)
    # b = np.tile(btrue, (niter+1, 1))
    
    # beta
    V_beta = 100*np.eye(2)
    mean_beta = np.zeros((2))
    beta = np.zeros((niter+1, 2))
    beta[0] = np.random.rand(2)
    betatrue = np.loadtxt("beta.txt")
    # beta = np.tile(betatrue, (niter+1, 1))
    
    # tau
    mean_tau = np.zeros((2))
    V_tau = 100*np.eye(2)
    tau = np.zeros((niter+1, 3, 6))
    tau[0, 0, [0, 1]] = 0 # np.random.rand(2)
    tau[0, 1, [2, 3]] = np.random.rand(2)
    tau[0, 2, [4, 5]] = np.random.rand(2)
    tautrue = np.loadtxt("tau.txt")
    # tau = np.tile(tautrue, (niter+1, 1, 1))
    
    
    z = np.zeros((niter+1, N, 6))
    ztrue = np.loadtxt("z.txt")
    # z = np.tile(ztrue, (niter+1, 1, 1))
    
    curr_loglike = loglike(beta[0], tau[0], z[0])
    start_time = time.time()
    for i in range(1, niter+1):
        if (i%200==0):
            print(i, "/", niter)
        # Ricardo's update
        if z_update == "ric":
            z[i] = draw_z_ric(b_to_B(b[i-1]), w, Psi, Lambda[i-1], Theta[i-1], I)
        # MH update
        else: 
            _, z[i], curr_loglike = draw_z_mh(z[i-1], b_to_B(b[i-1]), w, Psi, Lambda[i-1], Theta[i-1], I, beta[i-1], tau[i-1], curr_loglike)
        
        b[i] = draw_b_fast(V_b, mean_b, W, z[i], Psi)
        # Theta[i] = draw_Theta(I, Lambda[i-1], z[i])
        Lambda[i] = draw_Lambda(Theta[i], I, z[i], V_lambda, Lambdatrue)
        beta[i], curr_loglike = draw_beta(beta[i-1], tautrue, z[i], curr_loglike, V_beta)
        tau[i], curr_loglike = draw_tau(beta[i], mean_tau, V_tau, tau[i-1], z[i], curr_loglike)
    print("Time taken:", time.time() - start_time, "s")
    z_est = np.mean(z[-int(niter/2):], 0)
    beta_est = np.mean(beta[-int(niter/2):], 0)
    b_est = np.mean(b[-int(niter/2):], 0)
    Theta_est = np.mean(Theta[-int(niter/2):], 0)
    tau_est = np.mean(tau[-int(niter/2):], 0)
    Lambda_est = np.mean(Lambda[-int(niter/2):], 0)
    lambda_est = Lambda_to_lambda(Lambda_est)
    
    print("\nb bias")
    print(b_est - btrue)
    print("\nTheta bias")
    print(np.diag(Theta_est) - np.diag(Thetatrue))
    print("\nlambda bias")
    print(lambda_est - lambdatrue)
    print("\nbeta bias")
    print(beta_est - betatrue)
    print("\ntau bias")
    print(tau_est - tautrue)
    
    return b, Lambda, beta, tau

def run(z_update):
    dic = {}
    import drawer_ric
    dic["b1"], dic["Lambda1"], dic["beta1"], dic["tau1"] = main(z_update)
    import drawer_ric
    dic["b2"], dic["Lambda2"], dic["beta2"], dic["tau2"] = main(z_update)
    
    return dic