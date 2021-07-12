import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal, invwishart, invgamma
import warnings
warnings.filterwarnings('ignore')

import settings
from helper import lambda_to_Lambda

#--------------------------------CODE FOR ALL LATENT VARIABLES----------------

N = settings.N
def draw_z_ric(B, w, Psi, Lambda, Theta, I):
    intermediate = Lambda.T @ inv(Lambda @ Lambda.T + Theta)
    means = w.dot(B.T) + (I - w.dot((Lambda@B).T)).dot(intermediate.T)
    var = Psi - (Lambda.T) @ inv(Lambda @ Lambda.T + Theta) @ (Lambda)
    var = np.linalg.cholesky(var)
    draws = np.random.multivariate_normal(mean=np.zeros(means.shape[1]), cov=np.eye(var.shape[0]), size=N)
    draws = np.einsum('ij,jk->ij', draws, var)
    return draws+means

def draw_Theta(I, Lambda, z):
    R = 9
    alpha = 2 + N/2
    betas = 2 + 1/2*np.sum((I - (Lambda@z.T).T)**2, axis=0)
    return np.diag(1/np.random.gamma(alpha, 1/betas))

def draw_Psi():
    pass

# def draw_b_fast(V_b, mean_b, W, z, Psi):
#     Psi_inv = inv(Psi)
#     W_r = 1.0*W
#     W_r[:, 0] *= Psi_inv[0, 0]
#     W_r[:, 1] *= Psi_inv[1, 1]
#     z_r = 1.0*z
#     z_r[:, 0] *= Psi_inv[0, 0]
#     z_r[:, 1] *= Psi_inv[1, 1]
#     cov = inv(V_b) + np.einsum('ijk,ijl->kl', W, W_r)
#     cov = inv(cov)
#     mean = inv(V_b)@mean_b + np.einsum('ijk,ij->k', W, z_r)
#     mean = cov @ mean
#     return np.random.multivariate_normal(mean, cov)
    
def draw_b_fast(V_b, mean_b, W, z, Psi):
    Psi_inv = inv(Psi)
    W_r = 1.0*W
    W_r[:, 0] *= Psi_inv[0, 0]
    W_r[:, 1] *= Psi_inv[1, 1]
    z_r = 1.0*z
    z_r[:, 0] *= Psi_inv[0, 0]
    z_r[:, 1] *= Psi_inv[1, 1]
    # cov = inv(V_b) + np.einsum('ijk,ijl->kl', W, W_r)
    # cov = inv(cov)
    # mean = inv(V_b)@mean_b + np.einsum('ijk,ij->k', W, z_r)
    cov = inv(V_b) + np.einsum('ijk,ijl->kl', W, W)
    cov = inv(cov)
    mean = inv(V_b) @ mean_b + np.einsum('ijk,ij->k', W, z)
    mean = cov @ mean
    return np.random.multivariate_normal(mean, cov)
    
    
# def draw_Lambda(Theta, I, z, V_lambda, Lambdatrue):
#     inv_V_lambda = inv(V_lambda)
#     Theta = 1.0*np.diag(Theta)
#     R = 9
#     Lambda = np.zeros((9, 6))
#     term1 = np.einsum('ij,ik->jk', z, z)
#     for r in range(R):
#         ind = 2*(r//3)
#         var = inv(inv_V_lambda + term1[ind:ind+2, ind:ind+2]/Theta[r])
#         mean = np.einsum('ij,i->j', z[:, ind:ind+2], I[:, r])/Theta[r] @ var
#         Lambda[r, ind:ind+2] = np.random.multivariate_normal(mean, var)
#     # Lambda[0] = Lambdatrue[0]
#     # Lambda[6:9, 4:6] = 1.0*Lambda[3:6, 2:4]
#     return Lambda

col_ind = [0, 0, 1, 1, 2, 3, 3, 4, 5]
def draw_Lambda(Theta, I, z, V_lambda, Lambdatrue):
    inv_V_lambda = inv(V_lambda)
    Theta = 1.0*np.diag(Theta)
    R_dob = [0, 3, 6]
    R_single = [1, 2, 4, 5, 7, 8]
    Lambda = np.zeros((9, 6))
    term1 = np.einsum('ij,ik->jk', z, z)
    for r in R_dob:
        ind = 2*(r//3)
        var = inv(inv_V_lambda + term1[ind:ind+2, ind:ind+2]/Theta[r])
        mean = np.einsum('ij,i->j', z[:, ind:ind+2], I[:, r])/Theta[r] @ var
        Lambda[r, ind:ind+2] = np.random.multivariate_normal(mean, var)
    for r in R_single:
        ind = col_ind[r]
        var = 1/(1/100 + term1[ind,ind]/Theta[r])
        mean = np.sum(z[:, ind] * I[:, r])/Theta[r] * var
        Lambda[r, ind] = np.random.normal(mean, np.sqrt(var))
    return Lambda

def loglike(beta, tau, z):
    Beta = np.array([ASC[1], ASC[2], beta[0], beta[1]])
    uutils = np.zeros((N, 3))
    # for n in range(N):
    #     uutils[n] = x[n] @ Beta + tau @ z[n]
    utils1 = np.einsum('ijk,k->ij', x, Beta)
    utils2 = np.einsum('jk,ik->ij', tau, z)
    utils = utils1+utils2
    ref = np.tile(utils[np.arange(len(utils)), np.argmax(Y, axis=1)], (3, 1)).T
    utils -= ref
    
    utils = np.exp(utils)
    probs = np.log(1/np.sum(utils, 1))
    return probs

def tau_pdf(x):
    # mvn pdf in log form
    return -1/2 * 9.210340371976184 - 1/2 * (x @ x)/10000

def draw_tau(beta, mean_tau, V_tau, tau, z, curr_loglike):
    eta = np.random.normal(size=(3, 2))
    addend = eta @ (np.sqrt(settings.RHO_tau) * np.eye(2)).T
    accept_rate = 0
    comp = np.log(np.random.uniform(size=3)) # for comparison
    
    for j in [0, 1, 2]:
        tau_cand = np.copy(tau)
        tau_cand[j, 2*j:2*j+2] += addend[j]
        curr_loglike = loglike(beta, tau, z)
        cand_loglike = loglike(beta, tau_cand, z)
        curr_normal = tau_pdf(tau[j, 2*j:2*j+1]) # log mvn pdf
        cand_normal = tau_pdf(tau_cand[j, 2*j:2*j+1]) # log mvn pdf
        F = np.sum(cand_loglike - curr_loglike) + cand_normal - curr_normal
        if F >= comp[j]:
            tau = tau_cand
            curr_loglike = cand_loglike
            accept_rate += 1/2
            
    settings.RHO_tau = settings.RHO_tau - 0.0001 * (accept_rate < 0.3) + 0.0001 * (accept_rate > 0.3)
    return tau, curr_loglike
    

def draw_beta(beta, tau, z, curr_loglike, V_beta):
    eta = np.random.normal(size=2)
    addend = eta @ (np.sqrt(settings.RHO_beta) * np.linalg.cholesky(V_beta)).T #32.9 us
    
    beta_cand = beta + addend
    
    # conditional logit probs
    curr_loglike = loglike(beta, tau, z)
    cand_loglike = loglike(beta_cand, tau, z)
    
    # normal probs
    curr_normal = np.log(multivariate_normal.pdf(beta, np.zeros(2), V_beta)) 
    cand_normal = np.log(multivariate_normal.pdf(beta_cand, np.zeros(2), V_beta))
    
    r = np.sum(cand_loglike) + cand_normal - np.sum(curr_loglike) - curr_normal
    
    accept = np.log(np.random.uniform()) <= r
    if accept:
        beta = 1.0*beta_cand
        curr_loglike = cand_loglike
    
    settings.RHO_beta = settings.RHO_beta - 0.001 * (accept < 0.3) + 0.001 * (accept > 0.3)
    
    return beta, curr_loglike

RHO_z = 0.01
def draw_z_mh(z0, B, w, Psi, Lambda,Theta, I, beta, tau, curr_loglike):
    z = 1.0*z0 # copy
    
    # generate candidate
    eta = np.random.normal(size=(N, 6))
    addend = eta @ (np.sqrt(settings.RHO_z) * np.eye(6)).T
    z_cand = z + addend
    
    # calculate densities
    curr_loglike = loglike(beta, tau, z)
    cand_likelihood = loglike(beta, tau, z_cand)
    means_z = w.dot(B.T)
    means_I = z.dot(Lambda.T)
    means_I_cand = z_cand.dot(Lambda.T)
    Psi_inv = inv(Psi)
    Theta_inv = inv(Theta)
    
    t1 = z_cand - means_z
    cand_z_density = -0.5 * np.einsum('ij,ij->i', t1, t1*np.diag(Psi_inv))
    t2 = z - means_z
    curr_z_density = -0.5 * np.einsum('ij,ij->i', t2, t2*np.diag(Psi_inv))
    t3 = I - means_I_cand
    cand_I_density = -0.5 * np.einsum('ij,ij->i', t3, t3*np.diag(Theta_inv))
    t4 = I - means_I
    curr_I_density = -0.5 * np.einsum('ij,ij->i', t4, t4*np.diag(Theta_inv))
    
    F = (cand_likelihood - curr_loglike +
          cand_z_density - curr_z_density +
          cand_I_density - curr_I_density)
    
    # make updates
    idxAccept = np.log(np.random.uniform(size=N)) <= F
    z[idxAccept] = z_cand[idxAccept]
    curr_loglike[idxAccept] = cand_likelihood[idxAccept]
    
    # accept rate & rho
    accept_rate = np.mean(idxAccept)
    settings.RHO_z = settings.RHO_z - 0.0001 * (accept_rate < 0.4) + 0.0001 * (accept_rate > 0.4)
    
    return accept_rate, z, curr_loglike