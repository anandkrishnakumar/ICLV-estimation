import numpy as np


def B_to_b(B):
    x_indices = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    y_indices = [0, 0, 3, 1, 3, 1, 3, 2, 3, 2, 3]
    return np.array(B[x_indices, y_indices])

def b_to_B(b):
    B = np.zeros((6, 4))
    x_indices = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    y_indices = [0, 0, -1, 1, -1, 1, -1, 2, -1, 2, -1]
    B[x_indices, y_indices] = b
    return B

def Bchain_to_bchain(B_chain):
    chain_length = B_chain.shape[0]
    x_indices = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    y_indices = [0, 0, 3, 1, 3, 1, 3, 2, 3, 2, 3]
    b_chain = np.zeros((chain_length, 11))
    for sample in range(chain_length):
        b_chain[sample] = B_chain[sample, x_indices, y_indices]
    return b_chain

def Lambdachain_to_lambdachain(Lambda_chain):
    chain_length = Lambda_chain.shape[0]
    x_indices = [0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8]
    y_indices = [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5]
    # lambda_chain = np.zeros((chain_length, 12))
    # for sample in range(chain_length):
    #     lambda_chain[sample] = Lambda_chain[sample, x_indices, y_indices]
    lambda_chain = Lambda_chain[:, x_indices, y_indices]
    return lambda_chain

def Tauchain_to_tauchain(Tau_chain):
    chain_length = Tau_chain.shape[0]
    x_indices = [0, 0, 1, 1, 2, 2]
    y_indices = [0, 1, 2, 3, 4, 5]
    # tau_chain = np.zeros((chain_length, 3, 2))
    # for sample in range(chain_length):
    #     tau_chain[sample] = Tau_chain[sample, x_indices, y_indices].reshape((3, 2))
    tau_chain = Tau_chain[:, x_indices, y_indices].reshape(chain_length, 3, 2)
    return tau_chain

def Tau_to_tau(Tau):
    x_indices = [0, 0, 1, 1, 2, 2]
    y_indices = [0, 1, 2, 3, 4, 5]
    tau = Tau[x_indices, y_indices].reshape((3, 2))
    return tau

def w_to_W(w):
    row_indices = [0,1,1,2,2,3,3,4,4,5,5]
    col_indices = [0,1,2,3,4,5,6,7,8,9,10]
    N = w.shape[0]
    W = np.zeros((N, 6, 11))
    for n in range(N):
        li = w[n, [0,0,3,1,3,1,3,2,3,2,3]]
        W[n, row_indices, col_indices] = li
    return W

def lambda_to_Lambda(lambda_):
    Lambda = np.zeros((9, 6))
    li = lambda_[[0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]]
    row_indices = [0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8]
    col_indices = [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5]
    Lambda[row_indices, col_indices] = li 
    return Lambda

def Lambda_to_lambda(Lambda):
    row_indices = [0,0,1,2,3,3,4,5,6,6,7,8]
    col_indices = [0,1,0,1,2,3,2,3,4,5,4,5]
    return Lambda[row_indices, col_indices]