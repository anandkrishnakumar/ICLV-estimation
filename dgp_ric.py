import numpy as np
import sklearn.datasets

#------------------------------DGP FOR ALL LATENT VARIABLES-------------------
def dgp():
    N = 1000
    # alternative specific constants
    ASC = np.array([0, -0.6, 0.8])
    beta = np.array([-1.2, 0.8])
    X = np.random.normal(size=(N, 3, 2))
            
    ### LATENT VARIABLES
    b = np.array([0.5, 0.8, 1.5, 0.3, 1.8, 0.5, 1.0])
    B = np.zeros((6, 4))
    li = b[[0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6]]
    row_indices = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    col_indices = [0, 0, 3, 1, 3, 1, 3, 2, 3, 2, 3]
    B[row_indices, col_indices] = li
    w = np.random.normal(size=(N, 4))
    w[:, -1] = np.random.binomial(1, 0.5, size=N)
    # first three are alternative-specific
    # fourth attribute is individual-specific
    Psi = np.eye(6)
    z = np.empty((N, 6))
    for n in range(N):
        z[n] = B @ w[n] + np.random.multivariate_normal(np.zeros(6), Psi)
    tau = np.zeros((3, 6))
    tau[0, :2] = [-0.4, 0.5] # 0 # np.random.normal(size=2)
    tau[1, 2:4] = [0.5, -0.6] # np.random.normal(size=2)
    tau[2, 4:] = [0.5, 0.6] # np.random.normal(size=2)
    
    ### INDICATORS
    Lambda = np.zeros((9, 6))
    lambda_ = np.array([0.7, 0.5, 1.0, 0.7, 0.5, 0.8, 1.0, 1.0])
    li = lambda_[[0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]]
    row_indices = [0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8]
    col_indices = [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5]
    Lambda[row_indices, col_indices] = li
    I = np.empty((N, 9))
    Theta = np.eye(9)
    for n in range(N):
        I[n] = Lambda @ z[n].flatten() + np.random.multivariate_normal(np.zeros(9), Theta)
        
    ### MAKING CHOICES
    Y = np.zeros((N, 3))
    U = np.zeros((N, 3))
    for n in range(N):
        for j in range(3):
            U[n, j] = ASC[j] + X[n, j] @ beta + tau[j] @ z[n]
            U[n, j] += np.random.gumbel(0, 1)
        Y[n, np.argmax(U[n])] = 1
    
    
    ### SAVING
    np.savetxt("z.txt", z)
    np.savetxt("Psi.txt", Psi)
    np.savetxt("B.txt", B)
    np.savetxt("Theta.txt", Theta)
    np.savetxt("Lambda.txt", Lambda)
    np.savetxt("beta.txt", beta)
    np.savetxt("tau.txt", tau)
    
    np.savetxt("I.txt", I)
    np.savetxt("w.txt", w)
    np.savetxt("Y.txt", Y)
    np.savetxt("X.txt", X.reshape(N, 3*2)) # reshaped because 3D arrays can't be saved
    np.savetxt("ASC.txt", ASC)
    
    return 0.25