from importlib import reload
import dgp_ric; dgp_ric.dgp()  
import main_ric
import os
import time
import numpy as np
import settings
import shutil # to copy data files
from helper_ric import Lambda_to_lambda as L_to_l

# Gelman-Rubin diagnostic (Rhat)
def get_gelman(chain1, chain2):
    burn_in = int(settings.niter/2)
    length = burn_in
    n = chain1[burn_in:burn_in+length].shape[0]
    W = (chain1[burn_in:burn_in+length].std(0)**2 + chain2[burn_in:burn_in+length].std(0)**2)/2
    mean1 = chain1[burn_in:burn_in+length].mean(0)
    mean2 = chain2[burn_in:burn_in+length].mean(0)
    mean = (mean1 + mean2)/2
    B = n * ((mean1 - mean)**2 + (mean2 - mean)**2)
    var_theta = (1 - 1/n) * W + 1/n*B
    return np.sqrt(var_theta/W)

# Folder path to save samples
home = os.path.expanduser('~')
timestr = time.strftime("%Y%m%d-%H%M%S")
fpath = os.path.join(home, 'Documents\\ICLV Draws\\{}'.format(timestr))
# fpath = os.path.join(home, 'Documents\\ICLV Draws\\20210429-004551')
# if fpath does not exist, make directory
if not os.path.exists(fpath):
    os.makedirs(fpath)

def save(params, fpath, n, z_update):
    """Saves parameters as .npy files.
    params: dictionary of parameters
    fpath: folder path
    n: sample number
    z_update: type of z update
    """
    params_not_store = ["z1", "z2"]
    for param_name in params:
        if param_name not in params_not_store:
            foldpath = os.path.join(fpath, '{}_sample{}'.format(z_update, n))
            if not os.path.exists(foldpath):
                os.makedirs(foldpath)
            np.save(os.path.join(foldpath, '{}.npy'.format(param_name)), 
                    params[param_name])

def check_conv(params):
    """Check convergence of estimated parameters."""
    # need only check two variables
    mu_rhat = get_gelman(params["beta1"], params["beta2"])
    print(mu_rhat)
    mu_r = np.all(mu_rhat > 0.9) and np.all(mu_rhat < 1.4)
    b_rhat = get_gelman(params["b1"], params["b2"])
    print(b_rhat)
    b_r = np.all(b_rhat > 0.9) and np.all(b_rhat < 1.6)
    # lambda_rhat = get_gelman(L_to_l(params["Lambda1"]), L_to_l(params["Lambda2"]))
    # print(lambda_rhat)
    # lambda_r = np.all(lambda_rhat > 0.9) and np.all(lambda_rhat < 1.4)
    if mu_r and b_r: #Lambda_r:
        return True
    else:
        return False
    
# STORE TRUE VALS
data_files = ["B", "I", "Lambda", "Psi", "tau", "Theta", "z", "w", "beta"]
for file in data_files:
    orig = "{}.txt".format(file)
    targ = os.path.join(fpath, '{}.txt'.format(file))
    shutil.copyfile(orig, targ)

n_samples = 100
n = 0
run = 1
err_rate = np.zeros(n_samples)
while n < n_samples:
    print("\nSample", n+1)
    err_rate[n] = dgp_ric.dgp()
    print("Data generated")
    params_mh = main_ric.run("mh")
    
    # check Rhat
    mh_converged = check_conv(params_mh)
    if mh_converged:
        params_ric = main_ric.run("ric")
        ric_converged = check_conv(params_ric)
        print("Converged:", mh_converged and ric_converged)
        
        # if converges, save chains
        if mh_converged and ric_converged:
            save(params_mh, fpath, n, "mh")
            save(params_ric, fpath, n, "ric")
            n += 1
        else:
            print("Non converged")
    else:
        print("Not converged")
    run += 1

conv_rates = np.array([n_samples, run])
print("\n")
print(n_samples/run, "fraction of samples converged")
print("\n{} samples successfully stored.".format(n_samples))
print("Average data generation error rate:", err_rate.mean())
np.savetxt(os.path.join(fpath, "error_rates.txt"), err_rate)
np.savetxt(os.path.join(fpath, "converged-vs-not"), conv_rates)
print("Save directory:", fpath)


