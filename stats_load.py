import os
import settings; settings.init()
import numpy as np
import pandas as pd
import plotter
from helper_ric import Bchain_to_bchain, Lambdachain_to_lambdachain
from helper_ric import Lambda_to_lambda, Tauchain_to_tauchain, Tau_to_tau

# Gelman-Rubin diagnostic (Rhat)
def get_gelman(chain1, chain2, niter):
    burn_in = int(niter/2)
    length = burn_in
    n = chain1[burn_in:burn_in+length].shape[0]
    W = (chain1[burn_in:burn_in+length].std(0)**2 + chain2[burn_in:burn_in+length].std(0)**2)/2
    mean1 = chain1[burn_in:burn_in+length].mean(0)
    mean2 = chain2[burn_in:burn_in+length].mean(0)
    mean = (mean1 + mean2)/2
    B = n * ((mean1 - mean)**2 + (mean2 - mean)**2)
    var_theta = (1 - 1/n) * W + 1/n*B
    return np.sqrt(var_theta/W)


# enter folder path here (With saved chains)
fpath = r"C:\Users\anand\Documents\ICLV Draws\ricardo dgp, N1500, niter 3000"
fpath = os.path.normpath(fpath)

x_arr = ['txt', 'csv', 'pdf', 'png', 'pgf', 'eps']
sample_folders = [folder for folder in os.listdir(fpath) if all(x not in folder for x in x_arr)]
N_samples = max([int(folder.split('sample')[-1]) for folder in sample_folders]) + 1
print("{} samples found.".format(N_samples))


# Loading true values
betatrue = np.loadtxt(os.path.join(fpath, 'beta.txt'))
lambdatrue = Lambda_to_lambda(np.loadtxt(os.path.join(fpath, 'Lambda.txt')))
tautrue = Tau_to_tau(np.loadtxt(os.path.join(fpath, 'tau.txt')))
btrue = np.loadtxt(os.path.join(fpath, 'b.txt'))

true_params = {'beta': betatrue, 'lambda': lambdatrue,
               'b': btrue, 'tau': tautrue}

def do_stats(chain1, chain2, param):
    if param == "lambda":
        chain1 = Lambdachain_to_lambdachain(chain1)
        chain2 = Lambdachain_to_lambdachain(chain2)
    elif param == "tau":
        chain1 = Tauchain_to_tauchain(chain1)
        chain2 = Tauchain_to_tauchain(chain2)
    niter = chain1.shape[0]
    burn_in = int(-settings.niter/2)
    ch1 = chain1[burn_in:] # remove burn-in
    ch2 = chain2[burn_in:] # remove burn-in
    mean = (np.mean(ch1, 0) + np.mean(ch2, 0))/2
    std = (np.std(ch1, 0) + np.std(ch2, 0))/2
    median = (np.median(ch1, 0) + np.median(ch2, 0))/2
    quant1 = (np.quantile(ch1, 0.025, 0) + np.quantile(ch2, 0.025, 0))/2
    quant2 = (np.quantile(ch1, 0.975, 0) + np.quantile(ch2, 0.975, 0))/2
    if param in []: #['Lambda', 'tau']:
        abp = np.abs((mean.T - true_params[param])/true_params[param]) * 100
        count = ((quant1.T <= true_params[param]) & (quant2.T >= true_params[param])).astype("int")
    else:
        abp = np.abs((mean - true_params[param])/true_params[param]) * 100
        count = ((quant1 <= true_params[param]) & (quant2 >= true_params[param])).astype("int")
    # print("\n\n param", param)
    # print("mean \n", mean)
    # print("true \n", true_params[param])
    # print("abp \n", abp)
    # mean = np.mean(ch1, 0)
    # std = np.std(ch1, 0)
    # median = np.median(ch1, 0)
    # quant1 = np.quantile(ch1, 0.025, 0)
    # quant2 = np.quantile(ch1, 0.975, 0)
    rhat = get_gelman(chain1, chain2, niter)
    return (mean.flatten(), 
            std.flatten(),
            median.flatten(),
            quant1.flatten(),
            quant2.flatten(),
            abp.flatten(),
            rhat.flatten(),
            count.flatten())
            # np.mean(rhat[(0.9 < rhat) * (rhat < 1.1)]).flatten())


params = ["beta", "lambda", "b", "tau"]
params_n_elts = {"beta": 2,
                 "lambda": 12,
                 "b": 11,
                 "tau": 3*2}
mh_param_stats = dict((key, np.zeros(shape=(8, params_n_elts[key], N_samples))) 
                      for key in params) # empty arrays to store stats
# MEANING:
# 8 stats
# number of elements in param
# number of samples
ric_param_stats = dict((key, np.zeros(shape=(8, params_n_elts[key], N_samples))) 
                      for key in params) 

# start iterating over saved samples
for n in range(N_samples): # for each samples
    print(n)
    mh_sample_path = os.path.join(fpath, 'mh_sample{}'.format(n))
    ric_sample_path = os.path.join(fpath, 'ric_sample{}'.format(n))
    for param in params:
        mh_chain1 = np.load(os.path.join(mh_sample_path, "{}1.npy".format(param)))
        mh_chain2 = np.load(os.path.join(mh_sample_path, "{}2.npy".format(param)))
        ric_chain1 = np.load(os.path.join(ric_sample_path, "{}1.npy".format(param)))
        ric_chain2 = np.load(os.path.join(ric_sample_path, "{}2.npy".format(param)))
        xx = do_stats(mh_chain1, mh_chain2, param)
        mh_param_stats[param][0, :, n], mh_param_stats[param][1, :, n], mh_param_stats[param][2, :, n], mh_param_stats[param][3, :, n], mh_param_stats[param][4, :, n], mh_param_stats[param][5, :, n], mh_param_stats[param][6, :, n], mh_param_stats[param][7, :, n] = do_stats(mh_chain1, mh_chain2, param)
        # do_stats returns two arrays, each of length in params_n_elts
        ric_param_stats[param][0, :, n], ric_param_stats[param][1, :, n], ric_param_stats[param][2, :, n], ric_param_stats[param][3, :, n], ric_param_stats[param][4, :, n], ric_param_stats[param][5, :, n], ric_param_stats[param][6, :, n], ric_param_stats[param][7, :, n] = do_stats(ric_chain1, ric_chain2, param)
        
# EXAMPLE
# to get array of means of all samples for all elements of mu
print("\nExample")
print("Posterior means of beta")
print(mh_param_stats["beta"][0])
# FORMAT
# mu1 - sample1 post mean, sample2 post mean, ...
# mu2 - sample1 post mean, sample2 post mean, ...
# mu3 - sample1 post mean, sample2 post mean, ...
# mu4 - sample1 post mean, sample2 post mean, ...

df = pd.DataFrame()
# mean
df["Mean (mh)"] = [np.mean(mh_param_stats[param][0], axis=1) for param in params]
df["Mean (ric)"] = [np.mean(ric_param_stats[param][0], axis=1) for param in params]
# std
df["STD (mh)"] = [np.mean(mh_param_stats[param][1], axis=1) for param in params]
df["STD (ric)"] = [np.mean(ric_param_stats[param][1], axis=1) for param in params]
# median
df["Median (mh)"] = [np.mean(mh_param_stats[param][2], axis=1) for param in params]
df["Median (ric)"] = [np.mean(ric_param_stats[param][2], axis=1) for param in params]
# 2.5%
df["2.5% (mh)"] = [np.mean(mh_param_stats[param][3], axis=1) for param in params]
df["2.5% (ric)"] = [np.mean(ric_param_stats[param][3], axis=1) for param in params]
# 97.5%
df["97.5% (mh)"] = [np.mean(mh_param_stats[param][4], axis=1) for param in params]
df["97.5% (ric)"] = [np.mean(ric_param_stats[param][4], axis=1) for param in params]
# ABV
df["ABP (mh)"] = [np.mean(mh_param_stats[param][5], axis=1) for param in params]
df["ABP (ric)"] = [np.mean(ric_param_stats[param][5], axis=1) for param in params]
# rhat
df["Rhat (mh)"] = [np.mean(mh_param_stats[param][6], axis=1) for param in params]
df["Rhat (ric)"] = [np.mean(ric_param_stats[param][6], axis=1) for param in params]
# count (confidence)
df["MCount (mh)"] = [np.mean(mh_param_stats[param][7], axis=1) for param in params]
df["MCount (ric)"] = [np.mean(ric_param_stats[param][7], axis=1) for param in params]
df.index = params

print("\nMH beta mean of posterior means")
print(df["Mean (mh)"].loc["beta"])

print("\nRic beta mean of posterior means")
print(df["Mean (ric)"].loc["beta"])



# LOADING TRUE VALS
df["True"] = [[], [], [], []]
df["True"].loc["beta"] = betatrue
df["True"].loc["lambda"] = lambdatrue.flatten()
df["True"].loc["b"] = btrue.flatten()
df["True"].loc["tau"] = tautrue.flatten()


# Standard deviation of posterior means
df["SDPM (mh)"] = [[], [], [], []]
for param in params:
    df["SDPM (mh)"].loc[param] = np.std(mh_param_stats[param][0, :], axis=1)
df["SDPM (ric)"] = [[], [], [], []]
for param in params:
    df["SDPM (ric)"].loc[param] = np.std(ric_param_stats[param][0, :], axis=1)

###
# Calculating ABV value
###


my_df = pd.DataFrame(columns=df.columns.values)
beta_shp = [2]
lambda_shp = [12]
b_shp = [11]
tau_shp = [3,2]
# mu
for i in range(beta_shp[0]):
    my_df.loc["beta_{}".format(i+1)] = [df[col].loc["beta"][i] for col in df.columns]
    
# Lambda
for i in range(lambda_shp[0]):
    my_df.loc["lambda_{}".format(i+1)] = [df[col].loc["lambda"][i] for col in df.columns]
    
# b
for i in range(b_shp[0]):
    my_df.loc["b_{}".format(i+1)] = [df[col].loc["b"][i] for col in df.columns]
    
# tau
for i in range(np.product(tau_shp)):
    my_df.loc["tau_{}".format(i+1)] = [df[col].loc["tau"][i] for col in df.columns]


# my_df.to_csv(os.path.join(fpath, 'results.csv'), sep=',')

# plot posterior means
plotter.plot_hist(fpath, mh_param_stats["beta"][0], ric_param_stats["beta"][0], betatrue,
              mh_param_stats["lambda"][0], ric_param_stats["lambda"][0], lambdatrue, 
              mh_param_stats["b"][0], ric_param_stats["b"][0], btrue, 
              mh_param_stats["tau"][0], ric_param_stats["tau"][0], tautrue.flatten())