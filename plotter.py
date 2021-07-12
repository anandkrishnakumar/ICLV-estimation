import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.backends.backend_pdf # for pdf
# import matplotlib; matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
import seaborn as sns; sns.set()
sns.set_context('paper', font_scale=4)
# plt.style.use("ggplot"); import tikzplotlib
# from pandas.io import clipboard
import os

# from labellines import labelLine, labelLines

def reject_outliers(data, m=1):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def plot_trace(mu, mutrue, 
               Sigma, Sigmatrue, 
               b, btrue, 
               Lambda, Lambdatrue, 
               tau, tautrue, 
               niter):
    
    fig, ax = plt.subplots(4, 2, True)
    fig.set_size_inches(16, 7)
    
    
    for i in range(mu.shape[0]):
        ax[0, 0].plot(range(int(niter/2), niter+1), mu[int(niter/2):, i])
        ax[0, 0].axhline(y = mutrue[i],
                         color = ax[0, 0].lines[-1].get_color(),
                         label = "mu " + str(i))

    # labelLines(ax[0, 0].get_lines(),zorder=2.5)

    for i in range(4):
        l1 = int((i+1)/2)
        l2 = (i+1) % 2
        for j in range(4):
            ax[l1, l2].plot(range(int(niter/2), niter+1),
                            Sigma[int(niter/2):, i, j])
            ax[l1, l2].axhline(y = Sigmatrue[i, j],
                               color = ax[l1, l2].lines[-1].get_color(),
                               label="Sigma "+str(i)+","+str(j))
        # labelLines(ax[l1, l2].get_lines(),zorder=2.5)
            
    for i in range(b.shape[-1]):
        ax[2, 1].plot(range(int(niter/2), niter+1), b[int(niter/2):, i])
        ax[2, 1].axhline(y = btrue[i], 
                         color = ax[2, 1].lines[-1].get_color(),
                         label = "b " + str(i))
        # labelLines(ax[2, 1].get_lines(),zorder=2.5)

    
    for i in range(Lambda.shape[-2]):
        ax[3, 0].plot(range(int(niter/2), niter+1), Lambda[int(niter/2):, i])
        ax[3, 0].axhline(y = Lambdatrue[i], 
                         color = ax[3, 0].lines[-1].get_color(),
                         label = "Lambda " + str(i))
        # labelLines(ax[3, 0].get_lines(),zorder=2.5)

    for i in range(tau.shape[-2]):
        ax[3, 1].plot(range(int(niter/2), niter+1), tau[int(niter/2):, i])
        ax[3, 1].axhline(y = tautrue[i], 
                         color = ax[3, 1].lines[-1].get_color(),
                         label = "tau " + str(i))
        # labelLines(ax[3, 1].get_lines(),zorder=2.5)

    
    plt.tight_layout()
        
    pdf = matplotlib.backends.backend_pdf.PdfPages("Results/plots.pdf")
    for fig in range(1, 2):
        pdf.savefig(fig)
    pdf.close()   
    ####################
    
    
    
    
    
    
    
def plot_hist(fpath, beta_mh, beta_ric, betatrue, 
              lambda_mh, lambda_ric, lambdatrue, 
              b_mh, b_ric, btrue, 
              tau_mh, tau_ric, tautrue):
    
    beta_hist, ax = plt.subplots(1, 2)
    beta_hist.set_size_inches(16, 7)
    plt.tight_layout()
    bins = np.linspace(-3, 3, 100)
    for i in range(2):
        ax[i].hist(beta_mh[i], bins, alpha=0.7,  label="Metropolis-Hastings", color="blue")
        ax[i].hist(beta_ric[i], bins, alpha=0.7, label="Daziano and Bolduc (2013)", color="green")
        ax[i].axvline(x = betatrue[i], lw=6, label="True", linestyle='dotted', color="red")
        ax[i].set_xlim([betatrue[i]-1, betatrue[i]+1])
        ax[i].set_title(r'$\beta_{}$'.format(i+1))
    handles, labels = ax[0].get_legend_handles_labels()
    # beta_hist.legend(handles, labels, loc='lower center')
    beta_hist.savefig(os.path.join(fpath, 'beta.pdf'))
    
    for i in range(4):
        lambda_hist, ax = plt.subplots(1, 3)
        lambda_hist.set_size_inches(24, 7)
        plt.tight_layout()
        ax[(3*i)%3].hist(lambda_mh[3*i], bins, alpha=0.7, color="blue")
        ax[(3*i)%3].hist(lambda_ric[3*i], bins, alpha=0.7, color="green")
        ax[(3*i)%3].axvline(x = lambdatrue[3*i], lw=6, linestyle='dotted', color="red")
        ax[(3*i)%3].set_xlim([lambdatrue[3*i]-1, lambdatrue[3*i]+1])
        ax[(3*i)%3].set_title(r'$\lambda_{' + str(3*i + 1) + r'}$')
        
        ax[(3*i+1)%3].hist(lambda_mh[3*i + 1], bins, alpha=0.7, color="blue")
        ax[(3*i+1)%3].hist(lambda_ric[3*i + 1], bins, alpha=0.7, color="green")
        ax[(3*i+1)%3].axvline(x = lambdatrue[3*i + 1], lw=6, linestyle='dotted', color="red")
        ax[(3*i+1)%3].set_xlim([lambdatrue[3*i + 1]-1, lambdatrue[3*i + 1]+1])
        ax[(3*i+1)%3].set_title(r'$\lambda_{' + str(3*i + 2) + r'}$')
        
        ax[(3*i+2)%3].hist(lambda_mh[3*i + 2], bins, alpha=0.7, color="blue")
        ax[(3*i+2)%3].hist(lambda_ric[3*i + 2], bins, alpha=0.7, color="green")
        ax[(3*i+2)%3].axvline(x = lambdatrue[3*i + 2], lw=6, linestyle='dotted', color="red")
        ax[(3*i+2)%3].set_xlim([lambdatrue[3*i + 2]-1, lambdatrue[3*i + 2]+1])
        ax[(3*i+2)%3].set_title(r'$\lambda_{' + str(3*i + 3) + r'}$')
        
        # if i == 3:
        #     lambda_hist.legend(handles, labels, loc='lower center')
        lambda_hist.savefig(os.path.join(fpath, 'lambda_{}.pdf'.format(i)))
    
    
    for i in range(3):
        b_hist, ax = plt.subplots(1, 3)
        b_hist.set_size_inches(24, 7)
        plt.tight_layout()
        ax[(3*i)%3].hist(b_mh[3*i], bins, alpha=0.7, color="blue")
        ax[(3*i)%3].hist(b_ric[3*i], bins, alpha=0.7, color="green")
        ax[(3*i)%3].axvline(x = btrue[3*i], lw=6, linestyle='dotted', color="red")
        ax[(3*i)%3].set_xlim([btrue[3*i]-1, btrue[3*i]+1])
        ax[(3*i)%3].set_title(r'$b_{' + str(3*i + 1) + r'}$')
        
        ax[(3*i+1)%3].hist(b_mh[3*i + 1], bins, alpha=0.7, color="blue")
        ax[(3*i+1)%3].hist(b_ric[3*i + 1], bins, alpha=0.7, color="green")
        ax[(3*i+1)%3].axvline(x = btrue[3*i + 1], lw=6, linestyle='dotted', color="red")
        ax[(3*i+1)%3].set_xlim([btrue[3*i + 1]-1, btrue[3*i + 1]+1])
        ax[(3*i+1)%3].set_title(r'$b_{' + str(3*i + 2) + r'}$')
        
        ax[(3*i+2)%3].hist(b_mh[3*i + 2], bins, alpha=0.7, color="blue")
        ax[(3*i+2)%3].hist(b_ric[3*i + 2], bins, alpha=0.7, color="green")
        ax[(3*i+2)%3].axvline(x = btrue[3*i + 2], lw=6, linestyle='dotted', color="red")
        ax[(3*i+2)%3].set_xlim([btrue[3*i + 2]-1, btrue[3*i + 2]+1])
        ax[(3*i+2)%3].set_title(r'$b_{' + str(3*i + 3) + r'}$')
        b_hist.savefig(os.path.join(fpath, 'b_{}.pdf'.format(i)))
        
    
    b_hist, ax = plt.subplots(1, 2)
    b_hist.set_size_inches(16, 7)
    plt.tight_layout()
    i = 3
    ax[(3*i)%3].hist(b_mh[3*i], bins, alpha=0.7, color="blue")
    ax[(3*i)%3].hist(b_ric[3*i], bins, alpha=0.7, color="green")
    ax[(3*i)%3].axvline(x = btrue[3*i], lw=6, linestyle='dotted', color="red")
    ax[(3*i)%3].set_xlim([btrue[3*i]-1, btrue[3*i]+1])
    ax[(3*i)%3].set_title(r'$b_{' + str(3*i + 1) + r'}$')
    
    ax[(3*i+1)%3].hist(b_mh[3*i + 1], bins, alpha=0.7, color="blue")
    ax[(3*i+1)%3].hist(b_ric[3*i + 1], bins, alpha=0.7, color="green")
    ax[(3*i+1)%3].axvline(x = btrue[3*i + 1], lw=6, linestyle='dotted', color="red")
    ax[(3*i+1)%3].set_xlim([btrue[3*i + 1]-1, btrue[3*i + 1]+1])
    ax[(3*i+1)%3].set_title(r'$b_{' + str(3*i + 2) + r'}$')
    b_hist.savefig(os.path.join(fpath, 'b_{}.pdf'.format(i)))

    for i in range(3):
        tau_hist, ax = plt.subplots(1, 2)
        tau_hist.set_size_inches(16, 7)
        plt.tight_layout()
        ax[0].hist(tau_mh[2*i], bins, alpha=0.7, color="blue")
        ax[0].hist(tau_ric[2*i], bins, alpha=0.7, color="green")
        ax[0].axvline(x = tautrue[2*i], lw=6, linestyle='dotted', color="red")
        ax[0].set_xlim([tautrue[2*i]-1, tautrue[2*i]+1])
        ax[0].set_title(r'$\tau_{' + str(2*i + 1) + r'}$')
        
        ax[1].hist(tau_mh[2*i + 1], bins, alpha=0.7, color="blue")
        ax[1].hist(tau_ric[2*i + 1], bins, alpha=0.7, color="green")
        ax[1].axvline(x = tautrue[2*i + 1], lw=6, linestyle='dotted', color="red")
        ax[1].set_xlim([tautrue[2*i + 1]-1, tautrue[2*i + 1]+1])
        ax[1].set_title(r'$\tau_{' + str(2*i + 2) + r'}$')
        
        # if i == 2:
        #     tau_hist.legend(handles, labels, loc='lower center')
        tau_hist.savefig(os.path.join(fpath, 'tau_{}.pdf'.format(10*(2*i+1) + 2*i+2)))
        
    return None
    
    # ONLY LEGEND
    # import pylab
    # figlegend = pylab.figure(figsize=(8,3))
    # figlegend.legend(handles, labels, 'center')
    # figlegend.show()
    # figlegend.savefig(os.path.join(fpath, 'legend.pdf'))
    
    # saving as tikz
    # copy(Sigma_12_tikz)
    # copy(Sigma_34_tikz)
    # copy(Lambda_tikz)
    # copy(b_tikz)
    # copy(tau_tikz)
    

    # saving as png
    # mu_hist.savefig(os.path.join(fpath, 'mu.png'), dpi=300)
    # Sigma_hist.savefig(os.path.join(fpath, 'Sigma.png'), dpi=300)
    # Sigma_diagonal.savefig(os.path.join(fpath, 'Sigma_diag.png'), dpi=300)
    # Sigma_off.savefig(os.path.join(fpath, 'Sigma_off.png'), dpi=300)
    # Lambda_hist.savefig(os.path.join(fpath, 'Lambda.png'), dpi=300)
    # b_hist.savefig(os.path.join(fpath, 'b.png'), dpi=300)
    # tau_hist.savefig(os.path.join(fpath, 'tau.png'), dpi=300)
    
    # saving as pgf
    # mu_hist.savefig(os.path.join(fpath, 'mu.pgf'))
    # Sigma_hist.savefig(os.path.join(fpath, 'Sigma.pgf'))
    # Sigma_diagonal.savefig(os.path.join(fpath, 'Sigma_diag.pgf'))
    # Sigma_off.savefig(os.path.join(fpath, 'Sigma_off.pgf'))
    # Lambda_hist.savefig(os.path.join(fpath, 'Lambda.pgf'))
    # b_hist.savefig(os.path.join(fpath, 'b.pgf'))
    # tau_hist.savefig(os.path.join(fpath, 'tau.pgf'))
    
    # saving as pdf images
    # mu_hist.savefig(os.path.join(fpath, 'mu.pdf'))
    # Sigma_12.savefig(os.path.join(fpath, 'Sigma12.pdf'))
    # Sigma_34.savefig(os.path.join(fpath, 'Sigma34.pdf'))
    # Lambda_hist.savefig(os.path.join(fpath, 'Lambda.pdf'))
    # b_hist.savefig(os.path.join(fpath, 'b.pdf'))
    # tau_hist.savefig(os.path.join(fpath, 'tau.pdf'))
    
    # saving as pdf
    # import matplotlib.backends.backend_pdf
    # pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(fpath, "histogram.pdf"))
    # for j in range(1, 8):
    #     pdf.savefig(j)
    # pdf.close()
    
    
    #############################
def plot_histt(mu1, mu2, mutrue,
              Sigma1, Sigma2, Sigmatrue,
              lambda1, lambda2, lambdatrue,
              b1, b2, btrue,
              tau1, tau2, tautrue,
              z1, z2, ztrue):
    den = False
    line_height = 1000
    mu_hist, ax = plt.subplots(2, 2)
    plt.tight_layout()
    for i in range(2):
        for j in range(2):
            l = i*2 + j
            ax[i, j].hist(reject_outliers(mu1[:, l]), alpha=0.5,
                     color='blue', label="MH", density=den)
            ax[i, j].hist(reject_outliers(mu2[:, l]), alpha=0.5,
                     color='orange', label="Ricardo", density=den)
            ax[i, j].vlines(x=mutrue[l], ymin=0, ymax=line_height, color='black', label='True value')
            ax[i, j].set_xlim([mutrue[l]-1, mutrue[l]+1])
    handles, labels = ax[0, 0].get_legend_handles_labels()
    mu_hist.legend(handles, labels, loc='upper center')

    Sigma_hist1, ax = plt.subplots(2, 2)
    plt.tight_layout()
    for i in range(1):
        for j in range(4):
            l = i*4 + j
            ax[0+int(j/2), j%2].hist(reject_outliers(Sigma1[:, i, j]), alpha=0.5,
                     color='blue', label="MH", density=den)
            ax[0+int(j/2), j%2].hist(reject_outliers(Sigma2[:, i, j]), alpha=0.5,
                     color='orange', label="Ricardo", density=den)
            ax[0+int(j/2), j%2].vlines(x=Sigmatrue[i, j], ymin=0, ymax=line_height, color='black', label='True value')
            ax[0+int(j/2), j%2].set_xlim([Sigmatrue[i, j]-1, Sigmatrue[i, j]+1])
    handles, labels = ax[0, 0].get_legend_handles_labels()
    Sigma_hist1.legend(handles, labels, loc='upper center')    
    
    Sigma_hist2, ax = plt.subplots(2, 2)
    plt.tight_layout()
    for i in range(1, 2):
        for j in range(4):
            l = i*4 + j
            ax[0+int(j/2), j%2].hist(reject_outliers(Sigma1[:, i, j]), alpha=0.5,
                     color='blue', label="MH", density=den)
            ax[0+int(j/2), j%2].hist(reject_outliers(Sigma2[:, i, j]), alpha=0.5,
                     color='orange', label="Ricardo", density=den)
            ax[0+int(j/2), j%2].vlines(x=Sigmatrue[i, j], ymin=0, ymax=line_height, color='black', label='True value')
            ax[0+int(j/2), j%2].set_xlim([Sigmatrue[i, j]-1, Sigmatrue[i, j]+1])
    handles, labels = ax[0, 0].get_legend_handles_labels()
    Sigma_hist2.legend(handles, labels, loc='upper center')       
    
    Sigma_hist3, ax = plt.subplots(2, 2)
    plt.tight_layout()
    for i in range(2, 3):
        for j in range(4):
            l = i*4 + j
            ax[0+int(j/2), j%2].hist(reject_outliers(Sigma1[:, i, j]), alpha=0.5,
                     color='blue', label="MH", density=den)
            ax[0+int(j/2), j%2].hist(reject_outliers(Sigma2[:, i, j]), alpha=0.5,
                     color='orange', label="Ricardo", density=den)
            ax[0+int(j/2), j%2].vlines(x=Sigmatrue[i, j], ymin=0, ymax=line_height, color='black', label='True value')
            ax[0+int(j/2), j%2].set_xlim([Sigmatrue[i, j]-1, Sigmatrue[i, j]+1])
    handles, labels = ax[0, 0].get_legend_handles_labels()
    Sigma_hist3.legend(handles, labels, loc='upper center')        
    
    Sigma_hist4, ax = plt.subplots(2, 2)
    plt.tight_layout()
    for i in range(3, 4):
        for j in range(4):
            l = i*4 + j
            ax[0+int(j/2), j%2].hist(reject_outliers(Sigma1[:, i, j]), alpha=0.5,
                     color='blue', label="MH", density=den)
            ax[0+int(j/2), j%2].hist(reject_outliers(Sigma2[:, i, j]), alpha=0.5,
                     color='orange', label="Ricardo", density=den)
            ax[0+int(j/2), j%2].vlines(x=Sigmatrue[i, j], ymin=0, ymax=line_height, color='black', label='True value')
            ax[0+int(j/2), j%2].set_xlim([Sigmatrue[i, j]-1, Sigmatrue[i, j]+1])
    handles, labels = ax[0, 0].get_legend_handles_labels()
    Sigma_hist4.legend(handles, labels, loc='upper center')                
      
    b_hist, ax = plt.subplots(1, 2)
    plt.tight_layout()
    for i in range(2):
        ax[i].hist(reject_outliers(b1[:, i]), alpha=0.5,
                     color='blue', label="MH", density=den)
        ax[i].hist(reject_outliers(b2[:, i]), alpha=0.5,
                 color='orange', label="Ricardo", density=den)
        ax[i].vlines(x=btrue[i], ymin=0, ymax=line_height, color='black', label='True value')
        xmin = min(b2[:, i].mean()-0.5, b1[:, i].mean()-0.5, btrue[i]-1)
        xmax = max(b2[:, i].mean()+0.5, b1[:, i].mean()+0.5, btrue[i]+1)
        ax[i].set_xlim([xmin, xmax])
    handles, labels = ax[0].get_legend_handles_labels()
    b_hist.legend(handles, labels, loc='upper center')        
 
    Lambda_hist, ax = plt.subplots(1, 3)
    plt.tight_layout()
    for i in range(3):
        ax[i].hist(reject_outliers(lambda1[:, i]), alpha=1,
                     color='blue', label="MH", density=den)
        ax[i].hist(reject_outliers(lambda2[:, i]), alpha=1,
                 color='orange', label="Ricardo", density=den)
        ax[i].vlines(x=lambdatrue[i], ymin=0, ymax=line_height, color='black', label='True value')
        xmin = min(lambda2[:, i].mean()-0.5, lambda1[:, i].mean()-0.5, lambdatrue[i]-1)
        xmax = max(lambda2[:, i].mean()+0.5, lambda1[:, i].mean()+0.5, lambdatrue[i]+1)        
        ax[i].set_xlim([xmin, xmax])
    handles, labels = ax[0].get_legend_handles_labels()
    Lambda_hist.legend(handles, labels, loc='upper center')          

    tau_hist, ax = plt.subplots(1, 2)
    plt.tight_layout()
    for i in range(1, 3):
        ax[i-1].hist(reject_outliers(tau1[:, i]), alpha=0.5,
                     color='blue', label="MH", density=den)
        ax[i-1].hist(reject_outliers(tau2[:, i]), alpha=0.5,
                 color='orange', label="Ricardo", density=den)
        ax[i-1].vlines(x=tautrue[i], ymin=0, ymax=line_height, color='black', label='True value')
        xmin = min(tau2[:, i].mean()-0.5, tau1[:, i].mean()-0.5, tautrue[i]-1)
        xmax = max(tau2[:, i].mean()+0.5, tau1[:, i].mean()+0.5, tautrue[i]+1)
        ax[i-1].set_xlim([xmin, xmax])
    handles, labels = ax[0].get_legend_handles_labels()
    tau_hist.legend(handles, labels, loc='upper center')    
        
    pdf = matplotlib.backends.backend_pdf.PdfPages("Results/histograms.pdf")
    for fig in range(2, 10):
        pdf.savefig(fig)
    pdf.close()   
