import numpy as np
import matplotlib.pyplot as plt
import sys
import plotparams
plotparams.buba()

#selection = '_eBOSS'
#selection = '_DESI'
selection = '_'+sys.argv[1]
snap_dir = '_55'

z_dic = {'_55': 0.8, '_47': 1.1, '_41': 1.4}

fontsize = 16
lw = 3.
want_log = 0
env_types = ['_low','_high']
lss = ['--','-']

bin_centers = np.load("data/bin_env_centers.npy")

# definitions for the axes
left, width = 0.13, 0.85
bottom, height = 0.1, 0.2
spacing = 0.005

rect_scatter = [left, bottom + (height + spacing), width, 0.65]
rect_histx = [left, bottom, width, height]


# start with a rectangular Figure
plt.figure(figsize=(9, 7))

ax_scatter = plt.gca()

for i in range(len(env_types)):
    env_type = env_types[i]
    ls = lss[i]
    env_string = ''.join(env_type.split('_'))+'est'
    Rat_colsfg_mean = np.load("data/rat_colsfg"+env_type+snap_dir+selection+"_mean.npy")
    Rat_colsfg_err = np.load("data/rat_colsfg"+env_type+snap_dir+selection+"_err.npy")
    #Rat_oiisfg_mean = np.load("data/rat_oiisfg"+env_type+snap_dir+selection+"_mean.npy")
    #Rat_oiisfg_err = np.load("data/rat_oiisfg"+env_type+snap_dir+selection+"_err.npy")
    Corr_mean_col = np.load("data/corr_mean"+env_type+snap_dir+selection+"_col.npy")
    Corr_err_col = np.load("data/corr_err"+env_type+snap_dir+selection+"_col.npy")
    Corr_mean_sfg = np.load("data/corr_mean"+env_type+snap_dir+selection+"_sfg.npy")
    Corr_err_sfg = np.load("data/corr_err"+env_type+snap_dir+selection+"_sfg.npy")
    #Corr_mean_flux = np.load("data/corr_mean"+env_type+snap_dir+selection+"_flux.npy")
    #Corr_err_flux = np.load("data/corr_err"+env_type+snap_dir+selection+"_flux.npy")
    
    # the scatter plot:
    if want_log:
        #ax_scatter.errorbar(bin_centers, Corr_mean_sfg,yerr=Corr_err_sfg,color='#CC6677',linestyle=ls,linewidth=lw,label="SFR-selected ("+env_string+")",alpha=1.,fmt='o',capsize=4)
        ax_scatter.errorbar(bin_centers*1.05, Corr_mean_col,yerr=Corr_err_col,color='dodgerblue',linestyle=ls,linewidth=lw,label="20\% "+env_string+" environment",alpha=1.,fmt='o',capsize=4)
        #ax_scatter.errorbar(bin_centers, Corr_mean_flux,yerr=Corr_err_flux,color='lawngreen',linestyle=ls,linewidth=lw,label="OII-emitting"+env_string)
        ax_scatter.set_ylabel(r"$\xi(r)$")#,fontsize=fontsize)
        ax_scatter.set_yscale('log')
    else:

        #ax_scatter.errorbar(bin_centers, Corr_mean_sfg*bin_centers**2,yerr=Corr_err_sfg*bin_centers**2,color='#CC6677',linestyle=ls,linewidth=lw,label="SFR-selected",fmt='o',capsize=4)
        ax_scatter.errorbar(bin_centers, Corr_mean_col*bin_centers**2,yerr=Corr_err_col*bin_centers**2,color='dodgerblue',linestyle=ls,linewidth=lw,label="20\% "+env_string+" environment",fmt='o',capsize=4)
        ax_scatter.set_ylabel(r"$\xi \ r^2$")#,fontsize=fontsize)
    ax_scatter.set_xlabel(r"$r \ [{\rm Mpc}/h]$")#,fontsize=fontsize)
    ax_scatter.set_xscale('log')
    ax_scatter.text(1.3,25.,r'${\rm '+(''.join(selection.split('_')))+",} \ z = %.1f$"%z_dic[snap_dir])#,fontsize=fontsize)
    
    # now determine nice limits by hand:
    ax_scatter.set_xlim([0.4,bin_centers[-1]+3])
    ax_scatter.set_xlim([1.,32.])
    if want_log:
        ax_scatter.set_ylim([1.e-2,1000.])
    else:
        ax_scatter.set_ylim([-15.,200.])
    ax_scatter.legend()

#plt.savefig('figs/corr_env'+snap_dir+selection+'.png')
plt.savefig('paper/corr_env'+snap_dir+selection+'.pdf')
plt.show()
plt.close()
