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
lw = 2.
want_log = 1
env_types = ['_low','_high']
lss = ['--','-']

bin_centers = np.load("data/bin_centers.npy")

# definitions for the axes
left, width = 0.13, 0.85
bottom, height = 0.1, 0.2
spacing = 0.005

rect_scatter = [left, bottom + (height + spacing), width, 0.65]
rect_histx = [left, bottom, width, height]


# start with a rectangular Figure
plt.figure(figsize=(9, 9))

ax_scatter = plt.axes(rect_scatter)
#(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='out')#(direction='in', labelbottom=False)

for i in range(len(env_types)):
    env_type = env_types[i]
    ls = lss[i]
    env_string = ' '.join(env_type.split('_'))
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
        ax_scatter.errorbar(bin_centers, Corr_mean_sfg,yerr=Corr_err_sfg,color='orange',linestyle=ls,linewidth=lw,label="star-forming"+env_string)
        ax_scatter.errorbar(bin_centers, Corr_mean_col,yerr=Corr_err_col,color='dodgerblue',linestyle=ls,linewidth=lw,label="color-selected"+env_string)
        #ax_scatter.errorbar(bin_centers, Corr_mean_flux,yerr=Corr_err_flux,color='lawngreen',linestyle=ls,linewidth=lw,label="OII-emitting"+env_string)
        ax_scatter.set_ylabel(r"$\xi(r)$")#,fontsize=fontsize)
        ax_scatter.set_yscale('log')
    else:
        ax_scatter.errorbar(bin_centers, Corr_mean_col*bin_centers**2,yerr=Corr_err_col*bin_centers**2,linewidth=lw,label="color-selected")
        ax_scatter.errorbar(bin_centers, Corr_mean_sfg*bin_centers**2,yerr=Corr_err_sfg*bin_centers**2,linewidth=lw,label="star-forming")
        ax_scatter.set_ylabel(r"$\xi \ r^2$")#,fontsize=fontsize)
    ax_scatter.set_xlabel(r"$r \ [{\rm Mpc}/h]$")#,fontsize=fontsize)
    ax_scatter.set_xscale('log')
    ax_scatter.text(0.3,1000.,r'${\rm '+(''.join(selection.split('_')))+",} \ z = %.1f$"%z_dic[snap_dir])#,fontsize=fontsize)
    
    # now determine nice limits by hand:
    ax_scatter.set_xlim([bin_centers[0]-0.01,bin_centers[-1]+1])
    #ax_scatter.set_ylim([3.e-1,2.e8])
    ax_scatter.legend()

    ax_histx.errorbar(bin_centers,Rat_colsfg_mean-1,yerr=Rat_colsfg_err,color='dodgerblue',linestyle=ls,linewidth=lw,label='col/sfg-1')
    #ax_histx.errorbar(bin_centers,Rat_oiisfg_mean-1,yerr=Rat_oiisfg_err,color='lawngreen',linestyle=ls,linewidth=lw,label='oii/sfg-1')
    ax_histx.plot(bin_centers,np.zeros(len(bin_centers)),'orange',ls='--',linewidth=lw)

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histx.set_xscale('log')
    ax_histx.set_xlabel(r'$r \ [{\rm Mpc}/h]$')#,fontsize=fontsize)
    ax_histx.set_ylabel(r'${\rm Frac. \ Diff.}$')#,fontsize=fontsize)

plt.savefig('figs/corr_env'+snap_dir+selection+'.png')
#plt.show()
plt.close()
