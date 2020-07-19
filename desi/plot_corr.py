import numpy as np
import matplotlib.pyplot as plt
import sys

selection = '_'+sys.argv[1]
#selection = '_eBOSS'
#selection = '_DESI'
snap_dir = '_55'

fontsize = 16
lw = 2.
want_log = 1

bin_centers = np.load("data/bin_centers.npy")
Rat_colsfg_mean = np.load("data/rat_colsfg"+snap_dir+selection+"_mean.npy")
Rat_colsfg_err = np.load("data/rat_colsfg"+snap_dir+selection+"_err.npy")
Rat_oiisfg_mean = np.load("data/rat_oiisfg"+snap_dir+selection+"_mean.npy")
Rat_oiisfg_err = np.load("data/rat_oiisfg"+snap_dir+selection+"_err.npy")
Corr_mean_col = np.load("data/corr_mean"+snap_dir+selection+"_col.npy")
Corr_err_col = np.load("data/corr_err"+snap_dir+selection+"_col.npy")
Corr_mean_sfg = np.load("data/corr_mean"+snap_dir+selection+"_sfg.npy")
Corr_err_sfg = np.load("data/corr_err"+snap_dir+selection+"_sfg.npy")
#Corr_mean_flux = np.load("data/corr_mean"+snap_dir+selection+"_flux.npy")
#Corr_err_flux = np.load("data/corr_err"+snap_dir+selection+"_flux.npy")


# definitions for the axes
left, width = 0.1, 0.85
bottom, height = 0.1, 0.2
spacing = 0.005

rect_scatter = [left, bottom + (height + spacing), width, 0.65]
rect_histx = [left, bottom, width, height]


# start with a rectangular Figure
plt.figure(figsize=(8, 8))

ax_scatter = plt.axes(rect_scatter)
#(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='out')#(direction='in', labelbottom=False)

# the scatter plot:
if want_log:
    ax_scatter.errorbar(bin_centers, Corr_mean_sfg,yerr=Corr_err_sfg,color='orange',linewidth=lw,label="star-forming")
    ax_scatter.errorbar(bin_centers, Corr_mean_col,yerr=Corr_err_col,color='dodgerblue',linewidth=lw,label="color-selected")
    #ax_scatter.errorbar(bin_centers, Corr_mean_flux,yerr=Corr_err_flux,color='lawngreen',linewidth=lw,label="OII-emitting")#"most stellar massive")
    ax_scatter.set_ylabel(r"$\xi(r)$",fontsize=fontsize)
    ax_scatter.set_yscale('log')
else:
    ax_scatter.errorbar(bin_centers, Corr_mean_sfg*bin_centers**2,yerr=Corr_err_sfg*bin_centers**2,linewidth=lw,color='orange',label="star-forming ELG")
    ax_scatter.errorbar(bin_centers, Corr_mean_col*bin_centers**2,yerr=Corr_err_col*bin_centers**2,linewidth=lw,color='dodgerblue',label="color-selected ELG")
    #ax_scatter.errorbar(bin_centers, Corr_mean_flux*bin_centers**2,yerr=Corr_err_flux*bin_centers**2,linewidth=lw,color='lawngreen',label="OII-emitting")#"most stellar massive")
    ax_scatter.set_ylabel(r"$\xi \ r^2$",fontsize=fontsize)
ax_scatter.set_xlabel(r"$r$",fontsize=fontsize)
ax_scatter.set_xscale('log')

# now determine nice limits by hand:
ax_scatter.set_xlim([bin_centers[0]-0.01,bin_centers[-1]+1])
#ax_scatter.set_ylim([3.e-1,2.e8])
ax_scatter.legend()
ax_scatter.text(2,1000.,''.join(selection.split('_')),fontsize=fontsize)

ax_histx.errorbar(bin_centers,Rat_colsfg_mean-1,yerr=Rat_colsfg_err,color='dodgerblue',linewidth=lw,label='col/sfg-1')
ax_histx.errorbar(bin_centers,Rat_oiisfg_mean-1,yerr=Rat_oiisfg_err,color='lawngreen',linewidth=lw,label='oii/sfg-1')
ax_histx.plot(bin_centers,np.zeros(len(bin_centers)),'k--',linewidth=lw)

ax_histx.set_xlim(ax_scatter.get_xlim())
ax_histx.set_xscale('log')
ax_histx.set_xlabel(r'$r$',fontsize=fontsize)
ax_histx.set_ylabel(r'Frac. Diff.',fontsize=fontsize)
ax_histx.legend()
plt.savefig('figs/corr'+snap_dir+selection+'.png')
#plt.show()
plt.close()
