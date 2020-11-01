import numpy as np
import matplotlib.pyplot as plt
import sys
import plotparams
plotparams.buba()

selection = '_'+sys.argv[1]
#selection = '_eBOSS'
#selection = '_DESI'
snap_dir = '_%d'%(int(sys.argv[2]))

z_dic = {'_55': 0.8, '_47': 1.1, '_41': 1.4}

fontsize = 16
lw = 3.
want_log = 0

bin_centers = np.load("data/bin_centers.npy")
Rat_colsfg_mean = np.load("data/rat_colsfg"+snap_dir+selection+"_mean.npy")
Rat_colsfg_err = np.load("data/rat_colsfg"+snap_dir+selection+"_err.npy")
#Rat_oiisfg_mean = np.load("data/rat_oiisfg"+snap_dir+selection+"_mean.npy")
#Rat_oiisfg_err = np.load("data/rat_oiisfg"+snap_dir+selection+"_err.npy")
Corr_mean_col = np.load("data/corr_mean"+snap_dir+selection+"_col.npy")
Corr_err_col = np.load("data/corr_err"+snap_dir+selection+"_col.npy")
Corr_mean_sfg = np.load("data/corr_mean"+snap_dir+selection+"_sfg.npy")
Corr_err_sfg = np.load("data/corr_err"+snap_dir+selection+"_sfg.npy")
#Corr_mean_flux = np.load("data/corr_mean"+snap_dir+selection+"_flux.npy")
#Corr_err_flux = np.load("data/corr_err"+snap_dir+selection+"_flux.npy")


# definitions for the axes
left, width = 0.13, 0.85
bottom, height = 0.1, 0.2
spacing = 0.005

rect_scatter = [left, bottom + (height + spacing), width, 0.65]
rect_histx = [left, bottom, width, height]


# start with a rectangular Figure
plt.figure(figsize=(9, 9.4))

ax_scatter = plt.axes(rect_scatter)
#(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='out')#(direction='in', labelbottom=False)

# the scatter plot:
if want_log:
    ax_scatter.errorbar(bin_centers, Corr_mean_sfg,yerr=Corr_err_sfg,color='#CC6677',linewidth=lw,label="SFR-selected")
    ax_scatter.errorbar(bin_centers, Corr_mean_col,yerr=Corr_err_col,color='dodgerblue',linewidth=lw,label="color-selected")
    #ax_scatter.errorbar(bin_centers, Corr_mean_flux,yerr=Corr_err_flux,color='lawngreen',linewidth=lw,label="OII-emitting")#"most stellar massive")
    ax_scatter.set_ylabel(r"$\xi(r)$")#,fontsize=fontsize)
    ax_scatter.set_yscale('log')
else:
    ax_scatter.errorbar(bin_centers, Corr_mean_sfg*bin_centers**2,yerr=Corr_err_sfg*bin_centers**2,linewidth=lw,color='#CC6677',label="SFR-selected ELG",ls='-',fmt='o',capsize=4)
    ax_scatter.errorbar(bin_centers, Corr_mean_col*bin_centers**2,yerr=Corr_err_col*bin_centers**2,linewidth=lw,color='dodgerblue',label="color-selected ELG",ls='-',fmt='o',capsize=4)
    #ax_scatter.errorbar(bin_centers, Corr_mean_flux*bin_centers**2,yerr=Corr_err_flux*bin_centers**2,linewidth=lw,color='lawngreen',label="OII-emitting")#"most stellar massive")
    ax_scatter.set_ylabel(r"$\xi(r) \ r^2$")#,fontsize=fontsize)
ax_scatter.set_xlabel(r"$r \ [{\rm Mpc}/h]$")#,fontsize=fontsize)
ax_scatter.set_xscale('log')

# now determine nice limits by hand:
ax_scatter.set_xlim([bin_centers[0]-0.01,bin_centers[-1]+1])
#ax_scatter.set_ylim([3.e-1,2.e8])
ax_scatter.legend(loc='upper right')
if want_log:
    ax_scatter.text(0.3,1000.,r'${\rm '+(''.join(selection.split('_')))+",} \ z = %.1f$"%z_dic[snap_dir])#,fontsize=fontsize)
else:
    if snap_dir == '_55' and selection == '_eBOSS':
        ax_scatter.text(0.2,50.,r'${\rm '+(''.join(selection.split('_')))+",} \ z = %.1f$"%z_dic[snap_dir])#,fontsize=fontsize)
    elif snap_dir == '_55' and selection == '_DESI':
        ax_scatter.text(0.2,42.,r'${\rm '+(''.join(selection.split('_')))+",} \ z = %.1f$"%z_dic[snap_dir])#,fontsize=fontsize)
    elif snap_dir == '_47':
        ax_scatter.text(0.2,90.,r'${\rm '+(''.join(selection.split('_')))+",} \ z = %.1f$"%z_dic[snap_dir])#,fontsize=fontsize)
    elif snap_dir == '_41':
        ax_scatter.text(0.2,170.,r'${\rm '+(''.join(selection.split('_')))+",} \ z = %.1f$"%z_dic[snap_dir])#,fontsize=fontsize)
    
line = np.linspace(0,30,10)
ax_histx.plot(line,np.zeros(len(line)),'black',ls='--',linewidth=lw)
ax_histx.errorbar(bin_centers,Rat_colsfg_mean-1,yerr=Rat_colsfg_err,color='dodgerblue',linestyle='-',linewidth=lw,alpha=1.,fmt='o',capsize=4)#,label='col/sfg-1')
#ax_histx.errorbar(bin_centers,Rat_oiisfg_mean-1,yerr=Rat_oiisfg_err,color='lawngreen',linewidth=lw,label='oii/sfg-1')

ax_histx.set_xlim(ax_scatter.get_xlim())
ax_histx.set_xscale('log')
ax_histx.set_xlabel(r'$r \ [{\rm Mpc}/h]$')#,fontsize=fontsize)
ax_histx.set_ylabel(r'${\rm Frac. \ Diff.}$')#,fontsize=fontsize)
#ax_histx.legend()
if snap_dir in ['_47','_41']:
    #ax_histx.set_yticks([-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.8,1.])
    ax_histx.set_yticks([-0.4,0.,0.4,0.8])
    #ax_histx.set_yticklabels(['$-0.6$','$-0.4$','$-0.2$','$0$','$0.2$','$0.4$','$0.6$','$0.8$','$1$'])
    ax_histx.set_yticklabels(['$-0.4$','$0$','$0.4$','$0.8$'])
    ax_histx.set_ylim([-0.64,1.02])
else:
    ax_histx.set_ylim([-0.32,0.22])
#plt.savefig('figs/corr'+snap_dir+selection+'.png')
plt.savefig('paper/corr'+snap_dir+selection+'.pdf')
#plt.show()
plt.close()
