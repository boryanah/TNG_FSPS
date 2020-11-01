import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import Corrfunc
from random import choices
from util import get_density, smooth_density, get_jack_corr, get_counts, get_hist
import plotparams
#plotparams.buba()
plotparams.default()
import sys

np.random.seed(3000)

selection = '_DESI'
#selection = '_eBOSS'

redshift_dict = {'055':['sdss_desi','_0.0'],#-0.5
                 '047':['sdss_desi','_-0.0'],
                 '041':['sdss_des','']}

# snapshot number and dust model
snap = '0%d'%(55)#'041'#'047'#'055'#'050'#'059'
snap_dir = '_'+str(int(snap))
env_type = ''

# directory of the halo TNG data
box_name = "TNG300"
Lbox = 205.
root = "/mnt/gosling1/boryanah/"+box_name+"/"

bin_edges = np.logspace(np.log10(0.007),np.log10(5.),21)
vol_edges = 4./3*np.pi*bin_edges**3
vol_bins = vol_edges[1:]-vol_edges[:-1]

# get parent indices of the centrals and their subhalo indices in the original array
SubhaloGrNr_fp = np.load(root+'SubhaloGrNr_fp'+snap_dir+'.npy').astype(int)
SubhaloPos_fp = np.load(root+'SubhaloPos_fp'+snap_dir+'.npy')/1000.
GroupPos_fp = np.load(root+'GroupPos_fp'+snap_dir+'.npy')/1000.
unique_sub_grnr, firsts = np.unique(SubhaloGrNr_fp,return_index=True)

# load the elg sub ids
sub_id_col = np.load("data/sub_id"+env_type+snap_dir+selection+"_col.npy")
sub_id_sfg = np.load("data/sub_id"+env_type+snap_dir+selection+"_sfg.npy")
sub_id_all = np.load("data/sub_id"+env_type+snap_dir+selection+"_all.npy")

# factors for multiplying histogram
factor = 0.2#1.#10./9.6

# load DM density profile
hist_col_fp = np.load("data_prof/hist_col_fp_dm.npy")*factor
hist_col_hod = np.load("data_prof/hist_col_hod_dm.npy")*factor

lw = 3.
fs = (9,8)
proxies = ['SubhaloVmax','SubhaloVpeak','SubhaloVelDisp','SubhaloMass','SubhaloMassInMaxRad','SubhaloMassInRad']#'SubhaloMassInHalfRad'
names = ['vmax','vpeak','veldisp','mass','massinmax','massinrad']
lab_proxies = [r'$V_{\rm max}$',r'$V_{\rm peak}$',r'$V_{\rm disp}$',r'$M_{\rm SUBFIND}$',r'$M_{\rm max}$',r'$M_{\rm twice}$']
dm_exts = ['_fp','_dm']
#colors = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255']
colors = ['#0077BB','#33BBEE','#0099BB','#EE7733','#CC3311','#EE3377','#BBBBBB']
want_rescaled = 1

# og
d2_col = np.load("data_prof/d2_col_fp.npy")
pos_col = np.load("data_prof/pos_col_fp.npy")
# TESTING
'''
pos_center_col = GroupPos_fp[SubhaloGrNr_fp[sub_id_col]]
pos_col = SubhaloPos_fp[sub_id_col]
pos_diff = pos_col-pos_center_col
d2_col = np.sum(pos_diff**2,axis=1)
'''
for j in range(len(dm_exts)):
    for i in range(3):
        if j == 0:
            ls = '-'
        elif j == 1:
            ls = '-'
        dm_ext = dm_exts[j]
        
        i1 = 0+i*2
        i2 = 1+i*2
        
        color1 = colors[i1]
        color2 = colors[i2]
        proxy1 = proxies[i1]
        name_proxy1 = names[i1]
        proxy2 = proxies[i2]
        name_proxy2 = names[i2]

        lab_proxy1 = lab_proxies[i1]
        lab_proxy2 = lab_proxies[i2]
        
        d2_vpeak = np.load("data_prof/d2_"+name_proxy2+dm_ext+".npy")
        d2_vmax = np.load("data_prof/d2_"+name_proxy1+dm_ext+".npy")
        pos_vpeak = np.load("data_prof/pos_"+name_proxy2+dm_ext+".npy")
        pos_vmax = np.load("data_prof/pos_"+name_proxy1+dm_ext+".npy")
        w_col = np.ones(pos_col.shape[0],dtype=pos_col.dtype)
        w_vmax = np.ones(pos_vmax.shape[0],dtype=pos_vmax.dtype)
        w_vpeak = np.ones(pos_vpeak.shape[0],dtype=pos_vpeak.dtype)

        hist, bins = np.histogram(np.sqrt(d2_col),bins=bin_edges)
        hist_vmax, bins = np.histogram(np.sqrt(d2_vmax),bins=bin_edges)
        hist_vpeak, bins = np.histogram(np.sqrt(d2_vpeak),bins=bin_edges)
        print("contained within histogram for col and vmax = ",np.sum(hist),np.sum(hist_vmax))
        bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5

        plt.figure(2*j+1,figsize=fs)
        if i == 0 and j == 0:
            plt.plot(bin_cents,hist_col_fp/vol_bins,c='black',lw=lw,ls='--',label='DM profile')
        elif i == 0 and j == 1:
            plt.plot(bin_cents,hist_col_hod/vol_bins,c='black',lw=lw,ls='--')
        if j == 0:
            plt.plot(bin_cents,hist_vmax/vol_bins,c=color1,lw=lw,ls=ls,zorder=1,label=lab_proxy1)
            plt.plot(bin_cents,hist_vpeak/vol_bins,c=color2,lw=lw,ls=ls,zorder=2,label=lab_proxy2)
        else:
            plt.plot(bin_cents,hist_vmax/vol_bins,c=color1,lw=lw,ls=ls,zorder=1)
            plt.plot(bin_cents,hist_vpeak/vol_bins,c=color2,lw=lw,ls=ls,zorder=2)
        if i == 0: plt.scatter(bin_cents,hist/vol_bins,c='dodgerblue',s=120,zorder=3,marker='*')#,label='DESI ELGs')

        plt.figure(2*j+2,figsize=fs)
        
        line = np.linspace(0,20,3)
        plt.plot(line,np.ones(len(line)),'k--')

        Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(pos_col,w_col,pos_vmax,w_vmax,Lbox)
        
        if j == 0:
            #plt.errorbar(bin_centers*(1.+i*0.05),Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color=color1,ls=ls,label=lab_proxy1,alpha=1.,fmt='o',capsize=4)
            plt.plot(bin_centers,Rat_hodtrue_mean,color=color1,ls=ls,label=lab_proxy1)
            plt.fill_between(bin_centers,Rat_hodtrue_mean-Rat_hodtrue_err,Rat_hodtrue_mean+Rat_hodtrue_err,facecolor=color1,alpha=.1)
        else:
            #plt.errorbar(bin_centers*(1.+i*0.05),Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color=color1,ls=ls,alpha=1.,fmt='o',capsize=4)
            plt.plot(bin_centers,Rat_hodtrue_mean,color=color1,ls=ls)
            plt.fill_between(bin_centers,Rat_hodtrue_mean-Rat_hodtrue_err,Rat_hodtrue_mean+Rat_hodtrue_err,facecolor=color1,alpha=.1)
            
        Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(pos_col,w_col,pos_vpeak,w_vpeak,Lbox)
        
        if j == 0:
            #plt.errorbar(bin_centers*(1.-(1+i)*0.05),Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color=color2,ls=ls,label=lab_proxy2,alpha=1.,fmt='o',capsize=4)
            plt.plot(bin_centers,Rat_hodtrue_mean,color=color2,ls=ls,label=lab_proxy2)
            plt.fill_between(bin_centers,Rat_hodtrue_mean-Rat_hodtrue_err,Rat_hodtrue_mean+Rat_hodtrue_err,facecolor=color2,alpha=.1)
        else:
            #plt.errorbar(bin_centers*(1.-(1+i)*0.05),Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color=color2,ls=ls,alpha=1.,fmt='o',capsize=4)
            plt.plot(bin_centers,Rat_hodtrue_mean,color=color2,ls=ls)
            plt.fill_between(bin_centers,Rat_hodtrue_mean-Rat_hodtrue_err,Rat_hodtrue_mean+Rat_hodtrue_err,facecolor=color2,alpha=.1)

        if j == 0: plt.text(0.12,1.4,r'${\rm FP}$')
        if j == 1: plt.text(0.12,1.4,r'${\rm DMO}$')
        
    plt.figure(2*j+1)#,figsize=(8,6))
    plt.xscale('log')
    plt.yscale('log')
    if j == 0: plt.legend()
    if want_rescaled:
        plt.xlabel(r'$r/R_{\rm 200 m}$')
    else:
        plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
    plt.ylabel(r'$n(r)$')
    plt.xlim([0.007,5])
    plt.savefig("../paper/gal_distn"+dm_ext+".pdf")

    plt.figure(2*j+2)#,figsize=(8,6))
    plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
    plt.ylabel(r'$\xi(r)_{\rm model}/\xi(r)_{\rm DESI \ ELG}$')
    plt.xscale('log')
    if j == 0: plt.legend()
    plt.ylim([0.4,1.6])
    plt.xlim([0.095,12])
    plt.savefig("../paper/prof_ELG"+dm_ext+".pdf")

plt.show()
