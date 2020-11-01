import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfc
from scipy.optimize import curve_fit,minimize
import sys
import plotparams
plotparams.buba()

want_fit = 0; want_env = 0; show_only_total = 1; want_unnorm = 1
#want_fit = 0; want_env = 0; show_only_total = 1; want_unnorm = 0
#want_fit = 0; want_env = 1; show_only_total = 1

fs = 14
selection = '_'+sys.argv[1]
#selection = '_eBOSS'
#selection = '_DESI'
snap_dirs = ['_55','_47','_41']

z_dic = {'_55': 0.8, '_47': 1.1, '_41': 1.4}

bin_cen = np.load("data/bin_cen.npy")
if want_env:
    env_types = ['_low','_high']
    
else:
    env_types = ['']
    lss =  ['-','--',':']
    
plt.figure(1,figsize=(9,7))

for i in range(len(snap_dirs)):
    snap_dir = snap_dirs[i]
    env_type = env_types[0]
    env_string = ' '.join(env_type.split('_'))
    ls = lss[i]

    if want_unnorm:
        hist_col = np.load("data/hist_unnorm"+env_type+snap_dir+selection+"_col.npy")
        hist_sfg = np.load("data/hist_unnorm"+env_type+snap_dir+selection+"_sfg.npy")
        
    else:
        hist_cents_col = np.load("data/hist_cents"+env_type+snap_dir+selection+"_col.npy")
        hist_sats_col = np.load("data/hist_sats"+env_type+snap_dir+selection+"_col.npy")
        hist_cents_sfg = np.load("data/hist_cents"+env_type+snap_dir+selection+"_sfg.npy")
        hist_sats_sfg = np.load("data/hist_sats"+env_type+snap_dir+selection+"_sfg.npy")
        hist_col = hist_cents_col+hist_sats_col
        hist_sfg = hist_cents_sfg+hist_sats_sfg

    #hist_cents_flux = np.load("data/hist_cents"+env_type+snap_dir+selection+"_flux.npy")
    #hist_sats_flux = np.load("data/hist_sats"+env_type+snap_dir+selection+"_flux.npy")

    if want_env and i == 0:
        plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='k',ls='-',label="high environment")
        plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='k',ls='--',label="low environment")
    if i == 0:

        #plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='lawngreen',ls='-',label='OII-emitting')
        if show_only_total:
            plt.plot(10**bin_cen,np.zeros(len(hist_sfg)),lw=2.,color='k',ls='-',label=r'$z=0.8$')
            plt.plot(10**bin_cen,np.zeros(len(hist_sfg)),lw=2.,color='k',ls='--',label=r'$z=1.1$')
            plt.plot(10**bin_cen,np.zeros(len(hist_sfg)),lw=2.,color='k',ls=':',label=r'$z=1.4$')
            plt.plot(10**bin_cen,np.zeros(len(hist_sfg)),lw=2.,color='#CC6677',ls='-',label='SFR-selected')
            plt.plot(10**bin_cen,np.zeros(len(hist_sfg)),lw=2.,color='dodgerblue',ls='-',label='color-selected')
        else:
            plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.5,color='k',ls='-',label='centrals')
            plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=1.5,color='k',ls='-',label='satellites')
            plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='k',ls='-',label=r'$z=0.8$')
            plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='k',ls='--',label=r'$z=1.1$')
            plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='k',ls=':',label=r'$z=1.4$')
            plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='#CC6677',ls='-',label='SFR-selected')
            plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='dodgerblue',ls='-',label='color-selected')

    if show_only_total:
        plt.plot(10**bin_cen,hist_sfg,lw=3.,color='#CC6677',ls=ls)
        plt.plot(10**bin_cen,hist_col,lw=3.,color='dodgerblue',ls=ls)
        #plt.plot(10**bin_cen,hist_cents_flux+hist_sats_flux,lw=2.,color='lawngreen',ls=ls)
    else:
        #plt.plot(10**bin_cen,hist_cents_sfg,lw=2.5,color='#CC6677',ls=ls)
        #plt.plot(10**bin_cen,hist_sats_sfg,lw=1.5,color='#CC6677',ls=ls)
        plt.plot(10**bin_cen,hist_cents_col,lw=2.5,color='dodgerblue',ls=ls)
        plt.plot(10**bin_cen,hist_sats_col,lw=1.5,color='dodgerblue',ls=ls)
        #plt.plot(10**bin_cen,hist_cents_flux,lw=2.5,color='lawngreen',ls=ls)
        #plt.plot(10**bin_cen,hist_sats_flux,lw=1.5,color='lawngreen',ls=ls)

if want_fit:
    bincen_col, binsat_col, cents_col, sats_col = fit_hod(hist_cents_col,hist_sats_col,bin_cen)
    bincen_sfg, binsat_sfg, cents_sfg, sats_sfg = fit_hod(hist_cents_sfg,hist_sats_sfg,bin_cen)

    plt.plot(10**bincen_col,np.zeros(len(bincen_col)),lw=1.,color='k',ls='-.',label="polynomial fit")
    plt.plot(10**bincen_col,cents_col,lw=1.5,color='dodgerblue',ls='-.')
    plt.plot(10**binsat_col,sats_col,lw=0.5,color='dodgerblue',ls='-.')
    plt.plot(10**bincen_sfg,cents_sfg,lw=1.5,color='darkorange',ls='-.')
    plt.plot(10**binsat_sfg,sats_sfg,lw=0.5,color='darkorange',ls='-.')
    
plt.legend(loc='upper left',ncol=2,frameon=False)#,fontsize=fs-2)
if want_unnorm:
    plt.ylim([0.8,30000])
else:
    plt.ylim([0.001,40])
plt.xlim([1.e11,1.e15])#2.e14])
if want_unnorm:
    plt.ylabel(r"$ N_g \times dn/dM_{\rm halo}$")#,fontsize=fs)
else:
    plt.ylabel(r"$\langle N_g \rangle$")#,fontsize=fs)
plt.xlabel(r"$M_{\rm halo} \ [M_\odot/h]$")#,fontsize=fs)
plt.xscale('log')
plt.yscale('log')
if want_unnorm:
    plt.text(3.e12,2.,r'${\rm '+(''.join(selection.split('_')))+"}$")#,fontsize=fs)
else:
    plt.text(2.e11,0.2,r'${\rm '+(''.join(selection.split('_')))+"}$")#,fontsize=fs)
if want_env:
    #plt.savefig("figs/HOD_elg_env"+selection+".png")
    plt.savefig("paper/HOD_elg_env"+selection+".pdf")
else:
    #plt.savefig("figs/HOD_elg"+selection+".png")
    plt.savefig("paper/HOD_elg"+selection+".pdf")
