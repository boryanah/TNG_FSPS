import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfc
from scipy.optimize import curve_fit,minimize
import sys
import plotparams
plotparams.buba()

mode = sys.argv[3]
if mode == 'normal':
    want_fit = 0; want_env = 0; show_only_total = 0
elif mode == 'env':
    want_fit = 0; want_env = 1; show_only_total = 1

fs = 14
#selection = '_eBOSS'
#selection = '_DESI'
selection = '_'+sys.argv[1]
snap_dir = '_%d'%(int(sys.argv[2]))

z_dic = {'_55': 0.8, '_47': 1.1, '_41': 1.4}

def mean_Ncen(mhalo, mmin, sigma):
    return 0.5 * ( 1.0 + erf( (mhalo-mmin) / sigma ) )

def mean_Nsat(mhalo, mcut, m1, alpha):
    return  ( (10**mhalo - 10**mcut) / 10**m1 )**alpha

def mean_Ncen(mhalo,mb,md,fb,fd,sigma,ad):
    if mb < 0 or md < 0 or mb > 13.5 or md > 13.5 or sigma < 0 or fb < 0 or fd < 0:
        return np.inf*np.ones(len(mhalo))
    ncen = fb/2.*(1+erf((mhalo-mb)/sigma))+fd/2.*\
           np.exp(ad/2.*(2*md+ad*sigma**2-2.*mhalo))*\
           erfc(md+ad*sigma**2-mhalo/(np.sqrt(2.)*sigma))
    return ncen


def fit_hod(ncen,nsat,bincentre):
    fitmin = 11.; ncenmin = np.min(nsat[ncen>0.]); nsatmin = np.min(nsat[nsat>0.])#0.0005 0.006
        
    mask0 = np.isfinite(ncen) & (ncen >= ncenmin)
    mask1 = (np.isfinite(nsat)) & (bincentre >= fitmin) & (nsat >= nsatmin)
    
    zcen = np.polyfit(bincentre[mask0], ncen[mask0],8)
    pcen = np.poly1d(zcen)
    #print("central poly values: ",zcen)
    
    zsat = np.polyfit(bincentre[mask1], nsat[mask1],8)
    psat = np.poly1d(zsat)
    #print("satellite poly values: ",zsat)

    xcen = np.linspace(np.min(bincentre[ncen > 0]),np.max(bincentre[np.isfinite(ncen) & (ncen < 1.) & (ncen > ncenmin)]),100)
    print("xcen[-1] = ",xcen[-1])
    xsat = np.linspace(np.min(bincentre[nsat > 0]),np.max(bincentre[np.isfinite(nsat)]),100)
    print("xsat[0] = ",xcen[0])
    return xcen,xsat,pcen(xcen),psat(xsat)

bin_cen = np.load("data/bin_cen.npy")
if want_env:
    env_types = ['_low','_high']
    lss =  ['--','-']
else:
    env_types = ['']
    lss =  ['-']

plt.figure(1,figsize=(9,7))

for i in range(len(env_types)):
    env_type = env_types[i]
    env_string = ' '.join(env_type.split('_'))
    ls = lss[i]
    
    hist_cents_col = np.load("data/hist_cents"+env_type+snap_dir+selection+"_col.npy")
    hist_sats_col = np.load("data/hist_sats"+env_type+snap_dir+selection+"_col.npy")
    hist_cents_sfg = np.load("data/hist_cents"+env_type+snap_dir+selection+"_sfg.npy")
    hist_sats_sfg = np.load("data/hist_sats"+env_type+snap_dir+selection+"_sfg.npy")
    #hist_cents_flux = np.load("data/hist_cents"+env_type+snap_dir+selection+"_flux.npy")
    #hist_sats_flux = np.load("data/hist_sats"+env_type+snap_dir+selection+"_flux.npy")

    if want_env and i == 0:
        plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='k',ls='-',label="high environment")
        plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='k',ls='--',label="low environment")
    if i == 0:
        plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='orange',ls='-',label='star-forming')
        plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='dodgerblue',ls='-',label='color-selected')
        #plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='lawngreen',ls='-',label='OII-emitting')
        if show_only_total:
            True#plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.,color='k',ls='-',label='total')
        else:
            plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=2.5,color='k',ls='-',label='centrals')
            plt.plot(10**bin_cen,np.zeros(len(hist_sats_sfg)),lw=1.5,color='k',ls='-',label='satellites')

    if show_only_total:
        plt.plot(10**bin_cen,hist_cents_sfg+hist_sats_sfg,lw=2.,color='orange',ls=ls)
        plt.plot(10**bin_cen,hist_cents_col+hist_sats_col,lw=2.,color='dodgerblue',ls=ls)
        #plt.plot(10**bin_cen,hist_cents_flux+hist_sats_flux,lw=2.,color='lawngreen',ls=ls)
    else:
        plt.plot(10**bin_cen,hist_cents_sfg,lw=2.5,color='orange',ls=ls)
        plt.plot(10**bin_cen,hist_sats_sfg,lw=1.5,color='orange',ls=ls)
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
    
plt.legend(loc='upper left',ncol=2)#,fontsize=fs-2)
plt.ylim([0.001,10])
plt.xlim([1.e11,1.e15])#2.e14])
plt.ylabel(r"$\langle N_g \rangle$")#,fontsize=fs)
plt.xlabel(r"$M_{\rm halo} \ [M_\odot/h]$")#,fontsize=fs)
plt.xscale('log')
plt.yscale('log')
plt.text(2.e11,0.2,r'${\rm '+(''.join(selection.split('_')))+",} \ z = %.1f$"%z_dic[snap_dir])#,fontsize=fs)
if want_env:
    plt.savefig("figs/HOD_elg_env"+snap_dir+selection+".png")
else:
    plt.savefig("figs/HOD_elg"+snap_dir+selection+".png")

'''
def fit_hod(ncen,nsat,bincentre):
    fitmin = 11.5; fitnorm = 12.; ncenmin = 0.05; nsatmin = np.min(nsat[nsat>0.])#0.006
        
    mask0 = np.isfinite(ncen) & (ncen > ncenmin)
    mask1 = (np.isfinite(nsat)) & (bincentre >= fitmin) & (nsat > nsatmin)

    p0cen = [11.,1.0]
    p0cen = [11.7,11.3,0.05,1,0.09,1.7]
    p0sat = [fitmin, fitnorm, 0.49]
    noptcen, ncov = curve_fit(mean_Ncen, bincentre[mask0], ncen[mask0], p0 = p0cen, ftol = 1e-10, xtol = 1e-10)
    noptsat, ncov = curve_fit(mean_Nsat, bincentre[mask1], nsat[mask1], p0 = p0sat, ftol = 1e-10, xtol = 1e-10, sigma = np.sqrt(nsat[mask1]))

    print(noptcen)
    print(noptsat)

    #noptsat = (12.2,13.2,0.48)
    ncen_pred = mean_Ncen(bincentre, *noptcen)
    nsat_pred = mean_Nsat(bincentre, *noptsat)
    nsat_pred[np.isnan(nsat_pred)] = 0.

    return ncen_pred, nsat_pred
'''
