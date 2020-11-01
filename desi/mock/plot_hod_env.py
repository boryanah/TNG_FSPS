import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()

N_div = 5
bounds = np.linspace(0,N_div,N_div+1)/N_div

log_min = 11.
log_max = 15.
N_bins = 41
bin_edges = np.linspace(log_min,log_max,N_bins)
bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5

edges = np.load("data/edges.npy")
hist_col_all = np.load("data/hist_col_all.npy")
hist_cents_col_all = np.load("data/hist_cents_col_all.npy")
hist_sats_col_all = np.load("data/hist_sats_col_all.npy")
hist_sfg_all = np.load("data/hist_sfg_all.npy")
hist_cents_sfg_all = np.load("data/hist_cents_sfg_all.npy")
hist_sats_sfg_all = np.load("data/hist_sats_sfg_all.npy")
hist_all_all = np.load("data/hist_all_all.npy")
hist_cents_all_all = np.load("data/hist_cents_all_all.npy")
hist_sats_all_all = np.load("data/hist_sats_all_all.npy")


def plot_hod(hist_all,hist_cents_all,hist_sats_all,name):

     hist_mid = np.mean(hist_all,axis=1)
     hist_cents_mid = np.mean(hist_cents_all,axis=1)
     hist_sats_mid = np.mean(hist_sats_all,axis=1)

     hist_low = hist_all[:,0]; hist_high = hist_all[:,-1]
     hist_cents_low = hist_cents_all[:,0]; hist_cents_high = hist_cents_all[:,-1]
     hist_sats_low = hist_sats_all[:,0]; hist_sats_high = hist_sats_all[:,-1]

     plt.figure(figsize=(9,7))
     
     plt.plot(10.**bin_cents,hist_low,color=cs[0],ls=lss[0],label=r'20\% lowest environment')
     if 'all' in name:
          plt.plot(10.**bin_cents,hist_mid,color=cs[1],ls=lss[0],lw=1.5,label=r'full sample, mass-selected')
     elif 'col' in name:
          plt.plot(10.**bin_cents,hist_mid,color=cs[1],ls=lss[0],lw=1.5,label=r'full sample, color-selected')
     elif 'sfg' in name:
          plt.plot(10.**bin_cents,hist_mid,color=cs[1],ls=lss[0],lw=1.5,label=r'full sample, SFR-selected')
     plt.plot(10.**bin_cents,hist_high,color=cs[2],ls=lss[0],label=r'20\% highest environment')

     plt.plot(10.**bin_cents,hist_cents_low,color=cs[0],ls=lss[1])#,label=r'$%.1f$ lowest environment'%(1./N_div*100.))
     plt.plot(10.**bin_cents,hist_cents_mid,color=cs[1],lw=1.5,ls=lss[1])#,label=r'full sample')
     plt.plot(10.**bin_cents,hist_cents_high,color=cs[2],ls=lss[1])#,label=r'$%.1f$ highest environment'%(1./N_div*100.))

     plt.plot(10.**bin_cents,hist_sats_low,color=cs[0],ls=lss[2])#,label=r'$%.1f$ lowest environment'%(1./N_div*100.))
     plt.plot(10.**bin_cents,hist_sats_mid,color=cs[1],lw=1.5,ls=lss[2])#,label=r'full sample')
     plt.plot(10.**bin_cents,hist_sats_high,color=cs[2],ls=lss[2])#,label=r'$%.1f$ highest environment'%(1./N_div*100.))


     if 'all' in name:
          plt.ylim([1.e-2,30])
          plt.xlim([10**12,2.e14])
     else:
          plt.ylim([1.e-3,20])
          plt.xlim([10**11,2.e14])
     plt.ylabel(r"$\langle N_g \rangle$")
     plt.xlabel(r"$M_{\rm halo} \ [M_\odot/h]$")
     plt.xscale('log')
     plt.yscale('log')
     plt.legend()
     plt.savefig(name)
     plt.show()

lss = ['-',':','--']
cs = ['dodgerblue','black','#CC6677']
#plot_hod(hist_all_all,hist_cents_all_all,hist_sats_all_all,name='figs/hod_env_all.png')
plot_hod(hist_all_all,hist_cents_all_all,hist_sats_all_all,name='../paper/HOD_env_all.pdf')
#plot_hod(hist_col_all,hist_cents_col_all,hist_sats_col_all,name='figs/hod_env_col.png')
plot_hod(hist_col_all,hist_cents_col_all,hist_sats_col_all,name='../paper/HOD_env_col.pdf')
#plot_hod(hist_sfg_all,hist_cents_sfg_all,hist_sats_sfg_all,name='figs/hod_env_sfg.png')
