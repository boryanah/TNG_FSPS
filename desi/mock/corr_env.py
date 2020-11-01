import numpy as np
import matplotlib.pyplot as plt
import h5py
import Corrfunc
from util import get_density, get_smooth_density, smooth_density, get_jack_corr, get_counts, get_hist
import plotparams
plotparams.buba()

N_div = 5
bounds = np.linspace(0,N_div,N_div+1)/N_div

#selection = '_DESI'
selection = '_eBOSS'

redshift_dict = {'055':['sdss_desi','_0.0'],#-0.5
                 '047':['sdss_desi','_-0.0'],
                 '041':['sdss_des','']}

# snapshot number and dust model
snap = '0%d'%(55)#'041'#'047'#'055'#'050'#'059'
snap_dir = '_'+str(int(snap))
dust_index = redshift_dict[snap][1]
env_types = ['_low','_high']
env_type = ''

# directory of the halo TNG data
box_name = "TNG300"
Lbox = 205.
root = "/mnt/gosling1/boryanah/"+box_name+"/"

# load the elg sub ids
sub_id_col = np.load("data/sub_id"+env_type+snap_dir+selection+"_col.npy")
sub_id_sfg = np.load("data/sub_id"+env_type+snap_dir+selection+"_sfg.npy")
sub_id_all = np.load("data/sub_id"+env_type+snap_dir+selection+"_all.npy")


# loading the halo mass and group identification
Group_M_Mean200_fp = np.load(root+'Group_M_Mean200_fp'+snap_dir+'.npy')*1.e10
SubhaloGrNr_fp = np.load(root+'SubhaloGrNr_fp'+snap_dir+'.npy')
SubhaloPos_fp = np.load(root+'SubhaloPos_fp'+snap_dir+'.npy')/1.e3
GroupPos_fp = np.load(root+'GroupPos_fp'+snap_dir+'.npy')/1.e3
N_halos_fp = GroupPos_fp.shape[0]
inds_halo_fp = np.arange(N_halos_fp,dtype=int)
GroupEnv_fp = np.load(root+'GroupEnv_fp'+snap_dir+'.npy')

# get parent indices of the centrals and their subhalo indices in the original array
unique_sub_grnr, firsts = np.unique(SubhaloGrNr_fp,return_index=True)

count_halo_col_fp, count_halo_cents_col_fp, count_halo_sats_col_fp = get_counts(SubhaloGrNr_fp,firsts,N_halos_fp,sub_id_col)
count_halo_sfg_fp, count_halo_cents_sfg_fp, count_halo_sats_sfg_fp = get_counts(SubhaloGrNr_fp,firsts,N_halos_fp,sub_id_sfg)
count_halo_all_fp, count_halo_cents_all_fp, count_halo_sats_all_fp = get_counts(SubhaloGrNr_fp,firsts,N_halos_fp,sub_id_all)


def get_env_pos(gal_inds, sub_grnr, sub_pos, group_env, group_inds, group_mass):

    # define mass bins
    log_min = 11.
    log_max = 15.
    N_bins = 41
    bin_edges = np.linspace(log_min,log_max,N_bins)
    bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5

    # for the chosen galaxies what are their hosts
    gal_grnr = sub_grnr[gal_inds]
    gal_pos = sub_pos[gal_inds]
    cum_low = 0
    cum_high = 0
    
    for i in range(N_bins-1):
        M_lo = 10.**bin_edges[i]
        M_hi = 10.**bin_edges[i+1]

        mass_sel = (M_lo <= group_mass) & (M_hi > group_mass)
        if np.sum(mass_sel) == 0: continue
        
        for j in range(N_div):

            if j > 0 and j < N_div-1: continue
            
            lo_bound = j*1./N_div
            hi_bound = (j+1.)/N_div

            if lo_bound < 0.: lo_bound = 0.
            if hi_bound > 1.: hi_bound = 1.
            quant_lo = np.quantile(group_env[mass_sel], lo_bound)
            quant_hi = np.quantile(group_env[mass_sel], hi_bound)
            env_sel = (group_env > quant_lo) & (group_env <= quant_hi)

            halo_sel = mass_sel & env_sel
            N = int(np.sum(halo_sel))
            if N == 0: continue
            
            inds_sel = group_inds[halo_sel]
            
            inter, comm1, comm2 = np.intersect1d(inds_sel,gal_grnr,return_indices=1)

            n_gal = len(inter)
            if n_gal == 0: print("no gals in this mass bin and env bin"); continue
            
            if j == 0:
                try:
                    xyz_low = np.vstack((xyz_low,gal_pos[comm2]))
                except:
                    xyz_low = gal_pos[comm2]
                #xyz_low[cum_low:cum_low+n_gal] = gal_pos[comm2]
                cum_low += n_gal
            else:
                try:
                    xyz_high = np.vstack((xyz_high,gal_pos[comm2]))
                except:
                    xyz_high = gal_pos[comm2]
                #xyz_low[cum_low:cum_low+n_gal] = gal_pos[comm2]
                cum_high += n_gal
            

    print("number of low galaxies = ",cum_low)
    print("number of high galaxies = ",cum_high)
    assert xyz_low.shape[0] == cum_low, "something's up"
    assert xyz_high.shape[0] == cum_high, "something's up"
    
    
    w_low = np.ones(cum_low,dtype=xyz_low.dtype)
    w_high = np.ones(cum_high,dtype=xyz_high.dtype)
    
    return xyz_low, xyz_high, w_low, w_high

xyz_col_low, xyz_col_high, w_col_low, w_col_high = get_env_pos(sub_id_col, SubhaloGrNr_fp, SubhaloPos_fp, GroupEnv_fp, inds_halo_fp, Group_M_Mean200_fp)
xyz_sfg_low, xyz_sfg_high, w_sfg_low, w_sfg_high = get_env_pos(sub_id_sfg, SubhaloGrNr_fp, SubhaloPos_fp, GroupEnv_fp, inds_halo_fp, Group_M_Mean200_fp)

for i in range(2):
    env_type = env_types[i]
    if i == 0:
        Rat_colsfg_mean, Rat_colsfg_err, Corr_mean_col, Corr_err_col,  Corr_mean_sfg, Corr_err_sfg, bin_centers = get_jack_corr(xyz_sfg_low,w_sfg_low,xyz_col_low,w_col_low,Lbox)
    else:
        Rat_colsfg_mean, Rat_colsfg_err, Corr_mean_col, Corr_err_col,  Corr_mean_sfg, Corr_err_sfg, bin_centers = get_jack_corr(xyz_sfg_high,w_sfg_high,xyz_col_high,w_col_high,Lbox)
    plt.errorbar(bin_centers,Rat_colsfg_mean,yerr=Rat_colsfg_err,ls='-',label=' '.join(env_type.split('_')),alpha=1.,fmt='o',capsize=4)


    np.save("../data/rat_colsfg"+env_type+snap_dir+selection+"_mean.npy",Rat_colsfg_mean)
    np.save("../data/rat_colsfg"+env_type+snap_dir+selection+"_err.npy",Rat_colsfg_err)
    np.save("../data/corr_mean"+env_type+snap_dir+selection+"_col.npy",Corr_mean_col)
    np.save("../data/corr_err"+env_type+snap_dir+selection+"_col.npy",Corr_err_col)
    np.save("../data/corr_mean"+env_type+snap_dir+selection+"_sfg.npy",Corr_mean_sfg)
    np.save("../data/corr_err"+env_type+snap_dir+selection+"_sfg.npy",Corr_err_sfg)

np.save("../data/bin_env_centers.npy",bin_centers)

line = np.linspace(0.,100.,10)
# for seeing where we are
plt.plot(line,np.ones(len(line)),'k--')
'''
# COL
Rat_highlow_mean, Rat_highlow_err, Corr_mean_high, Corr_err_high,  Corr_mean_low, Corr_err_low, bin_centers = get_jack_corr(xyz_col_low,w_col_low,xyz_col_high,w_col_high,Lbox)
plt.errorbar(bin_centers,Rat_highlow_mean,yerr=Rat_highlow_err,color='dodgerblue',ls='-',label='color-selected',alpha=1.,fmt='o',capsize=4)

# SFG
Rat_highlow_mean, Rat_highlow_err, Corr_mean_high, Corr_err_high,  Corr_mean_low, Corr_err_low, bin_centers = get_jack_corr(xyz_sfg_low,w_sfg_low,xyz_sfg_high,w_sfg_high,Lbox)
plt.errorbar(bin_centers*1.05,Rat_highlow_mean,yerr=Rat_highlow_err,color='orange',ls='-',label='SFR-selected',alpha=1.,fmt='o',capsize=4)

# ALL
Rat_highlow_mean, Rat_highlow_err, Corr_mean_high, Corr_err_high,  Corr_mean_low, Corr_err_low, bin_centers = get_jack_corr(xyz_all_low,w_all_low,xyz_all_high,w_all_high,Lbox)
plt.plot(bin_centers,Rat_highlow_mean,color='black',ls='-',label='mass-selected')
plt.fill_between(bin_centers,Rat_highlow_mean-Rat_highlow_err,Rat_highlow_mean+Rat_highlow_err,color='black',alpha=0.1)
'''
plt.xlabel(r'$r [{\rm Mpc}/h]$')
plt.ylabel(r'$\xi(r)_{\rm HOD}/\xi(r)_{\rm TNG300}$')
plt.xscale('log')
plt.legend()
plt.ylim([0.6,1.4])
plt.xlim([1.e-1,10])
plt.savefig("figs/mock_ELG.png")
plt.show()
