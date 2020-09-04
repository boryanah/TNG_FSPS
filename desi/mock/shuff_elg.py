import numpy as np
import matplotlib.pyplot as plt
import h5py
import Corrfunc
from util import get_density, smooth_density, get_jack_corr, get_counts, get_hist
import plotparams
plotparams.buba()

np.random.seed(300)

selection = '_DESI'
#selection = '_eBOSS'

redshift_dict = {'055':['sdss_desi','_0.0'],#-0.5
                 '047':['sdss_desi','_-0.0'],
                 '041':['sdss_des','']}

# snapshot number and dust model
snap = '0%d'%(55)#'041'#'047'#'055'#'050'#'059'
snap_dir = '_'+str(int(snap))
dust_index = redshift_dict[snap][1]
env_type = ''

# directory of the halo TNG data
box_name = "TNG300"
Lbox = 205.
root = "/mnt/gosling1/boryanah/"+box_name+"/"

# load the elg sub ids
#sub_id_col = np.load("data/sub_id"+env_type+snap_dir+selection+"_all.npy")# SUPER INTERSTING RESUTL
sub_id_col = np.load("data/sub_id"+env_type+snap_dir+selection+"_col.npy")
sub_id_sfg = np.load("data/sub_id"+env_type+snap_dir+selection+"_sfg.npy")
sub_id_all = np.load("data/sub_id"+env_type+snap_dir+selection+"_all.npy")

# loading the halo mass and group identification
Group_M_Mean200_fp = np.load(root+'Group_M_Mean200_fp'+snap_dir+'.npy')*1.e10
SubhaloGrNr_fp = np.load(root+'SubhaloGrNr_fp'+snap_dir+'.npy').astype(int)
SubhaloPos_fp = np.load(root+'SubhaloPos_fp'+snap_dir+'.npy')/1.e3
GroupPos_fp = np.load(root+'GroupPos_fp'+snap_dir+'.npy')/1.e3
Group_M_Mean200_dm = np.load(root+'Group_M_Mean200_dm'+snap_dir+'.npy')*1.e10
#GroupFirst_dm = np.load(root+'GroupFirst_dm'+snap_dir+'.npy')
GroupPos_dm = np.load(root+'GroupPos_dm'+snap_dir+'.npy')/1.e3

N_halos_fp = GroupPos_fp.shape[0]
inds_halo = np.arange(N_halos_fp,dtype=int)
N_halos_dm = GroupPos_dm.shape[0]

'''
d_smooth = smooth_density(get_density(GroupPos_fp))
# finding who belongs where in the cosmic web
N_dim = 256
gr_size = Lbox/N_dim
#halo_x = SubhaloPos_fp[:,0]; halo_y = SubhaloPos_fp[:,1]; halo_z = SubhaloPos_fp[:,2]
halo_x = GroupPos_fp[:,0]; halo_y = GroupPos_fp[:,1]; halo_z = GroupPos_fp[:,2]

i_cw = (halo_x/gr_size).astype(int)
j_cw = (halo_y/gr_size).astype(int)
k_cw = (halo_z/gr_size).astype(int)
i_cw[i_cw == N_dim] = N_dim-1 # fixing floating point issue
j_cw[j_cw == N_dim] = N_dim-1 # fixing floating point issue
k_cw[k_cw == N_dim] = N_dim-1 # fixing floating point issue

# Environment definition
env_cw = d_smooth[i_cw,j_cw,k_cw]
'''


# get parent indices of the centrals and their subhalo indices in the original array
unique_sub_grnr, firsts = np.unique(SubhaloGrNr_fp,return_index=True)

count_halo_col_fp, count_halo_cents_col_fp, count_halo_sats_col_fp = get_counts(SubhaloGrNr_fp,firsts,N_halos_fp,sub_id_col)
count_halo_sfg_fp, count_halo_cents_sfg_fp, count_halo_sats_sfg_fp = get_counts(SubhaloGrNr_fp,firsts,N_halos_fp,sub_id_sfg)
count_halo_all_fp, count_halo_cents_all_fp, count_halo_sats_all_fp = get_counts(SubhaloGrNr_fp,firsts,N_halos_fp,sub_id_all)


def get_shuff_counts(count_halo_fp):
    log_min = 11.
    log_max = 15.
    N_bins = 41
    bin_edges = np.logspace(log_min,log_max,N_bins)
    bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5
    count_halo_shuff = np.zeros(N_halos_fp,dtype=int)
    for i  in range(N_bins-1):
        choice = (bin_edges[i] < Group_M_Mean200_fp) & (bin_edges[i+1] > Group_M_Mean200_fp)
        count = count_halo_fp[choice]
        np.random.shuffle(count)
        count_halo_shuff[choice] = count

    print(np.sum(count_halo_shuff))
    return count_halo_shuff

def get_xyz(sub_id,count_halo_fp,count_halo_shuff,want_pop_cent=0):
    if want_pop_cent:
        xyz_true = GroupPos_fp[count_halo_fp > 0]
        w_true = count_halo_fp[count_halo_fp > 0].astype(xyz_true.dtype)

        xyz_hod = GroupPos_fp[count_halo_shuff > 0]
        w_hod = count_halo_shuff[count_halo_shuff > 0].astype(xyz_hod.dtype)

    else:
        from random import choices

        population = np.load("data/population.npy")
        weights = np.load("data/weights.npy")

        overlap, comm1, comm2 = np.intersect1d(firsts,sub_id,return_indices=True)
        parents_cents = unique_sub_grnr[comm1]

        N_cen = len(overlap)
        N_gal = len(sub_id)
        N_sat = N_gal-N_cen


        theta = np.random.rand(N_gal)*np.pi
        phi = np.random.rand(N_gal)*2*np.pi
        radius = choices(population, weights, k=N_gal)
        x = radius*np.cos(phi)*np.sin(theta)
        y = radius*np.sin(phi)*np.sin(theta)
        z = radius*np.cos(theta)
        xyz = np.vstack((x,y,z)).T

        N_parents = np.sum(count_halo_shuff > 0)
        count = count_halo_shuff[count_halo_shuff > 0]
        pos = GroupPos_fp[count_halo_shuff > 0]
        ind = np.arange(len(count_halo_shuff))[count_halo_shuff > 0]
        xyz_hod = np.zeros((np.sum(count_halo_shuff),3))
        gal_sofar = 0
        offset = 0
        for i in range(N_parents):
            if np.sum(parents_cents == ind[i]) > 0:
                xyz_hod[gal_sofar] = pos[i]
                offset = 1
            c = count[i]
            xyz_hod[gal_sofar+offset:gal_sofar+c] = pos[i]+xyz[gal_sofar+offset:gal_sofar+c]

            gal_sofar += c
            offset = 0

        xyz_true = SubhaloPos_fp[sub_id]
        w_true = np.ones(xyz_true.shape[0],dtype=xyz_true.dtype)

        xyz_hod[xyz_hod >= Lbox] -= Lbox
        xyz_hod[xyz_hod <= 0.] += Lbox 

        w_hod = np.ones(xyz_hod.shape[0],dtype=xyz_hod.dtype)

        return xyz_true, w_true, xyz_hod, w_hod

def plot_ratio(xyz_true,w_true,xyz_hod,w_hod,Lbox,color,label):
    Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_hod,w_hod,Lbox)

    if color == 'black':
        plt.plot(bin_centers,Rat_hodtrue_mean,color=color,ls='-',label=label)
        plt.fill_between(bin_centers,Rat_hodtrue_mean-Rat_hodtrue_err,Rat_hodtrue_mean+Rat_hodtrue_err,color=color,alpha=0.5)
    else:
        plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color=color,ls='-',label=label,alpha=0.5,fmt='o',capsize=4)

    mean_val = np.mean(Rat_hodtrue_mean[np.isfinite(Rat_hodtrue_mean) & (bin_centers > 2.)])

    print('mean_val = ',mean_val)
    x = np.linspace(2,10,3)
    plt.plot(bin_centers,np.ones(len(bin_centers)),'k--')
    plt.plot(x,np.ones(len(x))*mean_val,color=color,ls='--',lw=1.5)

count_halo_col_shuff = get_shuff_counts(count_halo_col_fp)
count_halo_sfg_shuff = get_shuff_counts(count_halo_sfg_fp)
count_halo_all_shuff = get_shuff_counts(count_halo_all_fp)

xyz_col_true, w_col_true, xyz_col_hod, w_col_hod = get_xyz(sub_id_col,count_halo_col_fp,count_halo_col_shuff)
xyz_sfg_true, w_sfg_true, xyz_sfg_hod, w_sfg_hod = get_xyz(sub_id_sfg,count_halo_sfg_fp,count_halo_sfg_shuff)
xyz_all_true, w_all_true, xyz_all_hod, w_all_hod = get_xyz(sub_id_all,count_halo_all_fp,count_halo_all_shuff)

plt.figure(figsize=(10,8))
plot_ratio(xyz_col_true,w_col_true,xyz_col_hod,w_col_hod,Lbox=Lbox,color='dodgerblue',label='color-selected')
plot_ratio(xyz_sfg_true,w_sfg_true,xyz_sfg_hod,w_sfg_hod,Lbox=Lbox,color='orange',label='star-forming')
plot_ratio(xyz_all_true,w_all_true,xyz_all_hod,w_all_hod,Lbox=Lbox,color='black',label='stellar-mass')

plt.xlabel(r'$r [{\rm Mpc}/h]$')
plt.ylabel(r'$\xi(r)_{\rm HOD}/\xi(r)_{\rm TNG300}$')
plt.xscale('log')
plt.legend()
plt.ylim([0.6,1.4])
plt.savefig("figs/shuff_ELG.png")
plt.show()
quit()

# position of the selected galaxies
pos_col = SubhaloPos_fp[sub_id_col]
sub_grnr_col = SubhaloGrNr_fp[sub_id_col]
d2_col = np.zeros(len(sub_id_col))
gal_sofar = 0
sub_grnr_col_uni, inds = np.unique(sub_grnr_col,return_index=True)
pos_col_new = np.zeros((len(sub_id_col),3))
# this is better since we don't always have a central
for i in range(len(sub_grnr_col_uni)):
    pos = pos_col[sub_grnr_col_uni[i] == sub_grnr_col]
    pos_c = GroupPos_fp[sub_grnr_col_uni[i]]
    #pos_c = pos_col[inds[i]]
    pos_diff = pos-pos_c
    #plt.scatter(pos_diff[:,0],pos_diff[:,1],s=1)
    d2 = np.sum(pos_diff**2,axis=1)
    
    d2_col[gal_sofar:gal_sofar+len(d2)] = d2
    pos_col_new[gal_sofar:gal_sofar+len(d2)] = pos_c
    gal_sofar += len(d2)
'''
# this gives me the position of the centrals
overlap, comm1, comm2 = np.intersect1d(firsts,sub_id_col,return_indices=True)
pos_col_centrals = SubhaloPos_fp[overlap]
sub_grnr_col_centrals = unique_sub_grnr[comm1]
# should work with a mass sample
for i in range(pos_col_centrals.shape[0]):
    pos = pos_col[sub_grnr_col_centrals[i] == sub_grnr_col]
    pos_c = pos_col_centrals[i]
    pos_diff = pos-pos_c
    #plt.scatter(pos_diff[:,0],pos_diff[:,1],s=1)
    d2 = np.sum(pos_diff**2,axis=1)
        
    d2_col[gal_sofar:gal_sofar+len(d2)] = d2
    gal_sofar += len(d2)
'''
print(gal_sofar)
print(len(sub_id_col))
#plt.xlim([-2.,2.])
#plt.ylim([-2.,2.])
#plt.axis('equal')
#plt.show()

bin_edges = np.logspace(np.log10(0.007),np.log10(5.),41)
#bin_edges = np.linspace(0.0,5.,41)
vol_edges = 4./3*np.pi*bin_edges**3
vol_bins = vol_edges[1:]-vol_edges[:-1]
hist, bins = np.histogram(np.sqrt(d2_col),bins=bin_edges)
print(np.sum(hist))
bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5

from random import choices

weights = hist*1./np.sum(hist)
population = bin_cents
rad_distn = choices(population, weights, k=np.sum(hist))
hist_new, bins = np.histogram(rad_distn,bins=bin_edges)

plt.figure(2)
plt.plot(bin_cents,hist/vol_bins,label='original')
plt.plot(bin_cents,hist_new/vol_bins,label='drawn')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig("figs/gal_distn.png")
plt.close()

overlap, comm1, comm2 = np.intersect1d(firsts,sub_id_col,return_indices=True)
N_cen = len(overlap)
N_gal = len(sub_id_col)
N_sat = N_gal-N_cen

theta = np.random.rand(N_gal)*np.pi
phi = np.random.rand(N_gal)*2*np.pi
radius = choices(population, weights, k=N_gal)
x = radius*np.cos(phi)*np.sin(theta)
y = radius*np.sin(phi)*np.sin(theta)
z = radius*np.cos(theta)
xyz = np.vstack((x,y,z)).T


pos_col_new += xyz
pos_col_new[comm2] -= xyz[comm2]
pos_col_new[pos_col_new >= Lbox] -= Lbox
pos_col_new[pos_col_new <= 0.] += Lbox 

w_col = np.ones(pos_col.shape[0],dtype=pos_col.dtype)
w_col_new = np.ones(pos_col_new.shape[0],dtype=pos_col_new.dtype)

Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(pos_col,w_col,pos_col_new,w_col_new,Lbox)
plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='dodgerblue',ls='-',label='color-selected',alpha=0.5,fmt='o',capsize=4)

plt.xlabel(r'$r [{\rm Mpc}/h]$')
plt.ylabel(r'$\xi(r)_{\rm HOD}/\xi(r)_{\rm TNG300}$')
plt.xscale('log')
plt.legend()
plt.ylim([0.6,1.4])
plt.savefig("figs/prof_ELG.png")
plt.show()


quit()

hist_norm, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges)
edges,hist_col,hist_cents_col,hist_sats_col = get_hist(count_halo_col_fp,count_halo_cents_col_fp,count_halo_sats_col_fp,Group_M_Mean200_fp,hist_norm,bin_edges)
edges,hist_sfg,hist_cents_sfg,hist_sats_sfg = get_hist(count_halo_sfg_fp,count_halo_cents_sfg_fp,count_halo_sats_sfg_fp,Group_M_Mean200_fp,hist_norm,bin_edges)
edges,hist_all,hist_cents_all,hist_sats_all = get_hist(count_halo_all_fp,count_halo_cents_all_fp,count_halo_sats_all_fp,Group_M_Mean200_fp,hist_norm,bin_edges)

want_hist = 0
if want_hist:
    # all populations
    hist_norm, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges)

    # color-selected galaxies
    edges,hist_col,hist_cents_col,hist_sats_col = get_hist(count_halo_col_fp,count_halo_cents_col_fp,count_halo_sats_col_fp,Group_M_Mean200_fp,hist_norm,bin_edges)
    plt.plot(10.**bin_cents,hist_col,color='dodgerblue',label='all')

    # all populations
    hist_norm, edges = np.histogram(Group_M_Mean200_dm,bins=10**bin_edges)
    # shuffled color-selected galaxies
    edges,hist_col,hist_cents_col,hist_sats_col = get_hist(count_halo_col_hod,count_halo_cents_col_hod,count_halo_sats_col_hod,Group_M_Mean200_dm,hist_norm,bin_edges)
    plt.plot(10.**bin_cents,hist_col,color='dodgerblue',ls='--',label='HOD')
    # SFGs
    #edges,hist_sfg,hist_cents_sfg,hist_sats_sfg = get_hist(count_halo_sfg_fp,count_halo_cents_sfg_fp,count_halo_sats_sfg_fp,Group_M_Mean200_fp,N_bin)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()


xyz_col_true = GroupPos_fp[count_halo_col_fp > 0]
w_col_true = count_halo_col_fp[count_halo_col_fp > 0].astype(xyz_col_true.dtype)
xyz_sfg_true = GroupPos_fp[count_halo_sfg_fp > 0]
w_sfg_true = count_halo_sfg_fp[count_halo_sfg_fp > 0].astype(xyz_sfg_true.dtype)
xyz_all_true = GroupPos_fp[count_halo_all_fp > 0]
w_all_true = count_halo_all_fp[count_halo_all_fp > 0].astype(xyz_all_true.dtype)

# TESTING
xyz_col_gals = SubhaloPos_fp[sub_id_col]
w_col_gals = np.ones(xyz_col_gals.shape[0],dtype=xyz_col_gals.dtype)

def get_pos_hod(count_halo_elg_hod):
    xyz_hod = GroupPos_dm[count_halo_elg_hod > 0]
    w_hod = count_halo_elg_hod[count_halo_elg_hod > 0].astype(xyz_hod.dtype)

    print(np.sum(w_col_true),np.sum(w_hod))
    print(len(w_col_true),len(w_hod))

    return xyz_hod, w_hod

count_halo_col_hod = get_hod_counts(hist_col,hist_cents_col,hist_sats_col,N_halos_dm,Group_M_Mean200_dm)
xyz_col_hod, w_col_hod = get_pos_hod(count_halo_col_hod)

count_halo_sfg_hod = get_hod_counts(hist_sfg,hist_cents_sfg,hist_sats_sfg,N_halos_dm,Group_M_Mean200_dm)
xyz_sfg_hod, w_sfg_hod = get_pos_hod(count_halo_sfg_hod)

count_halo_all_hod = get_hod_counts(hist_all,hist_cents_all,hist_sats_all,N_halos_dm,Group_M_Mean200_dm)
xyz_all_hod, w_all_hod = get_pos_hod(count_halo_all_hod)

plt.figure(1,figsize=(8,6))

Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_col_true,w_col_true,xyz_col_hod,w_col_hod,Lbox)
plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='dodgerblue',ls='-',label='color-selected',alpha=0.5,fmt='o',capsize=4)

Rat_hodgals_mean, Rat_hodgals_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_gals, Corr_err_gals, bin_centers = get_jack_corr(xyz_col_gals,w_col_gals,xyz_col_hod,w_col_hod,Lbox)
plt.errorbar(bin_centers,Rat_hodgals_mean,yerr=Rat_hodgals_err,color='dodgerblue',ls='--',label='color-selected gals',alpha=0.5,fmt='o',capsize=4)

Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_tru, Corr_err_tru, bin_centers = get_jack_corr(xyz_sfg_true,w_sfg_true,xyz_sfg_hod,w_sfg_hod,Lbox)
plt.errorbar(bin_centers*1.05,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='orange',ls='-',label='SFR-selected',alpha=0.5,fmt='o',capsize=4)

Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_tr, Corr_err_tr, bin_centers = get_jack_corr(xyz_all_true,w_all_true,xyz_all_hod,w_all_hod,Lbox)
plt.plot(bin_centers*0.95,Rat_hodtrue_mean,color='gray',ls='-',label='stellar-mass-selected')
plt.fill_between(bin_centers*0.95,Rat_hodtrue_mean-Rat_hodtrue_err,Rat_hodtrue_mean+Rat_hodtrue_err,color='gray',alpha=0.5)

plt.xlabel(r'$r [{\rm Mpc}/h]$')
plt.ylabel(r'$\xi(r)_{\rm HOD}/\xi(r)_{\rm TNG300}$')
plt.xscale('log')
plt.legend()
plt.ylim([0.6,1.4])
plt.savefig("figs/mock_ELG.png")



plt.figure(2,figsize=(8,6))
plt.errorbar(bin_centers,Corr_mean_true*bin_centers**2,yerr=Corr_err_true*bin_centers**2,color='dodgerblue',ls='-',label='color-selected',alpha=0.5,fmt='o',capsize=4)
plt.errorbar(bin_centers,Corr_mean_gals*bin_centers**2,yerr=Corr_err_gals*bin_centers**2,color='dodgerblue',ls='--',label='color-selected gals',alpha=0.5,fmt='o',capsize=4)
plt.xlabel(r'$r [{\rm Mpc}/h]$')
plt.ylabel(r'$\xi(r) r^2$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
quit()


N_div = 3
lss = ['--',':','-.']
for i in range(N_div):
    lo_bound = i/N_div
    hi_bound = (i+1)/N_div
    quant_lo = np.quantile(env_cw, lo_bound)
    quant_hi = np.quantile(env_cw, hi_bound)
    inds_cw = inds_halo[(env_cw > quant_lo) & (env_cw < quant_hi)]
    #sub_id_col_cw = np.intersect1d(inds_cw,SubhaloGrNr[sub_id_col])
    #sub_id_sfg_cw = np.intersect1d(inds_cw,SubhaloGrNr[sub_id_sfg])

    #count_halo_col_fp[inds_cw]
    #count_halo_sfg_fp[inds_cw]
    #Group_M_Mean200_fp[inds_cw]

    print("number of galaxies here = ", np.sum(count_halo_col_fp[inds_cw]))
    N_bin, edges = np.histogram(Group_M_Mean200_fp[inds_cw],bins=10**bin_edges)
    # color-selected galaxies
    edges,hist_col,hist_cents_col,hist_sats_col = get_hist(count_halo_col_fp[inds_cw],count_halo_cents_col_fp[inds_cw],count_halo_sats_col_fp[inds_cw],Group_M_Mean200_fp[inds_cw],N_bin)
    # SFGs
    #edges,hist_sfg,hist_cents_sfg,hist_sats_sfg = get_hist(count_halo_sfg_fp[inds_cw],count_halo_cents_sfg_fp[inds_cw],count_halo_sats_sfg_fp[inds_cw],Group_M_Mean200_fp[inds_cw],N_bin)
    plt.plot(10.**bin_cents,hist_col,color='dodgerblue',ls=lss[i],label=r'$%.1f-%.1f$'%(lo_bound*100.,hi_bound*100))

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
