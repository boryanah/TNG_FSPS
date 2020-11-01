import numpy as np
import matplotlib.pyplot as plt
import h5py
import Corrfunc
from util import get_density, smooth_density, get_jack_corr, get_counts, get_counts_and_nstart, get_hist
import plotparams
plotparams.buba()

#np.random.seed(100000000)
#np.random.seed(1000000)
#np.random.seed(100000)
#np.random.seed(100)
np.random.seed(2)
#np.random.seed(3000000)

def plot_ratio(xyz_true,w_true,xyz_hod,w_hod,Lbox,color,label,offset=1.,rmin=2.):
    Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_true,w_true,xyz_hod,w_hod,Lbox)

    if color == 'black':
        plt.plot(bin_centers,Rat_hodtrue_mean,color=color,ls='-',label=label)
        plt.fill_between(bin_centers,Rat_hodtrue_mean-Rat_hodtrue_err,Rat_hodtrue_mean+Rat_hodtrue_err,color=color,alpha=.1)
    else:
        plt.errorbar(bin_centers*offset,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color=color,ls='-',label=label,alpha=1.,fmt='o',capsize=4)

    mean_val = np.mean(Rat_hodtrue_mean[np.isfinite(Rat_hodtrue_mean) & (bin_centers > rmin)])

    print('mean_val = ',mean_val)
    x = np.linspace(rmin,10,3)
    line = np.linspace(0.,20,3)
    plt.plot(line,np.ones(len(line)),'k--')
    plt.xlim([0.095,12])
    plt.plot(x,np.ones(len(x))*mean_val,color=color,ls='--',lw=1.5)


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

# START HERE
'''
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
#Group_M_Mean200_dm = np.load(root+'Group_M_Mean200_dm'+snap_dir+'.npy')*1.e10
#GroupFirst_dm = np.load(root+'GroupFirst_dm'+snap_dir+'.npy')
#GroupPos_dm = np.load(root+'GroupPos_dm'+snap_dir+'.npy')/1.e3

N_halos_fp = GroupPos_fp.shape[0]
inds_halo = np.arange(N_halos_fp,dtype=int)
#N_halos_dm = GroupPos_dm.shape[0]


# get parent indices of the centrals and their subhalo indices in the original array
unique_sub_grnr, firsts = np.unique(SubhaloGrNr_fp,return_index=True)

# TESTING alternative center def doesn't make a difference
#GroupPos_fp *= 0.
#GroupPos_fp[unique_sub_grnr] = SubhaloPos_fp[firsts]

want_fp_pos = 1
if want_fp_pos:
    count_halo_col_fp, count_halo_cents_col_fp, count_halo_sats_col_fp, nstart_halo_col_fp = get_counts_and_nstart(SubhaloGrNr_fp,firsts,N_halos_fp,sub_id_col)
    count_halo_sfg_fp, count_halo_cents_sfg_fp, count_halo_sats_sfg_fp, nstart_halo_sfg_fp = get_counts_and_nstart(SubhaloGrNr_fp,firsts,N_halos_fp,sub_id_sfg)
    count_halo_all_fp, count_halo_cents_all_fp, count_halo_sats_all_fp, nstart_halo_all_fp = get_counts_and_nstart(SubhaloGrNr_fp,firsts,N_halos_fp,sub_id_all)
else:
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
        ind = np.arange(np.sum(choice),dtype=int)
        np.random.shuffle(ind)
        count_halo_shuff[choice] = count[ind]

    print(np.sum(count_halo_shuff))
    return count_halo_shuff

def get_shuff_counts_and_nstart(count_halo_fp,nstart_halo_fp):
    log_min = 11.
    log_max = 15.
    N_bins = 41
    bin_edges = np.logspace(log_min,log_max,N_bins)
    bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5
    count_halo_shuff = np.zeros(N_halos_fp,dtype=int)
    nstart_halo_shuff = np.zeros(N_halos_fp,dtype=int)
    for i  in range(N_bins-1):
        choice = (bin_edges[i] < Group_M_Mean200_fp) & (bin_edges[i+1] > Group_M_Mean200_fp)
        count = count_halo_fp[choice]
        nst = nstart_halo_fp[choice]
        ind = np.arange(np.sum(choice),dtype=int)
        np.random.shuffle(ind)
        count_halo_shuff[choice] = count[ind]
        nstart_halo_shuff[choice] = nst[ind]

    print(np.sum(count_halo_shuff))
    return count_halo_shuff, nstart_halo_shuff

def get_xyz_fixed(sub_id,count_halo_fp,count_halo_shuff,nstart_halo_fp,nstart_halo_shuff):

    # sorting necessary cause we sort in get_counts_and_nstart
    sub_id = np.sort(sub_id)
    xyz_true = SubhaloPos_fp[sub_id]
    w_true = np.ones(xyz_true.shape[0],dtype=xyz_true.dtype)
    grnr_true = SubhaloGrNr_fp[sub_id]
    grpos_true = GroupPos_fp[grnr_true]
    
    # TESTING - is exactly consistent with 1!
    #positive = count_halo_fp > 0
    #nst = nstart_halo_fp[positive]
    #cou = count_halo_fp[positive]
    #gr_cen = GroupPos_fp[positive]
    # og
    positive = count_halo_shuff > 0
    nst = nstart_halo_shuff[positive]
    cou = count_halo_shuff[positive]
    gr_cen = GroupPos_fp[positive]
    

    xyz_hod = np.zeros(xyz_true.shape)
    for i in range(len(nst)):
        grc = gr_cen[i]
        grc_true = grpos_true[nst[i]]
        pos_diff = xyz_true[nst[i]:nst[i]+cou[i]]-grc_true#xyz_true[nst[i]]#grc#xyz_true[nst[i]]
        new_pos = grc+pos_diff

        xyz_hod[nst[i]:nst[i]+cou[i]] = new_pos
        

    xyz_hod[xyz_hod >= Lbox] -= Lbox
    xyz_hod[xyz_hod <= 0.] += Lbox 

    w_hod = np.ones(xyz_hod.shape[0],dtype=xyz_hod.dtype)

    return xyz_true, w_true, xyz_hod, w_hod
    
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

if want_fp_pos:
    count_halo_col_shuff, nstart_halo_col_shuff = get_shuff_counts_and_nstart(count_halo_col_fp,nstart_halo_col_fp)
    count_halo_sfg_shuff, nstart_halo_sfg_shuff = get_shuff_counts_and_nstart(count_halo_sfg_fp,nstart_halo_sfg_fp)
    count_halo_all_shuff, nstart_halo_all_shuff = get_shuff_counts_and_nstart(count_halo_all_fp,nstart_halo_all_fp)
else:
    count_halo_col_shuff = get_shuff_counts(count_halo_col_fp)
    count_halo_sfg_shuff = get_shuff_counts(count_halo_sfg_fp)
    count_halo_all_shuff = get_shuff_counts(count_halo_all_fp)


#want_fp_pos = 0
if want_fp_pos:
    xyz_col_true, w_col_true, xyz_col_hod, w_col_hod = get_xyz_fixed(sub_id_col,count_halo_col_fp,count_halo_col_shuff,nstart_halo_col_fp,nstart_halo_col_shuff)
    xyz_sfg_true, w_sfg_true, xyz_sfg_hod, w_sfg_hod = get_xyz_fixed(sub_id_sfg,count_halo_sfg_fp,count_halo_sfg_shuff,nstart_halo_sfg_fp,nstart_halo_sfg_shuff)
    xyz_all_true, w_all_true, xyz_all_hod, w_all_hod = get_xyz_fixed(sub_id_all,count_halo_all_fp,count_halo_all_shuff,nstart_halo_all_fp,nstart_halo_all_shuff)
else:
    xyz_col_true, w_col_true, xyz_col_hod, w_col_hod = get_xyz(sub_id_col,count_halo_col_fp,count_halo_col_shuff)
    xyz_sfg_true, w_sfg_true, xyz_sfg_hod, w_sfg_hod = get_xyz(sub_id_sfg,count_halo_sfg_fp,count_halo_sfg_shuff)
    xyz_all_true, w_all_true, xyz_all_hod, w_all_hod = get_xyz(sub_id_all,count_halo_all_fp,count_halo_all_shuff)

np.save("data_shuff/xyz_col_true.npy",xyz_col_true)
np.save("data_shuff/w_col_true.npy",w_col_true)
np.save("data_shuff/xyz_col_hod.npy",xyz_col_hod)
np.save("data_shuff/w_col_hod.npy",w_col_hod)
np.save("data_shuff/xyz_sfg_true.npy",xyz_sfg_true)
np.save("data_shuff/w_sfg_true.npy",w_sfg_true)
np.save("data_shuff/xyz_sfg_hod.npy",xyz_sfg_hod)
np.save("data_shuff/w_sfg_hod.npy",w_sfg_hod)
np.save("data_shuff/xyz_all_true.npy",xyz_all_true)
np.save("data_shuff/w_all_true.npy",w_all_true)
np.save("data_shuff/xyz_all_hod.npy",xyz_all_hod)
np.save("data_shuff/w_all_hod.npy",w_all_hod)

# END HERE
'''
xyz_col_true = np.load("data_shuff/xyz_col_true.npy")
w_col_true = np.load("data_shuff/w_col_true.npy")
xyz_col_hod = np.load("data_shuff/xyz_col_hod.npy")
w_col_hod = np.load("data_shuff/w_col_hod.npy")
xyz_sfg_true = np.load("data_shuff/xyz_sfg_true.npy")
w_sfg_true = np.load("data_shuff/w_sfg_true.npy")
xyz_sfg_hod = np.load("data_shuff/xyz_sfg_hod.npy")
w_sfg_hod = np.load("data_shuff/w_sfg_hod.npy")
xyz_all_true = np.load("data_shuff/xyz_all_true.npy")
w_all_true = np.load("data_shuff/w_all_true.npy")
xyz_all_hod = np.load("data_shuff/xyz_all_hod.npy")
w_all_hod = np.load("data_shuff/w_all_hod.npy")


plt.figure(figsize=(8,6))

plot_ratio(xyz_col_true,w_col_true,xyz_col_hod,w_col_hod,Lbox=Lbox,color='dodgerblue',label='color-selected',rmin=.8)
plot_ratio(xyz_sfg_true,w_sfg_true,xyz_sfg_hod,w_sfg_hod,Lbox=Lbox,offset=1.05,color='#CC6677',label='SFR-selected',rmin=.8)
plot_ratio(xyz_all_true,w_all_true,xyz_all_hod,w_all_hod,Lbox=Lbox,color='black',label='mass-selected',rmin=2.)

plt.xlabel(r'$r [{\rm Mpc}/h]$')
plt.ylabel(r'$\xi(r)_{\rm HOD}/\xi(r)_{\rm TNG300}$')
plt.xscale('log')
plt.legend()
plt.ylim([0.6,1.4])
#plt.savefig("figs/shuff_ELG.png")
plt.savefig("../paper/shuff_ELG.pdf")
plt.show()
