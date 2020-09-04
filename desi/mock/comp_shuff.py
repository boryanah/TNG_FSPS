import numpy as np
import matplotlib.pyplot as plt
import h5py
import Corrfunc
from util import get_density, smooth_density

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
# TESTING
sub_id_col = np.load("data/sub_id"+env_type+snap_dir+selection+"_col.npy")
#sub_id_col = np.load("data/sub_id"+env_type+snap_dir+selection+"_all.npy")# SUPER INTERSTING RESUTL
#sub_id_col = np.load("data/sub_id"+env_type+snap_dir+selection+"_sfg.npy")
sub_id_sfg = np.load("data/sub_id"+env_type+snap_dir+selection+"_sfg.npy")
sub_id_flux = np.load("data/sub_id"+env_type+snap_dir+selection+"_flux.npy")

# loading the halo mass and group identification
Group_M_Mean200_fp = np.load(root+'Group_M_Mean200_fp'+snap_dir+'.npy')*1.e10
SubhaloGrNr_fp = np.load(root+'SubhaloGrNr_fp'+snap_dir+'.npy')
SubhaloPos_fp = np.load(root+'SubhaloPos_fp'+snap_dir+'.npy')/1.e3
# Should be this
GroupPos_fp = np.load(root+'GroupPos_fp'+snap_dir+'.npy')/1.e3

# temporary solution
#sub_grnr, inds = np.unique(SubhaloGrNr_fp,return_index=True)
#GroupPos_fp = SubhaloPos_fp[inds]

N_halos = GroupPos_fp.shape[0]
Group_M_Mean200_fp = Group_M_Mean200_fp[:N_halos]
inds_halo = np.arange(N_halos,dtype=int)

d_smooth = smooth_density(get_density(GroupPos_fp))
#filename = '/home/boryanah/lars/LSSIllustrisTNG/CosmicWeb/WEB_CIC_256_DM_TNG'+box_name[-3:]+'-2.hdf5'
#f = h5py.File(filename, 'r')
#d_smooth = f['density_smooth'][:,:,:] 

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


def get_counts(sub_id_elg):

    # index of the parent halo and mass of parent halo for each elg
    parent_elg_fp = SubhaloGrNr_fp[sub_id_elg]
    parent_cents_elg_fp = SubhaloGrNr_fp[np.intersect1d(sub_id_elg,firsts)]
    parent_sats_elg_fp = SubhaloGrNr_fp[np.setdiff1d(sub_id_elg,firsts)]
    print(len(parent_elg_fp),len(parent_cents_elg_fp),len(parent_sats_elg_fp))

    # for each halo, how many elg point to it (excluding case of 0) and index of halo
    parent_elg_un_fp, counts = np.unique(parent_elg_fp,return_counts=True)
    parent_cents_elg_un_fp, counts_cents = np.unique(parent_cents_elg_fp,return_counts=True)
    parent_sats_elg_un_fp, counts_sats = np.unique(parent_sats_elg_fp,return_counts=True)

    # counts of elgs for all halos
    count_halo_elg_fp = np.zeros(len(Group_M_Mean200_fp),dtype=int)
    count_halo_cents_elg_fp = np.zeros(len(Group_M_Mean200_fp),dtype=int)
    count_halo_sats_elg_fp = np.zeros(len(Group_M_Mean200_fp),dtype=int)
    count_halo_elg_fp[parent_elg_un_fp] += counts
    count_halo_cents_elg_fp[parent_cents_elg_un_fp] += counts_cents
    count_halo_sats_elg_fp[parent_sats_elg_un_fp] += counts_sats
    print("sum halo fp = ",np.sum(count_halo_elg_fp))

    return count_halo_elg_fp, count_halo_cents_elg_fp, count_halo_sats_elg_fp

def get_hist(count_halo_elg_fp,count_halo_cents_elg_fp,count_halo_sats_elg_fp,group_mass,hist_norm):
    # create histograms for the ELGs
    hist, edges = np.histogram(group_mass,bins=10**bin_edges,weights=count_halo_elg_fp)
    hist_cents, edges = np.histogram(group_mass,bins=10**bin_edges,weights=count_halo_cents_elg_fp)
    hist_sats, edges = np.histogram(group_mass,bins=10**bin_edges,weights=count_halo_sats_elg_fp)
    hist_elg = hist/hist_norm
    hist_cents_elg = hist_cents/hist_norm
    hist_sats_elg = hist_sats/hist_norm
    return edges,hist_elg,hist_cents_elg,hist_sats_elg


# get parent indices of the centrals and their subhalo indices in the original array
unique_sub_grnr, firsts = np.unique(SubhaloGrNr_fp,return_index=True)

count_halo_col_fp, count_halo_cents_col_fp, count_halo_sats_col_fp = get_counts(sub_id_col)
#count_halo_sfg_fp, count_halo_cents_sfg_fp, count_halo_sats_sfg_fp = get_counts(sub_id_sfg)
#count_halo_flux_fp, count_halo_cents_flux_fp, count_halo_sats_flux_fp = get_counts(sub_id_flux)


# histogram
log_min = 11.
log_max = 15.
N_bins = 21
bin_edges = np.linspace(log_min,log_max,N_bins)
bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5

count_halo_col_fp_shuff = np.zeros(N_halos,dtype=int)
# in the subsequent code I will shuffle the occupations in mass bins
for i in range(N_bins-1):
    M_lo = 10.**bin_edges[i]
    M_hi = 10.**bin_edges[i+1]

    mass_sel = (M_lo < Group_M_Mean200_fp) & (M_hi > Group_M_Mean200_fp)

    count_sel = count_halo_col_fp[mass_sel]

    np.random.shuffle(count_sel)

    count_halo_col_fp_shuff[mass_sel] = count_sel


print(np.sum(count_halo_col_fp),np.sum(count_halo_col_fp_shuff))

x_fp = GroupPos_fp[count_halo_col_fp > 0,0]
y_fp = GroupPos_fp[count_halo_col_fp > 0,1]
z_fp = GroupPos_fp[count_halo_col_fp > 0,2]
w_fp = count_halo_col_fp[count_halo_col_fp > 0].astype(x_fp.dtype)

x_fp_shuff = GroupPos_fp[count_halo_col_fp_shuff > 0,0]
y_fp_shuff = GroupPos_fp[count_halo_col_fp_shuff > 0,1]
z_fp_shuff = GroupPos_fp[count_halo_col_fp_shuff > 0,2]
w_fp_shuff = count_halo_col_fp_shuff[count_halo_col_fp_shuff > 0].astype(x_fp_shuff.dtype)


# bins for the correlation function
N_bin = 21
bins = np.logspace(-1.,1.,N_bin)
bin_centers = (bins[:-1] + bins[1:])/2.


res_fp = Corrfunc.theory.xi(X=x_fp,Y=y_fp,Z=z_fp,weights=w_fp,boxsize=Lbox,nthreads=16,binfile=bins)['xi']
res_fp_shuff = Corrfunc.theory.xi(X=x_fp_shuff,Y=y_fp_shuff,Z=z_fp_shuff,weights=w_fp_shuff,boxsize=Lbox,nthreads=16,binfile=bins)['xi']


plt.figure(1)
plt.plot(bin_centers,res_fp,label='ELGs')
plt.plot(bin_centers,res_fp_shuff,label='shuffled')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.figure(2)
plt.plot(bin_centers,res_fp_shuff/res_fp,label='shuff/ELGs')
plt.xscale('log')
plt.legend()
plt.ylim([0.8,1.1])
plt.show()

'''
# all populations
N_bin, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges)

# color-selected galaxies
edges,hist_col,hist_cents_col,hist_sats_col = get_hist(count_halo_col_fp,count_halo_cents_col_fp,count_halo_sats_col_fp,Group_M_Mean200_fp,N_bin)
plt.plot(10.**bin_cents,hist_col,color='dodgerblue',label='all before')

# shuffled color-selected galaxies
edges,hist_col,hist_cents_col,hist_sats_col = get_hist(count_halo_col_fp_shuff,count_halo_cents_col_fp,count_halo_sats_col_fp,Group_M_Mean200_fp,N_bin)
# SFGs
#edges,hist_sfg,hist_cents_sfg,hist_sats_sfg = get_hist(count_halo_sfg_fp,count_halo_cents_sfg_fp,count_halo_sats_sfg_fp,Group_M_Mean200_fp,N_bin)
plt.plot(10.**bin_cents,hist_col,color='dodgerblue',label='all')

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
quit()
'''

N_div = 3
los = [0.,0.5,0.8]
his = [0.5,0.8,1.]
lss = ['--',':','-.']
for i in range(N_div):
    lo_bound = los[i]#i/N_div
    hi_bound = his[i]#(i+1)/N_div
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


quit()
# histogram
log_min = 11.
log_max = 15.
bin_edges = np.linspace(log_min,log_max,31)
N_bin, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges)

# color-selected galaxies
edges,hist_col,hist_cents_col,hist_sats_col = get_hist(count_halo_col_fp,count_halo_cents_col_fp,count_halo_sats_col_fp)
# SFGs
edges,hist_sfg,hist_cents_sfg,hist_sats_sfg = get_hist(count_halo_sfg_fp,count_halo_cents_sfg_fp,count_halo_sats_sfg_fp)
# OII emitters
edges,hist_flux,hist_cents_flux,hist_sats_flux = get_hist(count_halo_flux_fp,count_halo_cents_flux_fp,count_halo_sats_flux_fp)


bin_cen = .5*(bin_edges[1:]+bin_edges[:-1])


quit()
np.save("data/bin_cen.npy",bin_cen)
np.save("data/hist_cents"+env_type+snap_dir+selection+"_col.npy",hist_cents_col)
np.save("data/hist_sats"+env_type+snap_dir+selection+"_col.npy",hist_sats_col)
np.save("data/hist_cents"+env_type+snap_dir+selection+"_sfg.npy",hist_cents_sfg)
np.save("data/hist_sats"+env_type+snap_dir+selection+"_sfg.npy",hist_sats_sfg)
np.save("data/hist_cents"+env_type+snap_dir+selection+"_flux.npy",hist_cents_flux)
np.save("data/hist_sats"+env_type+snap_dir+selection+"_flux.npy",hist_sats_flux)

if True:
        # where are the galaxies located in real space
    xyz_col = SubhaloPos_fp[sub_id_col]
    xyz_sfg = SubhaloPos_fp[sub_id_sfg]
    xyz_flux = SubhaloPos_fp[sub_id_flux]
    print(xyz_flux.shape)
    print(xyz_sfg.shape)
    print(xyz_col.shape)

    # bins for the correlation function
    N_bin = 21
    bins = np.logspace(-1.,1.5,N_bin)
    bin_centers = (bins[:-1] + bins[1:])/2.


    # dimensions for jackknifing
    N_dim = 3

    # empty arrays to record data
    Rat_colsfg = np.zeros((N_bin-1,N_dim**3))
    Rat_oiisfg = np.zeros((N_bin-1,N_dim**3))
    Corr_col = np.zeros((N_bin-1,N_dim**3))
    Corr_sfg = np.zeros((N_bin-1,N_dim**3))
    Corr_flux = np.zeros((N_bin-1,N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                xyz_col_jack = xyz_col.copy()
                xyz_flux_jack = xyz_flux.copy()
                xyz_sfg_jack = xyz_sfg.copy()
            
                xyz = np.array([i_x,i_y,i_z],dtype=int)
                size = Lbox/N_dim

                bool_arr = np.prod((xyz == (xyz_col/size).astype(int)),axis=1).astype(bool)
                xyz_col_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_col_jack = xyz_col_jack[np.sum(xyz_col_jack,axis=1)!=0.]

                bool_arr = np.prod((xyz == (xyz_sfg/size).astype(int)),axis=1).astype(bool)
                xyz_sfg_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_sfg_jack = xyz_sfg_jack[np.sum(xyz_sfg_jack,axis=1)!=0.]


                bool_arr = np.prod((xyz == (xyz_flux/size).astype(int)),axis=1).astype(bool)
                xyz_flux_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_flux_jack = xyz_flux_jack[np.sum(xyz_flux_jack,axis=1)!=0.]

                res_col = Corrfunc.theory.xi(X=xyz_col_jack[:,0],Y=xyz_col_jack[:,1],Z=xyz_col_jack[:,2],boxsize=Lbox,nthreads=16,binfile=bins)
                res_sfg = Corrfunc.theory.xi(X=xyz_sfg_jack[:,0],Y=xyz_sfg_jack[:,1],Z=xyz_sfg_jack[:,2],boxsize=Lbox,nthreads=16,binfile=bins)
                res_flux = Corrfunc.theory.xi(X=xyz_flux_jack[:,0],Y=xyz_flux_jack[:,1],Z=xyz_flux_jack[:,2],boxsize=Lbox,nthreads=16,binfile=bins)

                rat_colsfg = res_col['xi']/res_sfg['xi']
                rat_oiisfg = res_flux['xi']/res_sfg['xi']
                Rat_colsfg[:,i_x+N_dim*i_y+N_dim**2*i_z] = rat_colsfg
                Rat_oiisfg[:,i_x+N_dim*i_y+N_dim**2*i_z] = rat_oiisfg
                Corr_col[:,i_x+N_dim*i_y+N_dim**2*i_z] = res_col['xi']
                Corr_sfg[:,i_x+N_dim*i_y+N_dim**2*i_z] = res_sfg['xi']
                Corr_flux[:,i_x+N_dim*i_y+N_dim**2*i_z] = res_flux['xi']

    # compute mean and error
    Rat_colsfg_mean = np.mean(Rat_colsfg,axis=1)
    Rat_colsfg_err = np.sqrt(N_dim**3-1)*np.std(Rat_colsfg,axis=1)
    Rat_oiisfg_mean = np.mean(Rat_oiisfg,axis=1)
    Rat_oiisfg_err = np.sqrt(N_dim**3-1)*np.std(Rat_oiisfg,axis=1)
    Corr_mean_col = np.mean(Corr_col,axis=1)
    Corr_err_col = np.sqrt(N_dim**3-1)*np.std(Corr_col,axis=1)
    Corr_mean_sfg = np.mean(Corr_sfg,axis=1)
    Corr_err_sfg = np.sqrt(N_dim**3-1)*np.std(Corr_sfg,axis=1)
    Corr_mean_flux = np.mean(Corr_flux,axis=1)
    Corr_err_flux = np.sqrt(N_dim**3-1)*np.std(Corr_flux,axis=1)
