import numpy as np
import matplotlib.pyplot as plt
import h5py
import Corrfunc
from util import get_density, get_smooth_density, smooth_density, get_jack_corr, get_counts, get_hist, circles
import plotparams
plotparams.buba()

np.random.seed(3000)

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
SubhaloGrNr_fp = np.load(root+'SubhaloGrNr_fp'+snap_dir+'.npy')
SubhaloPos_fp = np.load(root+'SubhaloPos_fp'+snap_dir+'.npy')/1.e3
SubhaloVmaxRad_fp = np.load(root+'SubhaloVmaxRad_fp'+snap_dir+'.npy')/1.e3
GroupPos_fp = np.load(root+'GroupPos_fp'+snap_dir+'.npy')/1.e3
N_halos_fp = GroupPos_fp.shape[0]
inds_halo_fp = np.arange(N_halos_fp,dtype=int)

Group_M_Mean200_dm = np.load(root+'Group_M_Mean200_dm'+snap_dir+'.npy')*1.e10
GroupPos_dm = np.load(root+'GroupPos_dm'+snap_dir+'.npy')/1.e3
#SubhaloVmax_dm = np.load(root+'SubhaloVmax_dm'+snap_dir+'.npy')
SubhaloVmax_dm = np.load(root+'SubhaloVpeak_dm'+snap_dir+'.npy')
GroupFirstSub_dm = np.load(root+'GroupFirstSub_dm'+snap_dir+'.npy')
SubhaloPos_dm = np.load(root+'SubhaloPos_dm'+snap_dir+'.npy')/1.e3
GroupNsubs_dm = np.load(root+'GroupNsubs_dm'+snap_dir+'.npy')
N_halos_dm = GroupPos_dm.shape[0]
inds_halo_dm = np.arange(N_halos_dm,dtype=int)

# one option is to compute the density field from the positions of the group centers
#d_smooth = smooth_density(get_density(GroupPos_fp))
# another is to use the particle positions
#filename = '/home/boryanah/lars/LSSIllustrisTNG/CosmicWeb/WEB_CIC_256_DM_TNG300-2.hdf5'
#f = h5py.File(filename, 'r')
#d_smooth = f['density_smooth'][:,:,:] 
#d_smooth = np.load(root+'smoothed_density_tng300-3'+snap_dir+'.npy')
pos_parts = np.load(root+'pos_parts_tng300-3'+snap_dir+'.npy')/1000.

def get_d_smooth(pos_parts,want_cosmic_web=1):
    if want_cosmic_web:
        N_dim = 128
        R = 4.
        lth = 0.1
        fac = 1
        depth = 0
        density = get_density(pos_parts,Lbox=Lbox,N_dim=N_dim)
        d_smooth, l1, l2, l3 = get_smooth_density(density,R=R,Lbox=Lbox,N_dim=N_dim,return_lambda=True)
        cosmic_web = np.zeros_like(d_smooth)
        cosmic_web[(l1 > lth)] = 3 # knot
        cosmic_web[(l2 > lth) & (l1 <= lth)] = 2 # filament
        cosmic_web[(l3 > lth) & (l2 <= lth)] = 1 # sheet
        cosmic_web[(l3 <= lth)] = 0 # void
        return d_smooth, cosmic_web, depth, N_dim, fac
    
    N_dim = 256
    R = 1.5
    fac = 2 # 256/128
    depth = 3
    density = get_density(pos_parts,Lbox=Lbox,N_dim=N_dim)
    d_smooth = get_smooth_density(density,R=R,Lbox=Lbox,N_dim=N_dim,return_lambda=False)

    return d_smooth, depth, N_dim, fac

want_cosmic_web = 1 # TESTING
if want_cosmic_web:
    d_smooth, cosmic_web, depth, N_dim, fac = get_d_smooth(pos_parts,want_cosmic_web=want_cosmic_web)
else:
    d_smooth, depth, N_dim, fac = get_d_smooth(pos_parts,want_cosmic_web=want_cosmic_web)

# finding who belongs where in the cosmic web
gr_size = Lbox/N_dim

xyz_col = SubhaloPos_fp[sub_id_col]/gr_size
xyz_sfg = SubhaloPos_fp[sub_id_sfg]/gr_size
xyz_all = SubhaloPos_fp[sub_id_all]/gr_size
rmax_col = SubhaloVmaxRad_fp[sub_id_col]/gr_size

ind_col = xyz_col.astype(int)
ind_all = xyz_all.astype(int)

def get_percentage(inds):
    i_cw = inds[:,0]
    j_cw = inds[:,1]
    k_cw = inds[:,2]
    env_cw = cosmic_web[i_cw,j_cw,k_cw]
    env, ind, count = np.unique(env_cw,return_index=1,return_counts=1)
    total = len(env_cw)
    for i in range(len(env)):
        print(env[i])
        print(count[i]*100./total)
    return

print("for the color-selected:")
get_percentage(ind_col)
print("for the mass-selected:")
get_percentage(ind_all)


chosen_slice = 30*fac


s = 120
al = 0.6
col_elg = 'dodgerblue'#'magenta'
col_all = 'red'#'tomato'#'midnightblue'
cm = 'Greys'#'jet'#cm = 'gist_stern'#'prism'#'jet'#'gist_stern'#'jet'

plt.figure(1,figsize=(8,7))
plt.imshow(np.log10(1.+d_smooth[:,:,chosen_slice]),cmap=cm)
chosen_inds = (xyz_col[:,2] > chosen_slice) & (xyz_col[:,2] <= chosen_slice+depth+1)
plt.scatter(xyz_col[chosen_inds,1],xyz_col[chosen_inds,0],color=col_elg,marker='*',s=s,alpha=al)
chosen_inds = (xyz_all[:,2] > chosen_slice) & (xyz_all[:,2] <= chosen_slice+depth+1)
plt.scatter(xyz_all[chosen_inds,1],xyz_all[chosen_inds,0],color=col_all,marker='*',s=s,alpha=al)
plt.xlim([-0.5,N_dim+0.5])
plt.ylim([-0.5,N_dim+0.5])
plt.xlabel(r"$X [\rm{Mpc}/h]$")
plt.ylabel(r"$Y [\rm{Mpc}/h]$")
plt.gca().axes.yaxis.set_ticklabels([])
plt.gca().axes.xaxis.set_ticklabels([])
if not want_cosmic_web:
    #plt.savefig("figs/density_2d.png")
    plt.savefig("../paper/density_2d.pdf")

if want_cosmic_web:
    plt.figure(2,figsize=(8,7))
    cm = 'Greys'#'jet'#cm = 'gist_stern'#'prism'#'jet'#'gist_stern'#'jet'
    plt.imshow(cosmic_web[:,:,chosen_slice],cmap=cm)
    chosen_inds = (xyz_col[:,2] > chosen_slice) & (xyz_col[:,2] <= chosen_slice+depth+1)
    plt.scatter(xyz_col[chosen_inds,1],xyz_col[chosen_inds,0],color=col_elg,marker='*',s=s,alpha=al)
    chosen_inds = (xyz_all[:,2] > chosen_slice) & (xyz_all[:,2] <= chosen_slice+depth+1)
    plt.scatter(xyz_all[chosen_inds,1],xyz_all[chosen_inds,0],color=col_all,marker='*',s=s,alpha=al)
    plt.xlim([-0.5,N_dim+0.5])
    plt.ylim([-0.5,N_dim+0.5])
    plt.xlabel(r"$X [\rm{Mpc}/h]$")
    plt.ylabel(r"$Y [\rm{Mpc}/h]$")
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.xaxis.set_ticklabels([])
    #plt.savefig("figs/cosmic_web_2d.png")
    plt.savefig("../paper/cosmic_web_2d.pdf")
plt.show()
