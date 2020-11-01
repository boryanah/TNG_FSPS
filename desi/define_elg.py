import numpy as np
import matplotlib.pyplot as plt
from util import get_scatter, load_fsps
import h5py
import sys
from scipy.special import erf
from scipy.optimize import curve_fit
import Corrfunc

selection = '_'+sys.argv[1]
#selection = '_DESI'
#selection = '_eBOSS'

redshift_dict = {'055':['sdss_desi','_0.0'],#-0.5
                 '047':['sdss_desi','_-0.0'],
                 '041':['sdss_des','']}

# user specified choices
want_corrfunc = 0#1 # saves corr function values
want_selection = 0 # saves plots with grz selections 
want_hod = 1 # saves HOD histogram
want_env = int(sys.argv[3]) # record for the high and low percentiles only
if want_env:
    env_type = sys.argv[4]#'_high'#'_low'
else:
    env_type = ''
# snapshot number and dust model
snap = '0%d'%(int(sys.argv[2]))#'041'#'047'#'055'#'050'#'059'
snap_dir = '_'+str(int(snap))
dust_index = redshift_dict[snap][1]

# photometric scatter
want_photo_scatter = 1
if want_photo_scatter:
    photo_scatter = ''
else:
    photo_scatter = '_noscatter'

# load the synthetic colors
sub_id, ugriz_grizy, stellar_mass, flux_oii = load_fsps(snap,dust_index,cam_filt=redshift_dict[snap][0])

# log mass of the tng galaxies
log_mass_star = np.log10(stellar_mass)
print("# objects over logM = 10: ",np.sum(log_mass_star > 10.))

# mag limits
# DESI limits: 24., 23.4, 22.5
# eBOSS limits: 24.7, 23.9, 23.0
if selection == '_DESI':
    g_lim = 24.
    r_lim = 23.4
    z_lim = 22.5
elif selection == '_eBOSS':
    g_lim = 24.7
    r_lim = 23.9
    z_lim = 23.0
    
if want_photo_scatter:
    fac = 1.
    g_dec = get_scatter(ugriz_grizy[:,5],g_lim,factor=fac)
    r_dec = get_scatter(ugriz_grizy[:,6],r_lim,factor=fac)
    z_dec = get_scatter(ugriz_grizy[:,8],z_lim,factor=fac)
else:
    g_dec = ugriz_grizy[:,5]
    r_dec = ugriz_grizy[:,6]
    z_dec = ugriz_grizy[:,8]

# these are unphysical values
selection_dec = g_dec < 100000.
g_dec = g_dec[selection_dec]
r_dec = r_dec[selection_dec]
z_dec = z_dec[selection_dec]
log_mass_star = log_mass_star[selection_dec]
sub_id = sub_id[selection_dec]
flux_oii = flux_oii[selection_dec]
print("# physical galaxies = ",len(flux_oii))



# rz and gr colors
x_dec = r_dec-z_dec
y_dec = g_dec-r_dec
q_dec = g_dec

if selection == '_eBOSS':
    
    # survey limit
    survey_selection = (g_dec < g_lim) & (r_dec < r_lim) & (z_dec < z_lim)
    
    # eBOSS/ELG
    y_line1 = -0.068*x_dec+0.457
    y_line2 = 0.112*x_dec+0.773
    x_line1 = 0.218*y_dec+0.571 #NGC 0.637*y_dec+0.399
    x_line2 = -0.555*y_dec+1.901
    q_line1 = 21.825
    q_line2 = 22.9#SGC 22.825 NGC 22.9
    q_selection = (q_dec < q_line2) & (q_dec > q_line1)
    
    # eBOSS color selection
    col_selection = (x_line1 < x_dec) & (x_line2 > x_dec) & (y_line1 < y_dec) & (y_line2 > y_dec) & (q_dec < q_line2) & (q_dec > q_line1)
    col_selection &= survey_selection
    
elif selection == '_DESI':
    
    # survey limit
    survey_selection = (g_dec < g_lim) & (r_dec < r_lim) & (z_dec < z_lim)
    print("applying mag limits = ",np.sum(survey_selection))
    
    # DESI/ELG
    # star formation rate vs color color coding
    # g band magnitude star formation rate
    # cut on color sorta sSFR
    # together do something different
    # color disfavors high mass
    # disfavors
    # high mass
    # not that high
    # which of these halos
    # photometric noise
    # red galaxies scatter b/n stellar mass and halo mass as you walk up mostly results in all high mass halos have a galaxies
    # not enough star formation to get into the mag cut
    # SFR color plot 3d plot halo mass centrals in terms of color and flux
    # low mass aren't bright and high mass are too red
    # g band vs halo mass until sfr turn off until z = 1 increase too red
    # halo mass not going faint enough to get to things that are mostly blue 1.4 not getting faint enough
    brfa_sel = (q_dec > 20) & (q_dec < 23.6) #23.5 or 23.6
    blre_sel = (x_dec > 0.3) & (x_dec < 1.6) # removing the red cut recovers the central population
    star_sel = (y_dec < 1.15*x_dec-0.15) #-0.15 or -0.35
    oii_sel =  (y_dec < -1.20*x_dec+1.6) # this screws up a lot the centrals
    q_selection = brfa_sel
    
    # combining all selections
    col_selection = brfa_sel & blre_sel & star_sel & oii_sel
    col_selection &= survey_selection

print("# color-selected galaxies = ",np.sum(col_selection))

# apply color selection
x_dec_col = x_dec[col_selection]
y_dec_col = y_dec[col_selection]
sub_id_col = sub_id[col_selection]
log_mass_col = log_mass_star[col_selection]

# directory of the halo TNG data
box_name = "TNG300"
Lbox = 205.
root = "/mnt/gosling1/boryanah/"+box_name+"/"

# loading the halo mass and group identification
Group_M_Mean200_fp = np.load(root+'Group_M_Mean200_fp'+snap_dir+'.npy')*1.e10
SubhaloGrNr_fp = np.load(root+'SubhaloGrNr_fp'+snap_dir+'.npy')
SubhaloPos_fp = np.load(root+'SubhaloPos_fp'+snap_dir+'.npy')/1.e3

# load the info about the star formation rate
fdir = '/mnt/store1/boryanah/IllustrisTNG/CosmicWeb'
filename = 'WEB_CIC_256_DM_TNG300-2.hdf5'
f = h5py.File(fdir+filename, 'r')
sub_SFR = f['catsh_SubhaloSFR'][:]
sub_star_mass = f['catsh_SubhaloMassType'][:,4]
sub_ids = f['catsh_id'][:]
f.close()

# get the indices of the top SFGs
sub_id_eligible = sub_id[q_selection]#[q_selection & survey_selection]#new
print(len(sub_id_eligible))
sub_ids_eligible, comm1, comm2 = np.intersect1d(sub_id_eligible,sub_ids,assume_unique=True,return_indices=True)#new
print(len(sub_ids_eligible))
sub_SFR_eligible = sub_SFR[comm2]#[sub_id_eligible]#new
sub_star_mass_eligible = sub_star_mass[comm2]

x_dec_eligible = (x_dec[q_selection])[comm1]
y_dec_eligible = (y_dec[q_selection])[comm1]

np.save("data/x_dec_eligible"+snap_dir+selection+".npy",x_dec_eligible)
np.save("data/y_dec_eligible"+snap_dir+selection+".npy",y_dec_eligible)
np.save("data/sub_star_mass_eligible"+snap_dir+selection+".npy",sub_star_mass_eligible)
np.save("data/sub_SFR_eligible"+snap_dir+selection+".npy",sub_SFR_eligible)

sub_sSFR_eligible = (sub_SFR_eligible/sub_star_mass_eligible)
log_sSFR_eligible = np.log10(sub_sSFR_eligible)
log_SFR_eligible = np.log10(sub_SFR_eligible)
# avoid division by 0
log_sSFR_eligible[sub_SFR_eligible == 0.] = -13.
log_SFR_eligible[sub_SFR_eligible == 0.] = -3.

# sorting and finding top shiners
#i_sort = np.argsort(sub_SFR_eligible)[::-1]
i_sort = np.argsort(sub_sSFR_eligible)[::-1]
n_top = np.sum(col_selection)
i_sort = i_sort[:n_top]
sub_id_sfg = sub_ids_eligible[i_sort]#new
mass_sfg = sub_star_mass_eligible[i_sort]
sfr_sfg = log_SFR_eligible[i_sort]
ssfr_sfg = log_sSFR_eligible[i_sort]
#sub_id_sfg = sub_ids[i_sort]#old
#mass_sfg = sub_star_mass[i_sort]#old

# Fluxinosity selection
#flux_min = 8.e-17#1.e-16#8.e-17
#flux_selection = flux_oii > flux_min
#flux_oii = flux_oii[survey_selection]
flux_selection = (np.argsort(flux_oii)[::-1])[:n_top]
sub_id_flux = sub_id[flux_selection]

#sub_id_flux = sub_id[flux_selection]
print("# flux-selected galaxies = ",len(flux_selection))

# Selection for the most massive galaxies
sub_id_all = sub_ids[(np.argsort(sub_star_mass)[::-1])[:n_top]]#[:12000]]#[:n_top]
np.save("mock/data/sub_id"+env_type+snap_dir+selection+"_all.npy",sub_id_all)

if want_env:
    filename = '/home/boryanah/lars/LSSIllustrisTNG/CosmicWeb/WEB_CIC_256_DM_TNG'+box_name[-3:]+'-2.hdf5'
    f = h5py.File(filename, 'r')
    d_smooth = f['density_smooth'][:,:,:] 

    # finding who belongs where in the cosmic web
    N_dim = 256
    gr_size = Lbox/N_dim
    halo_x = SubhaloPos_fp[:,0]; halo_y = SubhaloPos_fp[:,1]; halo_z = SubhaloPos_fp[:,2]
    
    i_cw = (halo_x/gr_size).astype(int)
    j_cw = (halo_y/gr_size).astype(int)
    k_cw = (halo_z/gr_size).astype(int)
    i_cw[i_cw == N_dim] = N_dim-1 # fixing floating point issue
    j_cw[j_cw == N_dim] = N_dim-1 # fixing floating point issue
    k_cw[k_cw == N_dim] = N_dim-1 # fixing floating point issue
    
    # Environment definition
    env_cw = d_smooth[i_cw,j_cw,k_cw]

    perc_low = 0.5
    perc_high = 0.5
    env_col = env_cw[sub_id_col]
    if env_type == '_high':
        quart_col = np.quantile(env_col, perc_high)
        ind_env_col = env_col > quart_col
    elif env_type == '_low':
        quart_col = np.quantile(env_col, perc_low)
        ind_env_col = env_col < quart_col
    env_sfg = env_cw[sub_id_sfg]
    if env_type == '_high':
        quart_sfg = np.quantile(env_sfg, perc_high)
        ind_env_sfg = env_sfg > quart_sfg
    elif env_type == '_low':
        quart_sfg = np.quantile(env_sfg, perc_low)
        ind_env_sfg = env_sfg < quart_sfg
    env_flux = env_cw[sub_id_flux]
    if env_type == '_high':
        quart_flux = np.quantile(env_flux, perc_high)
        ind_env_flux = env_flux > quart_flux
    elif env_type == '_low':
        quart_flux = np.quantile(env_flux, perc_low)
        ind_env_flux = env_flux < quart_flux

    sub_id_col = sub_id_col[ind_env_col]
    sub_id_sfg = sub_id_sfg[ind_env_sfg]
    sub_id_flux = sub_id_flux[ind_env_flux]

if want_corrfunc:
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

    # record all
    np.save("data/bin_centers.npy",bin_centers)
    np.save("data/rat_colsfg"+env_type+snap_dir+selection+"_mean.npy",Rat_colsfg_mean)
    np.save("data/rat_colsfg"+env_type+snap_dir+selection+"_err.npy",Rat_colsfg_err)
    np.save("data/rat_oiisfg"+env_type+snap_dir+selection+"_mean.npy",Rat_oiisfg_mean)
    np.save("data/rat_oiisfg"+env_type+snap_dir+selection+"_err.npy",Rat_oiisfg_err)
    np.save("data/corr_mean"+env_type+snap_dir+selection+"_col.npy",Corr_mean_col)
    np.save("data/corr_err"+env_type+snap_dir+selection+"_col.npy",Corr_err_col)
    np.save("data/corr_mean"+env_type+snap_dir+selection+"_sfg.npy",Corr_mean_sfg)
    np.save("data/corr_err"+env_type+snap_dir+selection+"_sfg.npy",Corr_err_sfg)
    np.save("data/corr_mean"+env_type+snap_dir+selection+"_flux.npy",Corr_mean_flux)
    np.save("data/corr_err"+env_type+snap_dir+selection+"_flux.npy",Corr_err_flux)


if want_selection:
    # color vs mass
    inter, comm1, comm2 = np.intersect1d(sub_id_sfg,sub_id,assume_unique=True,return_indices=True)
    y_dec_sfg = y_dec[comm2]
    x_dec_sfg = x_dec[comm2]
    m_dec_sfg = 10.**log_mass_star[comm2]

    # fontsize nrows and ncols for plotting
    fs = 16
    nrows = 1
    ncols = 3
    plt.subplots(nrows,ncols,figsize=(ncols*5,nrows*5.5))

    x_line = np.linspace(np.min(x_dec_col),np.max(x_dec_col),100)
    y_line = np.linspace(np.min(y_dec_col),np.max(y_dec_col),100)
    if selection == '_DESI':
        y_line1 = -1.2*x_line+1.6
        y_line2 = 1.15*x_line-0.15
        x_line1 = 0.3*np.ones(len(x_line))
        x_line2 = 1.6*np.ones(len(x_line))
    elif selection == '_eBOSS':
        y_line1 = -0.068*x_line+0.457
        y_line2 = 0.112*x_line+0.773
        x_line1 = 0.218*y_line+0.571 #NGC 0.637*y_line+0.399
        x_line2 = -0.555*y_line+1.901
        
    plt.subplot(nrows,ncols,1)
    plt.plot(x_line,y_line1,'k')
    plt.plot(x_line,y_line2,'k')
    plt.plot(x_line1,y_line,'k')
    plt.plot(x_line2,y_line,'k')
    
    plt.scatter(x_dec_sfg,y_dec_sfg,s=0.05,label="star-forming")
    plt.scatter(x_dec_col,y_dec_col,s=0.05,label="color-selected")
    plt.legend(fontsize=fs)
    plt.xlabel(r"$r-z$",fontsize=fs)
    plt.ylabel(r"$g-r$",fontsize=fs)
    plt.ylim([-0.2,1.4])
    
    plt.subplot(nrows,ncols,2)
    plt.scatter(m_dec_sfg,y_dec_sfg,s=0.05,label="star-forming")
    plt.scatter(10.**log_mass_col,y_dec_col,s=0.05,label="color-selected")
    plt.xscale('log')
    plt.xlabel(r"$\log(M)$",fontsize=fs)
    plt.ylabel(r"$g-r$",fontsize=fs)
    plt.ylim([-0.2,1.4])


    # mass vs sSFR
    inter, comm1, comm2 = np.intersect1d(sub_id_col,sub_ids,assume_unique=True,return_indices=True)
    ssfr_col = log_sSFR[comm2]
    sfr_col = log_SFR[comm2]
    mass_col = sub_star_mass[comm2]
    print("0 ssfr's = ",np.sum(ssfr_col == -13.))

    plt.subplot(nrows,ncols,3)
    plt.scatter(mass_sfg,sfr_sfg,s=0.05,label="star-forming")
    plt.scatter(mass_col,sfr_col,s=0.05,label="color-selected")
    plt.xscale('log')
    plt.xlabel(r"$\log(M)$",fontsize=fs)
    plt.ylabel(r"$\log({\rm SFR})$",fontsize=fs)
    plt.ylim([0,3])
    plt.savefig("figs/elg_selection"+snap_dir+selection+".png")
    
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

def get_hist(count_halo_elg_fp,count_halo_cents_elg_fp,count_halo_sats_elg_fp):
    # create histograms for the ELGs
    hist, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges,weights=count_halo_elg_fp)
    hist_cents, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges,weights=count_halo_cents_elg_fp)
    hist_sats, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges,weights=count_halo_sats_elg_fp)
    hist_elg = hist/N_bin
    hist_cents_elg = hist_cents/N_bin
    hist_sats_elg = hist_sats/N_bin
    return edges,hist,hist_cents_elg,hist_sats_elg

np.save("mock/data/sub_id"+env_type+snap_dir+selection+"_col.npy",sub_id_col)
np.save("mock/data/sub_id"+env_type+snap_dir+selection+"_sfg.npy",sub_id_sfg)
np.save("mock/data/sub_id"+env_type+snap_dir+selection+"_flux.npy",sub_id_flux)

if want_hod:
    # get parent indices of the centrals and their subhalo indices in the original array
    unique_sub_grnr, firsts = np.unique(SubhaloGrNr_fp,return_index=True)

    count_halo_col_fp, count_halo_cents_col_fp, count_halo_sats_col_fp = get_counts(sub_id_col)
    count_halo_sfg_fp, count_halo_cents_sfg_fp, count_halo_sats_sfg_fp = get_counts(sub_id_sfg)
    count_halo_flux_fp, count_halo_cents_flux_fp, count_halo_sats_flux_fp = get_counts(sub_id_flux)
    #count_halo_sfg_fp, count_halo_cents_sfg_fp, count_halo_sats_sfg_fp = get_counts(sub_id_all)


    # histogram
    log_min = 11.
    log_max = 15.
    bin_edges = np.linspace(log_min,log_max,31)
    N_bin, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges)

    # color-selected galaxies
    edges,hist_unnorm_col,hist_cents_col,hist_sats_col = get_hist(count_halo_col_fp,count_halo_cents_col_fp,count_halo_sats_col_fp)
    # SFGs
    edges,hist_unnorm_sfg,hist_cents_sfg,hist_sats_sfg = get_hist(count_halo_sfg_fp,count_halo_cents_sfg_fp,count_halo_sats_sfg_fp)
    # OII emitters
    edges,hist_unnorm_flux,hist_cents_flux,hist_sats_flux = get_hist(count_halo_flux_fp,count_halo_cents_flux_fp,count_halo_sats_flux_fp)


    bin_cen = .5*(bin_edges[1:]+bin_edges[:-1])
    np.save("data/bin_cen.npy",bin_cen)
    np.save("data/hist_cents"+env_type+snap_dir+selection+"_col.npy",hist_cents_col)
    np.save("data/hist_sats"+env_type+snap_dir+selection+"_col.npy",hist_sats_col)
    np.save("data/hist_cents"+env_type+snap_dir+selection+"_sfg.npy",hist_cents_sfg)
    np.save("data/hist_sats"+env_type+snap_dir+selection+"_sfg.npy",hist_sats_sfg)
    np.save("data/hist_cents"+env_type+snap_dir+selection+"_flux.npy",hist_cents_flux)
    np.save("data/hist_sats"+env_type+snap_dir+selection+"_flux.npy",hist_sats_flux)

    np.save("data/hist_unnorm"+env_type+snap_dir+selection+"_col.npy",hist_unnorm_col)
    np.save("data/hist_unnorm"+env_type+snap_dir+selection+"_sfg.npy",hist_unnorm_sfg)
    np.save("data/hist_unnorm"+env_type+snap_dir+selection+"_flux.npy",hist_unnorm_flux)
