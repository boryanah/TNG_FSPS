import numpy as np
import matplotlib.pyplot as plt
import fsps
import h5py

# parameter choices
tng = 'tng300'
snap = '055'#'041'#'055'#'047'#'055'#'050'#'059'
if snap == '040': redshift = 1.50; snap_dir = '_40'
if snap == '050': redshift = 1.00; snap_dir = '_50'
if snap == '059': redshift = 0.70; snap_dir = '_59'
if snap == '099': redshift = 0.06; snap_dir = ''

if snap == '036': redshift = 1.74234; snap_dir = '_36'
if snap == '041': redshift = 1.41131; snap_dir = '_41'; n_gal = 64400
if snap == '047': redshift = 1.11358; snap_dir = '_47'; n_gal = 72000
if snap == '055': redshift = 0.81947; snap_dir = '_55'; n_gal = 80000
if snap == '067': redshift = 0.50000; snap_dir = '_67'
cam_filt = 'sdss_des'
filts = cam_filt.split('_')
bands = []
for i in range(len(filts)):
    bands += fsps.find_filter(filts[i])
print(bands)
sfh_ap = '_30kpc'#'_3rad'#'_30kpc'#''
idx_start = 0
want_reddening = '_red'#'_red'
want_scatter = ''#'_scatter'

# use with TNG100
#n_gal = 20000
#fsps_output = np.load("../mags_data/big/"+tng+"_"+snap+"_"+cam_filt+"_id_ugriz_mass_"+str(idx_start)+"_"+str(idx_start+n_gal)+"_tauV_SFH"+sfh_ap+".npy")
# TNG300
fsps_output = np.load("../mags_data/big/"+tng+"_"+snap+"_"+cam_filt+"_id_ugriz_mass_lum_"+str(idx_start)+"_"+str(idx_start+n_gal)+"_tauV_SFH"+sfh_ap+want_reddening+want_scatter+".npy")

ugriz_grizy = fsps_output[:,1:len(bands)+1]
stellar_mass = fsps_output[:,1+len(bands)]
sub_id = fsps_output[:,0].astype(int)

g_dec_sp = ugriz_grizy[:,5]
r_dec_sp = ugriz_grizy[:,6]
z_dec_sp = ugriz_grizy[:,8]
flux_oii_sp = np.max(fsps_output[:,-2:],axis=1)
log_mass_star = np.log10(stellar_mass)

print("over logM = 10 = ",np.sum(log_mass_star > 10.))

selection_dec_sp = g_dec_sp < 100000.
g_dec_sp = g_dec_sp[selection_dec_sp]
r_dec_sp = r_dec_sp[selection_dec_sp]
z_dec_sp = z_dec_sp[selection_dec_sp]
log_mass_star = log_mass_star[selection_dec_sp]
sub_id = sub_id[selection_dec_sp]
flux_oii_sp = flux_oii_sp[selection_dec_sp]

print("Total gals = ",len(flux_oii_sp))

x_dec = r_dec_sp-z_dec_sp
y_dec = g_dec_sp-r_dec_sp
z_dec = g_dec_sp

# eBOSS
y_line1 = -0.068*x_dec+0.457
y_line2 = 0.112*x_dec+0.773
x_line1 = 0.218*y_dec+0.571
#x_line1 = 0.637*y_dec+0.399
x_line2 = -0.555*y_dec+1.901
z_line1 = 21.825#21.825
z_line2 = 22.9#22.825

# eBOSS color selection
#col_selection = (x_line1 < x_dec) & (x_line2 > x_dec) & (y_line1 < y_dec) & (y_line2 > y_dec) & (z_dec < z_line2) & (z_dec > z_line1) & (log_mass_star > 10.) 
#col_selection = (z_dec < z_line2) & (z_dec > z_line1) & (log_mass_star > 10.)

# DESI color selection
brfa_sel = (z_dec > 20) & (z_dec < 23.5) #23.6
blre_sel = (x_dec > 0.3) & (x_dec < 1.6) # removing the red cut recovers the central population
star_sel = (y_dec < 1.15*x_dec-0.15) #-0.35
oii_sel =  (y_dec < 1.20*x_dec+1.6)

col_selection = brfa_sel & blre_sel & star_sel & oii_sel
#col_selection = oii_sel
print("col galaxies = ",np.sum(col_selection))

# apply color selection
x_dec_col = x_dec[col_selection]
y_dec_col = y_dec[col_selection]
sub_id_col = sub_id[col_selection]
log_mass_col = log_mass_star[col_selection]


# Fluxinosity selection
flux_min = 0.#1.e-16#8.e-17
flux_selection = flux_oii_sp > flux_min
sub_id_flux = sub_id[flux_selection]
print("flux sel = ",np.sum(flux_selection))

# directory of the halo TNG data
box_name = "TNG"+tng[-3:]
root = "/mnt/gosling1/boryanah/"+box_name+"/"

# loading the halo mass and group identification
Group_M_Mean200_fp = np.load(root+'Group_M_Mean200_fp'+snap_dir+'.npy')*1.e10
SubhaloGrNr_fp = np.load(root+'SubhaloGrNr_fp'+snap_dir+'.npy')


# load the info about the star formation rate
#fname = '../data/galaxes_SFH_'+tng+"_"+snap+'.hdf5'
fname = '../data/galaxies_SFH_'+tng+"_"+snap+".hdf5"
f = h5py.File(fname, 'r')
sub_SFR = f['catsh_SubhaloSFR'][:]
sub_star_mass = f['catsh_SubhaloMassType'][:,4]
sub_ids = f['catsh_id'][:]
f.close()

# get the indices of the top 4000 sfgs
sub_sSFR = sub_SFR/sub_star_mass
log_sSFR = np.log10(sub_sSFR)
log_SFR = np.log10(sub_SFR)
# avoid division by 0
log_sSFR[sub_SFR == 0.] = -13.
log_SFR[sub_SFR == 0.] = -3.
i_sort = np.argsort(sub_SFR)[::-1]
n_top = np.sum(col_selection)#11449#4000
i_sort = i_sort[:n_top]
sub_id_sfg = sub_ids[i_sort]
mass_sfg = sub_star_mass[i_sort]
sfr_sfg = log_SFR[i_sort]
ssfr_sfg = log_sSFR[i_sort]


# color vs mass
inter, comm1, comm2 = np.intersect1d(sub_id_sfg,sub_id,assume_unique=True,return_indices=True)
y_dec_sfg = y_dec[comm2]
x_dec_sfg = x_dec[comm2]
m_dec_sfg = 10.**log_mass_star[comm2]

nrows = 1
ncols = 3
plt.subplots(nrows,ncols,figsize=(ncols*5,nrows*4))
plt.subplot(nrows,ncols,1)
plt.scatter(m_dec_sfg,y_dec_sfg,s=0.05,label="SFGs")
plt.scatter(10.**log_mass_col,y_dec_col,s=0.05,label="Color-selected")
plt.legend()
plt.xscale('log')
plt.xlabel("logM")
plt.ylabel("g-r")
plt.ylim([0.,1.4])

plt.subplot(nrows,ncols,2)
plt.scatter(x_dec_sfg,y_dec_sfg,s=0.05,label="SFGs")
plt.scatter(x_dec_col,y_dec_col,s=0.05,label="Color-selected")
plt.legend()
plt.xlabel("r-z")
plt.ylabel("g-r")
plt.ylim([0.,1.4])


# mass vs sSFR
inter, comm1, comm2 = np.intersect1d(sub_id_col,sub_ids,assume_unique=True,return_indices=True)
ssfr_col = log_sSFR[comm2]
sfr_col = log_SFR[comm2]
mass_col = sub_star_mass[comm2]
print("0 ssfr's = ",np.sum(ssfr_col == -13.))
'''
plt.figure()
plt.scatter(mass_sfg,ssfr_sfg,s=0.05,label="SFGs")
plt.scatter(mass_col,ssfr_col,s=0.05,label="Color-selected")
plt.legend()
plt.xscale('log')
plt.xlabel("logM")
plt.ylabel("log sSFR")
plt.ylim([-11,-7])
'''

plt.subplot(nrows,ncols,3)
plt.scatter(mass_sfg,sfr_sfg,s=0.05,label="SFGs")
plt.scatter(mass_col,sfr_col,s=0.05,label="Color-selected")
plt.legend()
plt.xscale('log')
plt.xlabel("logM")
plt.ylabel("log SFR")
plt.ylim([0,3])
plt.savefig("elg_selection.png")
plt.show()


# get the parent indices of the centrals and also the indices in the original array of the central subhalos
unique_sub_grnr, firsts = np.unique(SubhaloGrNr_fp,return_index=True)

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

count_halo_col_fp, count_halo_cents_col_fp, count_halo_sats_col_fp = get_counts(sub_id_col)
count_halo_sfg_fp, count_halo_cents_sfg_fp, count_halo_sats_sfg_fp = get_counts(sub_id_sfg)
#count_halo_flux_fp, count_halo_cents_flux_fp, count_halo_sats_flux_fp = get_counts(sub_id_flux)

def get_hist(count_halo_elg_fp,count_halo_cents_elg_fp,count_halo_sats_elg_fp):
    # create histograms for the ELGs
    hist, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges,weights=count_halo_elg_fp)
    hist_cents, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges,weights=count_halo_cents_elg_fp)
    hist_sats, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges,weights=count_halo_sats_elg_fp)
    hist_elg = hist/N_bin
    hist_cents_elg = hist_cents/N_bin
    hist_sats_elg = hist_sats/N_bin
    return edges,hist_elg,hist_cents_elg,hist_sats_elg

# histogram
log_min = 11
log_max = 14.4
bin_edges = np.linspace(log_min,log_max,21)
N_bin, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges)

# color-selected galaxies
edges,hist_col,hist_cents_col,hist_sats_col = get_hist(count_halo_col_fp,count_halo_cents_col_fp,count_halo_sats_col_fp)
# SFGs
edges,hist_sfg,hist_cents_sfg,hist_sats_sfg = get_hist(count_halo_sfg_fp,count_halo_cents_sfg_fp,count_halo_sats_sfg_fp)

bin_cen = .5*(bin_edges[1:]+bin_edges[:-1])
plt.plot(10**bin_cen,hist_col,'-',lw=2.,color='black',label='Total col-sel')
plt.plot(10**bin_cen,hist_cents_col,lw=2,color='red',label='Central col-sel')
plt.plot(10**bin_cen,hist_sats_col,lw=2,color='blue',label='Satellite col-sel')
plt.plot(10**bin_cen,hist_sfg,'-',lw=2.,color='black',ls='--',label='Total SFGs')
plt.plot(10**bin_cen,hist_cents_sfg,lw=2,color='red',ls='--',label='Central SFGs')
plt.plot(10**bin_cen,hist_sats_sfg,lw=2,color='blue',ls='--',label='Satellite SFGs')
plt.legend()
plt.ylim([0.01,10])
plt.xlim([3.e11,2.e14])
plt.ylabel(r"$\langle N \rangle_{\rm O II}$")
plt.xlabel(r"$M_{\rm halo} \ M_\odot/h$")
plt.xscale('log')
plt.yscale('log')
plt.savefig("HOD_elg.png")
plt.show()
