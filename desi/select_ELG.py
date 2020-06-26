import numpy as np
import matplotlib.pyplot as plt
from photometric_scatter import get_scatter
import h5py
import sys

# parameter choices
tng = 'tng300'
snap = sys.argv[1]#'041'#'047'#'055'#'050'#'059'
if snap == '040': redshift = 1.50; snap_dir = '_40'
if snap == '050': redshift = 1.00; snap_dir = '_50'; n_gal = 80000
if snap == '059': redshift = 0.70; snap_dir = '_59'
if snap == '099': redshift = 0.06; snap_dir = ''

if snap == '036': redshift = 1.74234; snap_dir = '_36'
if snap == '041': redshift = 1.41131; snap_dir = '_41'; n_gal = 64400
if snap == '047': redshift = 1.11358; snap_dir = '_47'; n_gal = 72000
if snap == '055': redshift = 0.81947; snap_dir = '_55'; n_gal = 80000
if snap == '067': redshift = 0.50000; snap_dir = '_67'
# color filters
cam_filt = 'sdss_desi'
bands = ['sdss_u0','sdss_g0','sdss_r0','sdss_i0','sdss_z0','decam_g','decam_r','decam_i','decam_z','decam_Y']

sfh_ap = '_30kpc'#'_3rad'#'_30kpc'#''
dust_index = '_0.0'#'_-0.5'#'_0.0'#'_-0.0'#'_-0.1'#'_-0.7'#'_-0.4'#''
idx_start = 0
if len(sys.argv) > 2:
    want_reddening = sys.argv[2]
else:
    want_reddening = ''#'_red'#''#'_red'
if want_reddening == '': zterm = '_nored'
if want_reddening == '_red': zterm = ''
# this is intrinsic FSPS scatter
want_scatter = ''#'_scatter'
want_photo_scatter = 1
if want_photo_scatter:
    photo_scatter = ''
else:
    photo_scatter = '_noscatter'
# for use with TNG100
#n_gal = 20000

#fsps_output = np.load("../mags_data/big/"+tng+"_"+snap+"_"+cam_filt+"_id_ugriz_mass_"+str(idx_start)+"_"+str(idx_start+n_gal)+"_tauV_SFH"+sfh_ap+".npy")
# TNG300
fsps_output = np.load("../mags_data/big/"+tng+"_"+snap+"_"+cam_filt+"_id_ugriz_mass_lum_"+str(idx_start)+"_"+str(idx_start+n_gal)+"_tauV_SFH"+dust_index+sfh_ap+want_reddening+want_scatter+".npy")

ugriz_grizy = fsps_output[:,1:len(bands)+1]
stellar_mass = fsps_output[:,1+len(bands)]
sub_id = fsps_output[:,0].astype(int)

# mag limits
g_lim = 24.
r_lim = 23.4 #23.4 is DESI#24.1 is DEEP2
z_lim = 22.5

if want_photo_scatter:
    fac = 1.#0.6
    g_dec_sp = get_scatter(ugriz_grizy[:,5],g_lim,factor=fac)
    r_dec_sp = get_scatter(ugriz_grizy[:,6],r_lim,factor=fac)
    z_dec_sp = get_scatter(ugriz_grizy[:,8],z_lim,factor=fac)
else:
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

'''
# TESTING DO WE WANT TO APPLY PROPER CUTS
color_selection = (g_dec < g_lim) & (r_dec < r_lim) & (z_dec < z_lim)
x_dec = x_dec[color_selection]
y_dec = y_dec[color_selection]
z_dec = z_dec[color_selection]
'''

# eBOSS
y_line1 = -0.068*x_dec+0.457
y_line2 = 0.112*x_dec+0.773
x_line1 = 0.218*y_dec+0.571
#x_line1 = 0.637*y_dec+0.399
x_line2 = -0.555*y_dec+1.901
z_line1 = 21.825
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
flux_min = 8.e-17#1.e-16#8.e-17
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


def HOD_cen(logM,logMmin,slog):
    return .5*(1+erf((logM-logMmin)/slog))

def HOD_sat(logM,logM0,logM1,alpha):
    M = 10**logM
    M0 = 10**logM0
    M1 = 10**logM1
    return ((M-M0)/M1)**alpha

def fit_HOD(hod_c,hod_s,bin_cen):
    logM_min = 12.63
    siglog = 0.28
    logMcut = 12.95
    logMone = 13.62
    al = 0.98

    hod_c[np.isinf(hod_c)] = 0.
    hod_c[np.isnan(hod_c)] = 0.
    popt, pcov = curve_fit(HOD_cen,bin_cen,hod_c,p0=(logM_min,siglog))
    print(popt)
    logM_min,siglog = popt

    hod_s = hist_s/N_bin
    hod_s[np.isinf(hod_s)] = 0.
    hod_s[np.isnan(hod_s)] = 0.
    N = 0#12
    print(bin_cen[N])
    popt, pcov = curve_fit(HOD_sat, bin_cen[N:], hod_s[N:],p0=(logMcut,logMone,al))
    print(popt)
    logMcut,logMone,al = popt

    cents = HOD_cen(logM=bin_cen,logMmin=logM_min,slog=siglog)
    sats = HOD_sat(logM=bin_cen,logM0=logMcut,logM1=logMone,alpha=al)
    return cents, sats

bin_cen = .5*(bin_edges[1:]+bin_edges[:-1])
np.save("data/bin_cen.npy",bin_cen)
np.save("data/hist_cents_col.npy",hist_cents_col)
np.save("data/hist_sats_col.npy",hist_sats_col)
np.save("data/hist_cents_sfg.npy",hist_cents_sfg)
np.save("data/hist_sats_sfg.npy",hist_sats_sfg)

cents_col, sats_col = fit_HOD(hist_cents_col,hist_sats_col,bin_cen)
cents_sfg, sats_sfg = fit_HOD(hist_cents_sfg,hist_sats_sfg,bin_cen)

plt.figure(1)
plt.plot(10**bin_cen,cents_col,ls='-')#,label='cents')
plt.plot(10**bin_cen,sats_col,ls='-')#,label='sats')
plt.plot(10**bin_cen,cents_sfg,ls='--')#,label='cents')
plt.plot(10**bin_cen,sats_sfg,ls='--')#,label='sats')
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
