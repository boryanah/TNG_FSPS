import h5py
import numpy as np
import matplotlib.pyplot as plt
import fsps
from mpi4py import MPI
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
# Usage mpirun -np 60 python get_mags_tauV_sfh.py
# check lowmass, tauV params, aperture, reddening, random, redshift, tng

# constants
solar_metal = 0.0127
h = 0.6774
omega_b = 0.0486
omega_l = 0.6911
omega_m = 0.3089

def get_lumdist(z):
    return cosmo.luminosity_distance(z).value # in Mpc

def get_age(z):
    return cosmo.age(z).value # in Gyr

# get cosmology
cosmo = FlatLambdaCDM(H0=h*100.*u.km/u.s/u.Mpc, Tcmb0=2.7255*u.K, Om0=omega_m)

# parameter choices
tng = 'tng300'
dust_reddening = '_red'#'_red'#''
skip_lowmass = 0
low_mass = 9.5
want_random = ''#''#'_scatter'
sfh_ap = '_30kpc'#'_30kpc'#'_3rad'#'_30kpc'#''
cam_filt = 'sdss_des'

# color bands
bands = []
filts = cam_filt.split('_')
for i in range(len(filts)):
    bands += fsps.find_filter(filts[i])

# which redshift, luminosity distance, snapshot and universe age
redshifts = np.array([1.5, 1.41131, 1.35539, 1.30228, 1.25173, 1.20355, 1.15755, 1.11358, 1.07147, 1.03111, 1, 0.95514, 0.91932, 0.88482, 0.85156, 0.81947, 0.78847, 0.7585, 0.7295, 0.7, 0.5, 0])
z_choice = np.abs(redshifts - 0.7) < 1.e-3
assert np.sum(z_choice) == 1, "Problem with picking redshift"
i_choice = np.argmax(z_choice)
snaps = np.linspace(40,60,20,endpoint=False).astype(int)
snaps = np.hstack((snaps,np.array([67,99])))
dists = get_lumdist(redshifts)
tobss = get_age(redshifts)
snap_dirs = ['_'+str(snaps[i]) for i in range(len(redshifts))]
snap_dirs[-1] = ''
snap = snaps[i_choice]
snap = format(snap,'03d')
redshift = redshifts[i_choice]
tobs = tobss[i_choice]
snap_dir = snap_dirs[i_choice]
dist = dists[i_choice]
if want_random == '_scatter':
    redshift_prev = redshifts[i_choice-1]
    redshift_next = redshifts[i_choice+1]
    z_min = redshift - 0.5*(redshift-redshift_next)
    z_max = redshift + 0.5*(redshift_prev-redshift)
#if snap == '040': redshift = 1.50; tobs = 4.293; snap_dir = '_40'
#if snap == '050': redshift = 1.00; tobs = 5.878; snap_dir = '_50'
#if snap == '059': redshift = 0.70; tobs = 7.314; snap_dir = '_59'
#if snap == '099': redshift = 0.06; tobs = 13.803; snap_dir = ''

# ELG emission lines
lambda1 = 3727.092 #3727.1
lambda2 = 3729.875 #3729.88
tol = 0.01


# MPI parameters
i_rank = MPI.COMM_WORLD.Get_rank()
n_gal = 20 # total is 22539 (12520 above logM of 9.5)
idx_start = i_rank*n_gal
inds = np.arange(idx_start,idx_start+n_gal,dtype=int)
print("start, end = ",idx_start,idx_start+n_gal)
id_ugriz_mass_gal = np.zeros((n_gal,1+len(bands)+1+len(bands)+len(bands)+2))

def get_SFH(sfh_insitu,sfh_exsitu,sfz_insitu,sfz_exsitu,idx_add):
    # interpolate SFH
    sfh_insitu_new = np.insert(sfh_insitu, idx_add, sfh_insitu[idx_add-1])
    sfh_exsitu_new = np.insert(sfh_exsitu, idx_add, sfh_exsitu[idx_add-1])
    sfz_insitu_new = np.insert(sfz_insitu, idx_add, sfz_insitu[idx_add-1])
    sfz_exsitu_new = np.insert(sfz_exsitu, idx_add, sfz_exsitu[idx_add-1])
    # calculate total SFH and SFZ
    sfh_tot = sfh_insitu_new + sfh_exsitu_new
    sfz_tot = (sfz_insitu_new*sfh_insitu_new + sfz_exsitu_new*sfh_exsitu_new)/sfh_tot
    # avoid division by 0
    sfz_tot[sfh_tot == 0.] = 1.e-16
    return sfh_tot, sfz_tot 

def get_idx_add(t_edge,t_obs,t_boundary):
    # add and update times at 30 Myr
    tadd = t_obs-t_boundary
    idx_add = np.where(t_edge > tadd)[0][0]
    tedge_new = np.insert(t_edge, idx_add, tadd)
    tbins_new = 0.5*np.diff(tedge_new) + tedge_new[:-1]
    return tbins_new, idx_add

def load_data(fname):
    # load hdf5
    f = h5py.File(file_name, 'r')

    '''
    # TESTING!!!!
    inds = np.arange(idx_start,idx_start+n_gal,dtype=int)
    sub_id = f['catsh_id'][:]
    sub_id = sub_id[inds]
    inds = [np.argmax(sub_id == 76086)]
    '''
    
    # load the bin center in Gyr
    tbins = f['info/sfh_tbins'][:]
    #dt = f['info/sfh_dt'][:]
    tedge = f['info/sfh_t_edges'][:]
    dt = tbins[1]-tbins[0]
    
    # load the in and ex situ sfr and sfz stellar history
    sfh_insitu_sfr = f['sfh_insitu'+sfh_ap+'_sfr'][inds,:]
    sfh_exsitu_sfr = f['sfh_exsitu'+sfh_ap+'_sfr'][inds,:]
    sfh_insitu_sfz = f['sfh_insitu'+sfh_ap+'_sfz'][inds,:]
    sfh_exsitu_sfz = f['sfh_exsitu'+sfh_ap+'_sfz'][inds,:]
    sfh_sfr = sfh_insitu_sfr+sfh_exsitu_sfr

    recent_time = tobs-.1 # select most recent 100 Myr
    time_selection = (tbins > recent_time) & (tbins < tobs)
    sfr_recent = np.mean(sfh_sfr[:,time_selection],axis=1)
    
    # load the SFR at present day
    sub_SFR = f['catsh_SubhaloSFR'][:]
    # numbers
    print("Number of galaxies = ",len(sub_SFR))
    sub_SFR = sub_SFR[inds]

    # load the stellar and gas mass
    sub_gas_mass = f['catsh_SubhaloMassType'][inds,0]# in Msol [not / h]
    sub_star_mass = f['catsh_SubhaloMassType'][:,4]# in Msol [not / h]
    print("Number of galaxies over 9.5 = ",np.sum(sub_star_mass > 10.**9.5))
    sub_star_mass = sub_star_mass[inds]
    sub_gas_halfr = f['catsh_SubhaloHalfmassRadType'][inds,0]
    sub_star_halfr = f['catsh_SubhaloHalfmassRadType'][inds,4]
    # avoid division by 0
    sub_gas_halfr[sub_gas_halfr == 0.] = np.min(sub_gas_halfr[sub_gas_halfr > 0.])
    # load the stellar and gas metallicity
    sub_star_metal = np.load("/mnt/gosling1/boryanah/TNG"+tng[-3:]+"/SubhaloStarMetallicity_fp"+snap_dir+".npy")
    sub_gas_metal = f['catsh_SubhaloGasMetallicity'][:]
    sub_gas_metal = sub_gas_metal[inds]
    # load the subhalo id in original catalog
    sub_id = f['catsh_id'][:]
    sub_id = sub_id[inds]
    f.close()
    
    # compute the log metallicity
    Z_gas_sol = (sub_gas_metal/solar_metal)
    sub_logzgas = np.log10(Z_gas_sol)
    # avoid division by 0
    sub_logzgas[Z_gas_sol == 0.] = -13.
    sub_logzstar = np.log10(sub_star_metal[sub_id]/solar_metal)
    # avoid division by 0
    sub_logzstar[np.isinf(sub_logzstar)] = -13.
    log_star_mass = np.log10(sub_star_mass)
    S_gas_norm = sub_gas_mass/sub_gas_halfr**2
    S_gas_norm /= np.mean(S_gas_norm)
    S_star_norm = sfr_recent/sub_star_halfr**2
    S_star_norm /= np.mean(S_star_norm)
    # avoid division by 0 (not used)
    log_sSFR = np.log10(sub_SFR/sub_star_mass)
    log_sSFR[sub_SFR == 0.] = -13.
    #print(Z_gas_sol, sub_logzgas, sub_logzstar, log_star_mass, S_gas_norm, log_sSFR, sub_id, (sfh_insitu_sfr*dt).sum(), (sfh_exsitu_sfr*dt).sum(), (sfh_insitu_sfz*dt).sum(), (sfh_exsitu_sfz*dt).sum())
    return tedge, Z_gas_sol, sub_logzgas, sub_logzstar, log_star_mass, S_gas_norm, S_star_norm, log_sSFR, sub_id, sfh_insitu_sfr, sfh_exsitu_sfr, sfh_insitu_sfz, sfh_exsitu_sfz

# load file
# large sample (cents+sats)
file_name = 'data/galaxies_SFH_'+tng+'_'+snap+'.hdf5'
# small sample (cents only)
#file_name = 'data/galaxies_tng100_099_1_SFH.hdf5'

# gather all data for this chunk
tedge, Z_gas_sol, sub_logzgas, sub_logzstar, log_star_mass, S_gas_norm, S_star_norm, log_sSFR, sub_id, sfh_insitu_sfr, sfh_exsitu_sfr, sfh_insitu_sfz, sfh_exsitu_sfz = load_data(file_name)

# tau_V parameters for small sample with S_gas_norm
#alpha, beta, gamma = [0.53067365, 0.22995855, 0.2252313]
# tau_V parameters for large sample with S_gas_norm
#alpha, beta, gamma = [ 0.83609055, -0.04689114,  0.16439263]
# tau_V parameters for large sample with S_star_norm
alpha, beta, gamma = [0.24567538, 0.04542672, 0.2688033]
# tau_V parameters for tng300 sample with S_star_norm
#alpha, beta, gamma = [0.06372212, 0.12692061, 0.28552597]

# tau_V ansatz
#tau_V = gamma*Z_gas_sol**alpha*S_gas_norm**beta
tau_V = gamma*Z_gas_sol**alpha*S_star_norm**beta
tau_V[np.isinf(tau_V)] = 0.
tau_V[np.isnan(tau_V)] = 0.

# add and update times at 30 Myr = 0.03 Gyr
tboundary = 30.*1e-3 
tbins_new, idx_add = get_idx_add(tedge,tobs,tboundary)
idx_continuum = tbins_new < tobs - tboundary

for idx_gal in range(n_gal):
    print("sub_id = ",sub_id[idx_gal],idx_gal)
    id_ugriz_mass_gal[idx_gal,0] = sub_id[idx_gal]

    # getting the SFH 
    sfh_insitu = sfh_insitu_sfr[idx_gal]
    sfh_exsitu = sfh_exsitu_sfr[idx_gal]
    sfz_insitu = sfh_insitu_sfz[idx_gal]
    sfz_exsitu = sfh_exsitu_sfz[idx_gal]
    
    sfh_tot, sfz_tot = get_SFH(sfh_insitu,sfh_exsitu,sfz_insitu,sfz_exsitu,idx_add)
    time = tbins_new

    # set SFH of contiuum (neb) to 0
    sfh_tot_continuum, sfz_tot_continuum = np.copy(sfh_tot), np.copy(sfz_tot)
    sfh_tot_neb, sfz_tot_neb = np.copy(sfh_tot), np.copy(sfz_tot)
    sfh_tot_continuum[~idx_continuum] = 0.0
    sfz_tot_continuum[~idx_continuum] = 0.0
    sfh_tot_neb[idx_continuum] = 0.0
    sfz_tot_neb[idx_continuum] = 0.0
    
    # getting stellar and gas metallicity
    logzsol = sub_logzstar[idx_gal]
    dust_index = 1.13-0.29*(log_star_mass[idx_gal]-10.)
    dust_index *= -1.
    gas_logz = sub_logzgas[idx_gal]
    dust2 = tau_V[idx_gal]
    if dust_reddening == '_red':
        dust2 *= (1.+redshift)**(-0.5)

    if skip_lowmass and log_star_mass[idx_gal] < low_mass:
        sfh_tot_continuum[:] = -1
        sfh_tot_neb[:] = -1

    if want_random == '_scatter':
        redshift = np.random.uniform(z_min,z_max)
        dist = get_lumdist(redshift)
        tobs = get_age(redshift)
        # TODO: might have to change this
        
    # fitting SED
    if np.sum(sfh_tot_continuum) > 0. and np.sum(sfz_tot_continuum) >= 0.:
        sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=3, \
                                    imf_type=1, add_neb_emission=False, gas_logu=-1.4, \
                                    gas_logz=gas_logz, sfh=3, logzsol=0.0,\
                                    dust_type=0, dust_index=dust_index, dust2=dust2,\
                                    dust1_index=-1,dust1=dust2)
        sp.set_tabular_sfh(time, sfh_tot_continuum, Z=sfz_tot_continuum)
    
        # convolving with a filter and getting the magnitudes at some redshift
        ugriz_continuum = sp.get_mags(tage=tobs, redshift=redshift, bands=bands)
        sp_stellar_mass = sp.stellar_mass
    else:
        # we can't predict the values accurately
        print("no exist cont")
        ugriz_continuum = np.ones(len(bands))*10000000000.
        sp_stellar_mass = 10.**log_star_mass[idx_gal]

    if np.sum(sfh_tot_neb) > 0.:
        logzstar_neb = np.log10(np.sum(sfh_tot_neb*sfz_tot_neb) / np.sum(sfh_tot_neb) / solar_metal)
        sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, \
                                    imf_type=1, add_neb_emission=True, gas_logu=-1.4, \
                                    gas_logz=gas_logz, sfh=3, logzsol=logzstar_neb,\
                                    dust_type=0, dust_index=dust_index, dust2=dust2,\
                                    dust1_index=-1,dust1=dust2)
        sp.set_tabular_sfh(time, sfh_tot_neb, Z=None)

        # convolving with a filter and getting the magnitudes at some redshift
        ugriz_neb = sp.get_mags(tage=tobs, redshift=redshift, bands=bands)

        # OII emission lines
        waves = sp.emline_wavelengths # in Angstroms
        lums = sp.emline_luminosity # in Lsun
        selection = (lambda1-tol < waves) & (lambda2+tol > waves)
        lumoii = lums[selection]*u.Lsun
        print(dist)
        dist *= u.Mpc
        fluxoii = lumoii/(4.*np.pi*dist**2)
        flux = (fluxoii.to(u.erg/(u.s*u.cm*u.cm))).value
        print(flux)
        id_ugriz_mass_gal[idx_gal,1+len(bands)+1+len(bands)+len(bands):] = flux[:]
    else:
        # we can't predict the values accurately
        print("no exist neb")
        ugriz_neb = np.ones(len(bands))*10000000000.

    # adding the two magnitudes
    ugriz = -2.5*np.log10(np.power(10, -0.4*ugriz_continuum) + np.power(10, -0.4*ugriz_neb))

    if skip_lowmass and log_star_mass[idx_gal] < low_mass:
        ugriz[:] = 10000000000.
    
    print("Stellar mass = ",format(np.log10(sp_stellar_mass),".2f"),format(log_star_mass[idx_gal],".2f"))
    for i_band in range(len(bands)):
        id_ugriz_mass_gal[idx_gal,i_band+1] = ugriz[i_band]
        id_ugriz_mass_gal[idx_gal,1+len(bands)+1+i_band] = ugriz_continuum[i_band]
        id_ugriz_mass_gal[idx_gal,1+len(bands)+1+len(bands)+i_band] = ugriz_neb[i_band]
    id_ugriz_mass_gal[idx_gal,1+len(bands)] = sp_stellar_mass
    print(id_ugriz_mass_gal[idx_gal])

# save the magnitudes and stellar masses
np.save("mags_data/"+tng+"_"+snap+"_"+cam_filt+"_id_ugriz_mass_lum_"+str(idx_start)+"_"+str(idx_start+n_gal)+"_tauV_SFH"+sfh_ap+dust_reddening+want_random+".npy",id_ugriz_mass_gal)

# all fields
#['catgrp_GroupNsubs','catgrp_Group_M_Crit200','catgrp_id','catgrp_is_primary','catsh_SubhaloBHMdot','catsh_SubhaloCM','catsh_SubhaloGasMetallicity','catsh_SubhaloGrNr','catsh_SubhaloHalfmassRadType','catsh_SubhaloMassInHalfRadType','catsh_SubhaloMassInRadType','catsh_SubhaloMassType','catsh_SubhaloPos','catsh_SubhaloSFR','catsh_SubhaloSpin','catsh_SubhaloStellarPhotometrics','catsh_SubhaloVel','catsh_SubhaloVmax','catsh_id','config','info','scalar_m_neutral_H','scalar_star_age_disk','scalar_star_age_light_wgtd','scalar_star_age_spheroid','sfh_exsitu_nstars','sfh_exsitu_nstars_tot','sfh_exsitu_sfr','sfh_exsitu_sfz',  'sfh_insitu_nstars',  'sfh_insitu_nstars_tot','sfh_insitu_sfr','sfh_insitu_sfz',  'tree_sh_idx']
