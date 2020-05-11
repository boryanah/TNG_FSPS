import h5py
import numpy as np
import matplotlib.pyplot as plt

# constants
solar_metal = 0.0127

# choices
sfh_ap = '_30kpc'
ind = 0

# load hdf5
tng = 'tng300'
snap = '099'
file_name = 'data/galaxies_SFH_'+tng+'_'+snap+'.hdf5'
f = h5py.File(file_name, 'r')
    
# load the bin center in Gyr
tbins = f['info/sfh_tbins'][:]
tedge = f['info/sfh_t_edges'][:]

# load the in and ex situ sfr and sfz stellar history
sfh_insitu_sfr = f['sfh_insitu'+sfh_ap+'_sfr'][ind,:]
sfh_exsitu_sfr = f['sfh_exsitu'+sfh_ap+'_sfr'][ind,:]
sfh_insitu_sfz = f['sfh_insitu'+sfh_ap+'_sfz'][ind,:]
sfh_exsitu_sfz = f['sfh_exsitu'+sfh_ap+'_sfz'][ind,:]
f.close()

# lookback times
tedge_last = tedge[-1]
lookback_tedges = np.arange(0.,tedge_last,0.1)
lookback_tedges = np.insert(lookback_tedges,1,0.3)
lookback_tbins = np.diff(lookback_tedges)*.5+lookback_tedges[:-1]
print(lookback_tedges)

# create final arrays
n_tbins = len(lookback_tedges)-1
sfz_new = np.zeros(n_tbins)
sfr_new = np.zeros(n_tbins)

# notice that i = 0 is today and i = -1 is the beginning snapshot
for i in range(n_tbins):
    # select the bins within that time frame
    idx_bin = (tbins > tedge_last-lookback_tedges[i+1]) & (tbins < tedge_last-lookback_tedges[i])

    # in/exsitu SFR and SFZ
    sfh_tot_bin = (sfh_insitu_sfr+sfh_exsitu_sfr)[idx_bin]
    sfz_tot_bin = (sfh_insitu_sfz*sfh_insitu_sfr+sfh_exsitu_sfz*sfh_exsitu_sfr)[idx_bin]/sfh_tot_bin
    # I think not this
    #sfz_tot_bin = (sfh_insitu_sfz+sfh_exsitu_sfz)[idx_bin]

    # bin the SFH's
    sfz_binned = np.sum(sfh_tot_bin*sfz_tot_bin)/np.sum(sfh_tot_bin)
    sfr_binned = np.mean(sfh_tot_bin)
    #logzstar_neb = np.log10(sfz_binned/solar_metal)

    # record to array
    sfz_new[i] = sfz_binned
    sfr_new[i] = sfr_binned

# order them in increasing time like original array
tbins_new = (tedge_last-lookback_tbins)[::-1]
sfz_new = sfz_new[::-1]
sfr_new = sfr_new[::-1]

# record these three arrays
