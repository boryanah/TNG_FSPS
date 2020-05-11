import h5py
import numpy as np
import matplotlib.pyplot as plt
# TODO: can certainly be optimized! Not a dire issue currently

def get_SFH_binned(tobs, tbins, sfh_insitu_sfr, sfh_exsitu_sfr, sfh_insitu_sfz, sfh_exsitu_sfz):

    # lookback times
    lookback_tedges = np.arange(0.,tobs,0.1)
    lookback_tedges = np.insert(lookback_tedges,1,0.03)
    lookback_tbins = np.diff(lookback_tedges)*.5+lookback_tedges[:-1]

    # create final arrays
    n_tbins = len(lookback_tedges)-1
    sfz_new = np.zeros(n_tbins)
    sfr_new = np.zeros(n_tbins)

    # note that i = 0 is today and i = -1 is the beginning snapshot
    for i in range(n_tbins):
        # select the bins within that time frame
        idx_bin = (tbins >= tobs-lookback_tedges[i+1]) & (tbins < tobs-lookback_tedges[i])
        
        # happens sometimes at the earliest times (i.e. furthest lookback) so not important
        if np.sum(idx_bin) == 0:
            sfz_new[i] = 0.
            sfr_new[i] = 0.
            continue
        
        # in/exsitu SFR and SFZ
        sfh_tot_bin = (sfh_insitu_sfr+sfh_exsitu_sfr)[idx_bin]
        sfz_tot_bin = (sfh_insitu_sfz*sfh_insitu_sfr+sfh_exsitu_sfz*sfh_exsitu_sfr)[idx_bin]/sfh_tot_bin
        sfz_tot_bin[sfh_tot_bin == 0.] = 0.

        # bin the SFH's
        sfz_binned = np.sum(sfh_tot_bin*sfz_tot_bin)/np.sum(sfh_tot_bin)
        if np.sum(sfh_tot_bin) == 0.: sfz_binned = 0. 
        sfr_binned = np.mean(sfh_tot_bin)

        # record to array
        sfz_new[i] = sfz_binned
        sfr_new[i] = sfr_binned

    # order them in increasing time like original array
    tbins_new = (tobs-lookback_tbins)[::-1]
    sfz_new = sfz_new[::-1]
    sfr_new = sfr_new[::-1]

    return tbins_new, sfr_new, sfz_new


def main():
    # choices
    sfh_ap = '_30kpc'
    ind = 30000

    # load hdf5
    tng = 'tng300'
    snap = '059'#'059'#'099'
    file_name = 'data/galaxies_SFH_'+tng+'_'+snap+'.hdf5'
    f = h5py.File(file_name, 'r')

    # load the bin center in Gyr
    tbins = f['info/sfh_tbins'][:]
    tedge = f['info/sfh_t_edges'][:]
    tobs = 7.309546074921499#tedge[-1]

    # load the in and ex situ sfr and sfz stellar history
    sfh_insitu_sfr = f['sfh_insitu'+sfh_ap+'_sfr'][ind,:]
    sfh_exsitu_sfr = f['sfh_exsitu'+sfh_ap+'_sfr'][ind,:]
    sfh_insitu_sfz = f['sfh_insitu'+sfh_ap+'_sfz'][ind,:]
    sfh_exsitu_sfz = f['sfh_exsitu'+sfh_ap+'_sfz'][ind,:]
    f.close()

    tbins_new, sfr_new, sfz_new = get_SFH_binned(tobs, tbins, sfh_insitu_sfr, sfh_exsitu_sfr, sfh_insitu_sfz, sfh_exsitu_sfz)
    
    # for the plots
    want_log = 0
    n_cols = 2
    n_rows = 2
    plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(7.5*n_cols, n_rows*5))

    # binning
    sfr = (sfh_insitu_sfr+sfh_exsitu_sfr)
    sfz = (sfh_insitu_sfz*sfh_insitu_sfr+sfh_exsitu_sfz*sfh_exsitu_sfr)/sfr
    sfz[sfr == 0.] = 0.
    time = tbins
    if want_log:
        sfz = np.log10(sfz);
        sfz[sfz == np.min(sfz)] = -9.
        sfr = np.log10(sfr);
        sfr[sfr == np.min(sfr)] = -4
        print("integrated sfr = ",format((np.trapz(10.**sfr, time*1.e9)).sum(),'.2e'))
    else:
        print("integrated sfr = ",format((np.trapz(sfr, time*1.e9)).sum(),'.2e'))

    # plotting
    i_plot = 0
    plt.subplot(n_rows, n_cols, i_plot*n_cols+1)
    plt.plot(time, sfr,lw=2, color='C'+str(0), alpha=0.5, label='SFR')
    plt.legend(frameon=False, loc=1, fontsize=14)
    if want_log:
        plt.ylabel(r'$\log \mathrm{SFR} [\mathrm{M}_{\odot} \mathrm{yr}^{-1}]$', fontsize=18)
    else:
        plt.ylabel(r'$\mathrm{SFR} [\mathrm{M}_{\odot} \mathrm{yr}^{-1}]$', fontsize=18)

    plt.subplot(n_rows, n_cols, i_plot*n_cols+2)
    plt.plot(time, sfz,lw=2, color='C'+str(0), alpha=0.5, label='SFZ')
    plt.legend(frameon=False, loc=1, fontsize=14)

    sfr = sfr_new
    sfz = sfz_new
    print(sfr[-1])
    time = tbins_new
    if want_log:
        sfz = np.log10(sfz);
        sfz[sfz == np.min(sfz)] = -9.
        sfr = np.log10(sfr);
        sfr[sfr == np.min(sfr)] = -4.
        print("integrated sfr = ",format((np.trapz(10.**sfr, time*1.e9)).sum(),'.2e'))
    else:
        print("integrated sfr = ",format((np.trapz(sfr, time*1.e9)).sum(),'.2e'))
    

    # plotting
    i_plot = 1
    plt.subplot(n_rows, n_cols, i_plot*n_cols+1)
    plt.plot(time, sfr,lw=2, color='C'+str(i_plot), alpha=0.5, label='SFR binned')
    plt.xlabel(r'$\mathrm{time} [\mathrm{Gyr}]$', fontsize=18)
    plt.legend(frameon=False, loc=1, fontsize=14)
    if want_log:
        plt.ylabel(r'$\log \mathrm{SFR} [\mathrm{M}_{\odot} \mathrm{yr}^{-1}]$', fontsize=18)
    else:
        plt.ylabel(r'$\mathrm{SFR} [\mathrm{M}_{\odot} \mathrm{yr}^{-1}]$', fontsize=18)

    plt.subplot(n_rows, n_cols, i_plot*n_cols+2)
    plt.plot(time, sfz,lw=2, color='C'+str(i_plot), alpha=0.5, label='SFZ binned')
    plt.xlabel(r'$\mathrm{time} [\mathrm{Gyr}]$', fontsize=18)
    plt.legend(frameon=False, loc=1, fontsize=14)
    # set axes
    plt.savefig("sfh.png")
    plt.show()
