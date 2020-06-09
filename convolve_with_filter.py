from sedpy.observate import getSED, load_filters
import numpy as np

# wave, spectrum come from FSPS (as well as redshift)
def get_mags(wave,spectrum,z,filterlist):
    # filter parameters
    filters = load_filters(filterlist)

    # redshift offset the spectrum
    a = 1+z
    wa, sa = wave*a, spectrum*a

    # get the color magnitudes
    mags = getSED(wa, lightspeed/wa**2 * sa * to_cgs, filters)
    return mags

# spectrum : Spectrum in Lsun/Hz per solar mass formed, restframe
#filters = load_filters(filternames, directory=filter_folder)
#filter_folder = ‘…../filters/’
#filterlist = ['decam_g.par', 'decam_r.par', 'decam_i.par', 'decam_z.par', 'decam_Y.par']

# constants (TODO: maybe use astropy instead)
lightspeed = 2.99792458e18  # AA/s
lsun = 3.846e33  # erg/s
pc = 3.085677581467192e18  # in cm

# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs_at_10pc = lsun/(4.0 * np.pi * (pc*10)**2)
to_cgs = to_cgs_at_10pc
#to_cgs =  lsun/(4.0 * np.pi * (pc)**2)
