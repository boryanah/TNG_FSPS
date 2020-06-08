import numpy as np


def get_scatter(band_mag,mag_limit,snr=5.):
    delta_m = 2.5*np.log10(1.+1./snr)
    # error [mag] = A_band x band-magnitude [mag]^2
    A_band = delta_m/mag_limit**2
    error = A_band*band_mag**2
    scat_mag = np.random.normal(band_mag,error)
    return scat_mag
