import numpy as np
import matplotlib.pyplot as plt

np.random.seed(300)

def get_scatter(band_mag,mag_limit,factor=1.,snr=5.):
    flux_limit = mag_to_nMgy(mag_limit)
    error = flux_limit/snr
    band_flux = mag_to_nMgy(band_mag)
    scat_flux = np.random.normal(band_flux,error*factor)
    scat_mag = nMgy_to_mag(scat_flux)
    #delta_m = 2.5*np.log10(1.+1./snr)
    # error [mag] = A_band x band-magnitude [mag]^[-1/2]
    #error = delta_m*10.**((-mag_limit+band_mag)/5.)
    #scat_mag = np.random.normal(band_mag,error*factor)
    return scat_mag

def load_fsps(snap,dust_index,tng='tng300',cam_filt='sdss_des'):
    # parameter choices
    if snap == '040': redshift = 1.50; snap_dir = '_40'
    if snap == '050': redshift = 1.00; snap_dir = '_50'; n_gal = 80000
    if snap == '059': redshift = 0.70; snap_dir = '_59'
    if snap == '099': redshift = 0.06; snap_dir = ''
    # for AS snaps
    if snap == '036': redshift = 1.74234; snap_dir = '_36'
    if snap == '041': redshift = 1.41131; snap_dir = '_41'; n_gal = 64400#64400#12000#64400
    if snap == '047': redshift = 1.11358; snap_dir = '_47'; n_gal = 72000#4880#16000#72000
    if snap == '055': redshift = 0.81947; snap_dir = '_55'; n_gal = 80000
    if snap == '067': redshift = 0.50000; snap_dir = '_67'

    # color filters
    bands = ['sdss_u0','sdss_g0','sdss_r0','sdss_i0','sdss_z0','decam_g','decam_r','decam_i','decam_z','decam_Y']
    sfh_ap = '_30kpc'#'_3rad'#'_30kpc'#''
    
    # reddening means (1+z)^-0.5 term
    want_reddening = ''#'_red'#''#'_red'
    if want_reddening == '': zterm = '_nored'
    if want_reddening == '_red': zterm = ''

    # this is intrinsic FSPS scatter which we don't really use
    want_scatter = ''#'_scatter'

    # load the fsps data
    fsps_output = np.load("../mags_data/big/"+tng+"_"+snap+"_"+cam_filt+
                          "_id_ugriz_mass_lum_0_"+str(n_gal)+"_tauV_SFH"+
                          dust_index+sfh_ap+want_reddening+want_scatter+".npy")

    # read out colors, mass, flux and ids
    ugriz_grizy = fsps_output[:,1:len(bands)+1]
    stellar_mass = fsps_output[:,1+len(bands)]
    sub_id = fsps_output[:,0].astype(int)
    flux_oii = np.max(fsps_output[:,-2:],axis=1)

    return sub_id, ugriz_grizy, stellar_mass, flux_oii

def make_scatter_histogram(x,y,z,w,figname,s=0.1,al=0.2,theta=None):

    if theta is not None:
        rot_str = r' $['+str(theta)+'^\circ'+']$ rotated'
        theta *= np.pi/180.
        rot_mat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        xy_rot = np.dot(rot_mat,np.vstack((x,y)))
        zw_rot = np.dot(rot_mat,np.vstack((z,w)))

        x = xy_rot[0]
        y = xy_rot[1]
        z = zw_rot[0]
        w = zw_rot[1]
    else:
        rot_str = ''
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005


    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the scatter plot:
    ax_scatter.scatter(x, y,s=s,color='dodgerblue',alpha=al,label='TNG-FSPS')
    ax_scatter.scatter(z, w,s=s,color='red',label='DEEP2-DR8')
    ax_scatter.set_xlabel("r-z"+rot_str)
    ax_scatter.set_ylabel("g-r"+rot_str)
    mean_z = np.mean(z[np.logical_not(np.isnan(z) | np.isinf(z))])
    mean_w = np.mean(w[np.logical_not(np.isnan(w) | np.isinf(w))])
    
    ax_scatter.scatter(np.mean(x), np.mean(y),s=100,color='blue',marker='x',label='avg TNG-FSPS')
    ax_scatter.scatter(mean_z, mean_w,s=100,color='darkorange',marker='x',label='avg DEEP2-DR8')
    
    # now determine nice limits by hand:
    binwidth = 0.05
    lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
    lim = 2.5
    ax_scatter.set_xlim((-0.5, lim))
    ax_scatter.set_ylim((-0.5, lim))
    ax_scatter.legend()
    
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x,bins=bins,histtype='step',color='b',density=True)
    ax_histy.hist(y,bins=bins,histtype='step',color='b',density=True,orientation='horizontal')
    ax_histx.hist(z,bins=bins,histtype='step',color='r',density=True)
    ax_histy.hist(w,bins=bins,histtype='step',color='r',density=True,orientation='horizontal')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    plt.savefig(figname)
    plt.close()
    return

def nMgy_to_mag(flux):
    mag = 22.5-2.5*np.log10(flux)
    return mag

def mag_to_nMgy(mag):
    flux = 10**(-0.4*(mag-22.5))
    return flux
