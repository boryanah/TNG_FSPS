import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import fsps

z_median = 0.85

hdu = fits.open('ELG_targets_SGC-dr3.fits')
dat = hdu[1].data

def nMgy_to_mag(f,color):
    f = np.clip(f,0.001,1.e11)
    return 22.5-2.5*np.log10(f)
                
g_dec = nMgy_to_mag(dat['decam_flux_g'],'g')
r_dec = nMgy_to_mag(dat['decam_flux_r'],'r')
z_dec = nMgy_to_mag(dat['decam_flux_z'],'z')

x_dec = r_dec-z_dec
y_dec = g_dec-r_dec

y_line1 = -0.068*x_dec+0.457
y_line2 = 0.112*x_dec+0.773
x_line1 = 0.218*y_dec+0.571
#x_line1 = 0.637*y_dec+0.399
x_line2 = -0.555*y_dec+1.901


# parameter choices
tng = 'tng300'
snap = '047'#'055'#'041'#'047'#'055'#'050'#'059'
if snap == '040': redshift = 1.50; snap_dir = '_40'
if snap == '050': redshift = 1.00; snap_dir = '_50'; n_gal = 80000
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
want_reddening = ''#''#'_red'
want_scatter = ''#'_scatter'




# use with TNG100
# for use with TNG100
#n_gal = 20000
#fsps_output = np.load("../mags_data/big/"+tng+"_"+snap+"_"+cam_filt+"_id_ugriz_mass_"+str(idx_start)+"_"+str(idx_start+n_gal)+"_tauV_SFH"+sfh_ap+".npy")
# TNG300
fsps_output = np.load("../mags_data/big/"+tng+"_"+snap+"_"+cam_filt+"_id_ugriz_mass_lum_"+str(idx_start)+"_"+str(idx_start+n_gal)+"_tauV_SFH"+sfh_ap+want_reddening+want_scatter+".npy")


ugriz_grizy = fsps_output[:,1:len(bands)+1]
stellar_mass = fsps_output[:,1+len(bands)]

g_dec_sp = ugriz_grizy[:,5]
r_dec_sp = ugriz_grizy[:,6]
z_dec_sp = ugriz_grizy[:,8]
log_mass_star = np.log10(stellar_mass)

selection_dec_sp = g_dec_sp < 100000.
g_dec_sp = g_dec_sp[selection_dec_sp]
r_dec_sp = r_dec_sp[selection_dec_sp]
z_dec_sp = z_dec_sp[selection_dec_sp]
log_mass_star = log_mass_star[selection_dec_sp]

print("Total gals = ",np.sum(selection_dec_sp))

x_dec_sp = r_dec_sp-z_dec_sp
y_dec_sp = g_dec_sp-r_dec_sp

plt.scatter(x_dec,y_dec,s=0.1)
plt.scatter(x_dec_sp,y_dec_sp,s=0.1)
plt.plot(x_line1,y_dec)
plt.plot(x_line2,y_dec)
plt.plot(x_dec,y_line1)
plt.plot(x_dec,y_line2)
plt.xlabel("r-z")
plt.ylabel("g-r")
plt.axis('equal')
plt.savefig("elg.png")
plt.close()

quit()
plt.scatter(log_mass_star,y_dec_sp,s=0.5,color='r')
plt.xlabel('log M$_{\star}$')
plt.ylabel('g-r')
plt.xlim([8.5, 12.0])
plt.ylim([0., 2.5])
plt.savefig("mass_col_dec.png")
plt.show()
#plt.scatter(x_dec,g_dec,s=0.1)
#plt.show()
#np.sqrt(1./dat['decam_flux_ivar_z'][0])
#print(dat.dtype)
#dtype((numpy.record, [('brickname', 'S8'), ('objid', '>i4'), ('ra', '>f8'), ('dec', '>f8'), ('ra_ivar', '>f4'), ('dec_ivar', '>f4'), ('ebv', '>f4'), ('decam_flux_g', '>f4'), ('decam_flux_r', '>f4'), ('decam_flux_z', '>f4'), ('decam_flux_ivar_g', '>f4'), ('decam_flux_ivar_r', '>f4'), ('decam_flux_ivar_z', '>f4'), ('decam_mw_transmission_g', '>f4'), ('decam_mw_transmission_r', '>f4'), ('decam_mw_transmission_z', '>f4'), ('sourcetype', 'S16'), ('EBOSS_TARGET1', '>i8'), ('syst_weight', '>f8')]))
