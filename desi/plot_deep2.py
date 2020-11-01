import numpy as np
import matplotlib.pyplot as plt
from util import load_fsps
from astropy.io import fits
from util import get_scatter, make_scatter_histogram, nMgy_to_mag
import sys
import plotparams
#plotparams.buba()
plotparams.default()

# fsps parameters
tng = 'tng300'#'tng300'
snap = sys.argv[1]#'041'#'047'#'055'#'050'#'059'
snap_dir = '_'+str(int(snap))
redshift_dict = {'055':['sdss_desi','_0.0'],#-0.5
                 '047':['sdss_desi','_-0.0'],
                 '041':['sdss_des','']}#['sdss_des','']['sdss_desi','_0.0']
dust_index = redshift_dict[snap][1]##'_0.5'#'_0.0'#'_-0.5'#'_-0.0'#'_-0.1'#'_-0.7'#'_-0.4'#''
want_hist = 0
# photometric scatter
want_photo_scatter = 1
if want_photo_scatter:
    photo_scatter = ''
else:
    photo_scatter = '_noscatter'

z_dic = {'_55': 0.8, '_47': 1.1, '_41': 1.4}
    
# deep2/decals parameters
nomask = ''#'-NoMask'#''
dr_v = '8'#'8'#'5'
deep2dir = '/mnt/gosling1/boryanah/TNG100/TNG_FSPS/DEEP2-DESI/data/derived/'
hdu = fits.open(deep2dir+'DR'+dr_v+'-matched-to-DEEP2-f2-glim24p25'+nomask+'.fits')
dat2 = hdu[1].data
hdu = fits.open(deep2dir+'DR'+dr_v+'-matched-to-DEEP2-f3-glim24p25'+nomask+'.fits')
dat3 = hdu[1].data
hdu = fits.open(deep2dir+'DR'+dr_v+'-matched-to-DEEP2-f4-glim24p25'+nomask+'.fits')
dat4 = hdu[1].data

def combine_col_fits(color):
    return np.hstack(((dat2['flux_'+color])[z_sel2],(dat3['flux_'+color])[z_sel3],(dat4['flux_'+color])[z_sel4]))


if snap == '041': z_min = 1.3; z_max = 1.5; 
if snap == '047': z_min = 1.04; z_max = 1.12;
if snap == '055': z_min = .79; z_max = .86;#z_min = .8; z_max = .85;

z_sel2 = (z_min < dat2['RED_Z']) & (dat2['RED_Z'] < z_max)
z_sel3 = (z_min < dat3['RED_Z']) & (dat3['RED_Z'] < z_max)
z_sel4 = (z_min < dat4['RED_Z']) & (dat4['RED_Z'] < z_max)

g_dec = nMgy_to_mag(combine_col_fits('g'))
r_dec = nMgy_to_mag(combine_col_fits('r'))
z_dec = nMgy_to_mag(combine_col_fits('z'))
selection_dec = (~np.isinf(g_dec)) & (~np.isinf(r_dec)) & (~np.isinf(z_dec)) & (~np.isnan(g_dec)) & (~np.isnan(r_dec)) & (~np.isnan(z_dec))
g_dec = g_dec[selection_dec]
r_dec = r_dec[selection_dec]
z_dec = z_dec[selection_dec]

print("total DEEP2/DECaLS gals = ",np.sum(selection_dec))

'''
z_all = np.hstack(((dat2['RED_Z'])[z_sel2],(dat3['RED_Z'])[z_sel3],(dat4['RED_Z'])[z_sel4]))
n_bins = 31
z_bins = np.linspace(z_min,z_max,n_bins)
z_cents = .5*(z_bins[1:]+z_bins[:-1])
hist_z, edges = np.histogram(z_all,bins=z_bins)
plt.plot(z_cents,hist_z)
#plt.savefig("hist_z_"+str(z_min)+"_"+str(z_max)+".png")
plt.show()
quit()
'''

x_dec = r_dec-z_dec
y_dec = g_dec-r_dec

g_lim = 24.
r_lim = 23.4#23.4 is DESI#24.1 is DEEP2
z_lim = 22.5#22.5
color_selection = (g_dec < g_lim) & (r_dec < r_lim) & (z_dec < z_lim)
x_dec = x_dec[color_selection]
y_dec = y_dec[color_selection]

# load the synthetic colors
sub_id, ugriz_grizy, stellar_mass, flux_oii = load_fsps(snap,dust_index,tng=tng,cam_filt=redshift_dict[snap][0])

# log mass of the tng galaxies
log_mass_star = np.log10(stellar_mass)
print("# objects over logM = 10: ",np.sum(log_mass_star > 10.))

if want_photo_scatter:
    fac = 1.
    g_dec_fsps = get_scatter(ugriz_grizy[:,5],g_lim,factor=fac)
    r_dec_fsps = get_scatter(ugriz_grizy[:,6],r_lim,factor=fac)
    z_dec_fsps = get_scatter(ugriz_grizy[:,8],z_lim,factor=fac)
else:
    g_dec_fsps = ugriz_grizy[:,5]
    r_dec_fsps = ugriz_grizy[:,6]
    z_dec_fsps = ugriz_grizy[:,8]


# selection of physical objects
selection_dec_fsps = (g_dec_fsps < 100000.)
g_dec_fsps = g_dec_fsps[selection_dec_fsps]
r_dec_fsps = r_dec_fsps[selection_dec_fsps]
z_dec_fsps = z_dec_fsps[selection_dec_fsps]
log_mass_star = log_mass_star[selection_dec_fsps]

def get_col_hist(grz_dec,grz_dec_fsps):
    grz_min = np.min(np.hstack((grz_dec,grz_dec_fsps)))
    grz_max = np.max(np.hstack((grz_dec,grz_dec_fsps)))
    grz_bins = np.linspace(grz_min,grz_max,n_bins)
    grz_cents = .5*(grz_bins[1:]+grz_bins[:-1])

    hist_grz, edges = np.histogram(grz_dec,bins=grz_bins,density=True)
    hist_grz_fsps, edges = np.histogram(grz_dec_fsps,bins=grz_bins,density=True)
    return grz_cents, hist_grz, hist_grz_fsps

if want_hist:
    n_bins = 41
    g_cents, hist_g, hist_g_fsps = get_col_hist(g_dec,g_dec_fsps)
    r_cents, hist_r, hist_r_fsps = get_col_hist(r_dec,r_dec_fsps)
    z_cents, hist_z, hist_z_fsps = get_col_hist(z_dec,z_dec_fsps)

    plt.subplots(1,3,figsize=(5.4*3,4.9))

    plt.subplot(1,3,1)
    plt.plot(g_cents,hist_g,label='DEEP2-DR8')
    plt.plot(g_cents,hist_g_fsps,label="TNG-FSPS")
    plt.legend()
    plt.xlabel("g")

    plt.subplot(1,3,2)
    plt.plot(r_cents,hist_r,label='DEEP2-DR8')
    plt.plot(r_cents,hist_r_fsps,label="TNG-FSPS")
    plt.legend()
    plt.xlabel("r")

    plt.subplot(1,3,3)
    plt.plot(z_cents,hist_z,label='DEEP2-DR8')
    plt.plot(z_cents,hist_z_fsps,label="TNG-FSPS")
    plt.legend()
    plt.xlabel("z")

    plt.savefig("figs/hist_grz.png")

print("# total galaxies = ",np.sum(selection_dec_fsps))

x_dec_fsps = r_dec_fsps-z_dec_fsps
y_dec_fsps = g_dec_fsps-r_dec_fsps
color_selection_fsps = (r_dec_fsps < r_lim) & (g_dec_fsps < g_lim) & (z_dec_fsps < z_lim)
x_dec_fsps = x_dec_fsps[color_selection_fsps]
y_dec_fsps = y_dec_fsps[color_selection_fsps]

# testing the rotation
theta = None
figname = "paper/deep2_dr"+dr_v+snap_dir+".png"
#figname = "paper/deep2_dr"+dr_v+snap_dir+".pdf"
make_scatter_histogram(x_dec_fsps,y_dec_fsps,x_dec,y_dec,figname=figname,redshift=z_dic[snap_dir],theta=theta)
