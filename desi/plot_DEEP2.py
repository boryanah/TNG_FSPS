import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photometric_scatter import get_scatter
import sys

nomask = ''#'-NoMask'#''
dr_v = '8'#'5'
deep2dir = '/mnt/gosling1/boryanah/TNG100/TNG_FSPS/DEEP2-DESI/data/derived/'
hdu = fits.open(deep2dir+'DR'+dr_v+'-matched-to-DEEP2-f2-glim24p25'+nomask+'.fits')
dat2 = hdu[1].data
hdu = fits.open(deep2dir+'DR'+dr_v+'-matched-to-DEEP2-f3-glim24p25'+nomask+'.fits')
dat3 = hdu[1].data
hdu = fits.open(deep2dir+'DR'+dr_v+'-matched-to-DEEP2-f4-glim24p25'+nomask+'.fits')
dat4 = hdu[1].data

def nMgy_to_mag(f):
    #f = np.clip(f,0.001,1.e11)
    return 22.5-2.5*np.log10(f)

def combine_col_fits(color):
    return np.hstack(((dat2['flux_'+color])[z_sel2],(dat3['flux_'+color])[z_sel3],(dat4['flux_'+color])[z_sel4]))

def make_scatter_histogram(x,y,z,w,s=0.1,al=0.3):

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
    ax_scatter.scatter(x, y,s=s,color='b',alpha=al,label='TNG-FSPS')
    ax_scatter.scatter(z, w,s=s,color='r',label='DEEP2-DR8')
    ax_scatter.set_xlabel("r-z")
    ax_scatter.set_ylabel("g-r")
    mean_z = np.mean(z[np.logical_not(np.isnan(z) | np.isinf(z))])
    mean_w = np.mean(w[np.logical_not(np.isnan(w) | np.isinf(w))])
    ax_scatter.scatter(mean_z, mean_w,s=100,color='orange',marker='x',label='avg DEEP2-DR8')
    ax_scatter.scatter(np.mean(x), np.mean(y),s=100,color='y',marker='x',label='avg TNG-FSPS')
    
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

    plt.savefig("deep2_dr"+dr_v+"_"+str(redshift)+zterm+photo_scatter+".png")
    plt.close()#plt.show()

snap = sys.argv[1]
if snap == '041': z_min = 1.3; z_max = 1.5; 
if snap == '047': z_min = 1.; z_max = 1.2;
if snap == '055': z_min = .8; z_max = .85;

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

print("total DEEP2/DESI gals = ",np.sum(selection_dec))

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

g_lim = 24.#3
r_lim = 23.4#4#3#23 is conservative cut #23.4 is DESI#24.1 is DEEP2
z_lim = 22.5
color_selection = (g_dec < g_lim) & (r_dec < r_lim) & (z_dec < z_lim)
x_dec = x_dec[color_selection]
y_dec = y_dec[color_selection]

y_line1 = -0.068*x_dec+0.457
y_line2 = 0.112*x_dec+0.773
x_line1 = 0.218*y_dec+0.571
#x_line1 = 0.637*y_dec+0.399
x_line2 = -0.555*y_dec+1.901


# parameter choices
tng = 'tng300'
#snap = '041'#'041'#'047'#'055'#'050'#'059'
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

if want_photo_scatter:
    fac = 1.#0.6
    g_dec_sp = get_scatter(ugriz_grizy[:,5],g_lim,factor=fac)
    r_dec_sp = get_scatter(ugriz_grizy[:,6],r_lim,factor=fac)
    z_dec_sp = get_scatter(ugriz_grizy[:,8],z_lim,factor=fac)
else:
    g_dec_sp = ugriz_grizy[:,5]
    r_dec_sp = ugriz_grizy[:,6]
    z_dec_sp = ugriz_grizy[:,8]


log_mass_star = np.log10(stellar_mass)
selection_dec_sp = (g_dec_sp < 100000.)
g_dec_sp = g_dec_sp[selection_dec_sp]
r_dec_sp = r_dec_sp[selection_dec_sp]
z_dec_sp = z_dec_sp[selection_dec_sp]
log_mass_star = log_mass_star[selection_dec_sp]

n_bins = 41

def get_col_hist(grz_dec,grz_dec_sp):
    grz_min = np.min(np.hstack((grz_dec,grz_dec_sp)))
    grz_max = np.max(np.hstack((grz_dec,grz_dec_sp)))
    grz_bins = np.linspace(grz_min,grz_max,n_bins)
    grz_cents = .5*(grz_bins[1:]+grz_bins[:-1])

    hist_grz, edges = np.histogram(grz_dec,bins=grz_bins,density=True)
    hist_grz_sp, edges = np.histogram(grz_dec_sp,bins=grz_bins,density=True)
    return grz_cents, hist_grz, hist_grz_sp

g_cents, hist_g, hist_g_sp = get_col_hist(g_dec,g_dec_sp)
r_cents, hist_r, hist_r_sp = get_col_hist(r_dec,r_dec_sp)
z_cents, hist_z, hist_z_sp = get_col_hist(z_dec,z_dec_sp)

plt.subplots(1,3,figsize=(5.4*3,1*5.))

plt.subplot(1,3,1)
plt.plot(g_cents,hist_g,label='DEEP2-DR8')
plt.plot(g_cents,hist_g_sp,label="TNG-FSPS")
plt.legend()
plt.xlabel("g")

plt.subplot(1,3,2)
plt.plot(r_cents,hist_r,label='DEEP2-DR8')
plt.plot(r_cents,hist_r_sp,label="TNG-FSPS")
plt.legend()
plt.xlabel("r")

plt.subplot(1,3,3)
plt.plot(z_cents,hist_z,label='DEEP2-DR8')
plt.plot(z_cents,hist_z_sp,label="TNG-FSPS")
plt.legend()
plt.xlabel("z")

plt.savefig("hist_grz.png")

print("Total gals = ",np.sum(selection_dec_sp))

x_dec_sp = r_dec_sp-z_dec_sp
y_dec_sp = g_dec_sp-r_dec_sp
color_selection_sp = (r_dec_sp < r_lim) & (g_dec_sp < g_lim) & (z_dec_sp < z_lim)
x_dec_sp = x_dec_sp[color_selection_sp]
y_dec_sp = y_dec_sp[color_selection_sp]

'''
# sandro's cuts
xlim = 0.9#0.8
y_dec_sp = y_dec_sp[x_dec_sp > xlim]
x_dec_sp = x_dec_sp[x_dec_sp > xlim]
y_dec = y_dec[x_dec > xlim]
x_dec = x_dec[x_dec > xlim]
'''

make_scatter_histogram(x_dec_sp,y_dec_sp,x_dec,y_dec)
quit()
plt.scatter(x_dec_sp,y_dec_sp,s=0.1,alpha=.3)
plt.scatter(x_dec,y_dec,color='k',s=0.1)
plt.plot(x_line1,y_dec)
plt.plot(x_line2,y_dec)
plt.plot(x_dec,y_line1)
plt.plot(x_dec,y_line2)
plt.xlabel("r-z")
plt.ylabel("g-r")
plt.axis('equal')
plt.xlim([-0.5,3])
plt.ylim([-0.5,3.4])
plt.savefig("deep2_dr"+dr_v+"_"+str(redshift)+zterm+photo_scatter+".png")
plt.close()
