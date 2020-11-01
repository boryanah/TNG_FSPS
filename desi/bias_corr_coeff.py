import numpy as np
import matplotlib.pyplot as plt
from corrfunc_util import get_jack, get_fast_jack
from halotools.utils import randomly_downsample_data

selection = '_DESI'
#selection = '_eBOSS'

redshift_dict = {'055':['sdss_desi','_0.0'],#-0.5
                 '047':['sdss_desi','_-0.0'],
                 '041':['sdss_des','']}

# snapshot number and dust model
snap = '0%d'%(55)#'041'#'047'#'055'#'050'#'059'
snap_dir = '_'+str(int(snap))
dust_index = redshift_dict[snap][1]
env_type = ''

# directory of the halo TNG data
box_name = "TNG300"
Lbox = 205.
root = "/mnt/gosling1/boryanah/"+box_name+"/"

# load the elg sub ids
mock_data = '/mnt/gosling1/boryanah/TNG100/TNG_FSPS/desi/mock/data/'
sub_id_col = np.load(mock_data+"sub_id"+env_type+snap_dir+selection+"_col.npy")
sub_id_sfg = np.load(mock_data+"sub_id"+env_type+snap_dir+selection+"_sfg.npy")
sub_id_all = np.load(mock_data+"sub_id"+env_type+snap_dir+selection+"_all.npy")


# loading the halo mass and group identification
# og
SubhaloPos_fp = np.load(root+'SubhaloPos_fp'+snap_dir+'.npy')/1.e3
# TESTING
#SubhaloPos_fp = np.load('/home/boryanah/lars/data_2500/SubhaloPos_fp.npy')/1.e3
#SubhaloMassType_fp = np.load('/home/boryanah/lars/data_2500/SubhaloMassType_fp.npy')*1.e10
#sub_mstar = SubhaloMassType_fp[:,4]
#sub_id_all = (np.argsort(sub_mstar)[::-1])[:12000]


N_m = 625**3#pos_m.shape[0]
down = 5000
N_m_down = N_m//down
print("Number of pcles = ",N_m_down)
print("Downsampling...")
# load particles
try:
    # TESTING
    #pos_m = np.load("/home/boryanah/lars/LSSIllustrisTNG/Lensing//pos_m_down_"+str(down)+".npy").astype(np.float32)
    # og
    pos_m = np.load("data_parts/pos_m_down_"+str(down)+".npy")
except:
    pos_m = np.load(root+'pos_parts_tng300-3'+snap_dir+'.npy')/1000.

    #x_m = pos_m[::down,0]
    #y_m = pos_m[::down,1]
    #z_m = pos_m[::down,2]
    #pos_m = np.vstack((x_m,y_m,z_m)).T

    pos_m = randomly_downsample_data(pos_m, N_m_down)

    print(pos_m.shape[0])
    np.save("data_parts/pos_m_down_"+str(down)+".npy",pos_m)
    plt.scatter(pos_m[:,0],pos_m[:,1],s=0.01)
    plt.show()
print("Downsampled O.o!")

# the positions of the galaxies
pos_col = SubhaloPos_fp[sub_id_col]
pos_sfg = SubhaloPos_fp[sub_id_sfg]
pos_all = SubhaloPos_fp[sub_id_all]

# TESTING
'''
pos_sfg = np.load("/home/boryanah//lars/LSSIllustrisTNG/Lensing/data_2dhod_pos/true_gals.npy").astype(np.float32)

pos_all = pos_all[:pos_sfg.shape[0]]


overlap, chosen1, chosen2 = np.intersect1d(np.round(np.sum(pos_sfg,axis=1),2),np.round(np.sum(pos_all,axis=1),2),return_indices=1)
print("overlap = ",len(overlap))
pos_sfg = pos_sfg[chosen1]
pos_all = pos_all[chosen2]

chosen_sfg = (pos_sfg[:,2] > 100.) & (pos_sfg[:,2] < 150.)
chosen_all = (pos_all[:,2] > 100.) & (pos_all[:,2] < 150.)
plt.scatter(pos_sfg[chosen_sfg,0],pos_sfg[chosen_sfg,1],s=1,label='mass-selected+bijective')
plt.scatter(pos_all[chosen_all,0],pos_all[chosen_all,1],s=1,label='mass-selected')
plt.axis('equal')
plt.legend()
plt.show()
'''

N_bin = 16
lb_min = -0.7
lb_max = 1.2
bins = np.logspace(lb_min,lb_max,N_bin)
bin_centers = .5*(bins[1:] + bins[:-1]) 

bias_col_mean, bias_col_err, corr_coeff_col_mean, corr_coeff_col_err = get_fast_jack('bias_corr_coeff',pos_col,pos_m,pos_col,pos_m,Lbox,bins)
bias_sfg_mean, bias_sfg_err, corr_coeff_sfg_mean, corr_coeff_sfg_err = get_fast_jack('bias_corr_coeff',pos_sfg,pos_m,pos_sfg,pos_m,Lbox,bins)
bias_all_mean, bias_all_err, corr_coeff_all_mean, corr_coeff_all_err = get_fast_jack('bias_corr_coeff',pos_all,pos_m,pos_all,pos_m,Lbox,bins)

print("Averaged bias at snapshot number ",snap)
print("Color-selected bias = ",np.mean(bias_col_mean[bin_centers > 1.]))
print("SFR-selected bias = ",np.mean(bias_sfg_mean[bin_centers > 1.]))
print("Mass-selected bias = ",np.mean(bias_all_mean[bin_centers > 1.]))

np.save("data_parts/bin_cents.npy",bin_centers)

def save_bias_corr_coeff(opt,bias_mean, bias_err, corr_coeff_mean, corr_coeff_err):

    np.save("data_parts/bias_mean_"+opt+".npy",bias_mean)
    np.save("data_parts/corr_coeff_mean_"+opt+".npy",corr_coeff_mean)
    np.save("data_parts/bias_error_"+opt+".npy",bias_err)
    np.save("data_parts/corr_coeff_error_"+opt+".npy",corr_coeff_err)

#save_bias_corr_coeff('col',bias_col_mean, bias_col_err, corr_coeff_col_mean, corr_coeff_col_err)
#save_bias_corr_coeff('sfg',bias_sfg_mean, bias_sfg_err, corr_coeff_sfg_mean, corr_coeff_sfg_err)
save_bias_corr_coeff('all',bias_all_mean, bias_all_err, corr_coeff_all_mean, corr_coeff_all_err)

plt.figure(1)
plt.plot(bin_centers,np.ones(len(bin_centers)),'k--')
#plt.errorbar(bin_centers,bias_col_mean,yerr=bias_col_err,label='color-selected')
plt.errorbar(bin_centers,bias_sfg_mean,yerr=bias_sfg_err,label='mass-selected+bijective')#TESTING'SFR-selected')
plt.errorbar(bin_centers,bias_all_mean,yerr=bias_all_err,label='mass-selected')
plt.legend()
plt.xscale('log')
plt.savefig('test_bias.png')
#plt.yscale('log')

plt.figure(2)
plt.plot(bin_centers,np.ones(len(bin_centers)),'k--')
#plt.errorbar(bin_centers,corr_coeff_col_mean,yerr=corr_coeff_col_err,label='color-selected')
plt.errorbar(bin_centers,corr_coeff_sfg_mean,yerr=corr_coeff_sfg_err,label='mass-selected+bijective')#TESTING'SFR-selected')
plt.errorbar(bin_centers,corr_coeff_all_mean,yerr=corr_coeff_all_err,label='mass-selected')
plt.xscale('log')
plt.legend()
#plt.yscale('log')
plt.savefig('test_corr.png')
plt.show()
