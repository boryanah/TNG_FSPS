import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import Corrfunc
from random import choices
from util import get_density, smooth_density, get_jack_corr, get_counts, get_hist
import plotparams
plotparams.buba()
import sys

np.random.seed(3000)

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

# particle directory
halo_parts_dir = "/mnt/gosling1/boryanah/TNG300/ELG_halo_parts/"
Omega_M = 0.3089
Omega_B = 0.0486
Omega_DM = Omega_M-Omega_B


# loading the halo mass and group identification
Group_M_Mean200_fp = np.load(root+'Group_M_Mean200_fp'+snap_dir+'.npy')*1.e10
Group_R_Mean200_fp = np.load(root+'Group_R_Mean200_fp'+snap_dir+'.npy')/1.e3
SubhaloGrNr_fp = np.load(root+'SubhaloGrNr_fp'+snap_dir+'.npy').astype(int)
SubhaloPos_fp = np.load(root+'SubhaloPos_fp'+snap_dir+'.npy')/1.e3
GroupPos_fp = np.load(root+'GroupPos_fp'+snap_dir+'.npy')/1.e3

Group_M_Mean200_dm = np.load(root+'Group_M_Mean200_dm'+snap_dir+'.npy')*1.e10
Group_R_Mean200_dm = np.load(root+'Group_R_Mean200_dm'+snap_dir+'.npy')/1.e3
SubhaloGrNr_dm = np.load(root+'SubhaloGrNr_dm'+snap_dir+'.npy').astype(int)
SubhaloPos_dm = np.load(root+'SubhaloPos_dm'+snap_dir+'.npy')/1.e3
GroupPos_dm = np.load(root+'GroupPos_dm'+snap_dir+'.npy')/1.e3

N_halos_fp = GroupPos_fp.shape[0]
N_halos_dm = GroupPos_dm.shape[0]

bin_edges = np.logspace(np.log10(0.007),np.log10(5.),21)
bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5

def get_hist_parts(inds,counts,dm_type='fp',delta='col'):

    nparts = np.load(halo_parts_dir+"nstart_halo_"+delta+"_"+dm_type+".npy")
    nstart = np.zeros(len(nparts),dtype=int)
    nstart[1:] = np.cumsum(nparts)[:-1]
    coords = np.load(halo_parts_dir+"coords_halo_"+delta+"_"+dm_type+".npy")/1000.

    if dm_type == 'fp':
        GroupPos = GroupPos_fp
        Group_R_Mean200 = Group_R_Mean200_fp
        f = Omega_M/Omega_DM
        
    elif dm_type == 'hod':
        GroupPos = GroupPos_dm
        Group_R_Mean200 = Group_R_Mean200_dm
        f = 1.

    hist = np.zeros(len(bin_cents))

    for i in range(len(inds)):

        if i % 100 == 0: print(i)
        
        pos_cen = GroupPos[inds[i]]
        r2_mean = Group_R_Mean200[inds[i]]**2

        #count = counts[inds[i]]
        count = nparts[i]
        pos_parts = coords[nstart[i]:nstart[i]+count]

        pos_diff = pos_parts-pos_cen
        d2 = np.sum(pos_diff**2,axis=1)

        d2 /= r2_mean

        hist_this, bins = np.histogram(np.sqrt(d2),bins=bin_edges)
        hist += hist_this

    hist /= len(inds)
    hist *= f
    return hist

delta = 'col'

count_col_fp = np.load("data_counts/count_halo_"+delta+"_fp.npy")
count_col_hod = np.load("data_counts/count_halo_"+delta+"_hod.npy")

print(len(count_col_fp),len(count_col_hod))

inds_fp = np.where(count_col_fp > 0)[0]
inds_hod = np.where(count_col_hod > 0)[0]

hist_fp = get_hist_parts(inds_fp,count_col_fp,dm_type='fp')
#hist_hod = get_hist_parts(inds_hod,count_col_hod,dm_type='hod')

np.save("data_prof/hist_col_fp_dm.npy",hist_fp)
#np.save("data_prof/hist_col_hod_dm.npy",hist_hod)

quit()

fp_dmo_halo_inds = np.load(root+'fp_dmo_halo_inds'+snap_dir+'.npy')
fp_halo_inds = fp_dmo_halo_inds[0]
dmo_halo_inds = fp_dmo_halo_inds[1]


# get parent indices of the centrals and their subhalo indices in the original array
unique_sub_grnr, firsts = np.unique(SubhaloGrNr_fp,return_index=True)
overlap, comm1, comm2 = np.intersect1d(firsts,sub_id_col,return_indices=True)
print("galaxies which are centrals = ",len(overlap))

# position of the selected galaxies
d2_col = np.zeros(len(sub_id_col))-2.
pos_col = SubhaloPos_fp[sub_id_col]
w_col = np.ones(pos_col.shape[0],dtype=pos_col.dtype)

sub_grnr_col = SubhaloGrNr_fp[sub_id_col]


d2_vmax = np.zeros(len(sub_id_col))-2.
pos_vmax = np.zeros((len(sub_id_col),3))
w_vmax = np.ones((len(sub_id_col)))
d2_vpeak = np.zeros(len(sub_id_col))-2.
pos_vpeak = np.zeros((len(sub_id_col),3))
w_vpeak = np.ones((len(sub_id_col)))

sub_grnr_col_uni, inds = np.unique(sub_grnr_col,return_index=True)
pos_col_new = np.zeros((len(sub_id_col),3))

match, inds_match, inds_uni = np.intersect1d(fp_halo_inds,sub_grnr_col_uni,return_indices=1)

missed = 0
central = 0
gal_sofar = 0
# START HERE

# this is better since we don't always have a central
for i in range(len(sub_grnr_col_uni)):
    if i % 100 == 0: print(i,len(sub_grnr_col_uni))
    # the first unique index is that of the first galaxy if it exists
    sub_id = sub_id_col[inds[i]]
    flag_c = int(sub_id in overlap)
    
    if not flag_c:
        #print("There is no central in this halo")
        central += 1
    assert flag_c <= 1, "wtf?"
        
    gal_grnr = sub_grnr_col_uni[i] == sub_grnr_col
    pos_fp = pos_col[gal_grnr]
    gal_num = np.sum(gal_grnr)
    num_sat = gal_num-flag_c
    
    pos_c_fp = GroupPos_fp[sub_grnr_col_uni[i]]
    
    
    if want_fp:
            
        fp_ind = sub_grnr_col_uni[i]
        first_sub = GroupFirstSub_fp[fp_ind]
        r2_mean = Group_R_Mean200_fp[fp_ind]**2
        n_sub = GroupNsubs_fp[fp_ind]

        all_grnr = np.arange(first_sub,first_sub+n_sub)
        poses = SubhaloPos_fp[all_grnr]
        vmaxes = SubhaloVmax_fp[all_grnr]
        vpeaks = SubhaloVpeak_fp[all_grnr]


        sats = np.ones(poses.shape[0],dtype=bool)
        sats[0] = False

        inds_vm = (np.argsort(vmaxes[sats])[::-1])[:num_sat]
        pos_vm = (poses[sats])[inds_vm]

        pos_diff = pos_vm-pos_c_fp
        d2 = np.sum(pos_diff**2,axis=1)
        if want_rescaled:
            d2 /= r2_mean
        d2_vmax[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = d2
        if flag_c:
            d2_vmax[gal_sofar] = -1.
        
        pos_vmax[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = pos_vm
        if flag_c:
            pos_vmax[gal_sofar] = SubhaloPos_fp[first_sub]

            
        inds_vp = (np.argsort(vpeaks[sats])[::-1])[:num_sat]
        pos_vp = (poses[sats])[inds_vp]

        pos_diff = pos_vp-pos_c_fp
        d2 = np.sum(pos_diff**2,axis=1)
        if want_rescaled:
            d2 /= r2_mean
        d2_vpeak[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = d2
        if flag_c:
            d2_vpeak[gal_sofar] = -1.
        

        pos_vpeak[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = pos_vp
        if flag_c:
            pos_vpeak[gal_sofar] = SubhaloPos_fp[first_sub]
            
    else:
        chosen = np.where(sub_grnr_col_uni[i] == match)[0]
        
        if len(chosen) == 0:
            missed += gal_num
            continue

        dmo_ind = dmo_halo_inds[inds_match[chosen]]
        
        first_sub = GroupFirstSub_dm[dmo_ind][0]
        r2_mean = Group_R_Mean200_dm[dmo_ind]**2
        n_sub = GroupNsubs_dm[dmo_ind]
        pos_c_dm = GroupPos_dm[dmo_ind]


        all_grnr = np.arange(first_sub,first_sub+n_sub)
        poses = SubhaloPos_dm[all_grnr]
        vmaxes = SubhaloVmax_dm[all_grnr]
        vpeaks = SubhaloVpeak_dm[all_grnr]

        sats = np.ones(poses.shape[0],dtype=bool)
        sats[0] = False
        # og
        #sats = np.sum((poses-pos_c_dm)**2,axis=1) >= 1.e-8
            
        vmaxes_sats = vmaxes[sats]
        vpeaks_sats = vpeaks[sats]
        inds_vm = (np.argsort(vmaxes_sats))
        inds_vp = (np.argsort(vpeaks_sats))
            

        if want_scatter1:
            vmaxes_sats = 10.**np.random.normal(np.log10(vmaxes_sats),delta1) 
            inds_vm = np.argsort(vmaxes_sats)
                
        if want_scatter2:
            vpeaks_sats = 10.**np.random.normal(np.log10(vpeaks_sats),delta2) 
            inds_vp = np.argsort(vpeaks_sats)
            
        inds_vm = inds_vm[::-1]
        inds_vp = inds_vp[::-1]
            
        inds_vm = inds_vm[:num_sat]
        inds_vp = inds_vp[:num_sat]
            
        pos_vm = (poses[sats])[inds_vm]
        pos_vp = (poses[sats])[inds_vp]

        pos_diff = pos_vm-pos_c_dm
        d2 = np.sum(pos_diff**2,axis=1)
        if want_rescaled:
            d2 /= r2_mean
        d2_vmax[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = d2
        if flag_c:
            d2_vmax[gal_sofar] = -1.
        pos_vmax[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = pos_vm
        if flag_c:
            pos_vmax[gal_sofar] = SubhaloPos_dm[first_sub]
            
        pos_diff = pos_vp-pos_c_dm
        d2 = np.sum(pos_diff**2,axis=1)
        if want_rescaled:
            d2 /= r2_mean
        d2_vpeak[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = d2
        if flag_c:
            d2_vpeak[gal_sofar] = -1.
        pos_vpeak[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = pos_vp
        if flag_c:
            pos_vpeak[gal_sofar] = SubhaloPos_dm[first_sub]
    
    
    pos_diff_fp = pos_fp-pos_c_fp
    # TESTING problem was with the rescaling
    d2_fp = np.sum(pos_diff_fp**2,axis=1)
    # og
    #d2_fp = np.sum(pos_diff_fp**2,axis=1)[flag_c:]

    if want_rescaled:
        r2_mean = Group_R_Mean200_fp[sub_grnr_col_uni[i]]**2
        d2_fp /= r2_mean

    # TESTING
    d2_col[gal_sofar:gal_sofar+num_sat+flag_c] = d2_fp
    # og
    #d2_col[gal_sofar:gal_sofar+num_sat] = d2_fp
    
    if want_weights:
        pos_col_new[gal_sofar:gal_sofar+num_sat] = pos_c_fp
    gal_sofar += num_sat+flag_c

missing = np.sum(pos_vmax,axis=1) != 0.
missing_central = d2_vmax > 0.#TESTING-1.
print("missing = ",len(sub_id_col)-np.sum(missing))
print("missed = ",missed)
print("halos without a central",central)
print("galaxies that are not centrals",np.sum(missing_central))

d2_col = d2_col[missing_central]
d2_vpeak = d2_vpeak[missing_central]
d2_vmax = d2_vmax[missing_central]
#TESTING
'''
pos_center_col = GroupPos_fp[SubhaloGrNr_fp[sub_id_col]]
pos_col = SubhaloPos_fp[sub_id_col]
pos_diff = pos_col-pos_center_col
d2_col = np.sum(pos_diff**2,axis=1)
'''
# og
pos_col = pos_col[missing]
pos_vpeak = pos_vpeak[missing]
pos_vmax = pos_vmax[missing]

np.save("data_prof/d2_col"+dm_ext+".npy",d2_col)
np.save("data_prof/d2_"+name_proxy2+dm_ext+".npy",d2_vpeak)
np.save("data_prof/d2_"+name_proxy1+dm_ext+".npy",d2_vmax)
np.save("data_prof/pos_col"+dm_ext+".npy",pos_col)
np.save("data_prof/pos_"+name_proxy2+dm_ext+".npy",pos_vpeak)
np.save("data_prof/pos_"+name_proxy1+dm_ext+".npy",pos_vmax)
# END HERE
quit()

d2_col = np.load("data_prof/d2_col"+dm_ext+".npy")
d2_vpeak = np.load("data_prof/d2_"+name_proxy2+dm_ext+".npy")
d2_vmax = np.load("data_prof/d2_"+name_proxy1+dm_ext+".npy")
pos_col = np.load("data_prof/pos_col"+dm_ext+".npy")
pos_vpeak = np.load("data_prof/pos_"+name_proxy2+dm_ext+".npy")
pos_vmax = np.load("data_prof/pos_"+name_proxy1+dm_ext+".npy")

w_col = np.ones(pos_col.shape[0],dtype=pos_col.dtype)
w_vmax = np.ones(pos_vmax.shape[0],dtype=pos_vmax.dtype)
w_vpeak = np.ones(pos_vpeak.shape[0],dtype=pos_vpeak.dtype)

if want_rescaled:
    bin_edges = np.logspace(np.log10(0.007),np.log10(3.),21)
else:
    bin_edges = np.logspace(np.log10(0.007),np.log10(5.),21)

#bin_edges = np.linspace(0.0,5.,41)
vol_edges = 4./3*np.pi*bin_edges**3
vol_bins = vol_edges[1:]-vol_edges[:-1]
hist, bins = np.histogram(np.sqrt(d2_col),bins=bin_edges)
hist_vmax, bins = np.histogram(np.sqrt(d2_vmax),bins=bin_edges)
hist_vpeak, bins = np.histogram(np.sqrt(d2_vpeak),bins=bin_edges)
print("contained within histogram for col and vmax = ",np.sum(hist),np.sum(hist_vmax))
bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5

if want_weights:
    # check that random number of points recovers distribution
    weights = hist*1./np.sum(hist)
    population = bin_cents
    rad_distn = choices(population, weights, k=np.sum(hist))
    hist_new, bins = np.histogram(rad_distn,bins=bin_edges)
    np.save("data/weights.npy",weights)
    np.save("data/population.npy",population)

plt.figure(2,figsize=(8,6))

plt.plot(bin_cents,hist_vmax/vol_bins,c='dimgray',lw=4,zorder=1,label=r'$V_{\rm max}$')
plt.plot(bin_cents,hist_vpeak/vol_bins,c='silver',lw=4,zorder=2,label=r'$V_{\rm peak}$')
plt.scatter(bin_cents,hist/vol_bins,c='dodgerblue',s=90,zorder=3,marker='*',label='DESI ELGs')
#plt.plot(bin_cents,hist_new/vol_bins,label='drawn')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
plt.ylabel(r'$n(r)$')
plt.xlim([0.007,5])
#plt.savefig("figs/gal_distn.png")
plt.savefig("../paper/gal_distn_"+name_proxy1+"_"+name_proxy2+".png")

# below we attemot ti draw from the distn to get the small-scale
overlap, comm1, comm2 = np.intersect1d(firsts,sub_id_col,return_indices=True)
N_cen = len(overlap)
N_gal = len(sub_id_col)
N_sat = N_gal-N_cen

if want_weights:
    theta = np.random.rand(N_gal)*np.pi
    phi = np.random.rand(N_gal)*2*np.pi
    
    radius = choices(population, weights, k=N_gal)
    x = radius*np.cos(phi)*np.sin(theta)
    y = radius*np.sin(phi)*np.sin(theta)
    z = radius*np.cos(theta)
    xyz = np.vstack((x,y,z)).T
    
    pos_col_new += xyz
    pos_col_new[comm2] -= xyz[comm2]
    pos_col_new[pos_col_new >= Lbox] -= Lbox
    pos_col_new[pos_col_new <= 0.] += Lbox 
    w_col_new = np.ones(pos_col_new.shape[0],dtype=pos_col_new.dtype)

plt.figure(1,figsize=(8,6))
# random draw
#Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(pos_col,w_col,pos_col_new,w_col_new,Lbox)

Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(pos_col,w_col,pos_vmax,w_vmax,Lbox)
plt.plot(bin_centers,np.ones(len(bin_centers)),'k--')
plt.errorbar(bin_centers*1.05,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='dimgray',ls='-',label=r'$V_{\rm max}$',alpha=1.,fmt='o',capsize=4)

Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(pos_col,w_col,pos_vpeak,w_vpeak,Lbox)
plt.errorbar(bin_centers,Rat_hodtrue_mean,yerr=Rat_hodtrue_err,color='silver',ls='-',label=r'$V_{\rm peak}$',alpha=1.,fmt='o',capsize=4)

plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
plt.ylabel(r'$\xi(r)_{\rm model}/\xi(r)_{\rm DESI \ ELG}$')
plt.xscale('log')
plt.legend()
plt.ylim([0.5,1.5])
#plt.savefig("figs/prof_ELG.png")
plt.savefig("../paper/prof_ELG_"+name_proxy1+"_"+name_proxy2+".png")
plt.show()
plt.close()
