import numpy as np
import matplotlib.pyplot as plt
import h5py
import Corrfunc
from random import choices
from util import get_density, smooth_density, get_jack_corr, get_counts, get_hist
import plotparams
plotparams.buba()

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

want_fp = 1 # do you wanna use dm or fp
if want_fp:
    dm_ext = '_fp'
else:
    dm_ext = '_dm'

# load the elg sub ids
sub_id_col = np.load("data/sub_id"+env_type+snap_dir+selection+"_col.npy")
sub_id_sfg = np.load("data/sub_id"+env_type+snap_dir+selection+"_sfg.npy")
sub_id_all = np.load("data/sub_id"+env_type+snap_dir+selection+"_all.npy")

# loading the halo mass and group identification
Group_M_Mean200_fp = np.load(root+'Group_M_Mean200_fp'+snap_dir+'.npy')*1.e10
SubhaloGrNr_fp = np.load(root+'SubhaloGrNr_fp'+snap_dir+'.npy').astype(int)
SubhaloPos_fp = np.load(root+'SubhaloPos_fp'+snap_dir+'.npy')/1.e3
GroupPos_fp = np.load(root+'GroupPos_fp'+snap_dir+'.npy')/1.e3

Group_M_Mean200_dm = np.load(root+'Group_M_Mean200_dm'+snap_dir+'.npy')*1.e10
SubhaloGrNr_dm = np.load(root+'SubhaloGrNr_dm'+snap_dir+'.npy').astype(int)
SubhaloPos_dm = np.load(root+'SubhaloPos_dm'+snap_dir+'.npy')/1.e3
GroupPos_dm = np.load(root+'GroupPos_dm'+snap_dir+'.npy')/1.e3

# for subhalo vmax
SubhaloVmax_fp = np.load(root+'SubhaloVmax_fp'+snap_dir+'.npy')
SubhaloVpeak_fp = np.load(root+'SubhaloVpeak_fp'+snap_dir+'.npy')
GroupFirstSub_fp = np.load(root+'GroupFirstSub_fp'+snap_dir+'.npy')
GroupNsubs_fp = np.load(root+'GroupNsubs_fp'+snap_dir+'.npy')

SubhaloVmax_dm = np.load(root+'SubhaloVmax_dm'+snap_dir+'.npy')
SubhaloVpeak_dm = np.load(root+'SubhaloVpeak_dm'+snap_dir+'.npy')
GroupFirstSub_dm = np.load(root+'GroupFirstSub_dm'+snap_dir+'.npy')
GroupNsubs_dm = np.load(root+'GroupNsubs_dm'+snap_dir+'.npy')

N_halos_fp = GroupPos_fp.shape[0]
N_halos_dm = GroupPos_dm.shape[0]

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

want_vmax = 1
if want_vmax:
    d2_vmax = np.zeros(len(sub_id_col))-2.
    pos_vmax = np.zeros((len(sub_id_col),3))
    w_vmax = np.ones((len(sub_id_col)))
    d2_vpeak = np.zeros(len(sub_id_col))-2.
    pos_vpeak = np.zeros((len(sub_id_col),3))
    w_vpeak = np.ones((len(sub_id_col)))

sub_grnr_col_uni, inds = np.unique(sub_grnr_col,return_index=True)
pos_col_new = np.zeros((len(sub_id_col),3))

missed = 0
central = 0
gal_sofar = 0
# START HERE
'''
# this is better since we don't always have a central
for i in range(len(sub_grnr_col_uni)):
    if i % 100 == 0: print(i,len(sub_grnr_col_uni))
    # the first unique index is that of the first galaxy if it exists
    sub_id = sub_id_col[inds[i]]
    flag_c = int(sub_id in firsts)
    
    if not flag_c:
        #print("There is no central in this halo")
        central += 1
    assert flag_c <= 1, "wtf?"
        
    gal_grnr = sub_grnr_col_uni[i] == sub_grnr_col
    pos = pos_col[gal_grnr]
    gal_num = np.sum(gal_grnr)

    pos_c = GroupPos_fp[sub_grnr_col_uni[i]]
    #pos_c = pos_col[inds[i]]
    
        
    if want_vmax:
        if want_fp:
            fp_ind = sub_grnr_col_uni[i]
            first_sub = GroupFirstSub_fp[fp_ind]
            n_sub = GroupNsubs_fp[fp_ind]

            # TESTING
            #all_grnr = np.arange(first_sub+flag_c,first_sub+n_sub)
            # og
            all_grnr = np.arange(first_sub,first_sub+n_sub)
            poses = SubhaloPos_fp[all_grnr]
            vmaxes = SubhaloVmax_fp[all_grnr]
            vpeaks = SubhaloVpeak_fp[all_grnr]
            sats = np.sum((poses-pos_c)**2,axis=1) >= 1.e-8
            
            inds_vm = (np.argsort(vmaxes[sats])[::-1])[:gal_num-flag_c]
            inds_vp = (np.argsort(vpeaks[sats])[::-1])[:gal_num-flag_c]
            # og
            pos_vm = (poses[sats])[inds_vm]
            pos_vp = (poses[sats])[inds_vp]
            # TESTING
            #pos_vm = (SubhaloPos_fp[all_grnr])[(np.argsort(vmaxes)[::-1])[flag_c:gal_num+flag_c]]
            #pos_vp = (SubhaloPos_fp[all_grnr])[(np.argsort(vpeaks)[::-1])[flag_c:gal_num+flag_c]]

            pos_diff = pos_vm-pos_c
            d2 = np.sum(pos_diff**2,axis=1)
            num_sat = len(d2)
            assert (num_sat+flag_c == gal_num), "wtf"
            d2_vmax[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = d2
            d2_vmax[gal_sofar:gal_sofar+flag_c] = -1.
            pos_vmax[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = pos_vm
            if flag_c:
                pos_vmax[gal_sofar] = SubhaloPos_fp[first_sub]
            
            pos_diff = pos_vp-pos_c
            d2 = np.sum(pos_diff**2,axis=1)
            d2_vpeak[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = d2
            d2_vpeak[gal_sofar:gal_sofar+flag_c] = -1.
            pos_vpeak[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = pos_vp
            if flag_c:
                pos_vpeak[gal_sofar] = SubhaloPos_fp[first_sub]
        else:
            match_fp = fp_halo_inds == sub_grnr_col_uni[i]
            if np.sum(match_fp) == 0:
                missed += gal_num
                continue

            dmo_ind = dmo_halo_inds[match_fp]
            first_sub = GroupFirstSub_dm[dmo_ind][0]
            n_sub = GroupNsubs_dm[dmo_ind]

            # TESTING
            #all_grnr = np.arange(first_sub+flag_c,first_sub+n_sub)
            # og
            all_grnr = np.arange(first_sub,first_sub+n_sub)
            poses = SubhaloPos_dm[all_grnr]
            vmaxes = SubhaloVmax_dm[all_grnr]
            vpeaks = SubhaloVpeak_dm[all_grnr]
            sats = np.sum((poses-pos_c)**2,axis=1) >= 1.e-8
            
            inds_vm = (np.argsort(vmaxes[sats])[::-1])[:gal_num-flag_c]
            inds_vp = (np.argsort(vpeaks[sats])[::-1])[:gal_num-flag_c]
            # og
            pos_vm = (poses[sats])[inds_vm]
            pos_vp = (poses[sats])[inds_vp]
            # TESTING
            #pos_vm = (SubhaloPos_dm[all_grnr])[(np.argsort(vmaxes)[::-1])[flag_c:gal_num+flag_c]]
            #pos_vp = (SubhaloPos_dm[all_grnr])[(np.argsort(vpeaks)[::-1])[flag_c:gal_num+flag_c]]

            pos_diff = pos_vm-pos_c
            d2 = np.sum(pos_diff**2,axis=1)
            num_sat = len(d2)
            assert (num_sat+flag_c == gal_num), "wtf"
            d2_vmax[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = d2
            d2_vmax[gal_sofar:gal_sofar+flag_c] = -1.
            pos_vmax[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = pos_vm
            if flag_c:
                pos_vmax[gal_sofar] = SubhaloPos_dm[first_sub]
            
            pos_diff = pos_vp-pos_c
            d2 = np.sum(pos_diff**2,axis=1)
            d2_vpeak[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = d2
            d2_vpeak[gal_sofar:gal_sofar+flag_c] = -1.
            pos_vpeak[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = pos_vp
            if flag_c:
                pos_vpeak[gal_sofar] = SubhaloPos_dm[first_sub]
            

    pos_diff = pos-pos_c
    #plt.scatter(pos_diff[:,0],pos_diff[:,1],s=1)
    d2 = np.sum(pos_diff**2,axis=1)[flag_c:]
    num_sat = len(d2)
    
    d2_col[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = d2
    pos_col_new[gal_sofar:gal_sofar+num_sat] = pos_c
    gal_sofar += num_sat+flag_c


missing = np.sum(pos_vmax,axis=1) != 0.
missing_central = d2_vmax > -1.
print("missing = ",len(sub_id_col)-np.sum(missing))
print("missed = ",missed)
print("halos without a central",central)
print("galaxies that are not centrals",np.sum(missing_central))

d2_col = d2_col[missing_central]
d2_vpeak = d2_vpeak[missing_central]
d2_vmax = d2_vmax[missing_central]
pos_col = pos_col[missing]
pos_vpeak = pos_vpeak[missing]
pos_vmax = pos_vmax[missing]


np.save("data_prof/d2_col"+dm_ext+".npy",d2_col)
np.save("data_prof/d2_vpeak"+dm_ext+".npy",d2_vpeak)
np.save("data_prof/d2_vmax"+dm_ext+".npy",d2_vmax)
np.save("data_prof/pos_col"+dm_ext+".npy",pos_col)
np.save("data_prof/pos_vpeak"+dm_ext+".npy",pos_vpeak)
np.save("data_prof/pos_vmax"+dm_ext+".npy",pos_vmax)
# END HERE
'''


d2_col = np.load("data_prof/d2_col"+dm_ext+".npy")
d2_vpeak = np.load("data_prof/d2_vpeak"+dm_ext+".npy")
d2_vmax = np.load("data_prof/d2_vmax"+dm_ext+".npy")
pos_col = np.load("data_prof/pos_col"+dm_ext+".npy")
pos_vpeak = np.load("data_prof/pos_vpeak"+dm_ext+".npy")
pos_vmax = np.load("data_prof/pos_vmax"+dm_ext+".npy")
w_col = np.ones(pos_col.shape[0],dtype=pos_col.dtype)
w_vmax = np.ones(pos_vmax.shape[0],dtype=pos_vmax.dtype)
w_vpeak = np.ones(pos_vpeak.shape[0],dtype=pos_vpeak.dtype)


print(len(d2_col))
print(len(d2_vmax))
print(len(d2_vpeak))
print(len(w_col))
print(len(w_vmax))
print(len(w_vpeak))
print(np.sum(d2_col==0.))
print(np.sum(d2_vmax==0.))
print(np.sum(d2_vpeak==0.))

bin_edges = np.logspace(np.log10(0.007),np.log10(5.),41)
#bin_edges = np.linspace(0.0,5.,41)
vol_edges = 4./3*np.pi*bin_edges**3
vol_bins = vol_edges[1:]-vol_edges[:-1]
hist, bins = np.histogram(np.sqrt(d2_col),bins=bin_edges)
hist_vmax, bins = np.histogram(np.sqrt(d2_vmax),bins=bin_edges)
hist_vpeak, bins = np.histogram(np.sqrt(d2_vpeak),bins=bin_edges)
print("contained within histogram for col and vmax = ",np.sum(hist),np.sum(hist_vmax))
bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5

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
plt.savefig("figs/gal_distn.png")

# below we attemot ti draw from the distn to get the small-scale
overlap, comm1, comm2 = np.intersect1d(firsts,sub_id_col,return_indices=True)
N_cen = len(overlap)
N_gal = len(sub_id_col)
N_sat = N_gal-N_cen

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
plt.savefig("figs/prof_ELG.png")
plt.show()
