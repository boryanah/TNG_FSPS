import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import Corrfunc
from util import get_density, get_smooth_density, smooth_density, get_jack_corr, get_counts, get_hist
import plotparams
#plotparams.buba()
plotparams.default()

np.random.seed(7) # i liked 7# 10 not horrible

# environment
N_div = 5
bounds = np.linspace(0,N_div,N_div+1)/N_div

selection = '_DESI'
#selection = '_eBOSS'

redshift_dict = {'055':['sdss_desi','_0.0'],#-0.5
                 '047':['sdss_desi','_-0.0'],
                 '041':['sdss_des','']}

# snapshot number and dust model
snap = '0%d'%(55)
snap_dir = '_'+str(int(snap))
dust_index = redshift_dict[snap][1]
env_type = ''

N_div_vmax = 1
want_scatter = 1
v_type = 'vmax'

log_min = 11.
log_max = 15.
N_bins = 41
bin_edges = np.linspace(log_min,log_max,N_bins)
bin_cents = (bin_edges[1:]+bin_edges[:-1])*.5

# directory of the halo TNG data
box_name = "TNG300"
Lbox = 205.
root = "/mnt/gosling1/boryanah/"+box_name+"/"

# directory of particle coordinates for halos belonging to random seed 7
halo_parts_dir = "/mnt/gosling1/boryanah/TNG300/ELG_halo_parts/"

# load the elg sub ids
sub_id_col = np.load("data/sub_id"+env_type+snap_dir+selection+"_col.npy")
sub_id_sfg = np.load("data/sub_id"+env_type+snap_dir+selection+"_sfg.npy")
sub_id_all = np.load("data/sub_id"+env_type+snap_dir+selection+"_all.npy")


# loading the halo mass and group identification
Group_M_Mean200_fp = np.load(root+'Group_M_Mean200_fp'+snap_dir+'.npy')*1.e10
SubhaloGrNr_fp = np.load(root+'SubhaloGrNr_fp'+snap_dir+'.npy')
SubhaloPos_fp = np.load(root+'SubhaloPos_fp'+snap_dir+'.npy')/1.e3
GroupPos_fp = np.load(root+'GroupPos_fp'+snap_dir+'.npy')/1.e3
N_halos_fp = GroupPos_fp.shape[0]
inds_halo_fp = np.arange(N_halos_fp,dtype=int)

Group_M_Mean200_dm = np.load(root+'Group_M_Mean200_dm'+snap_dir+'.npy')*1.e10
GroupPos_dm = np.load(root+'GroupPos_dm'+snap_dir+'.npy')/1.e3


SubhaloVmax_dm = np.load(root+'SubhaloVmax_dm'+snap_dir+'.npy'); start = 5.#good
#SubhaloVmax_dm = np.load(root+'SubhaloMassInRad_dm'+snap_dir+'.npy');  start = 1.e7
#SubhaloVmax_dm = np.load(root+'SubhaloMass_dm'+snap_dir+'.npy');  start = 1.e7
#SubhaloVmax_dm = np.load(root+'SubhaloVelDisp_dm'+snap_dir+'.npy'); start = 20.
SubhaloVpeak_dm = np.load(root+'SubhaloVpeak_dm'+snap_dir+'.npy')
#SubhaloVpeak_dm = np.load(root+'SubhaloVelDisp_dm'+snap_dir+'.npy'); start = 5.
delta_col = 0.1# works well w/ 0.4
delta_sfg = 0.1# works well w/ 0.4
delta_all = 0.02# works well w/ 0.02
want_scatter_col = 1
want_scatter_sfg = 1
want_scatter_all = 0
v_type_col = 'vmax' # works well with vmax and 0.23
v_type_sfg = 'vmax'#'vpeak' # works well with vdisp and 0.23
v_type_all = 'vmax' # works well with no scatter and vmax

log_start = np.log10(start)
GroupFirstSub_dm = np.load(root+'GroupFirstSub_dm'+snap_dir+'.npy')
SubhaloPos_dm = np.load(root+'SubhaloPos_dm'+snap_dir+'.npy')/1.e3
GroupNsubs_dm = np.load(root+'GroupNsubs_dm'+snap_dir+'.npy')
N_halos_dm = GroupPos_dm.shape[0]
inds_halo_dm = np.arange(N_halos_dm,dtype=int)

fp_dmo_halo_inds = np.load(root+'fp_dmo_halo_inds'+snap_dir+'.npy')
fp_halo_inds = fp_dmo_halo_inds[0]
dmo_halo_inds = fp_dmo_halo_inds[1]

# START HERE


show_match = 0
if show_match:
    plt.plot(np.array([1.e11,1.e15]),np.array([1.e11,1.e15]),'k--',lw=0.5)
    plt.scatter(Group_M_Mean200_fp[fp_halo_inds],Group_M_Mean200_dm[dmo_halo_inds],color='#CC6677',s=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1.e12,1.e15])
    plt.ylim([1.e12,1.e15])
    plt.show()


print("N halos fp dm = ",np.sum(Group_M_Mean200_fp>1.e11),np.sum(Group_M_Mean200_dm>1.e11))

#want_env = 1; match_dm = 0 
want_env = 1; match_dm = 1 # I think this is fixes things cause we are doing it self-consistently within DM
if want_env:

    # finding who belongs where in the cosmic web
    N_dim = 256
    gr_size = Lbox/N_dim
    R = 1.7
    R_fft = 1.4
    
    testing = 0
    if testing:
        # one option is to compute the density field from the positions of the group centers
        d_smooth = smooth_density(get_density(GroupPos_fp,weights=Group_M_Mean200_fp),R=2.)
    else:
        # another is to use the particle positions
        # wrong because at final redshift
        #filename = '/home/boryanah/lars/LSSIllustrisTNG/CosmicWeb/WEB_CIC_256_DM_TNG300-2.hdf5'
        #f = h5py.File(filename, 'r')
        #d_smooth = f['density_smooth'][:,:,:] 
        # want to recompute?
        pos_parts = np.load(root+'pos_parts_tng300-3'+snap_dir+'.npy')/1000.
        density = get_density(pos_parts,Lbox=Lbox,N_dim=N_dim)
        d_smooth = smooth_density(density,R=R)
        d_smooth_fft = get_smooth_density(density,R=R_fft,Lbox=Lbox,N_dim=N_dim)
        # take recorded value
        #d_smooth = np.load(root+'smoothed_density_tng300-3'+snap_dir+'.npy')
    
    # get the indices for each halo
    halo_x = GroupPos_fp[:,0]; halo_y = GroupPos_fp[:,1]; halo_z = GroupPos_fp[:,2]

    i_cw = (halo_x/gr_size).astype(int)%N_dim
    j_cw = (halo_y/gr_size).astype(int)%N_dim
    k_cw = (halo_z/gr_size).astype(int)%N_dim

    # Environment definition
    GroupEnv_fp = d_smooth[i_cw,j_cw,k_cw]
    np.save(root+'GroupEnv_fp'+snap_dir+'.npy',GroupEnv_fp)

    show_env = 0
    if show_env:
        chosen_slice = 20
        #chosen_inds = (halo_z/gr_size > chosen_slice) & (halo_z/gr_size <= chosen_slice+1.)
        plt.figure(1)
        plt.imshow(np.log10(d_smooth[:,:,chosen_slice]+1.),origin='lower left')
        plt.colorbar()
        
        plt.figure(2)
        plt.imshow(np.log10(d_smooth_fft[:,:,chosen_slice]+1.),origin='lower left')
        #plt.scatter(halo_y[chosen_inds]/gr_size,halo_x[chosen_inds]/gr_size,s=0.01,alpha=0.5)
        plt.colorbar()
        plt.show()
        quit()
        
    if testing:
        d_smooth = smooth_density(get_density(GroupPos_dm,weights=Group_M_Mean200_dm))
    else:
        d_smooth = d_smooth_fft
        
    # get the indices for each halo
    halo_x = GroupPos_dm[:,0]; halo_y = GroupPos_dm[:,1]; halo_z = GroupPos_dm[:,2]

    i_cw = (halo_x/gr_size).astype(int)%N_dim
    j_cw = (halo_y/gr_size).astype(int)%N_dim
    k_cw = (halo_z/gr_size).astype(int)%N_dim

    # Environment definition
    GroupEnv_dm = d_smooth[i_cw,j_cw,k_cw]
    np.save(root+'GroupEnv_dm'+snap_dir+'.npy',GroupEnv_dm)
    

# get parent indices of the centrals and their subhalo indices in the original array
unique_sub_grnr, firsts = np.unique(SubhaloGrNr_fp,return_index=True)

count_halo_col_fp, count_halo_cents_col_fp, count_halo_sats_col_fp = get_counts(SubhaloGrNr_fp,firsts,N_halos_fp,sub_id_col)
count_halo_sfg_fp, count_halo_cents_sfg_fp, count_halo_sats_sfg_fp = get_counts(SubhaloGrNr_fp,firsts,N_halos_fp,sub_id_sfg)
count_halo_all_fp, count_halo_cents_all_fp, count_halo_sats_all_fp = get_counts(SubhaloGrNr_fp,firsts,N_halos_fp,sub_id_all)


def get_hod_counts(hist_elg,hist_cents_elg,hist_sats_elg,N_halos,group_mass):

    count_halo_elg_hod = np.zeros(N_halos,dtype=int)
    count_halo_cents_elg_hod = np.zeros(N_halos,dtype=int)
    count_halo_sats_elg_hod = np.zeros(N_halos,dtype=int)
    for i in range(N_bins-1):
        M_lo = 10.**bin_edges[i]
        M_hi = 10.**bin_edges[i+1]

        mass_sel = (M_lo < group_mass) & (M_hi > group_mass)
        N = int(np.sum(mass_sel))
        if N == 0: continue

        # rough number gals
        N_g = N*hist_elg[i]

        prob_cen = hist_cents_elg[i]

        cens = np.random.binomial(1,prob_cen,N)
        N_c = int(np.sum(cens > 0.))
        sats = np.random.poisson(hist_sats_elg[i],N)
       
        N_s = np.sum(sats)
        N_t = N_c+N_s
        print("gals assigned and expected (out of N halos) = ",N_t,N_g,N)
        print("_____________________________________")
        # I think we can have a satellite without a central
        tot = sats+cens
        count_halo_elg_hod[mass_sel] = tot

    print("HOD gals = ",np.sum(count_halo_elg_hod))

    return count_halo_elg_hod

def get_hod_env_counts(hist_all, hist_cents_all, hist_sats_all, group_env, N_halos, group_mass):
    
    count_halo_elg_hod = np.zeros(N_halos,dtype=int)
    count_halo_cents_elg_hod = np.zeros(N_halos,dtype=int)
    count_halo_sats_elg_hod = np.zeros(N_halos,dtype=int)
    for i in range(N_bins-1):
        M_lo = 10.**bin_edges[i]
        M_hi = 10.**bin_edges[i+1]

        mass_sel = (M_lo <= group_mass) & (M_hi > group_mass)
        if np.sum(mass_sel) == 0: continue
        
        for j in range(N_div):
            lo_bound = j*1./N_div
            hi_bound = (j+1.)/N_div
            
            if lo_bound < 0.: lo_bound = 0.
            if hi_bound > 1.: hi_bound = 1.
            quant_lo = np.quantile(group_env[mass_sel], lo_bound)
            quant_hi = np.quantile(group_env[mass_sel], hi_bound)
            env_sel = (group_env > quant_lo) & (group_env <= quant_hi)

            hist_elg, hist_cents_elg, hist_sats_elg = hist_all[:,j], hist_cents_all[:,j], hist_sats_all[:,j]
            halo_sel = mass_sel & env_sel
                
            N = int(np.sum(halo_sel))
            if N == 0: continue

            # rough number gals
            N_g = N*hist_elg[i]

            prob_cen = hist_cents_elg[i]

            if np.isnan(prob_cen): prob_cen = 0.;
            cens = np.random.binomial(1,prob_cen,N)
            N_c = int(np.sum(cens > 0.))
            prob_sat = hist_sats_elg[i]
            if np.isnan(prob_sat): prob_sat = 0.;
            sats = np.random.poisson(prob_sat,N)
        
            N_s = np.sum(sats)
            N_t = N_c+N_s
            #print("gals assigned and expected (out of N halos) = ",N_t,N_g,N)
            #print("_____________________________________")
        
            tot = sats+cens
            count_halo_elg_hod[halo_sel] = tot
            count_halo_cents_elg_hod[halo_sel] = cens

    print("HOD gals = ",np.sum(count_halo_elg_hod))
    
    return count_halo_elg_hod, count_halo_cents_elg_hod


# just gotta change groupenv and groupM200 to not having fp
def get_hist_env(count_halo,count_halo_cents,count_halo_sats,group_mass,group_env,inds_halo,show_plot=0):
    hist_all = np.zeros((len(bin_edges)-1,N_div))
    hist_cents_all = np.zeros((len(bin_edges)-1,N_div))
    hist_sats_all = np.zeros((len(bin_edges)-1,N_div))
    
    edges = 10.**bin_edges
    lss = ['--','--','--']
    cs = ['dodgerblue','black','#CC6677']
    for j in range(len(bin_edges)-1):
        select_mass = (group_mass > 10**bin_edges[j]) & (group_mass <= 10**bin_edges[j+1])
        if np.sum(select_mass) == 0: continue
        for i in range(N_div):
            lo_bound = i*1./N_div
            hi_bound = (i+1.)/N_div
            
            if lo_bound <= 0.: lo_bound = 0.+1.e-12
            if hi_bound >= 1.: hi_bound = 1.-1.e-12
            quant_lo = np.quantile(group_env[select_mass], lo_bound)
            quant_hi = np.quantile(group_env[select_mass], hi_bound)
            # can be sped up
            select_env = (group_env > quant_lo) & (group_env <= quant_hi)

            select_all = select_mass & select_env
            inds_cw = inds_halo[select_all]

            tot_objs = len(inds_cw)
            counts = count_halo[inds_cw]
            counts_cents = count_halo_cents[inds_cw]
            counts_sats = count_halo_sats[inds_cw]
            masses = group_mass[inds_cw]
            tot_counts = np.sum(counts)
            tot_counts_cents = np.sum(counts_cents)
            tot_counts_sats = np.sum(counts_sats)

            hist = tot_counts/tot_objs
            hist_cents = tot_counts_cents/tot_objs
            hist_sats = tot_counts_sats/tot_objs
            
            hist_all[j,i] = hist
            hist_cents_all[j,i] = hist_cents
            hist_sats_all[j,i] = hist_sats
            

    if show_plot:
        hist_low = hist_all[:,0]; hist_mid = hist_all[:,N_div//2]; hist_high = hist_all[:,-1]
        i = 0; lo_bound = bounds[i];hi_bound = bounds[i+1]
        plt.plot(10.**bin_cents,hist_low,color=cs[0],ls=lss[0],label=r'$%.1f-%.1f$'%(lo_bound*100.,hi_bound*100))
        i = N_div//2; lo_bound = bounds[i];hi_bound = bounds[i+1]
        plt.plot(10.**bin_cents,hist_mid,color=cs[1],ls=lss[1],label=r'$%.1f-%.1f$'%(lo_bound*100.,hi_bound*100))
        i = N_div-1; lo_bound = bounds[i];hi_bound = bounds[i+1]
        plt.plot(10.**bin_cents,hist_high,color=cs[2],ls=lss[2],label=r'$%.1f-%.1f$'%(lo_bound*100.,hi_bound*100))
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()
        
    M_lo = 10.**bin_edges[0]
    M_hi = 10.**bin_edges[-1]
    mass_sel = (M_lo <= group_mass) & (M_hi > group_mass)
    
    return edges, hist_all, hist_cents_all, hist_sats_all

def prune_unmatched(count_halo_fp, count_halo_cents_fp, count_halo_sats_fp):
    count_halo_dm = np.zeros(N_halos_dm,dtype=int)
    count_halo_cents_dm = np.zeros(N_halos_dm,dtype=int)
    count_halo_sats_dm = np.zeros(N_halos_dm,dtype=int)
    count_halo_dm[dmo_halo_inds] = count_halo_fp[fp_halo_inds]
    count_halo_cents_dm[dmo_halo_inds] = count_halo_cents_fp[fp_halo_inds]
    count_halo_sats_dm[dmo_halo_inds] = count_halo_sats_fp[fp_halo_inds]
    print(np.sum(count_halo_dm),np.sum(count_halo_fp))
    print(np.sum(count_halo_cents_dm),np.sum(count_halo_cents_fp))
    print(np.sum(count_halo_sats_dm),np.sum(count_halo_sats_fp))
    count_halo_new_fp = np.zeros(N_halos_fp,dtype=int)
    count_halo_new_fp[fp_halo_inds] = count_halo_fp[fp_halo_inds]
    count_halo_fp = count_halo_new_fp
    print(np.sum(count_halo_dm),np.sum(count_halo_fp))
    return count_halo_dm, count_halo_cents_dm, count_halo_sats_dm, count_halo_fp

if want_env:
    if match_dm:
        count_halo_col_dm, count_halo_cents_col_dm, count_halo_sats_col_dm, count_halo_col_fp = prune_unmatched(count_halo_col_fp, count_halo_cents_col_fp, count_halo_sats_col_fp)
        count_halo_sfg_dm, count_halo_cents_sfg_dm, count_halo_sats_sfg_dm, count_halo_sfg_fp = prune_unmatched(count_halo_sfg_fp, count_halo_cents_sfg_fp, count_halo_sats_sfg_fp)
        count_halo_all_dm, count_halo_cents_all_dm, count_halo_sats_all_dm, count_halo_all_fp = prune_unmatched(count_halo_all_fp, count_halo_cents_all_fp, count_halo_sats_all_fp)


        edges, hist_col_all, hist_cents_col_all, hist_sats_col_all = get_hist_env(count_halo_col_dm,count_halo_cents_col_dm,count_halo_sats_col_dm,Group_M_Mean200_dm,GroupEnv_dm,inds_halo_dm)
        edges, hist_sfg_all, hist_cents_sfg_all, hist_sats_sfg_all = get_hist_env(count_halo_sfg_dm,count_halo_cents_sfg_dm,count_halo_sats_sfg_dm,Group_M_Mean200_dm,GroupEnv_dm,inds_halo_dm)
        edges, hist_all_all, hist_cents_all_all, hist_sats_all_all = get_hist_env(count_halo_all_dm,count_halo_cents_all_dm,count_halo_sats_all_dm,Group_M_Mean200_dm,GroupEnv_dm,inds_halo_dm)
        np.save("data/edges.npy",edges)
        np.save("data/hist_col_all.npy",hist_col_all)
        np.save("data/hist_cents_col_all.npy",hist_cents_col_all)
        np.save("data/hist_sats_col_all.npy",hist_sats_col_all)
        np.save("data/hist_sfg_all.npy",hist_sfg_all)
        np.save("data/hist_cents_sfg_all.npy",hist_cents_sfg_all)
        np.save("data/hist_sats_sfg_all.npy",hist_sats_sfg_all)
        np.save("data/hist_all_all.npy",hist_all_all)
        np.save("data/hist_cents_all_all.npy",hist_cents_all_all)
        np.save("data/hist_sats_all_all.npy",hist_sats_all_all)
        
    else:
        edges, hist_col_all, hist_cents_col_all, hist_sats_col_all = get_hist_env(count_halo_col_fp,count_halo_cents_col_fp,count_halo_sats_col_fp,Group_M_Mean200_fp,GroupEnv_fp,inds_halo_fp)
        edges, hist_sfg_all, hist_cents_sfg_all, hist_sats_sfg_all = get_hist_env(count_halo_sfg_fp,count_halo_cents_sfg_fp,count_halo_sats_sfg_fp,Group_M_Mean200_fp,GroupEnv_fp,inds_halo_fp)
        edges, hist_all_all, hist_cents_all_all, hist_sats_all_all = get_hist_env(count_halo_all_fp,count_halo_cents_all_fp,count_halo_sats_all_fp,Group_M_Mean200_fp,GroupEnv_fp,inds_halo_fp)

else:
    hist_norm, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges)
    edges, hist_col, hist_cents_col, hist_sats_col = get_hist(count_halo_col_fp,count_halo_cents_col_fp,count_halo_sats_col_fp,Group_M_Mean200_fp,hist_norm,bin_edges)
    edges, hist_sfg, hist_cents_sfg, hist_sats_sfg = get_hist(count_halo_sfg_fp,count_halo_cents_sfg_fp,count_halo_sats_sfg_fp,Group_M_Mean200_fp,hist_norm,bin_edges)
    edges, hist_all, hist_cents_all, hist_sats_all = get_hist(count_halo_all_fp,count_halo_cents_all_fp,count_halo_sats_all_fp,Group_M_Mean200_fp,hist_norm,bin_edges)


def get_pos_hod(count_halo_elg_hod,count_halo_cents_elg_hod,pos_type=v_type,want_scat=want_scatter,delta=delta_col):
    if pos_type == 'vmax' or pos_type == 'vpeak':
        if pos_type == 'vmax': SubhaloV_dm = SubhaloVmax_dm
        if pos_type == 'vpeak': SubhaloV_dm = SubhaloVpeak_dm
        inds_hod = inds_halo_dm[count_halo_elg_hod > 0]
        count_hod = count_halo_elg_hod[count_halo_elg_hod > 0]
        count_cents_hod = count_halo_cents_elg_hod[count_halo_elg_hod > 0]
        gals_num = np.sum(count_hod)
        xyz_hod = np.zeros((gals_num,3))
        w_hod = np.ones(gals_num)
        
        gal_sofar = 0
        for i in range(len(count_hod)):
            gal_num = count_hod[i]
            flag_c = count_cents_hod[i]
            num_sat = gal_num-flag_c
            first_sub = GroupFirstSub_dm[inds_hod[i]]
            n_sub = GroupNsubs_dm[inds_hod[i]]
            pos_c = GroupPos_dm[inds_hod[i]]

            all_grnr = np.arange(first_sub,first_sub+n_sub)
            vmaxes = SubhaloV_dm[all_grnr]
            poses = SubhaloPos_dm[all_grnr]
            sats = np.sum((poses-pos_c)**2,axis=1) >= 1.e-8

            vmaxes = vmaxes[sats]
            poses = poses[sats]
            inds_vm = np.argsort(vmaxes)
            

            if want_scat:
                vmaxes = 10.**np.random.normal(np.log10(vmaxes),delta) 
                inds_vm = np.argsort(vmaxes)
                                
            inds_vm = inds_vm[::-1]
            inds_vm = inds_vm[:gal_num-flag_c]
            
            pos_vm = (poses)[inds_vm]

            xyz_hod[gal_sofar+flag_c:gal_sofar+flag_c+num_sat] = pos_vm
            if flag_c:
                xyz_hod[gal_sofar] = pos_c
            
            gal_sofar += gal_num
    elif pos_type == 'center':
        xyz_hod = GroupPos_dm[count_halo_elg_hod > 0]
        w_hod = count_halo_elg_hod[count_halo_elg_hod > 0].astype(xyz_hod.dtype)

    elif pos_type == 'particles':
        nparts = np.load(halo_parts_dir+"nstart_halo_"+delta+"_hod.npy")
        nstart = np.zeros(len(nparts),dtype=int)
        nstart[1:] = np.cumsum(nparts)[:-1]
        coords = np.load(halo_parts_dir+"coords_halo_"+delta+"_hod.npy")/1000.
        #delta is col, sfg, all
        
        count_halo_sats_elg_hod = count_halo_elg_hod-count_halo_cents_elg_hod
        inds_hod = np.where(count_halo_elg_hod > 0)[0]

        for i in range(len(inds_hod)):
            nsats = count_halo_sats_elg_hod[inds_hod[i]]
            if nsats == 0: continue
            # because shuffled already
            xyz = (coords[nstart[i]:nstart[i]+nparts[i]])[:nsats]
            try:
                xyz_sats = np.vstack((xyz_sats,xyz))
            except:
                xyz_sats = xyz
        del coords
        xyz_cents = GroupPos_dm[count_halo_cents_elg_hod > 0]
        xyz_hod = np.vstack((xyz_sats,xyz_cents))
        w_hod = np.ones(xyz_hod.shape[0],dtype=xyz_hod.dtype)
        
    return xyz_hod, w_hod



true_pos_type = 'true_gal'#'vmax_dm'#'center'
if true_pos_type == 'center':
    xyz_col_true = GroupPos_fp[count_halo_col_fp > 0]
    w_col_true = count_halo_col_fp[count_halo_col_fp > 0].astype(xyz_col_true.dtype)
    xyz_sfg_true = GroupPos_fp[count_halo_sfg_fp > 0]
    w_sfg_true = count_halo_sfg_fp[count_halo_sfg_fp > 0].astype(xyz_sfg_true.dtype)
    xyz_all_true = GroupPos_fp[count_halo_all_fp > 0]
    w_all_true = count_halo_all_fp[count_halo_all_fp > 0].astype(xyz_all_true.dtype)
elif true_pos_type == 'vmax_dm':
    
    xyz_col_true, w_col_true = get_pos_hod(count_halo_col_dm,count_halo_cents_col_dm,pos_type=v_type)
    xyz_sfg_true, w_sfg_true = get_pos_hod(count_halo_sfg_dm,count_halo_cents_sfg_dm,pos_type=v_type)
    xyz_all_true, w_all_true = get_pos_hod(count_halo_all_dm,count_halo_cents_all_dm,pos_type=v_type)
elif true_pos_type == 'true_gal':
    xyz_col_true = SubhaloPos_fp[sub_id_col]
    w_col_true = np.ones(xyz_col_true.shape[0],dtype=xyz_col_true.dtype)
    xyz_sfg_true = SubhaloPos_fp[sub_id_sfg]
    w_sfg_true = np.ones(xyz_sfg_true.shape[0],dtype=xyz_sfg_true.dtype)
    xyz_all_true = SubhaloPos_fp[sub_id_all]
    w_all_true = np.ones(xyz_all_true.shape[0],dtype=xyz_all_true.dtype)


if want_env:
    count_halo_col_hod, count_halo_cents_col_hod  = get_hod_env_counts(hist_col_all, hist_cents_col_all, hist_sats_col_all, GroupEnv_dm,N_halos_dm, Group_M_Mean200_dm)
    count_halo_sfg_hod, count_halo_cents_sfg_hod = get_hod_env_counts(hist_sfg_all, hist_cents_sfg_all, hist_sats_sfg_all, GroupEnv_dm,N_halos_dm, Group_M_Mean200_dm)    
    count_halo_all_hod, count_halo_cents_all_hod = get_hod_env_counts(hist_all_all, hist_cents_all_all, hist_sats_all_all, GroupEnv_dm, N_halos_dm, Group_M_Mean200_dm)
else:
    count_halo_col_hod = get_hod_counts(hist_col,hist_cents_col,hist_sats_col,N_halos_dm,Group_M_Mean200_dm)
    count_halo_sfg_hod = get_hod_counts(hist_sfg,hist_cents_sfg,hist_sats_sfg,N_halos_dm,Group_M_Mean200_dm)    
    count_halo_all_hod = get_hod_counts(hist_all,hist_cents_all,hist_sats_all,N_halos_dm,Group_M_Mean200_dm)

'''
np.save("data_counts/count_halo_col_hod.npy",count_halo_col_hod)
np.save("data_counts/count_halo_col_fp.npy",count_halo_col_fp)
np.save("data_counts/count_halo_sfg_hod.npy",count_halo_sfg_hod)
np.save("data_counts/count_halo_sfg_fp.npy",count_halo_sfg_fp)
np.save("data_counts/count_halo_all_hod.npy",count_halo_all_hod)
np.save("data_counts/count_halo_all_fp.npy",count_halo_all_fp)
'''


# get weights and positions
'''
# method using subhalos ordered by property
xyz_col_hod, w_col_hod = get_pos_hod(count_halo_col_hod,count_halo_cents_col_hod,pos_type=v_type_col,delta=delta_col,want_scat=want_scatter_col)
xyz_sfg_hod, w_sfg_hod = get_pos_hod(count_halo_sfg_hod,count_halo_cents_sfg_hod,pos_type=v_type_sfg,delta=delta_sfg,want_scat=want_scatter_sfg)
xyz_all_hod, w_all_hod = get_pos_hod(count_halo_all_hod,count_halo_cents_all_hod,pos_type=v_type_all,delta=delta_all,want_scat=want_scatter_all)
'''
# new method with particles
xyz_col_hod, w_col_hod = get_pos_hod(count_halo_col_hod,count_halo_cents_col_hod,pos_type='particles',delta='col')
xyz_sfg_hod, w_sfg_hod = get_pos_hod(count_halo_sfg_hod,count_halo_cents_sfg_hod,pos_type='particles',delta='sfg')
xyz_all_hod, w_all_hod = get_pos_hod(count_halo_all_hod,count_halo_cents_all_hod,pos_type='particles',delta='all')

# TESTING
def shuffle_and_prune(xyz,w,lim):
    inds = np.arange(xyz.shape[0])
    np.random.shuffle(inds)
    inds = inds[:lim]
    xyz = xyz[inds]
    w = w[inds]
    return xyz, w

if xyz_col_hod.shape[0] > xyz_col_true.shape[0]:
    xyz_col_hod, w_col_hod = shuffle_and_prune(xyz_col_hod,w_col_hod,xyz_col_true.shape[0])
else:
    xyz_col_true, w_col_true = shuffle_and_prune(xyz_col_true,w_col_true,xyz_col_hod.shape[0])

if xyz_sfg_hod.shape[0] > xyz_sfg_true.shape[0]:
    xyz_sfg_hod, w_sfg_hod = shuffle_and_prune(xyz_sfg_hod,w_sfg_hod,xyz_sfg_true.shape[0])
else:
    xyz_sfg_true, w_sfg_true = shuffle_and_prune(xyz_sfg_true,w_sfg_true,xyz_sfg_hod.shape[0])

if xyz_all_hod.shape[0] > xyz_all_true.shape[0]:
    xyz_all_hod, w_all_hod = shuffle_and_prune(xyz_all_hod,w_all_hod,xyz_all_true.shape[0])
else:
    xyz_all_true, w_all_true = shuffle_and_prune(xyz_all_true,w_all_true,xyz_all_hod.shape[0])


# check that the hod's make sense
want_hist = 0
if want_hist:
    # all populations
    hist_norm, edges = np.histogram(Group_M_Mean200_fp,bins=10**bin_edges)

    # color-selected galaxies
    edges,hist_col,hist_cents_col,hist_sats_col = get_hist(count_halo_col_fp,count_halo_cents_col_fp,count_halo_sats_col_fp,Group_M_Mean200_fp,hist_norm,bin_edges)
    plt.plot(10.**bin_cents,hist_col,color='dodgerblue',label='all')

    # all populations
    hist_norm, edges = np.histogram(Group_M_Mean200_dm,bins=10**bin_edges)

    # shuffled color-selected galaxies
    edges,hist_col,hist_cents_col,hist_sats_col = get_hist(count_halo_col_hod,count_halo_cents_col_hod,count_halo_sats_col_hod,Group_M_Mean200_dm,hist_norm,bin_edges)
    plt.plot(10.**bin_cents,hist_col,color='dodgerblue',ls='--',label='HOD')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

# Correlation function


# get the ratios
Rat_col_hodtrue_mean, Rat_col_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_col_true,w_col_true,xyz_col_hod,w_col_hod,Lbox)
Rat_all_hodtrue_mean, Rat_all_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_all_true,w_all_true,xyz_all_hod,w_all_hod,Lbox)
Rat_sfg_hodtrue_mean, Rat_sfg_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers = get_jack_corr(xyz_sfg_true,w_sfg_true,xyz_sfg_hod,w_sfg_hod,Lbox)

np.save("data/Rat_col_hodtrue_mean.npy",Rat_col_hodtrue_mean)
np.save("data/Rat_col_hodtrue_err.npy",Rat_col_hodtrue_err)
np.save("data/Rat_sfg_hodtrue_mean.npy",Rat_sfg_hodtrue_mean)
np.save("data/Rat_sfg_hodtrue_err.npy",Rat_sfg_hodtrue_err)
np.save("data/Rat_all_hodtrue_mean.npy",Rat_all_hodtrue_mean)
np.save("data/Rat_all_hodtrue_err.npy",Rat_all_hodtrue_err)
np.save("data/bin_centers.npy",bin_centers)


# END HERE


N_bin = 12
bins = np.logspace(np.log10(0.12),1.,N_bin)
bin_centers = (bins[:-1] + bins[1:])/2.
#bin_centers = np.load("data/bin_centers.npy")
Rat_col_hodtrue_mean = np.load("data/Rat_col_hodtrue_mean.npy")
Rat_col_hodtrue_err = np.load("data/Rat_col_hodtrue_err.npy")
Rat_sfg_hodtrue_mean = np.load("data/Rat_sfg_hodtrue_mean.npy")
Rat_sfg_hodtrue_err = np.load("data/Rat_sfg_hodtrue_err.npy")
Rat_all_hodtrue_mean = np.load("data/Rat_all_hodtrue_mean.npy")
Rat_all_hodtrue_err = np.load("data/Rat_all_hodtrue_err.npy")



plt.figure(1,figsize=(9,7))

# for seeing where we are
line = np.linspace(0,20,3)
plt.plot(line,np.ones(len(line)),'k--')

# COL
plt.errorbar(bin_centers,Rat_col_hodtrue_mean,yerr=Rat_col_hodtrue_err,color='dodgerblue',ls='-',label='color-selected',alpha=1.,fmt='o',capsize=4)
r_min = .8; x_line = np.linspace(r_min,10,3)
mean_val = np.mean(Rat_col_hodtrue_mean[np.isfinite(Rat_col_hodtrue_mean) & (bin_centers > r_min)])
print('mean_val for col = ',mean_val)
plt.plot(x_line,np.ones(len(x_line))*mean_val,color='dodgerblue',ls='--',lw=1.5)

# SFG
plt.errorbar(bin_centers*1.05,Rat_sfg_hodtrue_mean,yerr=Rat_sfg_hodtrue_err,color='#CC6677',ls='-',label='SFR-selected',alpha=1.,fmt='o',capsize=4)
r_min = .8; x_line = np.linspace(r_min,10,3)
mean_val = np.mean(Rat_sfg_hodtrue_mean[np.isfinite(Rat_sfg_hodtrue_mean) & (bin_centers > r_min)])
print('mean_val for sfg = ',mean_val)
plt.plot(x_line,np.ones(len(x_line))*mean_val,color='#CC6677',ls='--',lw=1.5)

# ALL
plt.plot(bin_centers,Rat_all_hodtrue_mean,color='black',ls='-',label='mass-selected')
plt.fill_between(bin_centers,Rat_all_hodtrue_mean-Rat_all_hodtrue_err,Rat_all_hodtrue_mean+Rat_all_hodtrue_err,color='black',alpha=0.1)
r_min = 2.; x_line = np.linspace(r_min,10,3)
mean_val = np.mean(Rat_all_hodtrue_mean[np.isfinite(Rat_all_hodtrue_mean) & (bin_centers > r_min)])
print('mean_val for all = ',mean_val)
plt.plot(x_line,np.ones(len(x_line))*mean_val,color='black',ls='--',lw=1.5)

plt.xlabel(r'$r [{\rm Mpc}/h]$')
plt.ylabel(r'$\xi(r)_{\rm HOD}/\xi(r)_{\rm TNG300}$')
plt.xscale('log')
plt.legend()
plt.ylim([0.6,1.4])
plt.xlim([0.095,12])
#plt.savefig("figs/mock_ELG.png")
plt.savefig("../paper/mock_ELG.pdf")
plt.show()
