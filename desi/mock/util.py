import numpy as np
from scipy.ndimage import gaussian_filter
import Corrfunc
import numpy.linalg as la

def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence 
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, ) 
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence 
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)  
        `c` can be a 2-D array in which the rows are RGB or RGBA, however. 
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
        
    return collection

def Wg(k2, R):
    return np.exp(-k2*R*R/2.)


def get_smooth_density(D,R=4.,N_dim=256,Lbox=205.,return_lambda=False):
    karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim))
    dfour = np.fft.fftn(D)
    dksmo = np.zeros(shape=(N_dim, N_dim, N_dim),dtype=complex)
    ksq = np.zeros(shape=(N_dim, N_dim, N_dim),dtype=complex)
    ksq[:,:,:] = karr[None,None,:]**2+karr[None,:,None]**2+karr[:,None,None]**2
    dksmo[:,:,:] = Wg(ksq,R)*dfour
    if return_lambda:
        lambda1, lambda2, lambda3 = get_eig(dksmo,karr,N_dim)
        drsmo = np.real(np.fft.ifftn(dksmo))
        return drsmo, lambda1, lambda2, lambda3
    drsmo = np.real(np.fft.ifftn(dksmo))
    return drsmo

def get_eig(dfour,karr,N_dim):
    tfour = np.zeros(shape=(N_dim, N_dim, N_dim, 3, 3),dtype=complex)
    
    # computing tidal tensor and phi in fourier space
    # and smoothing using the window functions
    for a in range(N_dim):
        for b in range(N_dim):
            for c in range(N_dim):
                if (a, b, c) == (0, 0, 0):
                    #phifour[a, b, c] = 0.
                    1
                else:
                    ksq = karr[a]**2 + karr[b]**2 + karr[c]**2
                    #phifour[a, b, c] = -dfour[a, b, c]/ksq
                    # smoothed density Gauss fourier
                    # dksmo[a, b, c] = Wg(ksq)*dfour[a, b, c]
                    # smoothed density TH fourier
                    #dkth[a, b, c] = Wth(ksq)*dfour[a, b, c]
                    # all 9 components
                    tfour[a, b, c, 0, 0] = karr[a]*karr[a]*dfour[a, b, c]/ksq
                    tfour[a, b, c, 1, 1] = karr[b]*karr[b]*dfour[a, b, c]/ksq
                    tfour[a, b, c, 2, 2] = karr[c]*karr[c]*dfour[a, b, c]/ksq
                    tfour[a, b, c, 1, 0] = karr[a]*karr[b]*dfour[a, b, c]/ksq
                    tfour[a, b, c, 0, 1] = tfour[a, b, c, 1, 0]
                    tfour[a, b, c, 2, 0] = karr[a]*karr[c]*dfour[a, b, c]/ksq
                    tfour[a, b, c, 0, 2] = tfour[a, b, c, 2, 0]
                    tfour[a, b, c, 1, 2] = karr[b]*karr[c]*dfour[a, b, c]/ksq
                    tfour[a, b, c, 2, 1] = tfour[a, b, c, 1, 2]
                    # smoothed tidal Gauss fourier
                    # tksmo[a, b, c, :, :] = Wg(ksq)*tfour[a, b, c, :, :]
                    # smoothed tidal TH fourier
                    #tkth[a, b, c, :, :] = Wth(ksq)*tfour[a, b, c, :, :]

    tidt = np.real(np.fft.ifftn(tfour, axes = (0, 1, 2)))
    evals = np.zeros(shape=(N_dim, N_dim, N_dim, 3))

    for x in range(N_dim):
        for y in range(N_dim):
            for z in range(N_dim):
                # comute and sort evalues in ascending order, for descending add after argsort()[::-1]
                evals[x, y, z, :], evects = la.eig(tidt[x, y, z, :, :])
                idx = evals[x, y, z, :].argsort()
                evals[x, y, z] = evals[x, y, z, idx]
                #evects = evects[:, idx]

    lambda1 = evals[:,:,:,0]
    lambda2 = evals[:,:,:,1]
    lambda3 = evals[:,:,:,2]

    return lambda1, lambda2, lambda3
    
def get_density(pos,weights=None,N_dim=256,Lbox=205.):
    # x, y, and z position
    g_x = pos[:,0]
    g_y = pos[:,1]
    g_z = pos[:,2]

    if weights is None:
        # total number of objects
        N_g = len(g_x)
        # get a 3d histogram with number of objects in each cell
        D, edges = np.histogramdd(np.transpose([g_x,g_y,g_z]),bins=N_dim,range=[[0,Lbox],[0,Lbox],[0,Lbox]])
        # average number of particles per cell
        D_avg = N_g*1./N_dim**3
        D /= D_avg
        D -= 1.
    else:
        # get a 3d histogram with total mass of objects in each cell
        D, edges = np.histogramdd(np.transpose([g_x,g_y,g_z]),bins=N_dim,range=[[0,Lbox],[0,Lbox],[0,Lbox]],weights=weights)
        # average mass of particles per cell
        D_avg = np.sum(weights)/N_dim**3
        D /= D_avg
        D -= 1.
        
    return D

def smooth_density(D,R=1.1):
    # smoothing
    D_smo = gaussian_filter(D,R)
    return D_smo

def get_jack_corr(xyz_true,w_true,xyz_hod,w_hod,Lbox):
    # where are the galaxies located in real space
    
    # bins for the correlation function
    N_bin = 16
    bins = np.logspace(-1.,1.,N_bin)
    bin_centers = (bins[:-1] + bins[1:])/2.

    # dimensions for jackknifing
    N_dim = 3

    # empty arrays to record data
    Rat_hodtrue = np.zeros((N_bin-1,N_dim**3))
    Corr_hod = np.zeros((N_bin-1,N_dim**3))
    Corr_true = np.zeros((N_bin-1,N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                xyz_hod_jack = xyz_hod.copy()
                xyz_true_jack = xyz_true.copy()
                w_hod_jack = w_hod.copy()
                w_true_jack = w_true.copy()
            
                xyz = np.array([i_x,i_y,i_z],dtype=int)
                size = Lbox/N_dim

                bool_arr = np.prod((xyz == (xyz_hod/size).astype(int)),axis=1).astype(bool)
                xyz_hod_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_hod_jack = xyz_hod_jack[np.sum(xyz_hod_jack,axis=1)!=0.]
                w_hod_jack[bool_arr] = 0
                w_hod_jack = w_hod_jack[w_hod_jack!=0.]

                bool_arr = np.prod((xyz == (xyz_true/size).astype(int)),axis=1).astype(bool)
                xyz_true_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_true_jack = xyz_true_jack[np.sum(xyz_true_jack,axis=1)!=0.]
                w_true_jack[bool_arr] = 0
                w_true_jack = w_true_jack[w_true_jack!=0.]

                res_hod = Corrfunc.theory.xi(X=xyz_hod_jack[:,0],Y=xyz_hod_jack[:,1],Z=xyz_hod_jack[:,2],weights=w_hod_jack,weight_type='pair_product',boxsize=Lbox,nthreads=16,binfile=bins)
                res_true = Corrfunc.theory.xi(X=xyz_true_jack[:,0],Y=xyz_true_jack[:,1],Z=xyz_true_jack[:,2],weights=w_true_jack,weight_type='pair_product',boxsize=Lbox,nthreads=16,binfile=bins)

                rat_hodtrue = res_hod['xi']/res_true['xi']
                Rat_hodtrue[:,i_x+N_dim*i_y+N_dim**2*i_z] = rat_hodtrue
                Corr_hod[:,i_x+N_dim*i_y+N_dim**2*i_z] = res_hod['xi']
                Corr_true[:,i_x+N_dim*i_y+N_dim**2*i_z] = res_true['xi']

    # compute mean and error
    Rat_hodtrue_mean = np.mean(Rat_hodtrue,axis=1)
    Rat_hodtrue_err = np.sqrt(N_dim**3-1)*np.std(Rat_hodtrue,axis=1)
    Corr_mean_hod = np.mean(Corr_hod,axis=1)
    Corr_err_hod = np.sqrt(N_dim**3-1)*np.std(Corr_hod,axis=1)
    Corr_mean_true = np.mean(Corr_true,axis=1)
    Corr_err_true = np.sqrt(N_dim**3-1)*np.std(Corr_true,axis=1)

    return Rat_hodtrue_mean, Rat_hodtrue_err, Corr_mean_hod, Corr_err_hod,  Corr_mean_true, Corr_err_true, bin_centers

def get_counts(subgrnr,firsts,N_halo_fp,sub_id_elg):

    # index of the parent halo and mass of parent halo for each elg
    parent_elg_fp = subgrnr[sub_id_elg]
    parent_cents_elg_fp = subgrnr[np.intersect1d(sub_id_elg,firsts)]
    parent_sats_elg_fp = subgrnr[np.setdiff1d(sub_id_elg,firsts)]
    print(len(parent_elg_fp),len(parent_cents_elg_fp),len(parent_sats_elg_fp))

    # for each halo, how many elg point to it (excluding case of 0) and index of halo
    parent_elg_un_fp, counts = np.unique(parent_elg_fp,return_counts=True)
    parent_cents_elg_un_fp, counts_cents = np.unique(parent_cents_elg_fp,return_counts=True)
    parent_sats_elg_un_fp, counts_sats = np.unique(parent_sats_elg_fp,return_counts=True)

    # counts of elgs for all halos
    count_halo_elg_fp = np.zeros(N_halo_fp,dtype=int)
    count_halo_cents_elg_fp = np.zeros(N_halo_fp,dtype=int)
    count_halo_sats_elg_fp = np.zeros(N_halo_fp,dtype=int)
    count_halo_elg_fp[parent_elg_un_fp] += counts
    count_halo_cents_elg_fp[parent_cents_elg_un_fp] += counts_cents
    count_halo_sats_elg_fp[parent_sats_elg_un_fp] += counts_sats
    print("sum halo fp = ",np.sum(count_halo_elg_fp))

    return count_halo_elg_fp, count_halo_cents_elg_fp, count_halo_sats_elg_fp

def get_counts_and_nstart(subgrnr,firsts,N_halo_fp,sub_id_elg):

    # make sure subhalo array is sorted
    sub_id_elg = np.sort(sub_id_elg)
    
    # index of the parent halo and mass of parent halo for each elg
    parent_elg_fp = subgrnr[sub_id_elg]
    parent_cents_elg_fp = subgrnr[np.intersect1d(sub_id_elg,firsts)]
    parent_sats_elg_fp = subgrnr[np.setdiff1d(sub_id_elg,firsts)]
    print(len(parent_elg_fp),len(parent_cents_elg_fp),len(parent_sats_elg_fp))

    # for each halo, how many elg point to it (excluding case of 0) and index of halo
    parent_elg_un_fp, inds, counts = np.unique(parent_elg_fp,return_counts=True,return_index=True)
    parent_cents_elg_un_fp, counts_cents = np.unique(parent_cents_elg_fp,return_counts=True)
    parent_sats_elg_un_fp, counts_sats = np.unique(parent_sats_elg_fp,return_counts=True)

    # counts of elgs for all halos
    count_halo_elg_fp = np.zeros(N_halo_fp,dtype=int)
    count_halo_cents_elg_fp = np.zeros(N_halo_fp,dtype=int)
    count_halo_sats_elg_fp = np.zeros(N_halo_fp,dtype=int)
    count_halo_elg_fp[parent_elg_un_fp] += counts
    count_halo_cents_elg_fp[parent_cents_elg_un_fp] += counts_cents
    count_halo_sats_elg_fp[parent_sats_elg_un_fp] += counts_sats
    print("sum halo fp = ",np.sum(count_halo_elg_fp))

    # where do the galaxies start in their corresponding SubhaloPos[sub_id_eg] arrays
    nstart_halo_elg_fp = np.zeros(N_halo_fp,dtype=int)-1
    nstart_halo_elg_fp[parent_elg_un_fp] = inds
    
    return count_halo_elg_fp, count_halo_cents_elg_fp, count_halo_sats_elg_fp, nstart_halo_elg_fp

def get_hist(count_halo_elg_fp,count_halo_cents_elg_fp,count_halo_sats_elg_fp,group_mass,hist_norm,bin_edges):
    # create histograms for the ELGs
    hist, edges = np.histogram(group_mass,bins=10**bin_edges,weights=count_halo_elg_fp)
    hist_cents, edges = np.histogram(group_mass,bins=10**bin_edges,weights=count_halo_cents_elg_fp)
    hist_sats, edges = np.histogram(group_mass,bins=10**bin_edges,weights=count_halo_sats_elg_fp)
    hist_elg = hist/hist_norm
    hist_cents_elg = hist_cents/hist_norm
    hist_sats_elg = hist_sats/hist_norm
    return edges,hist_elg,hist_cents_elg,hist_sats_elg
