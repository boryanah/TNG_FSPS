import matplotlib.pyplot as plt
import numpy as np
import sys
import Corrfunc
from Corrfunc.theory.DD import DD
#from correlation_function import get_corrfunc

def get_RR_norm(bins,Lbox,N1=1,N2=1):
    vol_all = 4./3*np.pi*bins**3
    vol_bins = vol_all[1:]-vol_all[:-1]
    n2 = N2/Lbox**3
    n2_bin = vol_bins*n2
    pairs = N1*n2_bin
    pairs /= (N1*N2)
    return pairs

def get_pos(pos_g,xyz,size):
    pos_g_jack = pos_g.copy()
    bool_arr = np.prod((xyz == (pos_g/size).astype(int)),axis=1).astype(bool)
    pos_g_jack[bool_arr] = np.array([0.,0.,0.])
    pos_g_jack = pos_g_jack[np.sum(pos_g_jack,axis=1)!=0.]
    return pos_g_jack

# TESTING
#def get_cross(pos1,pos1_r,pos2,pos2_r,n1,n2,n_thread=16,periodic=True):
# og
def get_cross(pos1,pos1_r,pos2,pos2_r,Lbox,bins,n_thread=16,periodic=True):
    X_jack_g = pos1[:,0]
    Y_jack_g = pos1[:,1]
    Z_jack_g = pos1[:,2]

    X_jack_m = pos2[:,0]
    Y_jack_m = pos2[:,1]
    Z_jack_m = pos2[:,2]

    # TESTING
    #N_g = n1
    #N_m = n2
    # og
    N_g = len(X_jack_g)
    N_m = len(X_jack_m)


    X_jack_r_m = pos2_r[:,0]
    Y_jack_r_m = pos2_r[:,1]
    Z_jack_r_m = pos2_r[:,2]
    
    X_jack_r_g = pos1_r[:,0]
    Y_jack_r_g = pos1_r[:,1]
    Z_jack_r_g = pos1_r[:,2]

    # POSSIBLE BUG IN LENSING
    # og
    N_r_g = len(X_jack_r_g)
    N_r_m = len(X_jack_r_m)
    # TESTING
    #N_r_g = N_g*n_random
    #N_r_m = N_m*n_random
    
    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_jack_g, Y1=Y_jack_g, Z1=Z_jack_g,
                 X2=X_jack_m, Y2=Y_jack_m, Z2=Z_jack_m,
                 boxsize=Lbox,periodic=periodic)

    DD_gm = results['npairs'].astype(float)
    DD_gm /= (N_g*1.*N_m)

    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_jack_g, Y1=Y_jack_g, Z1=Z_jack_g,
                 X2=X_jack_r_m, Y2=Y_jack_r_m, Z2=Z_jack_r_m,
                 boxsize=Lbox,periodic=periodic)


    DR_gm = results['npairs'].astype(float)
    DR_gm /= (N_g*1.*N_r_m)

    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_jack_r_g, Y1=Y_jack_r_g, Z1=Z_jack_r_g,
                 X2=X_jack_m, Y2=Y_jack_m, Z2=Z_jack_m,
                 boxsize=Lbox,periodic=periodic)

    RD_gm = results['npairs'].astype(float)
    RD_gm /= (N_r_g*1.*N_m)

    autocorr = 0
    results = DD(autocorr,nthreads=n_thread,binfile=bins,
                 X1=X_jack_r_g, Y1=Y_jack_r_g, Z1=Z_jack_r_g,
                 X2=X_jack_r_m, Y2=Y_jack_r_m, Z2=Z_jack_r_m,
                 boxsize=Lbox,periodic=periodic)


    RR_gm = results['npairs'].astype(float)
    RR_gm /= (N_r_g*1.*N_r_m)

    Corr_gm = (DD_gm-DR_gm-RD_gm+RR_gm)/RR_gm

    return Corr_gm

def get_random(pos,Lbox,dtype=np.float64,n_random=35):
    N = pos.shape[0]
    N_r = N*n_random
    pos_r = np.random.uniform(0.,Lbox,(N_r,3)).astype(dtype)
    return pos_r
    
def get_jack(mode,pos1,pos2,pos3,pos4,Lbox,bins):
    if mode == 'cross' or mode == 'bias_corr_coeff':
        pos1_r = get_random(pos1,Lbox,pos1.dtype)
        pos2_r = get_random(pos2,Lbox,pos2.dtype)
        pos3_r = get_random(pos3,Lbox,pos3.dtype)
        pos4_r = get_random(pos4,Lbox,pos4.dtype)


    
    N_dim = 3
    size = Lbox/N_dim
    corr12 = np.zeros((len(bins)-1,N_dim**3))
    corr34 = np.zeros((len(bins)-1,N_dim**3))
    bias = np.zeros((len(bins)-1,N_dim**3))
    corr_coeff = np.zeros((len(bins)-1,N_dim**3))
    rat = np.zeros((len(bins)-1,N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                xyz = np.array([i_x,i_y,i_z],dtype=int)
                print(xyz)                

                xyz1_jack = get_pos(pos1,xyz,size)
                xyz3_jack = get_pos(pos3,xyz,size)
                
                if mode == 'cross' or mode == 'bias_corr_coeff':
                    xyz2_jack = get_pos(pos2,xyz,size)
                    xyz_r1_jack = get_pos(pos1_r,xyz,size)
                    xyz_r2_jack = get_pos(pos2_r,xyz,size)
                    xyz4_jack = get_pos(pos4,xyz,size)
                    xyz_r3_jack = get_pos(pos3_r,xyz,size)
                    xyz_r4_jack = get_pos(pos4_r,xyz,size)

                print(xyz1_jack.shape,xyz3_jack.shape)
                if mode == 'cross' or mode == 'bias_corr_coeff':

                    Corr12 = get_cross(xyz1_jack,xyz_r1_jack,xyz2_jack,xyz_r2_jack,Lbox,bins)
                elif mode == 'auto':
                    Corr12 = Corrfunc.theory.xi(X=xyz1_jack[:,0],Y=xyz1_jack[:,1],Z=xyz1_jack[:,2],boxsize=Lbox,nthreads=16,binfile=bins)['xi']
                    # for sonya:
                    #Corr12 = get_corrfunc(xyz1_jack,bins,Lbox)
                corr12[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr12
                if mode == 'cross':
                    # og
                    Corr34 = get_cross(xyz3_jack,xyz_r3_jack,xyz4_jack,xyz_r4_jack,Lbox,bins)
                    # TESTING
                    #Corr34 = get_cross(xyz3_jack,xyz_r3_jack,xyz4_jack,xyz_r4_jack,n1=N3,n2=N4,Lbox,bins)
                elif mode == 'auto':
                    Corr34 = Corrfunc.theory.xi(X=xyz3_jack[:,0],Y=xyz3_jack[:,1],Z=xyz3_jack[:,2],boxsize=Lbox,nthreads=16,binfile=bins)['xi']
                    # for sonya:
                    #Corr34 = get_corrfunc(xyz3_jack,bins,Lbox)
                elif mode == 'bias_corr_coeff':
                    # TESTING another thing
                    #Corr3 = Corrfunc.theory.xi(X=xyz3_jack[:,0],Y=xyz3_jack[:,1],Z=xyz3_jack[:,2],boxsize=Lbox,nthreads=16,binfile=bins)['xi']
                    #Corr4 = Corrfunc.theory.xi(X=xyz4_jack[:,0],Y=xyz4_jack[:,1],Z=xyz4_jack[:,2],boxsize=Lbox,nthreads=16,binfile=bins)['xi']
                    # og another thing
                    Corr3 = get_cross(xyz3_jack,xyz_r3_jack,xyz3_jack,xyz_r3_jack,Lbox,bins)
                    Corr4 = get_cross(xyz4_jack,xyz_r4_jack,xyz4_jack,xyz_r4_jack,Lbox,bins)                    
                    bias[:,i_x+N_dim*i_y+N_dim**2*i_z] = np.sqrt(Corr3/Corr4)
                    corr_coeff[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr12/np.sqrt(Corr3*Corr4)
                    Corr34 = Corr3
                    
                corr34[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr34

                rat[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr34/Corr12
                
    Corr12_mean = np.mean(corr12,axis=1)
    Corr12_error = np.sqrt(N_dim**3-1)*np.std(corr12,axis=1)
    Corr34_mean = np.mean(corr34,axis=1)
    Corr34_error = np.sqrt(N_dim**3-1)*np.std(corr34,axis=1)
    Rat_mean = np.mean(rat,axis=1)
    Rat_error = np.sqrt(N_dim**3-1)*np.std(rat,axis=1)

    Bias_mean = np.mean(bias,axis=1)
    Bias_error = np.sqrt(N_dim**3-1)*np.std(bias,axis=1)
    Corr_Coeff_mean = np.mean(corr_coeff,axis=1)
    Corr_Coeff_error = np.sqrt(N_dim**3-1)*np.std(corr_coeff,axis=1)
    if mode == 'bias_corr_coeff':
        return Bias_mean, Bias_error, Corr_Coeff_mean, Corr_Coeff_error
    return Corr12_mean, Corr12_error, Corr34_mean, Corr34_error, Rat_mean, Rat_error


def get_fast_jack(mode,pos1,pos2,pos3,pos4,Lbox,bins,n_thread=16,periodic=True):
    R1R2 = get_RR_norm(bins,Lbox)
        
    N_dim = 3
    size = Lbox/N_dim
    corr12 = np.zeros((len(bins)-1,N_dim**3))
    corr34 = np.zeros((len(bins)-1,N_dim**3))
    bias = np.zeros((len(bins)-1,N_dim**3))
    corr_coeff = np.zeros((len(bins)-1,N_dim**3))
    rat = np.zeros((len(bins)-1,N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                xyz = np.array([i_x,i_y,i_z],dtype=int)
                print(xyz)                

                xyz1_jack = get_pos(pos1,xyz,size)
                xyz3_jack = get_pos(pos3,xyz,size)

                N1 = xyz1_jack.shape[0]
                N3 = xyz3_jack.shape[0]
                
                print(xyz1_jack.shape,xyz3_jack.shape)
                
                if mode == 'cross' or mode == 'bias_corr_coeff':
                    xyz2_jack = get_pos(pos2,xyz,size)
                    xyz4_jack = get_pos(pos4,xyz,size)

                    N2 = xyz2_jack.shape[0]
                    N4 = xyz4_jack.shape[0]
                    
                    autocorr = 0
                    D1D2 = DD(autocorr,nthreads=n_thread,binfile=bins,
                              X1=xyz1_jack[:,0], Y1=xyz1_jack[:,1], Z1=xyz1_jack[:,2],
                              X2=xyz2_jack[:,0], Y2=xyz2_jack[:,1], Z2=xyz2_jack[:,2],
                              boxsize=Lbox,periodic=periodic)['npairs'].astype(float)
                    D1D2 /= (N1*N2)
                    Corr12 = D1D2/R1R2 - 1.

                elif mode == 'auto':
                    Corr12 = Corrfunc.theory.xi(X=xyz1_jack[:,0],Y=xyz1_jack[:,1],Z=xyz1_jack[:,2],boxsize=Lbox,nthreads=n_thread,binfile=bins)['xi']

                corr12[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr12

                if mode == 'cross':
                    Corr34 = get_cross(xyz3_jack,xyz_r3_jack,xyz4_jack,xyz_r4_jack,Lbox,bins)
                elif mode == 'auto':
                    Corr34 = Corrfunc.theory.xi(X=xyz3_jack[:,0],Y=xyz3_jack[:,1],Z=xyz3_jack[:,2],
                                                boxsize=Lbox,nthreads=n_thread,binfile=bins)['xi']
                elif mode == 'bias_corr_coeff':
                    Corr3 = Corrfunc.theory.xi(X=xyz3_jack[:,0],Y=xyz3_jack[:,1],Z=xyz3_jack[:,2],
                                               boxsize=Lbox,nthreads=n_thread,binfile=bins)['xi']
                    Corr4 = Corrfunc.theory.xi(X=xyz4_jack[:,0],Y=xyz4_jack[:,1],Z=xyz4_jack[:,2],
                                               boxsize=Lbox,nthreads=n_thread,binfile=bins)['xi']
                    
                    bias[:,i_x+N_dim*i_y+N_dim**2*i_z] = np.sqrt(Corr3/Corr4)
                    corr_coeff[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr12/np.sqrt(Corr3*Corr4)
                    Corr34 = Corr3
                    
                corr34[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr34

                rat[:,i_x+N_dim*i_y+N_dim**2*i_z] = Corr34/Corr12
                
    Corr12_mean = np.mean(corr12,axis=1)
    Corr12_error = np.sqrt(N_dim**3-1)*np.std(corr12,axis=1)
    Corr34_mean = np.mean(corr34,axis=1)
    Corr34_error = np.sqrt(N_dim**3-1)*np.std(corr34,axis=1)
    Rat_mean = np.mean(rat,axis=1)
    Rat_error = np.sqrt(N_dim**3-1)*np.std(rat,axis=1)

    Bias_mean = np.mean(bias,axis=1)
    Bias_error = np.sqrt(N_dim**3-1)*np.std(bias,axis=1)
    Corr_Coeff_mean = np.mean(corr_coeff,axis=1)
    Corr_Coeff_error = np.sqrt(N_dim**3-1)*np.std(corr_coeff,axis=1)
    if mode == 'bias_corr_coeff':
        return Bias_mean, Bias_error, Corr_Coeff_mean, Corr_Coeff_error
    return Corr12_mean, Corr12_error, Corr34_mean, Corr34_error, Rat_mean, Rat_error

