import numpy as np
import matplotlib.pyplot as plt
import plotparams
#plotparams.default()
plotparams.buba()

opts = np.array(['col','sfg','all'])[::-1]
lab_opts = np.array(['color-selected','SFR-selected','mass-selected'])[::-1]
col_opts = np.array(['dodgerblue','#CC6677','black'])[::-1]
types = ['bias','corr. coeff.']

bin_centers = np.load("data_parts/bin_cents.npy")
nprops = len(opts)
nrows = 2
ncols = 1#nprops
ntot = nrows*ncols
plt.subplots(nrows,ncols,figsize=(ncols*6.8,nrows*5))
plot_no = 0
for i in range(nprops):
    for i_type in range(2):
        opt = opts[i]
        lab_opt = lab_opts[i]#"-".join(opt.split('_'))+" "+types[i_type]
        col_opt = col_opts[i]
        
        print(opt)

            
        if i_type == 0:
            bias = np.load("data_parts/bias_mean_"+opt+".npy")
            bias_error = np.load("data_parts/bias_error_"+opt+".npy")
        else:    
            bias = np.load("data_parts/corr_coeff_mean_"+opt+".npy")
            bias_error = np.load("data_parts/corr_coeff_error_"+opt+".npy")
        
        plot_no = i_type*ncols+1

        plt.subplot(nrows,ncols,plot_no)
        plt.plot(bin_centers,np.ones(len(bias)),'k--',linewidth=2.)
        

        if i == 0:
            # blue ones
            shuff_color2 = 'black'#'#089FFF'
            shuff_color = 'black'#'#1B2ACC'
            lab_shuff = lab_opts[0]
            if i_type == 0:
                bias_shuff = bias.copy()
                bias_error_shuff = bias_error.copy()
                plt.plot(bin_centers,bias_shuff,linewidth=2.,color=shuff_color,label=lab_shuff)
                plt.fill_between(bin_centers,bias_shuff+bias_error_shuff,bias_shuff-bias_error_shuff,alpha=0.1, edgecolor=shuff_color, facecolor=shuff_color2)
            else:
                corr_coeff_shuff = bias.copy()
                corr_coeff_error_shuff = bias_error.copy()
                plt.plot(bin_centers,corr_coeff_shuff,linewidth=2.,color=shuff_color,label=lab_shuff)
                plt.fill_between(bin_centers,corr_coeff_shuff+corr_coeff_error_shuff,corr_coeff_shuff-corr_coeff_error_shuff,alpha=0.1, edgecolor=shuff_color, facecolor=shuff_color2)
        else:
            # orange ones
            shuff_color = '#CC6677'
            if i == 2:
                plt.errorbar(bin_centers*1.05,bias,yerr=bias_error,ls='-',c=col_opt,fmt='o',capsize=4,label=lab_opt)
            else:
                plt.errorbar(bin_centers,bias,yerr=bias_error,ls='-',c=col_opt,fmt='o',capsize=4,label=lab_opt)
        
        if i_type == 0:
            plt.ylim([0.7,2.5])
        if i_type == 1:
            plt.ylim([0.75,1.32])
            plt.legend()
        #origplt.xlim([.7,15])
        plt.xlim([.7,15])
        plt.xscale('log')

        if i_type == 0:
            plt.ylabel(r'$\tilde b(r) = (\xi_{\rm gg}/\xi_{\rm mm})^{1/2}$')
        else:
            plt.ylabel(r'$\tilde r(r) = \xi_{\rm gm}/(\xi_{\rm gg} \xi_{\rm mm})^{1/2}$')
            plt.xlabel(r'$r$ [Mpc/h]')
            
        if plot_no >= ntot-ncols+1:
            plt.xlabel(r'$r$ [Mpc/h]')
        if plot_no%ncols == 1 and i_type == 0:
            plt.ylabel(r'$\tilde b(r) = (\xi_{\rm gg}/\xi_{\rm mm})^{1/2}$')
        elif plot_no%ncols == 1 and i_type == 1:
            plt.ylabel(r'$\tilde r(r) = \xi_{\rm gm}/(\xi_{\rm gg} \xi_{\rm mm})^{1/2}$')
        else:
            1#plt.gca().axes.yaxis.set_ticklabels([])

plt.savefig("paper/bias_corr_coeff.pdf")
plt.show()
