import numpy as np
import matplotlib.pyplot as plt
import sys
import plotparams
plotparams.buba()

#selection = '_DESI'
selection = '_'+sys.argv[1]
#selection = '_eBOSS'

mode = r"sSFR-colored"
mode_str = "_"+mode.split('-')[0]

redshift_dict = {'055':['sdss_desi','_0.0'],#-0.5
                 '047':['sdss_desi','_-0.0'],
                 '041':['sdss_des','']}

# snapshot number and dust model
snap = '0%d'%(int(sys.argv[2]))#'041'#'047'#'055'#'050'#'059'
snap_dir = '_'+str(int(snap))
dust_index = redshift_dict[snap][1]
fs = 18

z_dic = {'_55': 0.8, '_47': 1.1, '_41': 1.4}

x_dec_eligible = np.load("data/x_dec_eligible"+snap_dir+selection+".npy")
y_dec_eligible = np.load("data/y_dec_eligible"+snap_dir+selection+".npy")
sub_star_mass_eligible = np.load("data/sub_star_mass_eligible"+snap_dir+selection+".npy")
sub_SFR_eligible = np.load("data/sub_SFR_eligible"+snap_dir+selection+".npy")

sub_sSFR_eligible = (sub_SFR_eligible/sub_star_mass_eligible)
log_sSFR_eligible = np.log10(sub_sSFR_eligible)
log_SFR_eligible = np.log10(sub_SFR_eligible)
# avoid division by 0
log_sSFR_eligible[sub_SFR_eligible == 0.] = -13.
log_SFR_eligible[sub_SFR_eligible == 0.] = -3.

#x_line = np.linspace(np.min(x_dec_eligible),np.max(x_dec_eligible),100)
#y_line = np.linspace(np.min(y_dec_eligible),np.max(y_dec_eligible),100)
x_line = np.linspace(-0.5,2.4,10)
y_line = np.linspace(-0.5,2.4,10)


if selection == '_DESI':
    y_line1 = -1.2*x_line+1.6
    y_line2 = 1.15*x_line-0.15
    x_line1 = 0.3*np.ones(len(x_line))
    x_line2 = 1.6*np.ones(len(x_line))
elif selection == '_eBOSS':
    y_line1 = -0.068*x_line+0.457
    y_line2 = 0.112*x_line+0.773
    x_line1 = 0.218*y_line+0.571 #NGC 0.637*y_line+0.399
    x_line2 = -0.555*y_line+1.901


plt.figure(1,figsize=(9,9))
'''
y_line1 = -1.2*x_line+1.6
y_line2 = 1.15*x_line-0.15
x_line1 = 0.3*np.ones(len(x_line))
x_line2 = 1.6*np.ones(len(x_line))

plt.plot(x_line,y_line1,'orange',label='DESI')
plt.plot(x_line,y_line2,'orange')
plt.plot(x_line1,y_line,'orange')
plt.plot(x_line2,y_line,'orange')

y_line1 = -0.068*x_line+0.457
y_line2 = 0.112*x_line+0.773
x_line1 = 0.218*y_line+0.571 #NGC 0.637*y_line+0.399
x_line2 = -0.555*y_line+1.901

plt.plot(x_line,y_line1,'dodgerblue',label='eBOSS')
plt.plot(x_line,y_line2,'dodgerblue')
plt.plot(x_line1,y_line,'dodgerblue')
plt.plot(x_line2,y_line,'dodgerblue')
'''
plt.plot(x_line,y_line1,'dodgerblue')
plt.plot(x_line,y_line2,'dodgerblue')
plt.plot(x_line1,y_line,'dodgerblue')
plt.plot(x_line2,y_line,'dodgerblue')

#plt.scatter(x_dec_eligible,y_dec_eligible,s=0.05,label="color-selected")
#plt.hexbin(x_dec_eligible, y_dec_eligible, C=np.abs(sub_SFR_eligible/np.mean(sub_SFR_eligible)), gridsize=50, bins='log', cmap='Greys')
if mode == "SFR-colored":
    #plt.hexbin(x_dec_eligible, y_dec_eligible, C=(sub_SFR_eligible/np.mean(sub_SFR_eligible)), gridsize=50, bins='log', cmap='Greys')#'inferno')#'Greys')
    plt.hexbin(x_dec_eligible, y_dec_eligible, C=(sub_SFR_eligible), gridsize=50, bins='log', cmap='Greys')
elif mode == "sSFR-colored":
    plt.hexbin(x_dec_eligible, y_dec_eligible, C=(sub_sSFR_eligible/np.mean(sub_sSFR_eligible)), gridsize=50, bins='log', cmap='Greys')

plt.xlim((-0.5, 2.4))
plt.ylim((-0.5, 2.4))
plt.text(.4,2.,r'${\rm '+(''.join(selection.split('_')))+",} \ z = %.1f"%z_dic[snap_dir]+r", \ {\rm "+mode+"}$")#,fontsize=fs)
#plt.text(.6,2.,r"$z = %.1f"%z_dic[snap_dir]+r", \ {\rm "+mode+"}$")#,fontsize=fs)
plt.xlabel(r"$r-z$")#,fontsize=fs)
plt.ylabel(r"$g-r$")#,fontsize=fs)
#plt.legend()
plt.savefig("figs/selection"+snap_dir+selection+mode_str+".png")
plt.close()
