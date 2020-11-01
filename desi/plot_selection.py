import numpy as np
import matplotlib.pyplot as plt
import sys
import plotparams
plotparams.buba()

#selection = '_DESI'
selection = '_'+sys.argv[1]
#selection = '_eBOSS'

mode = sys.argv[3]#r"sSFR-colored"
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

x_line3 = np.linspace(0.7,1.7,10)
x_line4 = np.linspace(0.2,0.8,10)#(-0.5,2.4,10) everyone was using this
y_line3 = np.linspace(-0.5,0.3,10)
y_line4 = np.linspace(-0.5,-0.2,10)


if selection == '_DESI':
    y_line1 = -1.2*x_line3+1.6
    y_line2 = 1.15*x_line4-0.15
    x_line1 = 0.3*np.ones(len(y_line3))
    x_line2 = 1.6*np.ones(len(y_line4))
elif selection == '_eBOSS':
    y_line1 = -0.068*x_line3+0.457
    y_line2 = 0.112*x_line4+0.773
    x_line1 = 0.218*y_line3+0.571 #NGC 0.637*y_line+0.399
    x_line2 = -0.555*y_line4+1.901


plt.figure(1,figsize=(11,7))
'''
y_line1 = -1.2*x_line+1.6
y_line2 = 1.15*x_line-0.15
x_line1 = 0.3*np.ones(len(x_line))
x_line2 = 1.6*np.ones(len(x_line))

plt.plot(x_line,y_line1,'#CC6677',label='DESI')
plt.plot(x_line,y_line2,'#CC6677')
plt.plot(x_line1,y_line,'#CC6677')
plt.plot(x_line2,y_line,'#CC6677')

y_line1 = -0.068*x_line+0.457
y_line2 = 0.112*x_line+0.773
x_line1 = 0.218*y_line+0.571 #NGC 0.637*y_line+0.399
x_line2 = -0.555*y_line+1.901

plt.plot(x_line,y_line1,'dodgerblue',label='eBOSS')
plt.plot(x_line,y_line2,'dodgerblue')
plt.plot(x_line1,y_line,'dodgerblue')
plt.plot(x_line2,y_line,'dodgerblue')
'''
plt.plot(x_line3,y_line1,lw=4.5,color='white')
plt.plot(x_line4,y_line2,lw=4.5,color='white')
plt.plot(x_line1,y_line3,lw=4.5,color='white')
plt.plot(x_line2,y_line4,lw=4.5,color='white')
plt.plot(x_line3,y_line1,lw=1.5,color='black')
plt.plot(x_line4,y_line2,lw=1.5,color='black')
plt.plot(x_line1,y_line3,lw=1.5,color='black')
plt.plot(x_line2,y_line4,lw=1.5,color='black')

cmap = 'coolwarm_r'

#plt.scatter(x_dec_eligible,y_dec_eligible,s=0.05,label="color-selected")
#plt.hexbin(x_dec_eligible, y_dec_eligible, C=np.abs(sub_SFR_eligible/np.mean(sub_SFR_eligible)), gridsize=50, bins='log', cmap='Greys')
if mode == "SFR":
    #plt.hexbin(x_dec_eligible, y_dec_eligible, C=(sub_SFR_eligible/np.mean(sub_SFR_eligible)), gridsize=50, bins='log', cmap='Greys')#'inferno')#'Greys')
    plt.hexbin(x_dec_eligible, y_dec_eligible, C=(sub_SFR_eligible), gridsize=50, bins='log', cmap=cmap)
    cb = plt.colorbar()
    cb.set_label(r"${\rm SFR} \ [M_{\odot}{\rm yr}^{-1}]$")
elif mode == "sSFR":
    #og
    plt.hexbin(x_dec_eligible, y_dec_eligible, C=(sub_sSFR_eligible/np.mean(sub_sSFR_eligible)), gridsize=50, bins='log', vmin = 0., vmax = 0.8, cmap=cmap)
    # TESTING
    #plt.hexbin(x_dec_eligible, y_dec_eligible, C=(sub_sSFR_eligible), gridsize=50, bins='log', cmap=cmap)

    cb = plt.colorbar()
    oldlabels = cb.ax.get_yticklabels()

    print("mean sSFR = ",np.mean(np.log10(sub_sSFR_eligible[sub_sSFR_eligible > 0.])))
    print("mean sSFR = ",np.mean((sub_sSFR_eligible)))
    
    y_new = np.linspace(0.,5.e-10,11)
    y_new /= np.mean(sub_sSFR_eligible)
    cb.ax.set_yticks(y_new)
    oldlabels = cb.ax.get_yticklabels()

    for label in oldlabels:
        print((label.get_text()))

    # TESTING
    cb.set_label(r"${\rm sSFR}/\langle {\rm sSFR} \rangle$")
    # og
    '''
    newlabels = map(lambda x: '$'+format(np.mean(sub_sSFR_eligible)*float((x.get_text()).split('$')[1]),'.1e')+'$', oldlabels)
    cb.ax.set_yticklabels(newlabels)
    cb.set_label(r"${\rm sSFR} \ [{\rm yr}^{-1}]$")
    '''

plt.xlim((-0.5, 2.4))
plt.ylim((-0.5, 2.4))
#plt.text(-.3,2.,r'${\rm '+(''.join(selection.split('_')))+",} \ z = %.1f"%z_dic[snap_dir]+r", \ {\rm "+mode+"}$")#,fontsize=fs)
plt.text(-.3,2.,r'${\rm '+(''.join(selection.split('_')))+",} \ z = %.1f"%z_dic[snap_dir]+r"$")#,fontsize=fs)

plt.xlabel(r"$r-z$")#,fontsize=fs)
plt.ylabel(r"$g-r$")#,fontsize=fs)
plt.gca().tick_params(axis='both', which='major', labelsize=24)
#plt.legend()
#plt.savefig("figs/selection"+snap_dir+selection+mode_str+".png")
plt.savefig("paper/selection"+snap_dir+selection+mode_str+".pdf")
plt.show()
plt.close()
