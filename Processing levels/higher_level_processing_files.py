import h5py
import matplotlib.pyplot as plt
##20180806_021244_0p2f_UVIS_LH1
import matplotlib
import numpy as np
import imageio
matplotlib.use('Agg')

file = h5py.File(r'C:\Users\kj4755\OneDrive - The Open University\SPIN\data\processed_data\20180806_021244_0p2f_UVIS_LH1.h5','r')

filenames = []

for i in range(0,180):
    
    if i > file['level_0p2e']['Science'].shape[2] :
        break

    fig,axs = plt.subplots(5,figsize = (10,10))
    
    
    im1 = axs[0].imshow(file['Science']['Y'][i],vmin = 0,vmax = 65535) #different format for science array
    axs[0].set_title('No processing')
    im2 = axs[1].imshow(file['level_0p2b']['Science'][:,:,i],vmin = 0,vmax = 65535)
    axs[1].set_title('level_0p2b')
    im3 = axs[2].imshow(file['level_0p2c']['Science'][:,:,i],vmin = 0,vmax = 65535)
    axs[2].set_title('level_0p2c')
    im4 = axs[3].imshow(file['level_0p2d']['Science'][:,:,i],vmin = 0,vmax = 65535)
    axs[3].set_title('level_0p2d')
    im5 = axs[4].imshow(file['level_0p2e']['Science'][:,:,i],vmin = 0,vmax = 65535)
    axs[4].set_title('level_0p2e')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im5, cax=cbar_ax)
    fig.suptitle('Frame %i' %i)

    plt.savefig(r'C:\Users\kj4755\OneDrive - The Open University\SPIN\Processing levels\20180806_021244_0p2f_UVIS_LH1\Frame %i.png' %i)
    filenames.append(r'C:\Users\kj4755\OneDrive - The Open University\SPIN\Processing levels\20180806_021244_0p2f_UVIS_LH1\Frame %i.png' % i)
    plt.clf()


images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(r'C:\Users\kj4755\OneDrive - The Open University\SPIN\Processing levels\20180806_021244_0p2f_UVIS_LH1\20180806_021244_0p2f_UVIS_LH1.gif', images)
#%%

##20181025_014256_0p2f_UVIS_LH1

import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import imageio

file = h5py.File(r'C:\Users\kj4755\OneDrive - The Open University\SPIN\data\processed_data\20181025_014256_0p2f_UVIS_LH1.h5','r')

filenames = []

for i in range(0,181):
    
    if i > file['level_0p2b']['Science'].shape[2]-1 :
        break

    fig,axs = plt.subplots(5,figsize = (10,10))
    
    
    im1 = axs[0].imshow(file['Science']['Y'][i],vmin = 0,vmax = 65535) #different format for science array
    axs[0].set_title('No processing')
    im2 = axs[1].imshow(file['level_0p2b']['Science'][:,:,i],vmin = 0,vmax = 65535)
    axs[1].set_title('level_0p2b')
    im3 = axs[2].imshow(file['level_0p2c']['Science'][:,:,i],vmin = 0,vmax = 65535)
    axs[2].set_title('level_0p2c')
    im4 = axs[3].imshow(file['level_0p2d']['Science'][:,:,i],vmin = 0,vmax = 65535)
    axs[3].set_title('level_0p2d')
    im5 = axs[4].imshow(file['level_0p2e']['Science'][:,:,i],vmin = 0,vmax = 65535)
    axs[4].set_title('level_0p2e')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im5, cax=cbar_ax)
    fig.suptitle('Frame %i' %i)

    plt.savefig(r'C:\Users\kj4755\OneDrive - The Open University\SPIN\Processing levels\20181025_014256_0p2f_UVIS_LH1\Frame %i.png' %i)
    filenames.append(r'C:\Users\kj4755\OneDrive - The Open University\SPIN\Processing levels\20181025_014256_0p2f_UVIS_LH1\Frame %i.png' % i)
    plt.clf()
    



images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(r'C:\Users\kj4755\OneDrive - The Open University\SPIN\Processing levels\20181025_014256_0p2f_UVIS_LH1\20181025_014256_0p2f_UVIS_LH1.gif', images)
#%%

