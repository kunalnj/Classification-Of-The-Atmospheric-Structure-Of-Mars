import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

rootDir = '/Users/kunal/OneDrive - The Open University/SPIN/data/level_1p0_data/occultation'

for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        #print('\t%s' % fname)

        file = h5py.File(r'/Users/kunal/OneDrive - The Open University/SPIN/data/level_1p0_data/occultation/%s' %fname ,'r')
        
        T = file['Science/Transmission']
        T_max = T.shape[1]
        
        trans = np.array([])
        for i in range(0,T_max):
            trans = np.append(trans,np.nansum(T[:,i]))
        
        plt.plot(np.arange(0,len(trans),1),trans)
        plt.ylabel('Transmission')
        plt.savefig(r'/Users/kunal/OneDrive - The Open University/SPIN/Transmission actual/%s.png' % fname)
        plt.clf()