import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('ggplot')

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

f_wav = 500

rootDir = r'C:\Users\kj4755\OneDrive - The Open University\SPIN\data\level_1p0_data\New occultations'

plt.clf()

for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        file = h5py.File(r'C:\Users\kj4755\OneDrive - The Open University\SPIN\data\level_1p0_data\New occultations\%s' %fname,'r')
        
        T = np.array(file['Science/Transmission'])
        TangAlt = np.array(file['Geometry/Point0/TangentAltSurface'])
        wav = np.array(file['Science/Wavelength'])
        
        avg_TangAlt = []
        
        for j in range(TangAlt.shape[0]):
            avg_TangAlt.append(np.mean(TangAlt[j,:]))
            
        
        
        T_wav = T[:,np.array(np.where(wav == find_nearest(wav,f_wav))).flatten()].reshape(-1,)
        #avg_TangAlt = np.array(avg_TangAlt)
       # alt_250 = avg_TangAlt[:,np.array(np.where(wav == min(abs(wav-250))+250)).flatten()].reshape(-1,)
        
        plt.plot(avg_TangAlt,T_wav,color = 'k')
        plt.ylim(0,1.1)
        plt.xlabel('Tangent Altitude (km)')
        plt.ylabel('Transmission')
        plt.title('Transmission vs TangAlt at %s nm' %f_wav)
        plt.savefig(r'C:\Users\kj4755\OneDrive - The Open University\SPIN\Transmission\Plot_%snm\%s.png' %(f_wav,fname))
        plt.clf()