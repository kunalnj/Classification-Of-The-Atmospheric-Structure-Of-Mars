import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import os

rootDir = r'C:\Users\kj4755\OneDrive - The Open University\SPIN\data\level_1p0_data\occultation'
plt.clf()
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        
        file = h5py.File(r'C:\Users\kj4755\OneDrive - The Open University\SPIN\data\level_1p0_data\occultation\%s' %fname,'r')
        
        T = np.array(file['Science/Transmission'])
        TangAlt = np.array(file['Geometry/Point0/TangentAltSurface'])
        wav = np.array(file['Science/Wavelength'])
        
        
        avg_TangAlt = []
        
        for j in range(TangAlt.shape[0]):
            avg_TangAlt.append(np.mean(TangAlt[j,:]))
        
        T_250 = T[:,np.array(np.where(wav == min(abs(wav-250))+250)).flatten()].reshape(-1,)
        
        plt.scatter(avg_TangAlt,T_250,marker = 'x')
        plt.ylim(0,1.1)
        plt.xlabel('Tangent Altitude (km)')
        plt.ylabel('Transmission')
        plt.title('Transmission vs TangAlt for 250 nm')
        plt.savefig(r'C:\Users\kj4755\OneDrive - The Open University\SPIN\Transmission\Scatter_250nm\%s.png' %fname)
        plt.clf()
