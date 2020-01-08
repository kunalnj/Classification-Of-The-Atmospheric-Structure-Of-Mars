"""
Developer: Kunal Jadhav
Implementation of a DBSCAN clustering algorithm for atmospheric tranmission profiles
"""

import numpy as np
import h5py
import os
from peakfinder import detect_peaks   #Credit: Graham Sellers
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import similaritymeasures             #pip install similaritymeasures if not available. Credit: https://github.com/cjekel/similarity_measures   #quicker than my implementation of DTW
import matplotlib.pyplot as plt
plt.style.use('ggplot')


transmission = []  #transmission placeholder list
altitude = []  #altitude placeholder list


##function to find nearest value in the HDF5 wavelength array to f_wav
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

f_wav = 250 #wavelength at which clustering is being applied

rootDir = r'C:\Users\kj4755\OneDrive - The Open University\SPIN\data\level_1p0_data\New occultations'  #location of data files
os.chdir(rootDir)
list_of_files = np.array(os.listdir(os.getcwd()))  #array containing filenames

print('Loading files:')
for each_file in tqdm(list_of_files):

    file = h5py.File(rootDir + '\%s' %each_file,'r')     #open each file

    T = np.array(file['Science/Transmission'])
    TangAlt = np.array(file['Geometry/Point0/TangentAltSurface'])
    wav = np.array(file['Science/Wavelength'])

    avg_TangAlt = []

    for j in range(TangAlt.shape[0]):
        avg_TangAlt.append(np.mean(TangAlt[j,:]))       #take mean of start and end observation altitudes

    T_wav = T[:,np.array(np.where(wav == find_nearest(wav,f_wav))).flatten()].reshape(-1,)  #transmission array of file at f_wav

    if T_wav[0] > 0.5:
        T_wav = T_wav[::-1]     ##reversing array if needed

    transmission.append(T_wav)

    if avg_TangAlt[0] > 100:
        avg_TangAlt = avg_TangAlt[::-1]        ##reversing array if needed

    altitude.append(avg_TangAlt)
            
transmission = np.array(transmission)
altitude = np.array(altitude)

bad_files_idx = []
for i,_ in enumerate(transmission):
    if _[0] > 0.8:
        bad_files_idx.append(i)
        
transmission = np.delete(transmission,bad_files_idx,0)
altitude = np.delete(altitude,bad_files_idx,0)

alt_interp = np.arange(0,max(np.array([len(_) for _ in transmission])),1)       #274
t_interp = []

for i in range(len(transmission)):
    t_interp.append(np.interp(alt_interp,altitude[i],transmission[i]))      #calculate interpolated transmission values
    
t_interp = np.array(t_interp)



##using peakfinder module to find point of highest 'structure' so that the smooth, approximately constant region
##at around transmission = 1 can be ignored


t_clipped = []
for j,t in enumerate(t_interp):
    clip = np.array([])
    peak_idx = detect_peaks(t,mph=0.05)
    for _ in t[peak_idx]:
        clip = np.append(clip,_)
        if _ > 0.963:
            break
    t_clipped.append(t[:np.where(t == clip[-1])[0][0]])     #appending transmission values up until the last peak detected
t_clipped = np.array(t_clipped)           


clipped_len = []
for t in t_clipped: 
    clipped_len.append(len(t))

t_clipped_max = t_interp[:,:max(clipped_len)]   #array of tranmission values clipped to the length of the smallest array


##manually increasing the transmission values at certain peaks and valleys detected, so that curve length of similar profiles is exaggerated and more differentiable
##makes it easier for DBSCAN clustering
for i,t in enumerate(t_clipped_max):
    
    peak_idx = detect_peaks(t,mph=0.05)     #only detect peaks bigger than 0.05. Avoids detecting the tiny peaks near the flat region close to 0
    valley_idx = detect_peaks(t,mph = 0.963, valley = True)     #detect valleys less than transmission = 0.963. This cuts off detecting valleys in the flat region of the profile

    p_structure_low = []
    p_structure_high = []
    v_structure_low = []
    v_structure_high = []
    
    for peak in peak_idx:
        if 0.05 < t[peak] <= 0.2:
            p_structure_low.append(peak)        #"low" structure peaks
        if 0.2 < t[peak] < 0.963:
            p_structure_high.append(peak)       #"high" structure peaks
            
    for valley in valley_idx:
        if 0.05 < t[valley] < 0.2:
            v_structure_low.append(valley)      #"low" structure valleys
        if t[valley] > 0.2:
            v_structure_high.append(valley)     #"high" structure valleys
        
            
    p_structure_high = np.array(p_structure_high)
    p_structure_low = np.array(p_structure_low)
    v_structure_low = np.array(v_structure_low)
    v_structure_high = np.array(v_structure_high)
    
    if len(p_structure_low) != 0:
        t_clipped_max[i,p_structure_low[0]] = 5     #change transmission at 'x' position of the first low peaks to 5
    
    if len(p_structure_high) != 0:
        t_clipped_max[i,p_structure_high[0]] = 10   #change transmission at 'x' position of the first high peaks to 10
    
    if len(v_structure_high) != 0:
        t_clipped_max[i,v_structure_high[0]] = 15   #change transmission at 'x' position of the first high valleys to 5
    


##restoring t_interp (the above process seems to have altered t_itnerp, restoring just as a precautionary measure)
alt_interp = np.arange(0,max(np.array([len(_) for _ in transmission])),1)
t_interp = []

for i in range(len(transmission)):
    t_interp.append(np.interp(alt_interp,altitude[i],transmission[i]))
    
t_interp = np.array(t_interp)



##function to generate a distance similarity distance matrix
def generate_similarity_dist_matrix(method,profiles):
    """
    method: method of computing similarity distances between pairs of signals. Implemented methods: ['curve_length', 'DTW']
    profiles: 2D array containing all transmission profiles
    # for best clustering of atmospheric transmission profiles, use method = 'cruve_length' and profiles = t_clipped_max
    """
    methods = ['curve_length', 'DTW']
    if method not in methods:
        raise ValueError('Invalid method. Implemented methods: %s or %s' %(methods[0],methods[1]))
        
    dm = np.zeros((len(profiles),len(profiles)))       #creating a similarity distance matrix to be populated
    print('Generating similarity distance matrix:')
    for i in tqdm(range(len(profiles))):
        
        t0 = np.zeros((len(profiles[i]),2))           #similaritymeasures requires input data in this form
        t0[:,0] = np.arange(len(profiles[i]))
        t0[:,1] = profiles[i]
        
        for j in range(len(profiles)):                #cycling through t_clipped_max, populating dm
            t1 = np.zeros((len(profiles[j]),2))       #similaritymeasures form
            t1[:,0] = np.arange(len(profiles[j]))
            t1[:,1] = profiles[j]
            
            #choose only one of the following, both shown just for demonstration of how to call the method
            if method == 'curve_length':
                dm[i,j] =similaritymeasures.curve_length_measure(t0,t1)     #calculate similarity between the two signals t0 and t1 based on curve length
            if method == 'DTW':
                dm[i,j] =similaritymeasures.dtw(t0,t1)[0]     #calculate dynamic time warping distance between t0 and t1
                
    return dm

##!NOTE!: depending on the size of the transmission array, in this case t_clipped_max, dm may take a while to be calculated
## implemetation of parallelization to calculate parts of the matrix in separate runs to be adjoined at the end, would be very beneficial in reducing computation time



##function to cluster profiles
def cluster(dm,eps,min_samples,directory, generate_clustered_files = True,generate_plots = False):
    """
    dm: precomputed distance matrix
    eps: epsilon, ie neighbourhood radius, refer to DBSCAN docs
    min_samples: min number of points defining a dense cluster
    directory: string of directory where clustered files.txt and plots are stored
    generate_clustered_files: To generate or not, a .txt file containing list of files in a cluster. Default = True
    generate_plots: To generate plots of clusters or not. Default = False
    """
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric = 'precomputed').fit(dm)       #DBSCAN clustering. For more info on how it is implemented, refer to https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)     #label '-1' represents noise. So effectively, one less cluster if noise is present
    print('Estimated number of clusters: %d' % n_clusters_)
    
    
    if generate_clustered_files:    
        print('Generating clustered files:')
        for i in tqdm(set(labels)):
            
            idx = np.array(np.where(labels == i)).flatten()
            sub_dir = directory + '\eps %s min_samples %s\Cluster %s' %(eps,min_samples,i)
            
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            
            if generate_plots:      #generate plots if called
                for _ in idx:
                    plt.clf()
                    plt.plot(transmission[_],color = 'k')
                    plt.savefig(sub_dir +'\%s.png' %list_of_files[_])
                
            np.savetxt(directory + '\\Cluster %s files.txt' %i,list_of_files[idx],fmt = '%s')   #generate clustered files.txt
    
    np.savetxt(directory + r'\eps %s min_samples %s\New Occultation dm 2.txt' %(eps,min_samples),dm)        #specify location to save dm given that it is very precious!!


#running the clustering
cluster(dm = generate_similarity_dist_matrix(profiles = t_clipped_max,method = 'curve_length'),eps = 3,min_samples = 5,generate_clustered_files = True,generate_plots = True,directory = r'C:\Users\kj4755\OneDrive - The Open University\SPIN\Transmission\Clusters\DTW')