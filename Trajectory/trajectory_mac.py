import os
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
import numpy as np
import h5py
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


rootDir = r'/Users/kunal/OneDrive - The Open University/SPIN/data/raw_data/limbs/full_limbs'




for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        
        file = h5py.File(r'/Users/kunal/OneDrive - The Open University/SPIN/data/raw_data/limbs/full_limbs/%s' %fname,'r')


        img = r'/Users/kunal/OneDrive - The Open University/SPIN/Trajectory/marssurface.jpg'
        
        
        fig1 = plt.figure(figsize=(8,6))
        
        lat_viewing_angle = 45
        lon_viewing_angle = 10
        
        #m = Basemap(projection='cyl',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
        m = Basemap(projection='ortho',lat_0 = lat_viewing_angle,lon_0 = lon_viewing_angle)
        
        m.warpimage(img)
        parallels = np.arange(-80,81,20)
        # labels = [left,right,top,bottom]
        m.drawparallels(parallels,labels=[False,True,True,False])
        meridians = np.arange(0.,361,20)
        m.drawmeridians(meridians,labels=[True,False,False,True])
        #plt.savefig(r'Mars_ortho_mac.jpg', bbox_inches='tight')
        
        lon = file["Geometry"]['Point0']['Lon']
        lat = file["Geometry"]['Point0']['Lat']
        
        init_lat = []
        init_lon = []
        for i in range(len(lat)-4):
            if -180 <= lon[i][0] <= 180 and -90 <= lat[i][0] <= 90:
                init_lon.append(lon[i][0])
                init_lat.append(lat[i][0])
        
        initial = np.transpose(np.array([init_lon[:],init_lat[:]]))
        
        initial_pos = initial[np.where(initial[:,0] > 0)]
        initial_neg = initial[np.where(initial[:,0] < 0)]
        
                
        final_lat = []
        final_lon = []
        for i in range(len(lat)-4):
            if -180 <= lon[i][0] <= 180 and -90 <= lat[i][0] <= 90:
                final_lon.append(lon[i][0])
                final_lat.append(lat[i][0])
        
        
        
        plt.scatter(initial[:,0],initial[:,1],color = 'y')
        
        #plt.plot(final_lon,final_lat,color = 'y')
        '''fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        ax2.plot(init_lon,init_lat)
        fig2.suptitle('initial')'''
        '''fig3 = plt.figure()
        ax3 = fig3.add_subplot(1,1,1)
        ax3.plot(final_lon,final_lat)
        fig3.suptitle('final')'''
        #save_path = os.mkdir(r'/Users/kunal/OneDrive - The Open University/SPIN/Trajectory/%s/' %fname)
        #plt.figure(figsize=(10,6))

        plt.savefig(r'/Users/kunal/OneDrive - The Open University/SPIN/Trajectory/%s.png' %fname)
        
        
        
