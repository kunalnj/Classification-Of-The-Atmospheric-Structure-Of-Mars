from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

img = r'C:\Users\kj4755\OneDrive - The Open University\SPIN\PIA02066.jpg'

map = Basemap(projection='ortho', lat_0=15, lon_0=0)
map.warpimage(img)

plt.savefig(r'Mars_ortho.jpg', bbox_inches='tight')
plt.show()