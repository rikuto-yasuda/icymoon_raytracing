# In[]


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as mpt
import pandas as pd
from matplotlib.colors import LogNorm

class test:

    def __init__ (self,ray):
        data = ray
        x = data[:,[1]]
        z = data[:,[3]]
        self.x = x
        self.z = z
    
    def plot_ray (self):

        plt.imshow(v, origin='lower', interpolation='nearest')
        plt.colorbar(extend='both')

        idx_list = np.array([])
        i_list = np.zeros([v.shape[1]])

        v0 = v[:,400]

        vv = v > 1.4E+10
        vv = vv.astype(np.uint8)

        up = np.roll(vv, 1, axis=0)
        down = np.roll(vv, -1, axis=0)
        right = np.roll(vv, 1, axis=1)
        left = np.roll(vv, -1, axis=1)

        vvv = up + down + right+ left - 4*vv

        vvv = np.where(vvv<1, np.nan, 1)

        data = v+0.000001


        plt.figure(dpi=200, figsize=(10,6))
        plt.imshow(v, norm=mpl.colors.LogNorm(), origin='lower',interpolation='nearest', extent=[-1000,997.5,-200,1997.5], vmin=1E+7, vmax=1E+12)
        plt.colorbar(extend='both')

        plt.title("europa_nonplume")
        plt.xlabel("x (km)")
        plt.ylabel("z (km)")

        plt.xlim(-1000,1000)
        plt.ylim(-200,2000)

        t = np.arange(-1601,1601,2)
        c = np.sqrt(2563201-t*t) - 1601
        
        plt.plot(t, c, color = "black")
        plt.plot(self.x, self.z, color = 'white')

        plt.show()




    
# In[]

plasma = np.genfromtxt('/Users/yasudarikuto/research/raytracing/raytrace.tohoku/src/rtc/tools/map_model/europa_nonplume_plasma_map')

l_2d = len(plasma)

idx = np.array(np.where(plasma[:,0]>-1000)) # np.whereは条件を満たす要素番号を返す　idxはnp.arrayは
print(idx[0,0])
r_size = idx[0,0]
c_size = int(l_2d/r_size)

x = plasma[:,0].reshape(c_size, r_size)
print(x.shape)
y = plasma[:,1].reshape(c_size, r_size)
print(y.shape)
z = plasma[:,2].reshape(c_size, r_size)
print(z.shape)
v = plasma[:,3].reshape(c_size, r_size).T
print(v.shape)

ray = np.genfromtxt('/Users/yasudarikuto/research/raytracing/tools/results/ray_stop/ray-Peuropa_nonplume-Mtest_simple-benchmark-LO-Z100-FR1e6')

data1 = test(ray)
data1.plot_ray()

# In[]
