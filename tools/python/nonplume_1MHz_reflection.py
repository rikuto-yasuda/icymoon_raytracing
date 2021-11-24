# In[]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as mpt
import pandas as pd
from matplotlib.colors import LogNorm

# In[]

df = np.genfromtxt('/Users/yasudarikuto/research/raytracing/raytrace.tohoku/src/rtc/tools/map_model/europa_nonplume_plasma_map')
l_2d = len(df)
idx = np.array(np.where(df[:,0]>-1000))
print(idx[0,0])
r_size = idx[0,0]
c_size = int(l_2d/r_size)


x = df[:,0].reshape(c_size, r_size)
print(x.shape)
y = df[:,1].reshape(c_size, r_size)
print(y.shape)
z = df[:,2].reshape(c_size, r_size)
print(z.shape)
v = df[:,3].reshape(c_size, r_size).T


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

#plt.imshow(vvv, cmap='spring', origin='lower', interpolation='nearest', extent=[-1000,997.5,-200,1997.5])


plt.title("europa_reflection_1MHz")
plt.xlabel("x (km)")
plt.ylabel("z (km)")

plt.xlim(-1000,1000)
plt.ylim(-200,2000)

t = np.arange(-1601,1601,2)
c = np.sqrt(2563201-t*t) - 1601
n = -1600+t*0
plt.plot(t, c, color = "black")
plt.plot(n, c, color = "black")

for i in range(100,1301,100):
    n = str(i)
    N = i
    filename = np.genfromtxt("/Users/yasudarikuto/research/raytracing/tools/results/for_JpGU_2021/europa_reflection_1MHz/ray-Peuropa_nonplume-Mtest_simple-benchmark-LO-Z"+n+"-FR1e6")
    
    x1 = filename[:,[1]]
    z1 = filename[:,[3]]
    plt.plot(x1,z1, color = 'white',linewidth = 0.5)





plt.show()

# %%
