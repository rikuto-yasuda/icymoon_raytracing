# In[]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as mpt
import pandas as pd
from matplotlib.colors import LogNorm

# In[]

df = np.genfromtxt('/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/tools/map_model/europa_nonplume_plasma_map')
l_2d = len(df)
idx = np.array(np.where(df[:,0]>-1500))
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

# 1e5Hz ライン
F_1e5 = v > 1.4E+8
F_1e5 = F_1e5.astype(np.uint8)

F_1e5_up = np.roll(F_1e5, 1, axis=0)
F_1e5_down = np.roll(F_1e5, -1, axis=0)
F_1e5_right = np.roll(F_1e5, 1, axis=1)
F_1e5_left = np.roll(F_1e5, -1, axis=1)

print_F_1e5 = F_1e5_up + F_1e5_down + F_1e5_right+ F_1e5_left - 4*F_1e5
print_F_1e5 = np.where(print_F_1e5<1, np.nan, 1)

# 5e5Hz ライン
F_5e5 = v > 3.53E+9
F_5e5 = F_5e5.astype(np.uint8)

F_5e5_up = np.roll(F_5e5, 1, axis=0)
F_5e5_down = np.roll(F_5e5, -1, axis=0)
F_5e5_right = np.roll(F_5e5, 1, axis=1)
F_5e5_left = np.roll(F_5e5, -1, axis=1)

print_F_5e5 = F_5e5_up + F_5e5_down + F_5e5_right+ F_5e5_left - 4*F_5e5
print_F_5e5 = np.where(print_F_5e5<1, np.nan, 1)

# 1e6Hz ライン
F_1e6 = v > 1.4E+10
F_1e6 = F_1e6.astype(np.uint8)

F_1e6_up = np.roll(F_1e6, 1, axis=0)
F_1e6_down = np.roll(F_1e6, -1, axis=0)
F_1e6_right = np.roll(F_1e6, 1, axis=1)
F_1e6_left = np.roll(F_1e6, -1, axis=1)

print_F_1e6 = F_1e6_up + F_1e6_down + F_1e6_right+ F_1e6_left - 4*F_1e6
print_F_1e6 = np.where(print_F_1e6<1, np.nan, 1)

data = v+0.000001



plt.imshow(v, norm=mpl.colors.LogNorm(), origin='lower',interpolation='nearest', extent=[-1500,1497.5,-750,1497.5], vmin=1E+7, vmax=1E+11)
plt.colorbar(extend='both')

plt.imshow(print_F_1e5, cmap='spring', origin='lower', interpolation='nearest', extent=[-1500,1497.5,-750,1497.5])
plt.imshow(print_F_5e5, cmap='autumn', origin='lower', interpolation='nearest', extent=[-1500,1497.5,-750,1497.5])
plt.imshow(print_F_1e6, cmap='winter', origin='lower', interpolation='nearest', extent=[-1500,1497.5,-750,1497.5])


plt.title("europa_occultation_5MHz")
plt.xlabel("x (km)")
plt.ylabel("z (km)")

plt.xlim(-1500,1500)
plt.ylim(-500,1500)

t = np.arange(-1601,1601,2)
c = np.sqrt(2563201-t*t) - 1601
n = -1600+t*0
plt.plot(t, c, color = "black",linewidth = 0.0001)
plt.plot(n, c, color = "black")
plt.fill_between(t, c, n, facecolor='black')

for i in range(100,1301,100):
    n = str(i)
    N = i
    filename = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/tools/results/for_JpGU_2021/europa_stop_5MHz/ray-Peuropa_nonplume-Mtest_simple-benchmark-LO-Z"+n+"-FR5e6")
    
    x1 = filename[:,[1]]
    z1 = filename[:,[3]]
    plt.plot(x1,z1, color = 'white')





plt.show();

# %%
