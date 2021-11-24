import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as mpt
import pandas as pd
from matplotlib.colors import LogNorm

df = np.genfromtxt('/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/tools/map_model/europa_plume_plasma_map')
data1 = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/testing/ray-Peuropa_plume-Mtest_simple-benchmark-LO-Z100-FR1e6")
data2 = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/testing/ray-Peuropa_plume-Mtest_simple-benchmark-LO-Z200-FR1e6")
data3 = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/testing/ray-Peuropa_plume-Mtest_simple-benchmark-LO-Z300-FR1e6")
data4 = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/testing/ray-Peuropa_plume-Mtest_simple-benchmark-LO-Z400-FR1e6")
data5 = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/testing/ray-Peuropa_plume-Mtest_simple-benchmark-LO-Z500-FR1e6")
data6 = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/testing/ray-Peuropa_plume-Mtest_simple-benchmark-LO-Z600-FR1e6")
data7 = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/testing/ray-Peuropa_plume-Mtest_simple-benchmark-LO-Z700-FR1e6")
data8 = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/testing/ray-Peuropa_plume-Mtest_simple-benchmark-LO-Z800-FR1e6")
data9 = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/testing/ray-Peuropa_plume-Mtest_simple-benchmark-LO-Z900-FR1e6")
data10 = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/testing/ray-Peuropa_plume-Mtest_simple-benchmark-LO-Z1000-FR1e6")
l_2d = len(df)


idx = np.array(np.where(df[:,0]>-1000))  #np.whereは条件を満たす要素番号を返す　idxはnp.arrayは
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


x1 = data1[:,[1]]
x2 = data2[:,[1]]
x3 = data3[:,[1]]
x4 = data4[:,[1]]
x5 = data5[:,[1]]
x6 = data6[:,[1]]
x7 = data7[:,[1]]
x8 = data8[:,[1]]
x9 = data9[:,[1]]
x10 = data10[:,[1]]

z1 = data1[:,[3]]
z2 = data2[:,[3]]
z3 = data3[:,[3]]
z4 = data4[:,[3]]
z5 = data5[:,[3]]
z6 = data6[:,[3]]
z7 = data7[:,[3]]
z8 = data8[:,[3]]
z9 = data9[:,[3]]
z10 = data10[:,[3]]

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
plt.imshow(vvv, cmap='spring', origin='lower', interpolation='nearest', extent=[-1000,997.5,-200,1997.5])


plt.title("europa_plume")
plt.xlabel("x (km)")
plt.ylabel("z (km)")

plt.xlim(-1000,1000)
plt.ylim(-200,2000)

t = np.arange(-1601,1601,2)
c = np.sqrt(2563201-t*t) - 1601
n = -1600+t*0
plt.plot(t, c, color = "black")
plt.plot(n, c, color = "black")
plt.plot(x1,z1, color = 'white')
plt.plot(x2,z2, color = 'white')
plt.plot(x3,z3, color = 'white')
plt.plot(x4,z4, color = 'white')
plt.plot(x5,z5, color = 'white')
plt.plot(x6,z6, color = 'white')
plt.plot(x7,z7, color = 'white')
plt.plot(x8,z8, color = 'white')
plt.plot(x9,z9, color = 'white')
plt.plot(x10,z10, color = 'white')
plt.fill_between(t, c, n, facecolor='black')




plt.show();
