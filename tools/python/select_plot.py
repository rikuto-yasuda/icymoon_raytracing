import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpt

df = np.genfromtxt ('/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/tools/map_model/europa_plume_plasma_map')
l_2d = len(df)

x = df[:,[0]]
y = df[:,[2]]
z = df[:,[3]]


plt.scatter(x,y,c=z,linewidths="0",norm=mpt.colors.LogNorm())

plt.title("europa_plume")
plt.xlabel("x (km)")
plt.ylabel("z (km)")

t = np.arange(-1601,1601,2)
c = np.sqrt(2563201-t*t) - 1601
n = -1600+t*0
plt.plot(t, c, color = "black")
plt.plot(n, c, color = "black")

plt.fill_between(t, c, n, facecolor='black')



"""

for i in range (l_2d):
    if 3.1*(10**11) < df[i,[3]] < 3.2*(10**11) :
        plt.scatter(df[i,[0]],df[i,[1]],c='black',linewidths="2")
"""
plt.xlim(-1000,1000)
plt.ylim(-200,1000)
plt.colorbar()
plt.show()
