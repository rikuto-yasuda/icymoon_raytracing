# In[]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as mpt
import pandas as pd
from matplotlib.colors import LogNorm

# In[]

df = np.genfromtxt('/Users/yasudarikuto/research/icymoon_raytracing/src_venv/rtc_cost_reduction/tools/map_model/pxz-normal')
l_2d = len(df)
x_min = df[0,0]
x_max = df[-1,0]
z_min = df[0,2]
z_max = df[-1,2]

idx = np.array(np.where(df[:,0]>x_min))
r_size = idx[0,0]
x_array = np.linspace(x_min, x_max, r_size)
c_size = int(l_2d/r_size)
z_array = np.linspace(z_min, z_max, c_size)

xx, zz = np.meshgrid(x_array, z_array)

print(xx[10][140])
print(zz[10][140])

v = df[:,3].reshape(c_size, r_size).T
v = v / 1e6 # m^-3 -> cm^-５

# カラーバーの範囲を指定
vmin = 1  # 最小値
vmax = 0.5e5  # 最大値



print(v[10][140])

plt.pcolor(
    xx,
    zz,
    v,
    cmap="jet",
    shading="auto",
    norm=LogNorm(vmin=vmin, vmax=vmax)  # カラーバーの範囲を指定
)

plt.xlim(-2000, 2000)
plt.ylim(-2000, 2000)

plt.title("Claire's Ganymede Ionosphere Model [cm^-3]")
plt.xlabel("x (km)")
plt.ylabel("z (km)")

plt.colorbar()
circle =plt.Circle((0,-2574.7), 2574.7, color='black', alpha=1)
plt.gca().add_artist(circle)

# plt.savefig("300_dpi_scatter.png", format="png", dpi=300)
plt.show()

# %%

df = np.genfromtxt('/Users/yasudarikuto/research/icymoon_raytracing/src_venv/rtc_cost_reduction/tools/map_model/pxz-normal')

x_list = np.where(df[:,0]==0)
y_list = np.where(df[:,1]==0)

select = np.intersect1d(x_list, y_list)

z = df[select,2]
ne = df[select,3]/1e6 # m^-3 -> cm^-3

plt.xlim(-500, 4000)
plt.ylim(0, 3000)

plt.plot(ne, z)
plt.show()
# %%
