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
print(v[10][140])

plt.pcolor(
    xx,
    zz,
    v,
    cmap="jet",
    shading="auto",
)

plt.xlim(-5000, 5000)
plt.ylim(-5000, 5000)

plt.title("Claire's Ganymede Ionosphere Model")
plt.xlabel("x (km)")
plt.ylabel("z (km)")

plt.colorbar()
circle =plt.Circle((0,0), 1560.8, color='black', alpha=1)
plt.gca().add_artist(circle)

# plt.savefig("300_dpi_scatter.png", format="png", dpi=300)
plt.show()

# %%
df = np.genfromtxt('/Users/yasudarikuto/research/icymoon_raytracing/src_venv/rtc_cost_reduction/tools/map_model/pxy_normal')
l_2d = len(df)
x_min = df[0,0]
x_max = df[-1,0]
y_min = df[0,1]
y_max = df[-1,1]

idx = np.array(np.where(df[:,0]>x_min))
r_size = idx[0,0]
x_array = np.linspace(x_min, x_max, r_size)
c_size = int(l_2d/r_size)
y_array = np.linspace(y_min, y_max, c_size)

xx, yy = np.meshgrid(x_array, y_array)

v = df[:,2].reshape(c_size, r_size).T
print(v[10][140])

plt.pcolor(
    xx,
    yy,
    v,
    cmap="jet",
    shading="auto",
)

plt.xlim(-5000, 5000)
plt.ylim(-5000, 5000)


plt.title("Claire's Ganymede Ionosphere Model")
plt.xlabel("x (km)")
plt.ylabel("y (km)")

plt.colorbar()
circle =plt.Circle((0,0), 1560.8, color='black', alpha=1)
plt.gca().add_artist(circle)

# plt.savefig("300_dpi_scatter.png", format="png", dpi=300)
plt.show()
# %%
