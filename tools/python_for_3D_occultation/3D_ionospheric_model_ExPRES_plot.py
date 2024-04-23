# In[]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as mpt
import pandas as pd
from matplotlib.colors import LogNorm

# In[]

df = np.genfromtxt(
    "/Users/yasudarikuto/research/icymoon_raytracing/src_local_env/rtc_3D/tools/map_model/pxz-normal"
)
l_2d = len(df)
idx = np.array(np.where(df[:, 0] > df[0][0]))
r_size = idx[0, 0]
c_size = int(l_2d / r_size)


x = df[:, 0].reshape(c_size, r_size)
print(x.shape)
y = df[:, 1].reshape(c_size, r_size)
print(y.shape)
z = df[:, 2].reshape(c_size, r_size)
print(z.shape)
v = df[:, 3].reshape(c_size, r_size).T

v = np.where(np.sqrt(x * x + y * y + z * z) <= 1560.8, float("NaN"), v)


plt.imshow(
    v,
    norm=mpl.colors.LogNorm(),
    origin="lower",
    interpolation="nearest",
    extent=[-3000, 3000, -3000, 3000],
)
plt.colorbar(extend="both")


plt.title("Europa 3D")
plt.xlabel("x (km)")
plt.ylabel("z (km)")

plt.xlim(-3000, 3000)
plt.ylim(-3000, 3000)


n, radii = 50, [0.01, 1560.8]
theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
xs = np.outer(radii, np.cos(theta))
ys = np.outer(radii, np.sin(theta))
xs[1, :] = xs[1, ::-1]
ys[1, :] = ys[1, ::-1]
plt.fill(np.ravel(xs), np.ravel(ys))


plt.show()

# %%
