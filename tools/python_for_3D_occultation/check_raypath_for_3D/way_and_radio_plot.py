# %%
import pprint
import cdflib
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import datetime
from multiprocessing import Pool
import matplotlib.colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from matplotlib.colors import LogNorm

# In[]
object_name = "europa_plume"  # ganydeme/
highest_plasma = "3e8"  # 単位は(/cc) 2e2/4e2/16e2
plasma_scaleheight = "2e7"  # 単位は(km) 1.5e2/3e2/6e2
Freq_str = "3.984813988208770752e5"

plot_hight_array = np.arange(1500, 2500, 10)

t = np.arange(-1560.8, 1560.8)
c = np.sqrt(1560.8 * 1560.8 - t * t)
n = t * 0
plt.plot(t, c, color="black", linewidth=0.0001)
plt.fill_between(t, c, n, facecolor="black")


plt.xlabel("x (km)")
plt.ylabel("z (km)")

plt.xlim(-5000, 5000)
plt.ylim(1000, 2500)

for i in plot_hight_array:
    filename = np.genfromtxt(
        "./ray_path_result/ray-P"
        + object_name
        + "_"
        + highest_plasma
        + "_"
        + plasma_scaleheight
        + "-Mtest_simple-benchmark-LO-Z"
        + str(i)
        + "-FR"
        + Freq_str
    )
    x1 = filename[:, [1]]
    z1 = filename[:, [3]]
    plt.plot(x1, z1, color="red", linewidth=0.5)
plt.show()


# %%
