# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
# %%
object_name = 'ganymede'  # ganydeme/europa/calisto``

# occultation color
occultation_detectable_longitude = [202.3, 23.1, 218]
occultation_detectable_latitude = [48.7, 50.3, -59]
occultation_detectable_density = [3200, 5000, 2000]

occultation_undetectable_longitude = [38]
occultation_undetectable_latitude = [20]

insitu_longitude = [105.55, 87.8]
insitu_longitude_range = [13.15, 38.9]
insitu_latitude = [30.7, 76.3]
insitu_latitude_range = [1.2, 3.6]
insitu_density = [100, 2500]

jovian_occultation_longitude = [263.4, 106.15]
jovian_occultation_longitude_range = [4.5, 6.05]
jovian_occultation_latitude = [25.9, 45.1]
jovian_occultation_latitude_range = [12.8, 5.5]
jovian_occultation_density = [62.5, 175]


fig, ax = plt.subplots()


ax.grid()
cm = plt.cm.get_cmap('rainbow')
mappable = ax.scatter(occultation_detectable_longitude, occultation_detectable_latitude,
                      c=occultation_detectable_density, vmin=0, vmax=5000, cmap=cm, marker="*", s=100)

ax.scatter(occultation_undetectable_longitude, occultation_undetectable_latitude,
           c='w', edgecolors='black', marker="*", s=100)

ax.scatter(insitu_longitude, insitu_latitude, c=insitu_density,
           vmin=0, vmax=5000, cmap=cm, marker="o", alpha=1)
ax.errorbar(insitu_longitude, insitu_latitude, xerr=insitu_longitude_range,
            yerr=insitu_latitude_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax.scatter(jovian_occultation_longitude, jovian_occultation_latitude,
           c=jovian_occultation_density, vmin=0, vmax=5000, cmap=cm, marker=",", alpha=1)
ax.errorbar(jovian_occultation_longitude, jovian_occultation_latitude, xerr=jovian_occultation_longitude_range,
            yerr=jovian_occultation_latitude_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)


# 緯度経度指定
ax.set_xlim(0, 360)
ax.set_ylim(-90, 90)
ax.invert_xaxis()
ax.set_xticks(np.linspace(0, 360, 5))
ax.set_xticks(np.linspace(0, 360, 9), minor=True)
ax.set_xticklabels(["", "90W", "180W", "270W", ""])

ax.set_yticks(np.linspace(-90, 90, 5))
ax.set_yticks(np.linspace(-90, 90, 13), minor=True)
ax.set_yticklabels(["", "45S", "0", "45N", ""])
ax.set_aspect('equal')
plt.title('Detected electron density')
ax.legend(['Radio occultation (detection)', 'Radio occultation (no detectable)', 'In situ',
          'Jovian radio occulatation'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="10%", pad=0.3)
fig.colorbar(mappable, cax=cax, orientation="horizontal")
plt.show()


# %%