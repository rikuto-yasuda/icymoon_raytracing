# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
# %%
object_name = 'callisto'  # ganydeme/europa/calisto``

# occultation magnetic field
occultation_detection_density = [4300, 5100, 15300, 8500, 17400, 3000]
occultation_detection_density_range = [440, 3300, 2300, 17000, 1500, 1600]
occultation_detection_ram = [82.8, 97.8, 80.9, 99.2, 81.7, 98.8]
occultation_detection_sza = [85.0, 94.0, 78.7, 101.3, 82.5, 97.6]

# occultation closed magnetic field
occultation_nodetection_ram = [105.7, 74.4]
occultation_nodetection_sza = [81.5, 98.5]
occultation_nodetection_density = [500, 500]
occultation_nodetection_density_range = [500, 500]

# in situ detection magnetic field
insitu_detection_density = [100, 400]
insitu_detection_ram = [120.7, 147.55]
insitu_detection_ram_range = [3.5, 27.95]
insitu_detection_sza = [85.95, 35.6]
insitu_detection_sza_range = [3.55, 31.2]

# in situ detection magnetic field
insitu_nodetection_density = [0]
insitu_nodetection_ram = [147.65]
insitu_nodetection_ram_range = [30.15]
insitu_nodetection_sza = [149.65]
insitu_nodetection_sza_range = [29.85]

# jovian occultation
jovian_occultation_density = [362.5, 2100, 375]

jovian_occultation_ram = [167.05, 18.55, 4.75]
jovian_occultation_ram_range = [3.65, 9.85, 3.05]
jovian_occultation_sza = [106.3, 70.6, 173.75]
jovian_occultation_sza_range = [3.9, 4, 2.85]

fig, ax = plt.subplots(figsize=(9.0, 5.0))


ax.grid()
cm = plt.cm.get_cmap('rainbow')
mappable = ax.scatter(occultation_detection_ram, occultation_detection_density,
                      c='red', marker="*", s=100)
ax.errorbar(occultation_detection_ram, occultation_detection_density,
            yerr=occultation_detection_density_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax.scatter(occultation_nodetection_ram, occultation_nodetection_density,
           c='blue', edgecolors='black', marker="*", s=100)
ax.errorbar(occultation_nodetection_ram, occultation_nodetection_density,
            yerr=occultation_nodetection_density_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)

"""
ax.scatter(insitu_detection_ram, insitu_detection_density, c='red', marker="o")
ax.errorbar(insitu_detection_ram, insitu_detection_density, xerr=insitu_detection_ram_range,
            capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax.scatter(insitu_nodetection_ram,
           insitu_nodetection_density, c='blue', marker="o")
ax.errorbar(insitu_nodetection_ram, insitu_nodetection_density, xerr=insitu_nodetection_ram_range,
            capsize=4, fmt='none', ecolor='black', alpha=0.3)
"""

ax.scatter(jovian_occultation_ram,
           jovian_occultation_density, c='red', marker=",", alpha=1)
ax.errorbar(jovian_occultation_ram, jovian_occultation_density,
            xerr=jovian_occultation_ram_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)


# 緯度経度指定
ax.set_xlim(0, 180)
ax.set_xticks(np.linspace(0, 180, 5))
ax.set_xticklabels(["0", "45", "90", "135", "180"])

#ax.set_yticks(np.linspace(0, 20000, 5))
#ax.set_yticklabels(["0", "5000", "10000", "15000", "20000"])
plt.title('Detected electron density')
ax.legend(['Radio occultation (detection)', 'Radio occultation (no detection)', 'In situ (detection)',
          'In situ (no detection)', 'Jovian radio occulatation'], loc='upper left')
divider = make_axes_locatable(ax)
ax.set_xlabel('Ram angle (deg)')
ax.set_ylabel('Electron density (/cc)')
plt.show()
# %%

fig, ax = plt.subplots(figsize=(9.0, 5.0))


ax.grid()
cm = plt.cm.get_cmap('rainbow')
mappable = ax.scatter(occultation_detection_sza, occultation_detection_density,
                      c='red', marker="*", s=100)
ax.errorbar(occultation_detection_sza, occultation_detection_density,
            yerr=occultation_detection_density_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax.scatter(occultation_nodetection_sza, occultation_nodetection_density,
           c='blue', edgecolors='black', marker="*", s=100)
ax.errorbar(occultation_nodetection_sza, occultation_nodetection_density,
            yerr=occultation_nodetection_density_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)

"""
ax.scatter(insitu_detection_sza, insitu_detection_density, c='red', marker="o")
ax.errorbar(insitu_detection_sza, insitu_detection_density, xerr=insitu_detection_sza_range,
            capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax.scatter(insitu_nodetection_sza,
           insitu_nodetection_density, c='blue', marker="o")
ax.errorbar(insitu_nodetection_sza, insitu_nodetection_density, xerr=insitu_nodetection_sza_range,
            capsize=4, fmt='none', ecolor='black', alpha=0.3)
"""

ax.scatter(jovian_occultation_sza,
           jovian_occultation_density, c='red', marker=",", alpha=1)
ax.errorbar(jovian_occultation_sza, jovian_occultation_density,
            xerr=jovian_occultation_sza_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)

# 緯度経度指定
ax.set_xlim(0, 180)
ax.set_xticks(np.linspace(0, 180, 5))
ax.set_xticklabels(["0", "45", "90", "135", "180"])
plt.title('Detected electron density')
#ax.legend(['Radio occultation (detection)', 'Radio occultation (no detection)','In situ (detection)', 'In situ (no detection)', 'Jovian radio occulatation'], loc='upper left')
ax.legend(['Radio occultation (detection)', 'Radio occultation (no detection)',
          'Jovian radio occulatation'], loc='upper left')
divider = make_axes_locatable(ax)
ax.set_yscale("log")
ax.set_xlabel('SZA angle (deg)')
ax.set_ylabel('Electron density (/cc)')
plt.show()
# %%

fig, ax = plt.subplots(3, 1, figsize=(7, 7))


ax[0].scatter(occultation_detection_sza, occultation_detection_density,
              c='red', marker="*", s=100)
ax[0].errorbar(occultation_detection_sza, occultation_detection_density,
               yerr=occultation_detection_density_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax[0].scatter(occultation_nodetection_sza, occultation_nodetection_density,
              c='blue', edgecolors='black', marker="*", s=100)
ax[0].errorbar(occultation_nodetection_sza, occultation_nodetection_density,
               yerr=occultation_nodetection_density_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)
ax[0].set_xlim(0, 180)
ax[0].set_title('Radio occultation', fontsize=10)

ax[0].legend(['Detection', 'No detection'], loc='upper right')

"""
ax[1].scatter(insitu_detection_sza,
              insitu_detection_density, c='red', marker="o")
ax[1].errorbar(insitu_detection_sza, insitu_detection_density, xerr=insitu_detection_sza_range,
               capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax[1].scatter(insitu_nodetection_sza,
              insitu_nodetection_density, c='blue', marker="o")
ax[1].errorbar(insitu_nodetection_sza, insitu_nodetection_density, xerr=insitu_nodetection_sza_range,
               capsize=4, fmt='none', ecolor='black', alpha=0.3)
"""
ax[1].set_title('In situ', fontsize=10)
ax[1].set_xlim(0, 180)

ax[1].legend(['Detection', 'No detection'], loc='upper right')

ax[2].scatter(jovian_occultation_sza,
              jovian_occultation_density, c='red', marker=",", alpha=1)
ax[2].errorbar(jovian_occultation_sza, jovian_occultation_density,
               xerr=jovian_occultation_sza_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)
ax[2].set_title('Jovian radio occultation', fontsize=10)
ax[2].set_xlim(0, 180)
fig.supxlabel('SZA angle (deg)')
fig.supylabel('Electron density (/cc)')

fig.subplots_adjust(left=0.13)
fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(bottom=0.07)

plt.show()
# %%
