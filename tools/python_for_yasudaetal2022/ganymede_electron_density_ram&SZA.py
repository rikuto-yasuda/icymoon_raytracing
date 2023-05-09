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

# occultation open magnetic field
occultation_open_density = [3200, 5000, 2000]
occultation_open_ram = [76.9, 103.1, 72]
occultation_open_sza = [82.8, 97.55, 95]

# occultation closed magnetic field
occultation_closed_ram = [125]
occultation_closed_sza = [80]
occultation_closed_density = [500]
occultation_closed_density_range = [500]

# in situ open magnetic field
insitu_density = [100, 2500]
insitu_ram = [144.6, 100.75]
insitu_ram_range = [3.3, 1.75]
insitu_sza = [68.4, 90.85]
insitu_sza_range = [9.4, 7.95]

# jovian occultation open
jovian_occultation_open_density = [175]
jovian_occultation_open_ram = [133.40]
jovian_occultation_open_ram_range = [6.60]
jovian_occultation_open_sza = [73.50]
jovian_occultation_open_sza_range = [6.00]

# jovian occultation closed
jovian_occultation_closed_density = [62.5]
jovian_occultation_closed_ram = [27.20]
jovian_occultation_closed_ram_range = [13.30]
jovian_occultation_closed_sza = [93.70]
jovian_occultation_closed_sza_range = [4.50]

# %%
fig, ax = plt.subplots(figsize=(9.0, 5.0))


ax.grid()
cm = plt.cm.get_cmap('rainbow')
mappable = ax.scatter(occultation_open_ram, occultation_open_density,
                      c='red', marker="*", s=100)

ax.scatter(occultation_closed_ram, occultation_closed_density,
           c='blue', edgecolors='black', marker="*", s=100)
ax.errorbar(occultation_closed_ram, occultation_closed_density,
            yerr=occultation_closed_density_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax.scatter(insitu_ram, insitu_density, c='red', marker="o")
ax.errorbar(insitu_ram, insitu_density, xerr=insitu_ram_range,
            capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax.scatter(jovian_occultation_open_ram,
           jovian_occultation_open_density, c='red', marker=",", alpha=1)
ax.errorbar(jovian_occultation_open_ram, jovian_occultation_open_density,
            xerr=jovian_occultation_open_ram_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax.scatter(jovian_occultation_closed_ram,
           jovian_occultation_closed_density, c='blue', marker=",", alpha=1)
ax.errorbar(jovian_occultation_closed_ram, jovian_occultation_closed_density,
            xerr=jovian_occultation_closed_ram_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)


# 緯度経度指定
ax.set_xlim(0, 180)
ax.set_xticks(np.linspace(0, 180, 5))
ax.set_xticklabels(["0", "45", "90", "135", "180"])

ax.set_yticks(np.linspace(0, 6000, 7))
ax.set_yticklabels(["0", "1000", "2000", "3000", "4000", "5000", "6000"])
plt.title('Ganymede ionospheric observation results')
ax.legend(['Radio occultation (open)', 'Radio occultation (cloesd & no detectable)', 'In situ (open)',
          'Jovian radio occulatation (open)', 'Jovian radio occulatation (closed)'], loc='upper left')
divider = make_axes_locatable(ax)
ax.set_xlabel('Ram angle (deg)')
ax.set_ylabel('Maximum density (cm-3)')
plt.savefig("../result_for_yasudaetal2022/observation_ppint_plot_for_paper/" +
            object_name+"_Ram_angle_plot.jpg", format="jpg", dpi=600)

plt.show()

# %%
fig, ax = plt.subplots(figsize=(9.0, 5.0))


ax.grid()
cm = plt.cm.get_cmap('rainbow')
mappable = ax.scatter(occultation_open_sza, occultation_open_density,
                      c='red', marker="*", s=100)

ax.scatter(occultation_closed_sza, occultation_closed_density,
           c='blue', edgecolors='black', marker="*", s=100)
ax.errorbar(occultation_closed_sza, occultation_closed_density,
            yerr=occultation_closed_density_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax.scatter(insitu_sza, insitu_density, c='red', marker="o")
ax.errorbar(insitu_sza, insitu_density, xerr=insitu_sza_range,
            capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax.scatter(jovian_occultation_open_sza,
           jovian_occultation_open_density, c='red', marker=",", alpha=1)
ax.errorbar(jovian_occultation_open_sza, jovian_occultation_open_density,
            xerr=jovian_occultation_open_sza_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax.scatter(jovian_occultation_closed_sza,
           jovian_occultation_closed_density, c='blue', marker=",", alpha=1)
ax.errorbar(jovian_occultation_closed_sza, jovian_occultation_closed_density,
            xerr=jovian_occultation_closed_sza_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)


# 緯度経度指定
ax.set_xlim(0, 180)
ax.set_xticks(np.linspace(0, 180, 5))
ax.set_xticklabels(["0", "45", "90", "135", "180"])

ax.set_yticks(np.linspace(0, 6000, 7))
ax.set_yticklabels(["0", "1000", "2000", "3000", "4000", "5000", "6000"])
plt.title('Ganymede ionospheric observation results')
ax.legend(['Radio occultation (open)', 'Radio occultation (cloesd & no detectable)', 'In situ (open)',
          'Jovian radio occulatation (open)', 'Jovian radio occulatation (closed)'], loc='upper left')
divider = make_axes_locatable(ax)
ax.set_xlabel('SZA angle (deg)')
ax.set_ylabel('Maximum density (cm-3)')
plt.savefig("../result_for_yasudaetal2022/observation_ppint_plot_for_paper/" +
            object_name+"_SZA_angle_plot.jpg", format="jpg", dpi=600)
plt.show()
# %%
fig, ax = plt.subplots(3, 1, figsize=(7, 7))

ax[0].scatter(occultation_open_ram, occultation_open_density,
              c='red', marker="*", s=100)

ax[0].scatter(occultation_closed_ram, occultation_closed_density,
              c='blue', edgecolors='black', marker="*", s=100)
ax[0].errorbar(occultation_closed_ram, occultation_closed_density,
               yerr=occultation_closed_density_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax[0].set_xlim(0, 180)
ax[0].set_title('Radio occultation', fontsize=10)
ax[0].legend(['Open magnetic field', 'Closed magnetic field'],
             loc='upper left')


ax[1].scatter(insitu_sza, insitu_density, c='red', marker="o")
ax[1].errorbar(insitu_sza, insitu_density, xerr=insitu_sza_range,
               capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax[1].set_title('In situ', fontsize=10)
ax[1].set_xlim(0, 180)


ax[2].scatter(jovian_occultation_open_ram,
              jovian_occultation_open_density, c='red', marker=",", alpha=1)
ax[2].errorbar(jovian_occultation_open_ram, jovian_occultation_open_density,
               xerr=jovian_occultation_open_ram_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)


ax[2].scatter(jovian_occultation_closed_ram,
              jovian_occultation_closed_density, c='blue', marker=",", alpha=1)
ax[2].errorbar(jovian_occultation_closed_ram, jovian_occultation_closed_density,
               xerr=jovian_occultation_closed_ram_range, capsize=4, fmt='none', ecolor='black', alpha=0.3)

ax[2].set_title('Jovian radio occultation', fontsize=10)
ax[2].set_xlim(0, 180)
ax[2].legend(['Open magnetic field', 'Closed magnetic field'],
             loc='upper left')

fig.supxlabel('ram angle (deg)')
fig.supylabel('Electron density (cm-3)')

fig.subplots_adjust(left=0.13)
fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(bottom=0.07)

plt.show()

# %%
