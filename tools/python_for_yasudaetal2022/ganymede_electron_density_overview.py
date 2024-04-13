# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from matplotlib.colors import LogNorm

# %%
object_name = "ganymede"  # ganydeme/europa/calisto``
# ガニメデの開いた磁力線と閉じた磁力線の境界を示したデータ(deg)
# 0 pper_boundary_north_OCFL_Wlon	1 Upper_boundary_north_OCFL_lat	2 Lower_boundary_north_OCFL_Wlon	3 Lower_boundary_north_OCFL_lat	4 Lower_boundary_south_OCFL_Wlon
# 5 Lower_boundary_south_OCFL_lat	6 Upper_boundary_south_OCFL_Wlon
Data_name_csv = "../result_for_yasudaetal2022/ganymede_OCFL_boundary.csv"
OCFL_boudary_data = pd.read_csv(Data_name_csv, header=0)
print(OCFL_boudary_data.Upper_boundary_north_OCFL_Wlon)

# occultation color
occultation_detectable_longitude = [202.3, 23.1, 218]
occultation_detectable_latitude = [48.7, 50.3, -59]
occultation_detectable_density = [3200, 5000, 2000]

occultation_undetectable_longitude = [38]
occultation_undetectable_latitude = [20]

insitu_longitude = [113.4, 123.6]
# insitu_longitude_range = [10.95, 38.9]
insitu_latitude = [30.4, 79.3]
# insitu_latitude_range = [1.2, 3.6]
insitu_density = [100, 2500]

jovian_occultation_longitude = [263.55, 105.45]
jovian_occultation_longitude_range = [4.65, 5.35]
jovian_occultation_latitude = [25.6, 44.3]
jovian_occultation_latitude_range = [13.2, 4.7]
jovian_occultation_density = [12.5, 150]


fig, ax = plt.subplots()
ax.grid()
cm = plt.cm.get_cmap("rainbow")

#### 密度のプロット分布　#####
norm = LogNorm(vmin=10, vmax=5000)
mappable = ax.scatter(
    occultation_detectable_longitude,
    occultation_detectable_latitude,
    c=occultation_detectable_density,
    cmap=cm,
    marker="*",
    s=50,
    norm=norm,
)

ax.scatter(
    occultation_undetectable_longitude,
    occultation_undetectable_latitude,
    c="w",
    edgecolors="black",
    marker="*",
    s=50,
)

ax.scatter(
    insitu_longitude,
    insitu_latitude,
    c=insitu_density,
    cmap=cm,
    marker="o",
    alpha=1,
    norm=norm,
    s=30,
)
"""
ax.errorbar(
    insitu_longitude,
    insitu_latitude,
    xerr=insitu_longitude_range,
    yerr=insitu_latitude_range,
    capsize=4,
    fmt="none",
    ecolor="black",
    alpha=0.3,
)
"""
ax.scatter(
    jovian_occultation_longitude,
    jovian_occultation_latitude,
    c=jovian_occultation_density,
    cmap=cm,
    marker=",",
    alpha=1,
    norm=norm,
    s=40,
)

ax.errorbar(
    jovian_occultation_longitude,
    jovian_occultation_latitude,
    xerr=jovian_occultation_longitude_range,
    yerr=jovian_occultation_latitude_range,
    capsize=4,
    fmt="none",
    ecolor="black",
    alpha=0.3,
)

#### Open/Closed field line boudara（開いた磁力線領域・閉じた磁力線領域の境界線をひくコード）　#####


def plot_OFCL(lon, lat):
    ax.plot(lon, lat, linewidth=0.5, color="black", linestyle="--")


plot_OFCL(
    OCFL_boudary_data.Upper_boundary_north_OCFL_Wlon,
    OCFL_boudary_data.Upper_boundary_north_OCFL_lat,
)
plot_OFCL(
    OCFL_boudary_data.Lower_boundary_north_OCFL_Wlon,
    OCFL_boudary_data.Lower_boundary_north_OCFL_lat,
)

plot_OFCL(
    OCFL_boudary_data.Upper_boundary_south_OCFL_Wlon,
    OCFL_boudary_data.Upper_boundary_south_OCFL_lat,
)

plot_OFCL(
    OCFL_boudary_data.Lower_boundary_south_OCFL_Wlon,
    OCFL_boudary_data.Lower_boundary_south_OCFL_lat,
)


ax.text(
    OCFL_boudary_data.Upper_boundary_north_OCFL_Wlon[72] - 2,
    OCFL_boudary_data.Upper_boundary_north_OCFL_lat[72] + 3,
    "Open",
    fontsize=8,
)

ax.text(
    OCFL_boudary_data.Lower_boundary_north_OCFL_Wlon[72] - 2,
    OCFL_boudary_data.Lower_boundary_north_OCFL_lat[72] + 3,
    "Boundary region",
    fontsize=8,
)

ax.text(
    358,
    3,
    "Closed",
    fontsize=8,
)

ax.text(
    OCFL_boudary_data.Upper_boundary_south_OCFL_Wlon[72] - 2,
    OCFL_boudary_data.Upper_boundary_south_OCFL_lat[72] + 3,
    "Boundary region",
    fontsize=8,
)

ax.text(
    OCFL_boudary_data.Upper_boundary_south_OCFL_Wlon[72] - 2,
    OCFL_boudary_data.Upper_boundary_south_OCFL_lat[72] - 20,
    "Open",
    fontsize=8,
)

# 緯度経度指定
ax.set_xlim(0, 360)
ax.set_ylim(-90, 90)
ax.set_xticks(np.linspace(0, 360, 5))
ax.set_xticks(np.linspace(0, 360, 9), minor=True)
ax.set_xticklabels(["", "90W", "180W", "270W", ""])
ax.set_xlabel("Longitude (deg)")

ax.set_yticks(np.linspace(-90, 90, 5))
ax.set_yticks(np.linspace(-90, 90, 13), minor=True)
ax.set_yticklabels(["", "45S", "0", "45N", ""])
ax.set_ylabel("Latitude (deg)")
ax.set_aspect("equal")
ax.set_title("Ganymede ionospheric observation results")
ax.invert_xaxis()
# ax.legend(['Radio occultation (detection)', 'Radio occultation (no detectable)', 'In situ','Jovian radio occulatation'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
ax.legend(
    [
        "Radio occultation (detection)",
        "Radio occultation (no detectable)",
        "In situ",
        "Jovian radio occulatation",
    ],
    loc="lower right",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="6%", pad=0.48)
cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
cb.set_label("Maximum density (cm-3)")
fig.savefig(
    "../result_for_yasudaetal2022/observation_ppint_plot_for_paper/"
    + object_name
    + "_overview_plot.jpg",
    format="jpg",
    dpi=600,
)

plt.show()


# %%
