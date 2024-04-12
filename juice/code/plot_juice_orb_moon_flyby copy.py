#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import juice_lib
from astropy.constants import au
import sys
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from matplotlib import patches
from datetime import datetime

# In[2]:


# SPICE test
spice.tkvrsn("TOOLKIT")


# Import JUICE Lib
# In[3]:
# import JUICE lib
sys.path.append("/Users/yasudarikuto/research/icymoon_raytracing/juice/lib")
# Load SPICE kernels
import juice_lib

# In[4]:


# load SPICE ketnels
spice_dir = "/Users/yasudarikuto/spice/JUICE/kernels/"
juice_lib.spice_ini(spice_dir)


save_folder_path = "/Users/yasudarikuto/research/icymoon_raytracing/juice/results/"
# JUICE orbit near Moon

# In[5]:
# set date/time
utctim = "2024-08-19T20:00:00"  # start date/time
et_ex = spice.str2et(utctim)  # seconds
nd = 60 * 60 * 2  # number of data
dt = 1  # time step [second]
et = et_ex + dt * np.arange(0, nd)

# some parameters to calculate
rm = 1737.4  # km 月半径
re = 6400.0  # km 地球半径
full_theta = np.linspace(0, 2 * np.pi, 360)
half_theta = np.linspace(np.pi / 2, np.pi * 3 / 2, 180)

# select radio source type
"""
radio_x, radio_y, radio_z, type = (
    re * np.cos(full_theta),
    re * np.sin(full_theta),
    np.full(360, 2 * re),
    "(Re*cos(theta), Re*sin(theta), 2Re)",
)  # north auroral area

radio_x, radio_y, radio_z, type = (
    np.zeros(1),
    np.zeros(1),
    np.array([re]),
    "(0,0,Re)",
)  # north pole

radio_x, radio_y, radio_z, type = (
    np.zeros(1),
    np.zeros(1),
    np.array([-re]),
    "(0,0,Re)",
)  # south pole

radio_x, radio_y, radio_z, type = (
    re * np.cos(full_theta),
    re * np.sin(full_theta),
    np.full(360, -2 * re),
    "(Re*cos(theta), Re*sin(theta), -2Re)",
)  # south auroral area
"""
radio_x, radio_y, radio_z, type = (
    np.concatenate(
        [re * np.cos(full_theta), np.zeros(1), np.zeros(1), re * np.cos(full_theta)]
    ),
    np.concatenate(
        [re * np.sin(full_theta), np.zeros(1), np.zeros(1), re * np.sin(full_theta)]
    ),
    np.concatenate(
        [np.full(360, 2 * re), np.array([re]), np.array([-re]), np.full(360, -2 * re)]
    ),
    "all conditions",
)  # north auroral area

plt.scatter(
    radio_x / re,
    radio_y / re,
)
plt.title("Radio source xy (Re)")
plt.axis("equal")
plt.show()
plt.scatter(radio_x / re, radio_z / re)
plt.title("Radio source xz (Re)")
plt.axis("equal")
plt.show()

# calculate target position with spice
x_j_m, y_j_m, z_j_m, r_j_m, lat_j_m, lon_j_m = juice_lib.get_juice_pos_moon(
    et, x_ref="SUN"
)

# calculate target position with spice
x_e_m, y_e_m, z_e_m, r_e_m, lat_e_m, lon_e_m = juice_lib.get_earth_pos_moon(
    et, x_ref="SUN"
)

minimum_ind = np.argmin(r_j_m)


# In[6]:
# plot x-y figure
plt.plot(
    np.cos(full_theta),
    np.sin(full_theta),
    color="black",
    alpha=0.5,
    label="Filled Circle",
)
plt.fill(np.cos(half_theta), np.sin(half_theta), color="black", alpha=0.5)
plt.arrow(
    0,
    0,
    (x_e_m[int(len(x_e_m) / 2)] / r_e_m[int(len(r_e_m) / 2)]),
    (y_e_m[int(len(y_e_m) / 2)] / r_e_m[int(len(r_e_m) / 2)]),
    head_width=0.2,
    head_length=0.3,
    fc="red",
    ec="black",
)

plt.plot(x_j_m / rm, y_j_m / rm)  # x 地球から見た太陽方向
plt.scatter(x_j_m[minimum_ind] / rm, y_j_m[minimum_ind] / rm)
plt.text(
    x_j_m[minimum_ind] / rm,
    y_j_m[minimum_ind] / rm,
    "CA "
    + str(spice.et2datetime(et[minimum_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[minimum_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[minimum_ind]).second),
)
plt.title("JUICE orbit @ Moon: GSE X-Y")
plt.xlabel("GSE-X: Distance from Moon [Rm] (sun direction)")
plt.ylabel("GSE-Y: Distance from Moon [Rm]")
plt.text(
    x_j_m[0] / rm,
    y_j_m[0] / rm,
    str(spice.et2datetime(et[0]).hour)
    + ":"
    + str(spice.et2datetime(et[0]).minute)
    + ":"
    + str(spice.et2datetime(et[0]).second),
)
plt.text(
    x_j_m[-1] / rm,
    y_j_m[-1] / rm,
    str(spice.et2datetime(et[-1]).hour)
    + ":"
    + str(spice.et2datetime(et[-1]).minute)
    + ":"
    + str(spice.et2datetime(et[-1]).second),
)
plt.axis("equal")
plt.savefig(save_folder_path + "moon_flyby_xy.png")
plt.show()


# %%
# plot x-z figure
plt.plot(
    np.cos(full_theta),
    np.sin(full_theta),
    color="black",
    alpha=0.5,
    label="Filled Circle",
)
plt.fill(np.cos(half_theta), np.sin(half_theta), color="black", alpha=0.5)
plt.arrow(
    0,
    0,
    (x_e_m[int(len(x_e_m) / 2)] / r_e_m[int(len(r_e_m) / 2)]),
    (z_e_m[int(len(z_e_m) / 2)] / r_e_m[int(len(r_e_m) / 2)]),
    head_width=0.2,
    head_length=0.3,
    fc="red",
    ec="black",
)
plt.plot(x_j_m / rm, z_j_m / rm)  # x 地球から見た太陽方向
plt.scatter(x_j_m[minimum_ind] / rm, z_j_m[minimum_ind] / rm)
plt.text(
    x_j_m[minimum_ind] / rm,
    z_j_m[minimum_ind] / rm,
    "CA "
    + str(spice.et2datetime(et[minimum_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[minimum_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[minimum_ind]).second),
)
plt.title("JUICE orbit @ Moon: GSE X-Z")
plt.xlabel("GSE-X: Distance from Moon [Rm] / sun direction")
plt.ylabel("GSE-Z: Distance from Moon [Rm] / IAU_EARTH north")
plt.text(
    x_j_m[0] / rm,
    z_j_m[0] / rm,
    str(spice.et2datetime(et[0]).hour)
    + ":"
    + str(spice.et2datetime(et[0]).minute)
    + ":"
    + str(spice.et2datetime(et[0]).second),
)
plt.text(
    x_j_m[-1] / rm,
    z_j_m[-1] / rm,
    str(spice.et2datetime(et[-1]).hour)
    + ":"
    + str(spice.et2datetime(et[-1]).minute)
    + ":"
    + str(spice.et2datetime(et[-1]).second),
)
plt.axis("equal")
plt.savefig(save_folder_path + "moon_flyby_xz.png")
plt.show()

# %%
# plot t-r figure
plt.plot(et, r_j_m / rm)  # x 地球から見た太陽方向

plt.scatter(et[minimum_ind], r_j_m[minimum_ind] / rm)
plt.text(
    et[minimum_ind],
    r_j_m[minimum_ind] / rm,
    "CA "
    + str(spice.et2datetime(et[minimum_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[minimum_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[minimum_ind]).second),
)
plt.title("JUICE orbit @ Moon: distance - time")
plt.xlabel("Date")
plt.ylabel("Distance from Moon [Rm]")

x_values = [et[0], et[-1]]
x_labels = [
    str(spice.et2datetime(et[0]).hour)
    + ":"
    + str(spice.et2datetime(et[0]).minute)
    + ":"
    + str(spice.et2datetime(et[0]).second),
    str(spice.et2datetime(et[-1]).hour)
    + ":"
    + str(spice.et2datetime(et[-1]).minute)
    + ":"
    + str(spice.et2datetime(et[-1]).second),
]
plt.xticks(x_values, x_labels)
plt.savefig(save_folder_path + "moon_flyby_tr.png")
plt.show()


# JUICE orbit near Moon caluculate tangential distance ()
# In[5]:
# calculate target position with spice
x_j, y_j, z_j, r_j, lat_j, lon_j = juice_lib.get_juice_pos_earth(et, x_ref="SUN")
x_m, y_m, z_m, r_m, lat_m, lon_m = juice_lib.get_moon_pos_earth(et, x_ref="SUN")


start_occulataion_ind_arr = np.zeros(len(radio_x))
end_occulataion_ind_arr = np.zeros(len(radio_x))
for j in range(len(radio_x)):
    tangential_dis = np.zeros(len(x_j))
    tangential_dis_abs = np.zeros(len(x_j))
    for i in range(len(x_j)):
        r_radio2juice = np.array([x_j[i], y_j[i], z_j[i]]) - np.array(
            [radio_x[j], radio_y[j], radio_z[j]]
        )
        r_radio2moon = np.array([x_m[i], y_m[i], z_m[i]]) - np.array(
            [radio_x[j], radio_y[j], radio_z[j]]
        )
        tangential_cross = np.cross(r_radio2juice, r_radio2moon)
        vn = np.linalg.norm(tangential_cross) / np.linalg.norm(r_radio2juice)
        tangential_dis[i] = vn - rm
        tangential_dis_abs[i] = abs(vn - rm)
    closest_ind = np.argmin(tangential_dis)

    if tangential_dis[closest_ind] < 0:
        # print(tangential_dis_abs[np.argmin(tangential_dis_abs[:closest_ind])])
        # print(tangential_dis_abs[closest_ind + np.argmin(tangential_dis_abs[closest_ind:])])
        start_occulataion_ind_arr[j] = np.argmin(tangential_dis_abs[:closest_ind])
        end_occulataion_ind_arr[j] = closest_ind + np.argmin(
            tangential_dis_abs[closest_ind:]
        )
    else:
        start_occulataion_ind_arr[j] = np.nan
        end_occulataion_ind_arr[j] = np.nan

start_penumbra_ind = int(np.min(start_occulataion_ind_arr))
start_umbra_ind = int(np.max(start_occulataion_ind_arr))
end_umbra_ind = int(np.min(end_occulataion_ind_arr))
end_penumbra_ind = int(np.max(end_occulataion_ind_arr))

# %%
# plot
# x&y plot with occultation
plt.plot(
    np.cos(full_theta),
    np.sin(full_theta),
    color="black",
    alpha=0.5,
    label="Filled Circle",
)
plt.fill(np.cos(half_theta), np.sin(half_theta), color="black", alpha=0.5)
plt.arrow(
    0,
    0,
    (x_e_m[int(len(x_e_m) / 2)] / r_e_m[int(len(r_e_m) / 2)]),
    (y_e_m[int(len(y_e_m) / 2)] / r_e_m[int(len(r_e_m) / 2)]),
    head_width=0.2,
    head_length=0.3,
    fc="red",
    ec="black",
)


plt.plot(x_j_m / rm, y_j_m / rm)  # x 地球から見た太陽方向
plt.scatter(x_j_m[minimum_ind] / rm, y_j_m[minimum_ind] / rm, c="black")
plt.text(
    x_j_m[minimum_ind] / rm,
    y_j_m[minimum_ind] / rm,
    "CA "
    + str(spice.et2datetime(et[minimum_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[minimum_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[minimum_ind]).second),
    c="black",
)

plt.scatter(
    x_j_m[start_penumbra_ind] / rm, y_j_m[start_penumbra_ind] / rm, c="orangered"
)
plt.text(
    x_j_m[start_penumbra_ind] / rm,
    y_j_m[start_penumbra_ind] / rm,
    "Penumbra"
    + str(spice.et2datetime(et[start_penumbra_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[start_penumbra_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[start_penumbra_ind]).second),
    c="orangered",
)

plt.scatter(x_j_m[start_umbra_ind] / rm, y_j_m[start_umbra_ind] / rm, c="red")
plt.text(
    x_j_m[start_umbra_ind] / rm,
    y_j_m[start_umbra_ind] / rm + 0.4,
    "Umbra "
    + str(spice.et2datetime(et[start_umbra_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[start_umbra_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[start_umbra_ind]).second),
    c="red",
)

plt.scatter(x_j_m[end_penumbra_ind] / rm, y_j_m[end_penumbra_ind] / rm, c="orangered")
plt.text(
    x_j_m[end_penumbra_ind] / rm,
    y_j_m[end_penumbra_ind] / rm,
    "Penumbra "
    + str(spice.et2datetime(et[end_penumbra_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[end_penumbra_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[end_penumbra_ind]).second),
    c="orangered",
)

plt.scatter(x_j_m[end_umbra_ind] / rm, y_j_m[end_umbra_ind] / rm, c="red")
plt.text(
    x_j_m[end_umbra_ind] / rm,
    y_j_m[end_umbra_ind] / rm - 0.4,
    "Umbra "
    + str(spice.et2datetime(et[end_umbra_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[end_umbra_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[end_umbra_ind]).second),
    c="red",
)

plt.title("JUICE orbit @ Moon: GSE X-Y " + type)
plt.xlabel("GSE-X: Distance from Moon [Rm] (sun direction)")
plt.ylabel("GSE-Y: Distance from Moon [Rm]")
plt.text(
    x_j_m[0] / rm,
    y_j_m[0] / rm,
    str(spice.et2datetime(et[0]).hour)
    + ":"
    + str(spice.et2datetime(et[0]).minute)
    + ":"
    + str(spice.et2datetime(et[0]).second),
)
plt.text(
    x_j_m[-1] / rm,
    y_j_m[-1] / rm,
    str(spice.et2datetime(et[-1]).hour)
    + ":"
    + str(spice.et2datetime(et[-1]).minute)
    + ":"
    + str(spice.et2datetime(et[-1]).second),
)
plt.axis("equal")
plt.savefig(save_folder_path + type + "moon_flyby_xy_.png")
plt.show()

# %%
# plot x&z with occulataion
plt.plot(
    np.cos(full_theta),
    np.sin(full_theta),
    color="black",
    alpha=0.5,
    label="Filled Circle",
)
plt.fill(np.cos(half_theta), np.sin(half_theta), color="black", alpha=0.5)
plt.arrow(
    0,
    0,
    (x_e_m[int(len(x_e_m) / 2)] / r_e_m[int(len(r_e_m) / 2)]),
    (z_e_m[int(len(z_e_m) / 2)] / r_e_m[int(len(r_e_m) / 2)]),
    head_width=0.2,
    head_length=0.3,
    fc="red",
    ec="black",
)
plt.plot(x_j_m / rm, z_j_m / rm)  # x 地球から見た太陽方向
plt.scatter(x_j_m[minimum_ind] / rm, z_j_m[minimum_ind] / rm, c="black")
plt.text(
    x_j_m[minimum_ind] / rm,
    z_j_m[minimum_ind] / rm,
    "CA "
    + str(spice.et2datetime(et[minimum_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[minimum_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[minimum_ind]).second),
    c="black",
)

plt.scatter(
    x_j_m[start_penumbra_ind] / rm, z_j_m[start_penumbra_ind] / rm, c="orangered"
)
plt.text(
    x_j_m[start_penumbra_ind] / rm,
    z_j_m[start_penumbra_ind] / rm - 0.4,
    "Penumbra"
    + str(spice.et2datetime(et[start_penumbra_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[start_penumbra_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[start_penumbra_ind]).second),
    c="orangered",
)

plt.scatter(x_j_m[start_umbra_ind] / rm, z_j_m[start_umbra_ind] / rm, c="red")
plt.text(
    x_j_m[start_umbra_ind] / rm,
    z_j_m[start_umbra_ind] / rm - 0.8,
    "Umbra "
    + str(spice.et2datetime(et[start_umbra_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[start_umbra_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[start_umbra_ind]).second),
    c="red",
)

plt.scatter(x_j_m[end_penumbra_ind] / rm, z_j_m[end_penumbra_ind] / rm, c="orangered")
plt.text(
    x_j_m[end_penumbra_ind] / rm,
    z_j_m[end_penumbra_ind] / rm - 0.4,
    "Penumbra "
    + str(spice.et2datetime(et[end_penumbra_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[end_penumbra_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[end_penumbra_ind]).second),
    c="orangered",
)

plt.scatter(x_j_m[end_umbra_ind] / rm, z_j_m[end_umbra_ind] / rm, c="red")
plt.text(
    x_j_m[end_umbra_ind] / rm,
    z_j_m[end_umbra_ind] / rm - 0.8,
    "Umbra "
    + str(spice.et2datetime(et[end_umbra_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[end_umbra_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[end_umbra_ind]).second),
    c="red",
)
plt.title("JUICE orbit @ Moon: GSE X-Z " + type)
plt.xlabel("GSE-X: Distance from Moon [Rm] / sun direction")
plt.ylabel("GSE-Z: Distance from Moon [Rm] / IAU_EARTH north")
plt.text(
    x_j_m[0] / rm,
    z_j_m[0] / rm,
    str(spice.et2datetime(et[0]).hour)
    + ":"
    + str(spice.et2datetime(et[0]).minute)
    + ":"
    + str(spice.et2datetime(et[0]).second),
)
plt.text(
    x_j_m[-1] / rm,
    z_j_m[-1] / rm,
    str(spice.et2datetime(et[-1]).hour)
    + ":"
    + str(spice.et2datetime(et[-1]).minute)
    + ":"
    + str(spice.et2datetime(et[-1]).second),
)
plt.axis("equal")
plt.savefig(save_folder_path + type + "moon_flyby_xz_.png")
plt.show()

# %%
# t&r plot with occultation
plt.plot(et, r_j_m / rm)  # x 地球から見た太陽方向

plt.scatter(et[minimum_ind], r_j_m[minimum_ind] / rm)
"""
plt.text(
    et[minimum_ind],
    r_j_m[minimum_ind] / rm,
    "CA "
    + str(spice.et2datetime(et[minimum_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[minimum_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[minimum_ind]).second),
)
"""

plt.scatter(et[start_penumbra_ind], r_j_m[start_penumbra_ind] / rm, c="orangered")
plt.text(
    et[start_penumbra_ind],
    r_j_m[start_penumbra_ind] / rm - 0.4,
    "Penumbra"
    + str(spice.et2datetime(et[start_penumbra_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[start_penumbra_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[start_penumbra_ind]).second),
    c="orangered",
)

plt.scatter(et[start_umbra_ind], r_j_m[start_umbra_ind] / rm, c="red")
plt.text(
    et[start_umbra_ind],
    r_j_m[start_umbra_ind] / rm - 0.8,
    "Umbra "
    + str(spice.et2datetime(et[start_umbra_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[start_umbra_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[start_umbra_ind]).second),
    c="red",
)

plt.scatter(et[end_penumbra_ind], r_j_m[end_penumbra_ind] / rm, c="orangered")
plt.text(
    et[end_penumbra_ind],
    r_j_m[end_penumbra_ind] / rm - 0.8,
    "Penumbra "
    + str(spice.et2datetime(et[end_penumbra_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[end_penumbra_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[end_penumbra_ind]).second),
    c="orangered",
)

plt.scatter(et[end_umbra_ind], r_j_m[end_umbra_ind] / rm, c="red")
plt.text(
    et[end_umbra_ind],
    r_j_m[end_umbra_ind] / rm - 0.4,
    "Umbra "
    + str(spice.et2datetime(et[end_umbra_ind]).hour)
    + ":"
    + str(spice.et2datetime(et[end_umbra_ind]).minute)
    + ":"
    + str(spice.et2datetime(et[end_umbra_ind]).second),
    c="red",
)
plt.title("JUICE orbit @ Moon: distance - time " + type)
plt.xlabel("Date")
plt.ylabel("Distance from Moon [Rm]")

x_values = [et[0], et[-1]]
x_labels = [
    str(spice.et2datetime(et[0]).hour)
    + ":"
    + str(spice.et2datetime(et[0]).minute)
    + ":"
    + str(spice.et2datetime(et[0]).second),
    str(spice.et2datetime(et[-1]).hour)
    + ":"
    + str(spice.et2datetime(et[-1]).minute)
    + ":"
    + str(spice.et2datetime(et[-1]).second),
]
plt.xticks(x_values, x_labels)
plt.savefig(save_folder_path + type + "moon_flyby_tr_.png")
plt.show()
