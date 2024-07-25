#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import GLL_lib
from astropy.constants import au
import sys
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from matplotlib import patches
from mpl_toolkits import mplot3d


# In[2]:


# load SPICE ketnels
spice.tkvrsn("TOOLKIT")
sys.path.append("../lib")
import GLL_lib
spice_dir = "/Users/yasudarikuto/spice/GLL/kernels/"
GLL_lib.spice_ini(spice_dir)


########################################################################
# JUICE orbit near Europa
# In[5]:
# set date/time
utctim = "1997-12-16T11:40:00"  # start date/time
et_ex = spice.str2et(utctim)  # seconds
nd = 60 * 50  # number of data 
dt = 1  # time step [second]
et = et_ex + dt * np.arange(0, nd)
set_y_manually = True
# calculate target position with spice
x, y, z, r, lat, lon = GLL_lib.get_GLL_pos_claire_definition(et, y_ref="JUPITER")


# set ray path parameters
path_name = "/Users/yasudarikuto/research/icymoon_raytracing/tools/python_for_3D_occultation/raytracing_results/unsymmetry_2_variables_diffusion_fitting_results/"
Freq_str = "6.510338783264160156e5"

initial_ray_x_array = np.arange(-3200, 3200, 100)
initial_ray_z_array = np.arange(-3200, 3200, 100)


########################################################################
# intersection calculation
def Calculation_intersection(x,y,intersect_x):
    # calculate the intersection of the y position of ray path at intersect_x
    # x: x-coordinate of the ray path
    # y: y-coordinate of the ray path
    # intersect_x: x-coordinate of the xz-plane
    # return: y-coordinate of the intersection

    # calculate the slope of the ray path
    if x[0]>intersect_x and x[-1]<intersect_x:
        x_before_intersection_pos = np.where(x>intersect_x)[0][-1]

        x_before_intersection = x[x_before_intersection_pos]
        x_after_intersection = x[x_before_intersection_pos+1]
        y_before_intersection = y[x_before_intersection_pos]
        y_after_intersection = y[x_before_intersection_pos+1]

        x_distance_ray_step = np.abs(x_after_intersection - x_before_intersection)
        x_distance_to_intersect = np.abs(intersect_x - x_before_intersection)

        y_intersection = y_before_intersection + (y_after_intersection - y_before_intersection) * (x_distance_to_intersect / x_distance_ray_step)

    else:
        y_intersection = np.nan

    return y_intersection


########################################################################
#　plot GLL orbit simply
# In[6]:
# plot
re = 1560.8  # km
full_theta = np.linspace(0, 2 * np.pi, 360)
plt.plot(x / re, y / re)  # x 地球から見た太陽方向
plt.plot(
    np.cos(full_theta),
    np.sin(full_theta),
    color="black",
    alpha=0.5,
    label="Filled Circle",
)
plt.title("GLL orbit @ Europa")
plt.xlabel("X: Distance from Europa [Re] (Traling direction)")
plt.ylabel("Y: Distance from Europa [Re] (Jupiter direction)")
plt.text(int(x[0] / re), int(y[0] / re), utctim)
plt.text(int(x[-1] / re), int(y[-1] / re), spice.et2datetime(et[-1]))
plt.axis("equal")
plt.show()

# %%
plt.plot(x / re, z / re)  # x 地球から見た太陽方向
plt.plot(
    np.cos(full_theta),
    np.sin(full_theta),
    color="black",
    alpha=0.5,
    label="Filled Circle",
)
plt.title("GLL orbit @ Europa")
plt.xlabel("X: Distance from Europa [Re] (Traling direction)")
plt.ylabel("Z: Distance from Europa [Re] (North direction)")
plt.text(int(x[0] / re), int(z[0] / re), utctim)
plt.text(int(x[-1] / re), int(z[-1] / re), spice.et2datetime(et[-1]))
plt.axis("equal")
plt.show()


########################################################################
#　plot GLL orbit and ray path in 3D
# %%
# 3Dプロットを作成する
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# エウロパ球をプロットする
# 球のパラメータ
radius = 1  # 球の半径
center = (0, 0, 0)  # 球の中心座標

# 球面上の点の数
num_points = 100

# 球面上の点の座標を計算する
u = np.linspace(0, 2 * np.pi, num_points)
v = np.linspace(0, np.pi, num_points)
x_surface = center[0] + radius * np.outer(np.cos(u), np.sin(v))
y_surface = center[1] + radius * np.outer(np.sin(u), np.sin(v))
z_surfece = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

# 3Dプロットを作成する
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 球体をプロットする
ax.plot_surface(x_surface, y_surface, z_surfece, color='black', alpha=0.2)


# 軌道データをプロットする
ax.scatter(x / re, y / re, z /re , c='r', marker='o', s=0.05)

# レイパスをプロットする
for initial_ray_z in initial_ray_z_array:
    for initial_ray_x in initial_ray_x_array:
        filename = np.genfromtxt(
            path_name
            + "ray-Pclaire_240507-Mtest_simple-benchmark-LO-X"
            + str(initial_ray_x)
            + "-Z"
            + str(initial_ray_z)
            + "-FR"
            + Freq_str
        )
        ray_x = filename[:, [1]]
        ray_y = filename[:, [2]]
        ray_z = filename[:, [3]]

        ax.scatter(ray_x / re, ray_y / re, ray_z /re , c='b', marker='o',s=0.01)

# 軸ラベルを設定する
ax.set_xlabel('X (Traling direction)')
ax.set_ylabel('Y (Jupiter direction)')
ax.set_zlabel('Z (North)')

# プロット範囲を指定する
ax.set_xlim([-3, 3])  # X軸の範囲
ax.set_ylim([-3, 3])  # Y軸の範囲
ax.set_zlim([-3, 3])  # Z軸の範囲

# グラフを表示する
plt.show()

# %%
########################################################################
#　plot GLL orbit and ray path in 2D
# set the xz-plane to plot from orbit data (Re)
# set date/time
utctim = "1997-12-16T11:00:00"  # start date/time
et_ex = spice.str2et(utctim)  # seconds
nd = 60 * 60 * 2  # number of data 
dt = 1  # time step [second]
et = et_ex + dt * np.arange(0, nd)

# calculate target position with spice
x, y, z, r, lat, lon = GLL_lib.get_GLL_pos_claire_definition(et, y_ref="JUPITER")

if set_y_manually == True:
    # set the xz-plane manually (Re)
    plot_surface_y = -2.7
    GLL_pos_x = Calculation_intersection(y/re,x/re,plot_surface_y)
    GLL_pos_z = Calculation_intersection(y/re,z/re,plot_surface_y)
    title_str = "GLL orbit and ray path @ y = " + "{:.2g}".format(plot_surface_y) + "Re"
else:
    # set date/time
    utctim_GLL_plot = "1997-12-16T12:15:00"  # start date/time
    et_ex_GLL_plot = spice.str2et(utctim_GLL_plot)  # seconds
    nd_GLL_plot = 2  # number of data 
    dt_GLL_plot = 1  # time step [second]
    et_GLL_plot = et_ex_GLL_plot + dt_GLL_plot * np.arange(0, nd_GLL_plot)

    # calculate target position with spice
    x_GLL_plot, y_GLL_plot, z_GLL_plot, r_GLL_plot, lat_GLL_plot, lon_GLL_plot = GLL_lib.get_GLL_pos_claire_definition(et_GLL_plot, y_ref="JUPITER")
    plot_surface_y = y_GLL_plot[0] / re
    GLL_pos_x = x_GLL_plot[0] / re
    GLL_pos_z = z_GLL_plot[0] / re
    title_str = "GLL orbit and ray path @ y = " + "{:.2g}".format(plot_surface_y) + "Re" +"  " + utctim_GLL_plot



plt.scatter(GLL_pos_x,GLL_pos_z,c="r",marker="*",s=30)

# レイパスをプロットする
for initial_ray_z in initial_ray_z_array:
    for initial_ray_x in initial_ray_x_array:
        filename = np.genfromtxt(
            path_name
            + "ray-Pclaire_240507-Mtest_simple-benchmark-LO-X"
            + str(initial_ray_x)
            + "-Z"
            + str(initial_ray_z)
            + "-FR"
            + Freq_str
        )
        ray_x = filename[:, [1]]
        ray_y = filename[:, [2]]
        ray_z = filename[:, [3]]

        intersection_x = Calculation_intersection(ray_y/re,ray_x/re,plot_surface_y)
        intersection_z = Calculation_intersection(ray_y/re,ray_z/re,plot_surface_y)

        plt.scatter(intersection_x, intersection_z , c='b', marker='o',s=0.1)

full_theta = np.linspace(0, 2 * np.pi, 360)
plt.plot(
    np.cos(full_theta),
    np.sin(full_theta),
    color="black",
    alpha=0.5,
    label="Filled Circle",
)

plt.title(title_str)
plt.xlabel("X: Distance from Europa [Re] (Traling direction)")
plt.ylabel("Z: Distance from Europa [Re] (North direction)")


plt.axis("equal")
plt.xlim([-2, 2])  # X軸の範囲
plt.ylim([-2, 2])  # Y軸の範囲
plt.show()




########################################################################
# JUICE orbit near Europa
# %%

# In[5]:
# set date/time
utctim = "1997-12-16T11:40:00"  # start date/time
et_ex = spice.str2et(utctim)  # seconds
nd = 60 * 50  # number of data 
dt = 1  # time step [second]
et = et_ex + dt * np.arange(0, nd)

# calculate target position with spice
x, y, z, r, lat, lon = GLL_lib.get_pos_yref(et, tar="JUPITER")

print(np.rad2deg(np.arcsin(x[0] / r[0])))
print(np.rad2deg(np.arcsin(y[0] / r[0])))
print(np.rad2deg(np.arcsin(z[0] / r[0])))
print(np.rad2deg(np.arcsin(x[-1] / r[-1])))
print(np.rad2deg(np.arcsin(y[-1] / r[-1])))
print(np.rad2deg(np.arcsin(z[-1] / r[-1])))

# %%
