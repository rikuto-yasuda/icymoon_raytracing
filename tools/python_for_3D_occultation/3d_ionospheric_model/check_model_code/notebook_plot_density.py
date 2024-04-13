# %%
#######################
# plot_density_ne_europa.py
# ---------------------
# This routine reads the density
#  file and plot the map in
# the XY and XZ plane
#
# C. Baskevitch
# UVSQ-LATMOS
# claire.baskevitch@latmos.ipsl.fr
# November 2020
#######################

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import math
import sys, os
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
import time
from scipy.special import sph_harm
from scipy.optimize import curve_fit
from scipy.special import lpmv


# %%
def plot_density_ne(src_dir, typefile, rundate, diagtime, zoom=True):
    ncfile = (
        src_dir + typefile + rundate + "_t" + diagtime + ".nc"
    )  #'_extract_4Re_grid.nc'#

    ncid = Dataset(ncfile)
    var_nc = ncid.variables
    # print(var_nc)

    # planetname = var_nc['planetname'][:]
    centr = var_nc["s_centr"][:]  # 3dim: planet center position
    radius = var_nc["r_planet"][:]  # 1dim: planet radius
    # print(radius)
    gs = var_nc["gstep"][:]  # space_dim(depend on the results?): grid length?
    nptot = var_nc["nptot"][:]  # number of particles in the simulation
    Dn = var_nc["Density"][:]  # /cc
    # nrm        = var_nc['phys_density'][:]
    nrm_len = var_nc["phys_length"][:]
    # X_axis = var_nc["X_axis"][:]
    # Y_axis = var_nc["Y_axis"][:]
    # Z_axis = var_nc["Z_axis"][:]

    # radius=1
    nc = [len(Dn[0][0]), len(Dn[0]), len(Dn)]  # like [len(x),len(y),len(z)]
    # print(var_nc["X_axis"][:])

    Dn = np.where(Dn <= 0, float("NaN"), Dn)
    # maximum and minimum
    if radius > 1.0:
        min_val = 0.0
        max_val = 5.0
    else:
        min_val = 1.5  # log(cm-3) for run without planete
        max_val = 1.6  # log(sm-3) for run without planete

    # -- Creation of axis values centered on the planet ( normalized to planet radius)
    X_XY, Y_XY = np.meshgrid(
        np.arange(0, (nc[0]) * gs[0], gs[0]), np.arange(0, (nc[1]) * gs[1], gs[1])
    )  # unit..phys_length
    # X_XY, Y_XY = np.meshgrid((X_axis-X_axis[0])/nrm_len,(Y_axis-Y_axis[0])/nrm_len)
    X_XY = np.divide(np.matrix.transpose(X_XY), radius) - np.divide(
        centr[0] * np.ones((nc[0], nc[1])), radius
    )  # unit..Re origin..center
    Y_XY = np.divide(np.matrix.transpose(Y_XY), radius) - np.divide(
        centr[1] * np.ones((nc[0], nc[1])), radius
    )

    X_XZ, Z_XZ = np.meshgrid(
        np.arange(0, (nc[0]) * gs[0], gs[0]), np.arange(0, (nc[2]) * gs[2], gs[2])
    )
    # X_XZ, Z_XZ = np.meshgrid((X_axis-X_axis[0])/nrm_len,(Z_axis-Z_axis[0])/nrm_len)
    X_XZ = np.divide(np.matrix.transpose(X_XZ), radius) - np.divide(
        centr[0] * np.ones((nc[0], nc[2])), radius
    )
    Z_XZ = np.divide(np.matrix.transpose(Z_XZ), radius) - np.divide(
        centr[2] * np.ones((nc[0], nc[2])), radius
    )

    Y_YZ, Z_YZ = np.meshgrid(
        np.arange(0, (nc[1]) * gs[1], gs[1]), np.arange(0, (nc[2]) * gs[2], gs[2])
    )
    # Y_YZ, Z_YZ = np.meshgrid((Y_axis-Y_axis[0])/nrm_len,(Z_axis-Z_axis[0])/nrm_len)
    Y_YZ = np.divide(np.matrix.transpose(Y_YZ), radius) - np.divide(
        centr[1] * np.ones((nc[1], nc[2])), radius
    )
    Z_YZ = np.divide(np.matrix.transpose(Z_YZ), radius) - np.divide(
        centr[2] * np.ones((nc[1], nc[2])), radius
    )

    # planet center in cell number to plot on the plane through center (NB: cell number start at 1
    icentr = int(np.fix(centr[0] / gs[0]))
    jcentr = int(np.fix(centr[1] / gs[1]))
    kcentr = int(np.fix(centr[2] / gs[2]))
    iwake = int(icentr + np.fix(1.5 * radius / gs[0]))

    Dn_XY = np.zeros((nc[0], nc[1]))
    Dn_XY[:, :] = np.matrix.transpose(Dn[kcentr, :, :])

    Dn_XZ = np.zeros((nc[0], nc[2]))
    Dn_XZ[:, :] = np.matrix.transpose(Dn[:, jcentr, :])

    Dn_YZ_term = np.zeros((nc[1], nc[2]))
    Dn_YZ_term[:, :] = np.matrix.transpose(Dn[:, :, icentr])

    Dn_YZ_wake = np.zeros((nc[1], nc[2]))
    Dn_YZ_wake[:, :] = np.matrix.transpose(Dn[:, :, iwake])

    Dne_1D = np.zeros(nc[0])
    Dne_1D[:] = Dn[kcentr, jcentr, :]
    # x = np.divide(np.arange(0, nc[0], 1.0) * gs[0]-centr[0], radius)
    # x = ([0:nc(1)-1]*gs(1)-centr(1))./radius;

    # planet drawing
    theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
    xp = np.cos(theta)
    yp = np.sin(theta)

    fig_size = [[6, 9.5], [6, 7.5], [8, 6], [10, 6]]  # differentes tailles de fenetres
    figsize_Xnum = 2  # numero de la taille de la fenetre pour les plans XZ et XY
    figsize_Ynum = 2  # numero de la taille de la fenetre pour les plans YZ

    if zoom == True:
        Xmin = -4
        Xmax = 4
        Ymin = -4
        Ymax = 4
        Zmin = -4
        Zmax = 4
    else:
        Xmin = X_XY[0][0]
        Xmax = X_XY[len(X_XY) - 1][len(X_XY[0]) - 1]
        Ymin = Y_XY[0][0]
        Ymax = Y_XY[len(Y_XY) - 1][len(Y_XY[0]) - 1]
        Zmin = Z_XZ[0][0]
        Zmax = Z_XZ[len(Z_XZ) - 1][len(Z_XZ[0]) - 1]

    # -- Figure 1 & 2 -- Dn
    # **************************************************************************

    if zoom == True:
        fig, ax = plt.subplots(figsize=fig_size[figsize_Xnum])
    else:
        fig, ax = plt.subplots(figsize=fig_size[figsize_Xnum])
    c = ax.pcolor(
        X_XY,
        Y_XY,
        np.log10(Dn_XY),
        vmin=min_val,
        vmax=max_val,
        cmap="jet",
        shading="auto",
    )
    if radius != 1:
        ax.plot(xp, yp, c="black")
    fig.colorbar(c, ax=ax)
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("X [R_E]")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Y [R_E]")  # ,'fontsize',12,'fontweight','b');

    # plt.savefig(dest_dir+"Dn_ne_XY_Europa_"+rundate+"_t"+diagtime+".png")

    # --

    if zoom == True:
        fig, ax = plt.subplots(figsize=fig_size[figsize_Xnum])
    else:
        fig, ax = plt.subplots(figsize=fig_size[figsize_Xnum])
    c = ax.pcolor(
        X_XZ,
        Z_XZ,
        np.log10(Dn_XZ),
        vmin=min_val,
        vmax=max_val,
        cmap="jet",
        shading="auto",
    )
    if radius != 1:
        ax.plot(xp, yp, c="black")
    fig.colorbar(c, ax=ax)
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Zmin, Zmax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("X [R_E]")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Z [R_E]")  # ,'fontsize',12,'fontweight','b');

    # plt.savefig(dest_dir+"Dn_ne_XZ_Europa_"+rundate+"_t"+diagtime+".png")

    # =========== figure in YZ plane ==============================
    # Dn in X=0 and X=1.5Rm

    # figure 3 & 4
    if zoom == True:
        fig, ax = plt.subplots(figsize=fig_size[figsize_Xnum])
    else:
        fig, ax = plt.subplots(figsize=fig_size[figsize_Ynum])

    print(Dn_YZ_term)
    c = ax.pcolor(
        Y_YZ,
        Z_YZ,
        np.log10(Dn_YZ_term),
        vmin=min_val,
        vmax=max_val,
        cmap="jet",
        shading="auto",
    )
    if radius != 1:
        ax.plot(xp, yp, c="black")
    fig.colorbar(c, ax=ax)
    ax.set_xlim(Ymin, Ymax)
    ax.set_ylim(Zmin, Zmax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("Y [R_E]")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Z [R_E]")  # ,'fontsize',12,'fontweight','b');

    # plt.savefig(dest_dir+"Dn_ne_YZ_Europa_"+rundate+"_t"+diagtime+".png")

    # --
    if zoom == True:
        fig, ax = plt.subplots(figsize=fig_size[figsize_Ynum])
    else:
        fig, ax = plt.subplots(figsize=fig_size[figsize_Ynum])
    c = ax.pcolor(
        Y_YZ,
        Z_YZ,
        np.log10(Dn_YZ_wake),
        vmin=min_val,
        vmax=max_val,
        cmap="jet",
        shading="auto",
    )
    if radius != 1:
        ax.plot(xp, yp, c="black")
    fig.colorbar(c, ax=ax)
    ax.set_xlim(Ymin, Ymax)
    ax.set_ylim(Zmin, Zmax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("Y [R_E]")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Z [R_E]")  # ,'fontsize',12,'fontweight','b');

    plt.show()
    # plt.savefig(dest_dir+"Dn_ne_YZ_wake_Europa_"+rundate+"_t"+diagtime+".png")

    # plt.close('all')


# %%
def density_plot(src_dir, typefile, rundate, diagtime):
    ncfile = (
        src_dir + typefile + rundate + "_t" + diagtime + ".nc"
    )  #'_extract_4Re_grid.nc'#

    ncid = Dataset(ncfile)
    var_nc = ncid.variables
    # print(var_nc)

    centr = var_nc["s_centr"][:]  # 3dim: planet center position
    radius = var_nc["r_planet"][:]  # 1dim: planet radius

    gs = var_nc["gstep"][:]  # space_dim(depend on the results?): grid length?
    nptot = var_nc["nptot"][:]  # number of particles in the simulation
    Dn = var_nc["Density"][:]  # /cc [z_num][y_num][x_num]
    # nrm        = var_nc['phys_density'][:]
    nrm_len = var_nc["phys_length"][:]
    # X_axis = var_nc["X_axis"][:]
    # Y_axis = var_nc["Y_axis"][:]
    # Z_axis = var_nc["Z_axis"][:]

    # radius=1
    nc = [len(Dn[0][0]), len(Dn[0]), len(Dn)]  # like [len(x),len(y),len(z)]
    # print(var_nc["X_axis"][:])

    Dn = np.where(Dn <= 0, float("NaN"), Dn)
    # maximum and minimum
    min_val = 0  # log(cm-3) for run without planete
    max_val = 5  # log(sm-3) for run without planete

    # position array from moon center (unit..phys_length)
    x_array_moon_center_phylen = np.arange(0, (nc[0]) * gs[0], gs[0]) - centr[0]
    y_array_moon_center_phylen = np.arange(0, (nc[1]) * gs[1], gs[1]) - centr[1]
    z_array_moon_center_phylen = np.arange(0, (nc[2]) * gs[2], gs[2]) - centr[2]

    # position array from moon center (unit..Re)
    x_array_moon_center_radius = x_array_moon_center_phylen / radius
    y_array_moon_center_radius = y_array_moon_center_phylen / radius
    z_array_moon_center_radius = z_array_moon_center_phylen / radius

    # position array from moon center (unit..km)
    x_array_moon_center_km = x_array_moon_center_phylen * nrm_len
    y_array_moon_center_km = y_array_moon_center_phylen * nrm_len
    z_array_moon_center_km = z_array_moon_center_phylen * nrm_len

    def plot_in_xy_plane(x_array, y_array, z_array, Dn, radius, z_pos=0):
        z_ind = (np.abs(z_array - z_pos)).argmin()
        X_XY, Y_XY = np.meshgrid(x_array, y_array)
        X_XY = np.matrix.transpose(X_XY)
        Y_XY = np.matrix.transpose(Y_XY)

        # print(np.abs(z_array[z_ind] - z_pos))
        Dn_XY = np.zeros((nc[0], nc[1]))
        Dn_XY[:, :] = np.matrix.transpose(Dn[z_ind, :, :])

        # Xmin = X_XY[0][0]
        # Xmax = X_XY[len(X_XY) - 1][len(X_XY[0]) - 1]
        # Ymin = Y_XY[0][0]
        # Ymax = Y_XY[len(Y_XY) - 1][len(Y_XY[0]) - 1]

        fig, ax = plt.subplots()

        c = ax.pcolor(
            X_XY,
            Y_XY,
            np.log10(Dn_XY),
            vmin=min_val,
            vmax=max_val,
            cmap="jet",
            shading="auto",
        )

        if z_array[z_ind] < radius:
            # planet drawing
            rp = np.sqrt(radius * radius - z_array[z_ind] * z_array[z_ind])
            theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
            xp = rp * np.cos(theta)
            yp = rp * np.sin(theta)
            ax.plot(xp, yp, c="black")

        fig.colorbar(c, ax=ax)
        # ax.set_xlim(Xmin, Xmax)
        # ax.set_ylim(Ymin, Ymax)

        titre = "Density ne log[cm-3] time: " + diagtime
        plt.title(titre)  # ,'fontsize',12,'fontweight','b');
        ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
        ax.set_ylabel("Y")  # ,'fontsize',12,'fontweight','b');
        plt.plot()

    def plot_in_xz_plane(x_array, y_array, z_array, Dn, radius, y_pos=0):
        y_ind = (np.abs(y_array - y_pos)).argmin()
        X_XZ, Z_XZ = np.meshgrid(x_array, z_array)
        X_XZ = np.matrix.transpose(X_XZ)
        Z_XZ = np.matrix.transpose(Z_XZ)
        # print(np.abs(y_array[y_ind] - y_pos))
        Dn_XZ = np.zeros((nc[0], nc[2]))
        Dn_XZ[:, :] = np.matrix.transpose(Dn[:, y_ind, :])

        # Xmin = X_XY[0][0]
        # Xmax = X_XY[len(X_XY) - 1][len(X_XY[0]) - 1]
        # Zmin = Z_XZ[0][0]
        # Zmax = Z_XZ[len(Z_XZ) - 1][len(Z_XZ[0]) - 1]

        fig, ax = plt.subplots()

        c = ax.pcolor(
            X_XZ,
            Z_XZ,
            np.log10(Dn_XZ),
            vmin=min_val,
            vmax=max_val,
            cmap="jet",
            shading="auto",
        )

        if y_array[y_ind] < radius:
            # planet drawing
            rp = np.sqrt(radius * radius - y_array[y_ind] * y_array[y_ind])
            theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
            xp = rp * np.cos(theta)
            yp = rp * np.sin(theta)
            ax.plot(xp, yp, c="black")

        fig.colorbar(c, ax=ax)

        titre = "Density ne log[cm-3] time: " + diagtime
        plt.title(titre)  # ,'fontsize',12,'fontweight','b');
        ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
        ax.set_ylabel("Z")  # ,'fontsize',12,'fontweight','b');
        plt.plot()

    def plot_in_yz_plane(x_array, y_array, z_array, Dn, radius, x_pos=0):
        x_ind = (np.abs(x_array - x_pos)).argmin()
        Y_YZ, Z_YZ = np.meshgrid(y_array, z_array)
        Y_YZ = np.matrix.transpose(Y_YZ)
        Z_YZ = np.matrix.transpose(Z_YZ)
        # print(np.abs(x_array[x_ind] - x_pos))
        Dn_YZ = np.zeros((nc[1], nc[2]))
        Dn_YZ[:, :] = np.matrix.transpose(Dn[:, :, x_ind])

        # Ymin = Y_YZ[0][0]S
        # Ymax = Y_YZ[len(Y_YZ) - 1][len(Y_YZ[0]) - 1]
        # Zmin = Z_YZ[0][0]
        # Zmax = Z_YZ[len(Z_YZ) - 1][len(Z_YZ[0]) - 1]

        fig, ax = plt.subplots()

        print(Dn_YZ.shape)
        c = ax.pcolor(
            Y_YZ,
            Z_YZ,
            np.log10(Dn_YZ),
            vmin=min_val,
            vmax=max_val,
            cmap="jet",
            shading="auto",
        )

        if x_array[x_ind] < radius:
            # planet drawing
            rp = np.sqrt(radius * radius - x_array[x_ind] * x_array[x_ind])
            theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
            xp = rp * np.cos(theta)
            yp = rp * np.sin(theta)
            ax.plot(xp, yp, c="black")

        fig.colorbar(c, ax=ax)
        # ax.set_xlim(Ymin, Ymax)
        # ax.set_ylim(Zmin, Zmax)

        titre = "Density ne log[cm-3] time: " + diagtime
        plt.title(titre)  # ,'fontsize',12,'fontweight','b');
        ax.set_xlabel("Y")  # ,'fontsize',12,'fontweight','b');
        ax.set_ylabel("Z")  # ,'fontsize',12,'fontweight','b');
        plt.plot()

    plot_in_yz_plane(
        x_array_moon_center_radius,
        y_array_moon_center_radius,
        z_array_moon_center_radius,
        Dn,
        1,
        x_pos=0,
    )

    plot_in_xy_plane(
        x_array_moon_center_radius,
        y_array_moon_center_radius,
        z_array_moon_center_radius,
        Dn,
        1,
        z_pos=0,
    )

    plot_in_xz_plane(
        x_array_moon_center_radius,
        y_array_moon_center_radius,
        z_array_moon_center_radius,
        Dn,
        1,
        y_pos=0,
    )

    plot_in_xz_plane(
        x_array_moon_center_km,
        y_array_moon_center_km,
        z_array_moon_center_km,
        Dn,
        1560.8,
        y_pos=0,
    )


def plot_in_xy_plane(x_array, y_array, z_array, Dn, radius, x_range, y_range, z_pos=0):
    z_ind = (np.abs(z_array - z_pos)).argmin()
    X_XY, Y_XY = np.meshgrid(x_array, y_array)
    X_XY = np.matrix.transpose(X_XY)
    Y_XY = np.matrix.transpose(Y_XY)

    # print(np.abs(z_array[z_ind] - z_pos))
    Dn_XY = np.zeros((len(x_array), len(y_array)))
    Dn_XY[:, :] = np.matrix.transpose(Dn[z_ind, :, :])

    # Xmin = X_XY[0][0]
    # Xmax = X_XY[len(X_XY) - 1][len(X_XY[0]) - 1]
    # Ymin = Y_XY[0][0]
    # Ymax = Y_XY[len(Y_XY) - 1][len(Y_XY[0]) - 1]

    fig, ax = plt.subplots()

    c = ax.pcolor(
        X_XY,
        Y_XY,
        np.log10(Dn_XY),
        vmin=0,
        vmax=5,
        cmap="jet",
        shading="auto",
    )
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    if z_array[z_ind] < radius:
        # planet drawing
        rp = np.sqrt(radius * radius - z_array[z_ind] * z_array[z_ind])
        theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
        xp = rp * np.cos(theta)
        yp = rp * np.sin(theta)
        ax.plot(xp, yp, c="black")

    fig.colorbar(c, ax=ax)
    # ax.set_xlim(Xmin, Xmax)
    # ax.set_ylim(Ymin, Ymax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Y")  # ,'fontsize',12,'fontweight','b');
    plt.show()


def plot_in_xz_plane(x_array, y_array, z_array, Dn, radius, x_range, z_range, y_pos=0):
    y_ind = (np.abs(y_array - y_pos)).argmin()
    X_XZ, Z_XZ = np.meshgrid(x_array, z_array)
    X_XZ = np.matrix.transpose(X_XZ)
    Z_XZ = np.matrix.transpose(Z_XZ)
    # print(np.abs(y_array[y_ind] - y_pos))
    Dn_XZ = np.zeros((len(x_array), len(z_array)))
    Dn_XZ[:, :] = np.matrix.transpose(Dn[:, y_ind, :])

    # Xmin = X_XY[0][0]
    # Xmax = X_XY[len(X_XY) - 1][len(X_XY[0]) - 1]
    # Zmin = Z_XZ[0][0]
    # Zmax = Z_XZ[len(Z_XZ) - 1][len(Z_XZ[0]) - 1]

    fig, ax = plt.subplots()

    c = ax.pcolor(
        X_XZ,
        Z_XZ,
        Dn_XZ,
        vmin=0,
        vmax=100,
        cmap="jet",
        shading="auto",
    )

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(z_range[0], z_range[1])

    if y_array[y_ind] < radius:
        # planet drawing
        rp = np.sqrt(radius * radius - y_array[y_ind] * y_array[y_ind])
        theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
        xp = rp * np.cos(theta)
        yp = rp * np.sin(theta)
        ax.plot(xp, yp, c="black")

    fig.colorbar(c, ax=ax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Z")  # ,'fontsize',12,'fontweight','b');
    plt.show()


def plot_in_yz_plane(x_array, y_array, z_array, Dn, radius, y_range, z_range, x_pos=0):
    x_ind = (np.abs(x_array - x_pos)).argmin()
    Y_YZ, Z_YZ = np.meshgrid(y_array, z_array)
    Y_YZ = np.matrix.transpose(Y_YZ)
    Z_YZ = np.matrix.transpose(Z_YZ)
    # print(np.abs(x_array[x_ind] - x_pos))
    Dn_YZ = np.zeros((len(y_array), len(z_array)))
    Dn_YZ[:, :] = np.matrix.transpose(Dn[:, :, x_ind])

    # Ymin = Y_YZ[0][0]S
    # Ymax = Y_YZ[len(Y_YZ) - 1][len(Y_YZ[0]) - 1]
    # Zmin = Z_YZ[0][0]
    # Zmax = Z_YZ[len(Z_YZ) - 1][len(Z_YZ[0]) - 1]

    fig, ax = plt.subplots()

    print(Dn_YZ.shape)
    c = ax.pcolor(
        Y_YZ,
        Z_YZ,
        np.log10(Dn_YZ),
        vmin=0,
        vmax=5,
        cmap="jet",
        shading="auto",
    )

    ax.set_xlim(y_range[0], y_range[1])
    ax.set_ylim(z_range[0], z_range[1])

    if x_array[x_ind] < radius:
        # planet drawing
        rp = np.sqrt(radius * radius - x_array[x_ind] * x_array[x_ind])
        theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
        xp = rp * np.cos(theta)
        yp = rp * np.sin(theta)
        ax.plot(xp, yp, c="black")

    fig.colorbar(c, ax=ax)
    # ax.set_xlim(Ymin, Ymax)
    # ax.set_ylim(Zmin, Zmax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("Y")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Z")  # ,'fontsize',12,'fontweight','b');
    plt.show()


def density_interpolate(src_dir, typefile, rundate, diagtime):
    start = time.time()  # 現在時刻（処理開始前）を取得
    ncfile = (
        src_dir + typefile + rundate + "_t" + diagtime + ".nc"
    )  #'_extract_4Re_grid.nc'#

    ncid = Dataset(ncfile)
    var_nc = ncid.variables
    # print(var_nc)

    centr = var_nc["s_centr"][:]  # 3dim: planet center position
    radius = var_nc["r_planet"][:]  # 1dim: planet radius

    gs = var_nc["gstep"][:]  # space_dim(depend on the results?): grid length?
    nptot = var_nc["nptot"][:]  # number of particles in the simulation
    Dn = var_nc["Density"][:]  # /cc [z_num][y_num][x_num]
    # nrm        = var_nc['phys_density'][:]
    nrm_len = var_nc["phys_length"][:]
    # X_axis = var_nc["X_axis"][:]
    # Y_axis = var_nc["Y_axis"][:]
    # Z_axis = var_nc["Z_axis"][:]

    # radius=1
    nc = [len(Dn[0][0]), len(Dn[0]), len(Dn)]  # like [len(x),len(y),len(z)]
    # print(var_nc["X_axis"][:])

    # Dn = np.where(Dn <= 0, float("NaN"), Dn)
    # maximum and minimum

    # position array from moon center (unit..phys_length)
    x_array_moon_center_phylen = np.arange(0, (nc[0]) * gs[0], gs[0]) - centr[0]
    y_array_moon_center_phylen = np.arange(0, (nc[1]) * gs[1], gs[1]) - centr[1]
    z_array_moon_center_phylen = np.arange(0, (nc[2]) * gs[2], gs[2]) - centr[2]

    # position array from moon center (unit..km)
    x_array_moon_center_km = x_array_moon_center_phylen * nrm_len
    y_array_moon_center_km = y_array_moon_center_phylen * nrm_len
    z_array_moon_center_km = z_array_moon_center_phylen * nrm_len

    interp_func = RegularGridInterpolator(
        (z_array_moon_center_km, y_array_moon_center_km, x_array_moon_center_km), Dn
    )

    interp_func.method = "nearest"

    # position array from moon center (unit..km)
    interpolate_grid_length = 50
    x_array_moon_center_km_interpolation = np.arange(
        x_array_moon_center_km[0], x_array_moon_center_km[-1], interpolate_grid_length
    )
    y_array_moon_center_km_interpolation = np.arange(
        y_array_moon_center_km[0], y_array_moon_center_km[-1], interpolate_grid_length
    )
    z_array_moon_center_km_interpolation = np.arange(
        z_array_moon_center_km[0], z_array_moon_center_km[-1], interpolate_grid_length
    )

    x_interpolate_grid, y_interpolate_grid, z_interpolate_grid = np.meshgrid(
        x_array_moon_center_km_interpolation,
        y_array_moon_center_km_interpolation,
        z_array_moon_center_km_interpolation,
        indexing="ij",
    )

    Dn_interpolate = interp_func(
        (
            np.matrix.transpose(z_interpolate_grid),
            np.matrix.transpose(y_interpolate_grid),
            np.matrix.transpose(x_interpolate_grid),
        )
    )
    print(x_interpolate_grid)

    end = time.time()  # 現在時刻（処理完了後）を取得
    time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
    print(time_diff)  # 処理にかかった時間データを使用

    plot_in_xz_plane(
        x_array_moon_center_km,
        y_array_moon_center_km,
        z_array_moon_center_km,
        Dn,
        1560.8,
        y_pos=0,
    )
    plot_in_xz_plane(
        x_array_moon_center_km_interpolation,
        y_array_moon_center_km_interpolation,
        z_array_moon_center_km_interpolation,
        Dn_interpolate,
        1560.8,
        y_pos=0,
    )


def density_fitting(src_dir, typefile, rundate, diagtime):
    start = time.time()  # 現在時刻（処理開始前）を取得
    ncfile = (
        src_dir + typefile + rundate + "_t" + diagtime + ".nc"
    )  #'_extract_4Re_grid.nc'#

    ncid = Dataset(ncfile)
    var_nc = ncid.variables
    # print(var_nc)

    centr = var_nc["s_centr"][:]  # 3dim: planet center position
    radius = var_nc["r_planet"][:]  # 1dim: planet radius

    gs = var_nc["gstep"][:]  # space_dim(depend on the results?): grid length?
    nptot = var_nc["nptot"][:]  # number of particles in the simulation
    Dn = var_nc["Density"][:]  # /cc [z_num][y_num][x_num]
    # nrm        = var_nc['phys_density'][:]
    nrm_len = var_nc["phys_length"][:]
    # X_axis = var_nc["X_axis"][:]
    # Y_axis = var_nc["Y_axis"][:]
    # Z_axis = var_nc["Z_axis"][:]

    # radius=1
    nc = [len(Dn[0][0]), len(Dn[0]), len(Dn)]  # like [len(x),len(y),len(z)]
    # print(var_nc["X_axis"][:])

    Dn = np.where(Dn <= 0, float("NaN"), Dn)
    # maximum and minimum

    # position array from moon center (unit..phys_length)
    x_array_moon_center_phylen = np.arange(0, (nc[0]) * gs[0], gs[0]) - centr[0]
    y_array_moon_center_phylen = np.arange(0, (nc[1]) * gs[1], gs[1]) - centr[1]
    z_array_moon_center_phylen = np.arange(0, (nc[2]) * gs[2], gs[2]) - centr[2]

    # position array from moon center (unit..km)
    x_array_moon_center_km = x_array_moon_center_phylen * nrm_len
    y_array_moon_center_km = y_array_moon_center_phylen * nrm_len
    z_array_moon_center_km = z_array_moon_center_phylen * nrm_len

    x_meshgrid, y_meshgrid, z_meshgrid = np.meshgrid(
        x_array_moon_center_km,
        y_array_moon_center_km,
        z_array_moon_center_km,
        indexing="ij",
    )
    x_meshgrid = np.matrix.transpose(x_meshgrid)
    y_meshgrid = np.matrix.transpose(y_meshgrid)
    z_meshgrid = np.matrix.transpose(z_meshgrid)

    x_meshgrid_1d = x_meshgrid.flatten()
    y_meshgrid_1d = y_meshgrid.flatten()
    z_meshgrid_1d = z_meshgrid.flatten()
    Dn_1d = Dn.flatten()
    not_nan_indices = np.where(~np.isnan(Dn_1d))[0]

    def cartesian_to_spherical(x, y, z):
        """
        Converts Cartesian coordinates to spherical coordinates.
        Args:
        - x, y, z: Cartesian coordinates
        Returns:
        - r, theta, phi: Spherical coordinates
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    def cartesian_to_spherical2(x, y):
        """
        Converts Cartesian coordinates to spherical coordinates.
        Args:
        - x, y, z: Cartesian coordinates
        Returns:
        - r, theta, phi: Spherical coordinates
        """
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return r, phi

    def added_spherical_function(
        X,
        a100,
        a110,
        a111,
        a200,
        a210,
        a211,
        a220,
        a221,
        a300,
        a310,
        a311,
        a320,
        a321,
        a330,
        a331,
        A100,
        A110,
        A111,
        A200,
        A210,
        A211,
    ):
        r, theta, phi = X
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)

        total = a100 * lpmv(0, 1, cos_theta) / (r * r)
        +a110 * cos_phi * lpmv(1, 1, cos_theta) / (r * r)
        +a111 * sin_phi * lpmv(1, 1, cos_theta) / (r * r)
        +a200 * lpmv(0, 2, cos_theta) / (r * r * r)
        +a210 * cos_phi * lpmv(1, 2, cos_theta) / (r * r * r)
        +a211 * sin_phi * lpmv(1, 2, cos_theta) / (r * r * r)
        +a220 * np.cos(2 * phi) * lpmv(2, 2, cos_theta) / (r * r * r)
        +a221 * np.sin(2 * phi) * lpmv(2, 2, cos_theta) / (r * r * r)
        +a300 * lpmv(0, 3, cos_theta) / (r * r * r * r)
        +a310 * cos_phi * lpmv(1, 3, cos_theta) / (r * r * r * r)
        +a311 * sin_phi * lpmv(1, 3, cos_theta) / (r * r * r * r)
        +a320 * np.cos(2 * phi) * lpmv(2, 3, cos_theta) / (r * r * r * r)
        +a321 * np.sin(2 * phi) * lpmv(2, 3, cos_theta) / (r * r * r * r)
        +a330 * np.cos(3 * phi) * lpmv(3, 3, cos_theta) / (r * r * r * r)
        +a331 * np.sin(3 * phi) * lpmv(3, 3, cos_theta) / (r * r * r * r)
        +A100 * lpmv(0, 1, cos_theta) * r
        +A110 * cos_phi * lpmv(1, 1, cos_theta) * r
        +A111 * sin_phi * lpmv(1, 1, cos_theta) * r
        +A200 * lpmv(0, 2, cos_theta) * r * r
        +A210 * cos_phi * lpmv(1, 2, cos_theta) * r * r
        +A211 * sin_phi * lpmv(1, 2, cos_theta) * r * r

        return total

    r_modeled, theta_modeled, phi_modeled = cartesian_to_spherical(
        x_meshgrid_1d, y_meshgrid_1d, z_meshgrid_1d
    )
    Dn_remove_nan = Dn_1d[not_nan_indices]
    r_remove_nan = r_modeled[not_nan_indices]
    print(r_remove_nan)
    theta_remove_nan = theta_modeled[not_nan_indices]
    phi_remove_nan = phi_modeled[not_nan_indices]

    good_r_pos = np.where(r_remove_nan < 1560.8 * 6)[0]
    Dn_fitted = Dn_remove_nan[good_r_pos]
    r_fitted = r_remove_nan[good_r_pos]
    theta_fitted = theta_remove_nan[good_r_pos]
    phi_fitted = phi_remove_nan[good_r_pos]
    print("aaa")
    popt, pcov = curve_fit(
        added_spherical_function,
        (
            r_fitted,
            theta_fitted,
            phi_fitted,
        ),
        Dn_fitted,
        maxfev=200000,
    )  # poptは最適推定値、pcovは共分散が出力される

    end = time.time()  # 現在時刻（処理完了後）を取得
    time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
    print(time_diff)  # 処理にかかった時間データを使用

    plot_in_xy_plane(
        x_array_moon_center_km,
        y_array_moon_center_km,
        z_array_moon_center_km,
        Dn,
        1560.8,
        [-1560.8 * 4, 1560.8 * 4],
        [-1560.8 * 4, 1560.8 * 4],
        z_pos=0,
    )
    print(popt)
    """
    plot_in_xz_plane(
        x_array_moon_center_km,
        y_array_moon_center_km,
        z_array_moon_center_km,
        Dn,
        1560.8,
        [-1560.8 * 4, 1560.8 * 4],
        [-1560.8 * 4, 1560.8 * 4],
        y_pos=0,
    )
    """
    XX, YY = np.meshgrid(x_array_moon_center_km, y_array_moon_center_km)
    r_test, phi_test = cartesian_to_spherical2(XX, YY)
    fig, ax = plt.subplots()
    c = ax.pcolor(
        XX,
        YY,
        np.log10(
            added_spherical_function(
                (r_test, 0, phi_test),
                popt[0],
                popt[1],
                popt[2],
                popt[3],
                popt[4],
                popt[5],
                popt[6],
                popt[7],
                popt[8],
                popt[9],
                popt[10],
                popt[11],
                popt[12],
                popt[13],
                popt[14],
                popt[15],
                popt[16],
                popt[17],
                popt[18],
                popt[19],
                popt[20],
            )
        ),
        cmap="jet",
        shading="auto",
        vmin=0.0,
        vmax=5.0,
    )
    plt.xlim(-1560.8 * 4, 1560.8 * 4)
    plt.ylim(-1560.8 * 4, 1560.8 * 4)
    fig.colorbar(c, ax=ax)
    plt.show()


# %%
# src_dir = "../LatHyS_simu/Europa/RUN_A3/"
src_dir = "../LatHyS_simu/Europa/RUN_A3/"
typefile = "Ojv_"
rundate = "19_04_23"
diagtime = "00600"
# plot_density_ne(src_dir, typefile, rundate, diagtime, zoom=True)
# density_interpolate(src_dir, typefile, rundate, diagtime)

density_fitting(src_dir, typefile, rundate, diagtime)
# %%
