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
    max_val = 3  # log(sm-3) for run without planete

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

    def plot_in_xy_plane(x_array, y_array, z_array, Dn, radius, x_lim, y_lim, z_pos=0):
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

        fig.colorbar(c, ax=ax)
        # ax.set_xlim(Xmin, Xmax)
        # ax.set_ylim(Ymin, Ymax)

        titre = "Density ne log[cm-3] time: " + diagtime
        plt.title(titre)  # ,'fontsize',12,'fontweight','b');
        ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
        ax.set_ylabel("Y")  # ,'fontsize',12,'fontweight','b');
        circle =plt.Circle((0,0), radius, color='black', alpha=1)
        plt.gca().add_artist(circle)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])    
        plt.plot()

    def plot_in_xz_plane(x_array, y_array, z_array, Dn, radius, x_lim, z_lim,  y_pos=0):
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

        fig.colorbar(c, ax=ax)

        titre = "Density ne log[cm-3] time: " + diagtime
        plt.title(titre)  # ,'fontsize',12,'fontweight','b');
        ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
        ax.set_ylabel("Z")  # ,'fontsize',12,'fontweight','b');
        circle =plt.Circle((0,0), radius, color='black', alpha=1)
        plt.gca().add_artist(circle)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(z_lim[0], z_lim[1])     
        plt.plot()

    def plot_in_yz_plane(x_array, y_array, z_array, Dn, radius, y_lim, z_lim,  x_pos=0):
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

        fig.colorbar(c, ax=ax)
        # ax.set_xlim(Ymin, Ymax)
        # ax.set_ylim(Zmin, Zmax)

        titre = "Density ne log[cm-3] time: " + diagtime
        plt.title(titre)  # ,'fontsize',12,'fontweight','b');
        ax.set_xlabel("Y")  # ,'fontsize',12,'fontweight','b');
        ax.set_ylabel("Z")  # ,'fontsize',12,'fontweight','b');
        circle =plt.Circle((0,0), radius, color='black', alpha=1)
        plt.gca().add_artist(circle)
        ax.set_xlim(y_lim[0], y_lim[1])
        ax.set_ylim(z_lim[0], z_lim[1])    
        plt.plot()

    def plot_in_xy_plane_scatter(x_array, y_array, z_array, Dn, radius, x_lim, y_lim, z_pos=0):
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

        c = ax.scatter(
            X_XY,
            Y_XY,
            c = np.log10(Dn_XY),
            s = 10,
            vmin=min_val,
            vmax=max_val,
            cmap="jet",
        )

        fig.colorbar(c, ax=ax)
        # ax.set_xlim(Xmin, Xmax)
        # ax.set_ylim(Ymin, Ymax)

        titre = "Density ne log[cm-3] time: " + diagtime
        plt.title(titre)  # ,'fontsize',12,'fontweight','b');
        ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
        ax.set_ylabel("Y")  # ,'fontsize',12,'fontweight','b');
        circle =plt.Circle((0,0), radius, color='black', alpha=1)
        plt.gca().add_artist(circle)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])    
        plt.plot()

    def plot_in_xz_plane_scatter(x_array, y_array, z_array, Dn, radius, x_lim, z_lim,  y_pos=0):
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

        c = ax.scatter(
            X_XZ,
            Z_XZ,
            c = np.log10(Dn_XZ),
            s = 10,
            vmin=min_val,
            vmax=max_val,
            cmap="jet",
        )

        fig.colorbar(c, ax=ax)

        titre = "Density ne log[cm-3] time: " + diagtime
        plt.title(titre)  # ,'fontsize',12,'fontweight','b');
        ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
        ax.set_ylabel("Z")  # ,'fontsize',12,'fontweight','b');
        circle =plt.Circle((0,0), radius, color='black', alpha=1)
        plt.gca().add_artist(circle)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(z_lim[0], z_lim[1])     
        plt.plot()

    def plot_in_yz_plane_scatter(x_array, y_array, z_array, Dn, radius, y_lim, z_lim,  x_pos=0):
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
        c = ax.scatter(
            Y_YZ,
            Z_YZ,
            c = np.log10(Dn_YZ),
            s = 10,
            vmin=min_val,
            vmax=max_val,
            cmap="jet",
        )

        fig.colorbar(c, ax=ax)
        # ax.set_xlim(Ymin, Ymax)
        # ax.set_ylim(Zmin, Zmax)

        titre = "Density ne log[cm-3] time: " + diagtime
        plt.title(titre)  # ,'fontsize',12,'fontweight','b');
        ax.set_xlabel("Y")  # ,'fontsize',12,'fontweight','b');
        ax.set_ylabel("Z")  # ,'fontsize',12,'fontweight','b');
        circle =plt.Circle((0,0), radius, color='black', alpha=1)
        plt.gca().add_artist(circle)
        ax.set_xlim(y_lim[0], y_lim[1])
        ax.set_ylim(z_lim[0], z_lim[1])    
        plt.plot()

    plot_in_yz_plane(
        x_array_moon_center_km,
        y_array_moon_center_km,
        z_array_moon_center_km,
        Dn,
        1560.8,
        y_lim=[-1560.8*3, 1560.8*3],
        z_lim=[-1560.8*3, 1560.8*3],
        x_pos=0,
    )

    plot_in_xy_plane(
        x_array_moon_center_km,
        y_array_moon_center_km,
        z_array_moon_center_km,
        Dn,
        1560.8,
        x_lim=[-1560.8*3, 1560.8*3],
        y_lim=[-1560.8*3, 1560.8*3],
        z_pos=0,
    )

    plot_in_xz_plane(
        x_array_moon_center_km,
        y_array_moon_center_km,
        z_array_moon_center_km,
        Dn,
        1560.8,
        x_lim=[-1560.8*3, 1560.8*3],
        z_lim=[-1560.8*3, 1560.8*3],
        y_pos=0,
    )    

    plot_in_yz_plane_scatter(
        x_array_moon_center_km,
        y_array_moon_center_km,
        z_array_moon_center_km,
        Dn,
        1560.8,
        y_lim=[0, 500],
        z_lim=[1500,2000],
        x_pos=0,
    )

    plot_in_xy_plane_scatter(
        x_array_moon_center_km,
        y_array_moon_center_km,
        z_array_moon_center_km,
        Dn,
        1560.8,
        x_lim=[0, 500],
        y_lim=[1500,2000],
        z_pos=0,
    )

    plot_in_xz_plane_scatter(
        x_array_moon_center_km,
        y_array_moon_center_km,
        z_array_moon_center_km,
        Dn,
        1560.8,
        x_lim=[0, 500],
        z_lim=[1500,2000],
        y_pos=0,
    )        

# %%
# src_dir = "../LatHyS_simu/Europa/RUN_A3/"
src_dir = "../LatHyS_simu/Europa/RUN_A3/"
typefile = "O2pl_"
rundate = "19_04_23"
diagtime = "00600"
# plot_density_ne(src_dir, typefile, rundate, diagtime, zoom=True)
# density_interpolate(src_dir, typefile, rundate, diagtime)

density_plot(src_dir, typefile, rundate, diagtime)
# %%
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
    np.log10(v),
    vmin=0,
    vmax=3,
    cmap="jet",
    shading="auto",
)

plt.xlim(-1560.8*3, 1560.8*3)
plt.ylim(-1560.8*3, 1560.8*3)

plt.title("Claire's Ganymede Ionosphere Model")
plt.xlabel("x (km)")
plt.ylabel("z (km)")

plt.colorbar()
circle =plt.Circle((0,0), 1560.8, color='black', alpha=1)
plt.gca().add_artist(circle)

# plt.savefig("300_dpi_scatter.png", format="png", dpi=300)
plt.show()

# %%
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


plt.scatter(
    xx,
    zz,
    c = np.log10(v),
    s = 10,
    vmin=0,
    vmax=3,
    cmap="jet"
)

plt.xlim(0, 500)
plt.ylim(1500, 2000)

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
    np.log10(v),
    vmin=0,
    vmax=3,
    cmap="jet",
    shading="auto",
)

plt.xlim(-1560.8*3, 1560.8*3)
plt.ylim(-1560.8*3, 1560.8*3)


plt.title("Claire's Ganymede Ionosphere Model")
plt.xlabel("x (km)")
plt.ylabel("y (km)")

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

plt.scatter(
    xx,
    yy,
    c = np.log10(v),
    s = 10, 
    vmin=0,
    vmax=3,
    cmap="jet",
)

plt.xlim(0, 500)
plt.ylim(1500, 2000)


plt.title("Claire's Ganymede Ionosphere Model")
plt.xlabel("x (km)")
plt.ylabel("y (km)")

plt.colorbar()
circle =plt.Circle((0,0), 1560.8, color='black', alpha=1)
plt.gca().add_artist(circle)

# plt.savefig("300_dpi_scatter.png", format="png", dpi=300)
plt.show()
# %%
