# %%
#######################
# (Source code)plot_density_ne_europa.py
# This routine reads the density file and plot the map in the XY and XZ plane
# C. Baskevitch
# UVSQ-LATMOS
# claire.baskevitch@latmos.ipsl.fr
# November 2020
#######################
# (develop) notebook_plot_density5.py 2024/05/03
# Add routines..
# 1. to plot the model density along the X, Y, Z axis
# 2. to fit the density profile with a spherical symmetric function (a0 + a1 * np.exp(-1 * (r / a2) ** a3)) using 1~4 Re
# 3. to plot the fitted density in the XY, XZ, YZ plane
# 4. to plot the difference between the model density and the fitted density in the XY, XZ, YZ plane
# 5. downgrade the fitting method from spherical symmetric function to diffusion equlibrium function n(r) = a0 + a1 * np.exp(-1 * (r / a2))
#######################

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import time
from scipy.special import lpmv
import scipy.optimize as so
import gc
from scipy.optimize import curve_fit


# %%
def plot_in_xaxis(x_array, y_array, z_array, Dn, radius, x_range, y_pos=0, z_pos=0):
    y_ind = (np.abs(y_array - y_pos)).argmin()
    z_ind = (np.abs(z_array - z_pos)).argmin()

    # print(np.abs(z_array[z_ind] - z_pos))
    Dn_X = np.zeros(len(x_array))
    Dn_X[:] = np.matrix.transpose(Dn[z_ind, y_ind, :])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.vlines(radius, 0, 300, linestyles="solid", colors="k")
    ax.vlines(-1 * radius, 0, 300, linestyles="solid", colors="k")
    ax.plot(
        x_array,
        Dn_X,
    )
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(0, 300)
    # ax.set_xlim(Xmin, Xmax)
    # ax.set_ylim(Ymin, Ymax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("X(Re)")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Density ne log (cm-3)")  # ,'fontsize',12,'fontweight','b');

    plt.show()

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
    ax.plot(x_array, np.ones(len(x_array)) * y_pos, c="red")

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(x_range[0], x_range[1])
    if z_array[z_ind] < radius:
        # planet drawing
        rp = np.sqrt(radius * radius - z_array[z_ind] * z_array[z_ind])
        theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
        xp = rp * np.cos(theta)
        yp = rp * np.sin(theta)
        ax.plot(xp, yp, c="black")

    fig.colorbar(c, ax=ax, label="Density ne log (cm-3)")
    # ax.set_xlim(Xmin, Xmax)
    # ax.set_ylim(Ymin, Ymax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Y")  # ,'fontsize',12,'fontweight','b');
    plt.show()


def plot_in_yaxis(x_array, y_array, z_array, Dn, radius, y_range, x_pos=0, z_pos=0):
    x_ind = (np.abs(x_array - x_pos)).argmin()
    z_ind = (np.abs(z_array - z_pos)).argmin()

    # print(np.abs(z_array[z_ind] - z_pos))
    Dn_Y = np.zeros(len(y_array))
    Dn_Y[:] = np.matrix.transpose(Dn[z_ind, :, x_ind])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.vlines(radius, 0, 300, linestyles="solid", colors="k")
    ax.vlines(-1 * radius, 0, 300, linestyles="solid", colors="k")
    ax.plot(
        y_array,
        Dn_Y,
    )
    ax.set_xlim(y_range[0], y_range[1])
    ax.set_ylim(0, 300)
    # ax.set_xlim(Xmin, Xmax)
    # ax.set_ylim(Ymin, Ymax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("Y(Re)")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Density ne log (cm-3)")  # ,'fontsize',12,'fontweight','b');

    plt.show()

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
    ax.plot(np.ones(len(y_array)) * x_pos, y_array, c="red")

    ax.set_xlim(y_range[0], y_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    if z_array[z_ind] < radius:
        # planet drawing
        rp = np.sqrt(radius * radius - z_array[z_ind] * z_array[z_ind])
        theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
        xp = rp * np.cos(theta)
        yp = rp * np.sin(theta)
        ax.plot(xp, yp, c="black")

    fig.colorbar(c, ax=ax, label="Density ne log (cm-3)")
    # ax.set_xlim(Xmin, Xmax)
    # ax.set_ylim(Ymin, Ymax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Y")  # ,'fontsize',12,'fontweight','b');
    plt.show()


def plot_in_xy_diagonal_axis(
    x_array, y_array, z_array, Dn, radius, xy_range, x_pos=0, y_pos=0, z_pos=0
):
    x_ind = (np.abs(x_array - x_pos)).argmin()
    y_ind = (np.abs(y_array - y_pos)).argmin()
    z_ind = (np.abs(z_array - z_pos)).argmin()

    x_total_num = len(x_array)
    x_plot_array = np.arange(
        x_ind - int(x_total_num / 2.1), x_ind + int(x_total_num / 2.1)
    )
    y_plot_array = np.arange(
        y_ind - int(x_total_num / 2.1), y_ind + int(x_total_num / 2.1)
    )
    xy_plot_array = np.sign(x_array[x_plot_array]) * np.sqrt(
        x_array[x_plot_array] ** 2 + y_array[y_plot_array] ** 2
    )

    # print(np.abs(z_array[z_ind] - z_pos))
    Dn_Z = np.zeros(len(x_plot_array))
    for i in range(len(x_plot_array)):
        Dn_Z[i] = Dn[z_ind, y_plot_array[i], x_plot_array[i]]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.vlines(radius, 0, 1000, linestyles="solid", colors="k")
    ax.vlines(-1 * radius, 0, 1000, linestyles="solid", colors="k")
    ax.plot(
        xy_plot_array,
        Dn_Z,
    )
    ax.set_xlim(xy_range[0], xy_range[1])
    ax.set_ylim(0, 1000)
    # ax.set_xlim(Xmin, Xmax)
    # ax.set_ylim(Ymin, Ymax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("XY(Re)")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Density ne log (cm-3)")  # ,'fontsize',12,'fontweight','b');

    plt.show()

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
    ax.plot(x_array[x_plot_array], y_array[y_plot_array], c="red")

    ax.set_xlim(xy_range[0], xy_range[1])
    ax.set_ylim(xy_range[0], xy_range[1])
    if z_array[z_ind] < radius:
        # planet drawing
        rp = np.sqrt(radius * radius - z_array[z_ind] * z_array[z_ind])
        theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
        xp = rp * np.cos(theta)
        yp = rp * np.sin(theta)
        ax.plot(xp, yp, c="black")

    fig.colorbar(c, ax=ax, label="Density ne log (cm-3)")
    # ax.set_xlim(Xmin, Xmax)
    # ax.set_ylim(Ymin, Ymax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Y")  # ,'fontsize',12,'fontweight','b');
    plt.show()


def plot_in_zaxis(x_array, y_array, z_array, Dn, radius, z_range, x_pos=0, y_pos=0):
    x_ind = (np.abs(x_array - x_pos)).argmin()
    y_ind = (np.abs(y_array - y_pos)).argmin()

    # print(np.abs(z_array[z_ind] - z_pos))
    Dn_Z = np.zeros(len(z_array))
    Dn_Z[:] = np.matrix.transpose(Dn[:, y_ind, x_ind])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.vlines(radius, 0, 300, linestyles="solid", colors="k")
    ax.vlines(-1 * radius, 0, 300, linestyles="solid", colors="k")
    ax.plot(
        z_array,
        Dn_Z,
    )
    ax.set_xlim(z_range[0], z_range[1])
    ax.set_ylim(0, 300)
    # ax.set_xlim(Xmin, Xmax)
    # ax.set_ylim(Ymin, Ymax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("Z(Re)")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Density ne log (cm-3)")  # ,'fontsize',12,'fontweight','b');

    plt.show()

    X_XZ, Z_XZ = np.meshgrid(x_array, z_array)
    X_XZ = np.matrix.transpose(X_XZ)
    Z_XZ = np.matrix.transpose(Z_XZ)
    # print(np.abs(z_array[z_ind] - z_pos))
    Dn_XZ = np.zeros((len(x_array), len(z_array)))
    Dn_XZ[:, :] = np.matrix.transpose(Dn[:, y_ind, :])

    # Xmin = X_XY[0][0]
    # Xmax = X_XY[len(X_XY) - 1][len(X_XY[0]) - 1]
    # Ymin = Y_XY[0][0]
    # Ymax = Y_XY[len(Y_XY) - 1][len(Y_XY[0]) - 1]

    fig, ax = plt.subplots()

    c = ax.pcolor(
        X_XZ,
        Z_XZ,
        np.log10(Dn_XZ),
        vmin=0,
        vmax=5,
        cmap="jet",
        shading="auto",
    )
    ax.plot(np.ones(len(z_array)) * x_pos, z_array, c="red")

    ax.set_xlim(z_range[0], z_range[1])
    ax.set_ylim(z_range[0], z_range[1])
    if y_array[y_ind] < radius:
        # planet drawing
        rp = np.sqrt(radius * radius - y_array[y_ind] * y_array[y_ind])
        theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
        xp = rp * np.cos(theta)
        yp = rp * np.sin(theta)
        ax.plot(xp, yp, c="black")

    fig.colorbar(c, ax=ax, label="Density ne log (cm-3)")
    # ax.set_xlim(Xmin, Xmax)
    # ax.set_ylim(Ymin, Ymax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Z")  # ,'fontsize',12,'fontweight','b');
    plt.show()


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

    fig.colorbar(c, ax=ax, label="Density ne log (cm-3)")
    # ax.set_xlim(Xmin, Xmax)
    # ax.set_ylim(Ymin, Ymax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Y")  # ,'fontsize',12,'fontweight','b');
    plt.show()


def plot_in_xy_plane_simple(
    x_array, y_array, z_array, Dn, radius, x_range, y_range, z_pos=0
):
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
    # print(X_XY)
    # print(Dn_XY)

    c = ax.pcolor(
        X_XY,
        Y_XY,
        Dn_XY,
        vmin=-300,
        vmax=300,
        cmap="jet",
        shading="auto",
    )

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    """
    if z_array[z_ind] < radius:
        # planet drawing
        rp = np.sqrt(radius * radius - z_array[z_ind] * z_array[z_ind])
        theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
        xp = rp * np.cos(theta)
        yp = rp * np.sin(theta)
        ax.plot(xp, yp, c="black")

    """

    fig.colorbar(c, ax=ax, label="Density ne log (cm-3)")
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
        np.log10(Dn_XZ),
        vmin=0,
        vmax=5,
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

    fig.colorbar(c, ax=ax, label="Density ne log (cm-3)")

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("X")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Z")  # ,'fontsize',12,'fontweight','b');
    plt.show()


def plot_in_xz_plane_simple(
    x_array, y_array, z_array, Dn, radius, x_range, z_range, y_pos=0
):
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
        vmin=-300,
        vmax=300,
        cmap="jet",
        shading="auto",
    )

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(z_range[0], z_range[1])

    """
    if y_array[y_ind] < radius:
        # planet drawing
        rp = np.sqrt(radius * radius - y_array[y_ind] * y_array[y_ind])
        theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
        xp = rp * np.cos(theta)
        yp = rp * np.sin(theta)
        ax.plot(xp, yp, c="black")
    """

    fig.colorbar(c, ax=ax, label="Density ne log (cm-3)")

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

    fig.colorbar(c, ax=ax, label="Density ne log (cm-3)")
    # ax.set_xlim(Ymin, Ymax)
    # ax.set_ylim(Zmin, Zmax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("Y")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Z")  # ,'fontsize',12,'fontweight','b');

    plt.show()


def plot_in_yz_plane_simple(
    x_array, y_array, z_array, Dn, radius, y_range, z_range, x_pos=0
):
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

    c = ax.pcolor(
        Y_YZ,
        Z_YZ,
        Dn_YZ,
        vmin=-300,
        vmax=300,
        cmap="jet",
        shading="auto",
    )

    ax.set_xlim(y_range[0], y_range[1])
    ax.set_ylim(z_range[0], z_range[1])

    """
    if x_array[x_ind] < radius:
        # planet drawing
        rp = np.sqrt(radius * radius - x_array[x_ind] * x_array[x_ind])
        theta = np.divide(2.0 * math.pi * np.arange(1, 101, 1.0), 100.0)
        xp = rp * np.cos(theta)
        yp = rp * np.sin(theta)
        ax.plot(xp, yp, c="black")
    """

    fig.colorbar(c, ax=ax, label="Density ne log (cm-3)")
    # ax.set_xlim(Ymin, Ymax)
    # ax.set_ylim(Zmin, Zmax)

    titre = "Density ne log[cm-3] time: " + diagtime
    plt.title(titre)  # ,'fontsize',12,'fontweight','b');
    ax.set_xlabel("Y")  # ,'fontsize',12,'fontweight','b');
    ax.set_ylabel("Z")  # ,'fontsize',12,'fontweight','b');

    plt.show()


# %%
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


def spherical_symmetric_function(r1, a0, a1, a2):

    r = r1 - 1
    total = a0 + a1 * np.exp(-1 * (r / a2))

    return total


def density_fitting(src_dir, typefile, rundate, diagtime):

    for i in range(len(typefile)):
        if i == 0:
            ncfile = (
                src_dir + typefile[i] + rundate + "_t" + diagtime + ".nc"
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

        else:
            ncfile1 = (
                src_dir + typefile[i] + rundate + "_t" + diagtime + ".nc"
            )  #'_extract_4Re_grid.nc'#
            ncid1 = Dataset(ncfile1)
            var_nc1 = ncid1.variables

            ncfile0 = (
                src_dir + typefile[i - 1] + rundate + "_t" + diagtime + ".nc"
            )  #'_extract_4Re_grid.nc'#
            ncid0 = Dataset(ncfile1)
            var_nc0 = ncid1.variables

            if (
                np.allclose(var_nc0["s_centr"][:], var_nc1["s_centr"][:])
                & np.allclose(var_nc0["r_planet"][:], var_nc1["r_planet"][:])
                & np.allclose(var_nc0["gstep"][:], var_nc1["gstep"][:])
                & np.allclose(var_nc0["nptot"][:], var_nc1["nptot"][:])
                & np.allclose(var_nc0["phys_length"][:], var_nc1["phys_length"][:])
            ):
                Dn += var_nc1["Density"][:]  # /cc [z_num][y_num][x_num]

            else:
                print(
                    "stop stop stop stop stop stop stop stop stop stop stop stop stop stop stop!"
                )

    # radius=1
    nc = [len(Dn[0][0]), len(Dn[0]), len(Dn)]  # like [len(x),len(y),len(z)]
    # print(var_nc["X_axis"][:])

    Dn = np.where(Dn <= 0, float("NaN"), Dn)
    # maximum and minimum

    # position array from moon center (unit..phys_length)
    x_array_moon_center_phylen = np.arange(0, (nc[0]) * gs[0], gs[0]) - centr[0]
    y_array_moon_center_phylen = np.arange(0, (nc[1]) * gs[1], gs[1]) - centr[1]
    z_array_moon_center_phylen = np.arange(0, (nc[2]) * gs[2], gs[2]) - centr[2]

    # position array from moon center (unit..Re)
    x_array_moon_center_Re = x_array_moon_center_phylen * nrm_len / 1560.8
    y_array_moon_center_Re = y_array_moon_center_phylen * nrm_len / 1560.8
    z_array_moon_center_Re = z_array_moon_center_phylen * nrm_len / 1560.0

    """
    plot_in_xy_plane(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        Dn,
        1,
        [-4, 4],
        [-4, 4],
        z_pos=0,
    )

    plot_in_xz_plane(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        Dn,
        1,
        [-4, 4],
        [-4, 4],
        y_pos=0,
    )

    plot_in_yz_plane(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        Dn,
        1,
        [-4, 4],
        [-4, 4],
        x_pos=0,
    )

    plot_in_xaxis(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        Dn,
        1,
        [-4, 4],
        y_pos=0,
        z_pos=0,
    )

    plot_in_yaxis(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        Dn,
        1,
        [-4, 4],
        x_pos=0,
        z_pos=0,
    )

    plot_in_zaxis(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        Dn,
        1,
        [-4, 4],
        x_pos=0,
        y_pos=0,
    )

    plot_in_xy_diagonal_axis(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        Dn,
        1,
        [-4, 4],
        x_pos=0,
        y_pos=0,
        z_pos=0,
    )
    """
    # position array from moon center meshgrid (unit..Re)
    x_meshgrid, y_meshgrid, z_meshgrid = np.meshgrid(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        indexing="ij",
    )

    x_meshgrid = np.matrix.transpose(x_meshgrid)
    y_meshgrid = np.matrix.transpose(y_meshgrid)
    z_meshgrid = np.matrix.transpose(z_meshgrid)

    x_meshgrid_1d = x_meshgrid.flatten()
    y_meshgrid_1d = y_meshgrid.flatten()
    z_meshgrid_1d = z_meshgrid.flatten()

    Dn_1d = Dn.flatten()
    # print(Dn_1d)
    not_nan_indices = np.where(~np.isnan(Dn_1d))[0]

    r_modeled, theta_modeled, phi_modeled = cartesian_to_spherical(
        x_meshgrid_1d, y_meshgrid_1d, z_meshgrid_1d
    )

    Dn_fitted = Dn_1d[not_nan_indices]
    # print(Dn_fitted)
    r_fitted = r_modeled[not_nan_indices]
    # print(r_fitted)
    theta_fitted = theta_modeled[not_nan_indices]
    phi_fitted = phi_modeled[not_nan_indices]

    fitted_pos = np.where((1 < r_fitted) & (r_fitted < 4))[0]
    Dn_fitted = Dn_fitted[fitted_pos]
    # print(Dn_fitted)
    r_fitted = r_fitted[fitted_pos]
    theta_fitted = theta_fitted[fitted_pos]
    phi_fitted = phi_fitted[fitted_pos]

    A0_ini = (
        20,
        200,
        0.1,
    )  # 初期値
    A0_bound = (
        [10, 10, 0],
        [100, 1000, 1],
    )  # 拘束条件

    # leastsqの戻り値は、最適化したパラメータのリストと、最適化の結果
    start = time.time()  # 現在時刻（処理開始前）を取得
    print("start")
    A1, pcov = curve_fit(
        spherical_symmetric_function,
        r_fitted,
        Dn_fitted,
        p0=A0_ini,
        maxfev=10000000,
        bounds=A0_bound,
    )

    end = time.time()  # 現在時刻（処理完了後）を取得
    time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
    print(time_diff)  # 処理にかかった時間データを使用
    print(A1)
    r_meshgrid, theta_meshgrid, phi_meshgrid = cartesian_to_spherical(
        x_meshgrid, y_meshgrid, z_meshgrid
    )

    fit_dense = np.zeros([len(Dn), len(Dn[0]), len(Dn[0][0])])
    for z in range(len(Dn)):
        print(z)
        for y in range(len(Dn[0])):
            for x in range(len(Dn[0][0])):
                r, theta, phi = cartesian_to_spherical(
                    x_array_moon_center_Re[x],
                    y_array_moon_center_Re[y],
                    z_array_moon_center_Re[z],
                )
                imput = np.array([r, theta, phi])

                fit_dense[z, y, x] = spherical_symmetric_function(
                    r,
                    A1[0],
                    A1[1],
                    A1[2],
                )
    np.save("../Europa/fiting_dense_1", fit_dense)

    D1 = Dn - fit_dense
    np.save("../Europa/error_fiting_dense_1", D1)
    # print(x_array_moon_center_Reｐ
    plot_in_xy_plane(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        fit_dense,
        1,
        [-4, 4],
        [-4, 4],
        z_pos=0,
    )
    plot_in_xz_plane(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        fit_dense,
        1,
        [-4, 4],
        [-4, 4],
        y_pos=0,
    )
    plot_in_yz_plane(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        fit_dense,
        1,
        [-4, 4],
        [-4, 4],
        x_pos=0,
    )

    plot_in_xy_plane_simple(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        D1,
        1,
        [-4, 4],
        [-4, 4],
        z_pos=0,
    )
    plot_in_xz_plane_simple(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        D1,
        1,
        [-4, 4],
        [-4, 4],
        y_pos=0,
    )
    plot_in_yz_plane_simple(
        x_array_moon_center_Re,
        y_array_moon_center_Re,
        z_array_moon_center_Re,
        D1,
        1,
        [-4, 4],
        [-4, 4],
        x_pos=0,
    )


# %%
# src_dir = "../LatHyS_simu/Europa/RUN_A3/"
src_dir = "../LatHyS_simu/Europa/RUN_A3/"
typefile = ["O2pl_", "H2Opl_", "Ojv_", "O2pl_"]
# typefile = ["H2Opl_"]
# typefile = ["H2pl_"]
# typefile = ["O2pl_"]
# typefile = ["Ojv_"]
rundate = "19_04_23"
diagtime = "00600"
# plot_density_ne(src_dir, typefile, rundate, diagtime, zoom=True)
# density_interpolate(src_dir, typefile, rundate, diagtime)

density_fitting(src_dir, typefile, rundate, diagtime)
# %%
