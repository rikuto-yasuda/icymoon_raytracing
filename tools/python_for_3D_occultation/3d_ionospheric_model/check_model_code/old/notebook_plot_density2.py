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
import matplotlib as mpl
import math
import time
from scipy.special import lpmv
from scipy import optimize
import gc


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


"""
def added_spherical_function(a, r, theta, phi):
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)

    total = a[0] * lpmv(0, 1, cos_theta) / (r * r)
    +a[1] * cos_phi * lpmv(1, 1, cos_theta) / (r * r)
    +a[2] * sin_phi * lpmv(1, 1, cos_theta) / (r * r)
    +a[3] * lpmv(0, 2, cos_theta) / (r * r * r)
    +a[4] * cos_phi * lpmv(1, 2, cos_theta) / (r * r * r)
    +a[5] * sin_phi * lpmv(1, 2, cos_theta) / (r * r * r)
    +a[6] * np.cos(2 * phi) * lpmv(2, 2, cos_theta) / (r * r * r)
    +a[7] * np.sin(2 * phi) * lpmv(2, 2, cos_theta) / (r * r * r)
    +a[8] * lpmv(0, 3, cos_theta) / (r * r * r * r)
    +a[9] * cos_phi * lpmv(1, 3, cos_theta) / (r * r * r * r)
    +a[10] * sin_phi * lpmv(1, 3, cos_theta) / (r * r * r * r)
    +a[11] * np.cos(2 * phi) * lpmv(2, 3, cos_theta) / (r * r * r * r)
    +a[12] * np.sin(2 * phi) * lpmv(2, 3, cos_theta) / (r * r * r * r)
    +a[13] * np.cos(3 * phi) * lpmv(3, 3, cos_theta) / (r * r * r * r)
    +a[14] * np.sin(3 * phi) * lpmv(3, 3, cos_theta) / (r * r * r * r)

    return total


def added_spherical_function2(a, r1, theta, phi):

    r = r1 - 1
    total = a[0] + a[1] * np.exp(-1 * (r / a[2]) ** a[3])

    return total
"""


def mimus_pi(arr, threshold):
    result = np.empty(0)
    for element in arr:
        if element > threshold:
            result = np.append(result, element - np.pi)
        else:
            result = np.append(result, element)
    return result


def added_gaussinan_diffusion_function(a, r, theta, phi):

    r_array = r - 1
    nx = np.cos(theta)
    ny = np.sin(theta)
    n0x = np.cos(a[3])
    n0y = np.sin(a[3])
    theta_array = np.arcsin(
        (nx * n0y - ny * n0x)
        / ((np.sqrt(nx * nx + ny * ny)) * (np.sqrt(n0x * n0x + n0y * n0y)))
    )
    phi_array = phi - a[4]
    theta_theta_array = theta_array * theta_array
    theta_phi_array = theta_array * phi_array
    phi_phi_array = phi_array * phi_array

    determinant = a[5] * a[8] - a[6] * a[7]

    f = (
        a[2]
        * np.exp(
            (
                a[8] * theta_theta_array
                + (-1 * (a[6] + a[7]) * theta_phi_array)
                + a[5] * phi_phi_array
            )
            / (-2 * determinant)
        )
        / (np.sqrt(np.abs(determinant)))
    )

    total = a[0] + ((a[1] + f) * np.exp(-1 * ((r_array / a[9]) ** np.abs(a[10]))))
    print(a[3])

    return total


"""
def residual(a, r, theta, phi, den):
    res = den - added_spherical_function2(a, r, theta, phi)
    return res
"""


def residual(a, r, theta, phi, den):
    res = den - added_gaussinan_diffusion_function(a, r, theta, phi)
    return res


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

    """
    plt.scatter(
        surface_phi,
        surface_theta,
        c=surface_Dn,
        cmap="jet",
        vmin=0,
        vmax=1000,
    )

    plt.colorbar()
    print(surface_phi[np.argmax(surface_Dn)], surface_theta[np.argmax(surface_Dn)])

    plt.scatter(
        surface_phi[np.argmax(surface_Dn)],
        surface_theta[np.argmax(surface_Dn)],
        s=30,
        marker="*",
        vmin=0,
        vmax=1000,
    )

    plt.show()
    """
    # a0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #  theta_phi_array=np.array([theta-a[3],phi-a[4]])
    #  matrix = np.array([[a[5], a[6]], [a[7], a[8]]])
    # total = a[0] + (a[1] + (a[2] / (np.abs(np.linalg.det(matrix))**1/2)) * np.exp((-1 / 2) * theta_phi_array.T * np.linalg.inv(matrix) * theta_phi_array))* np.exp(-1 * (r / a[9]) ** a[10])
    a0 = [
        20,
        200,
        800,
        1.4404480651226226,
        -2.5899376710612465,
        1,
        0,
        0,
        1,
        100,
        1,
    ]
    # leastsqの戻り値は、最適化したパラメータのリストと、最適化の結果
    start = time.time()  # 現在時刻（処理開始前）を取得
    print("start")
    a1, ret = optimize.leastsq(
        residual,
        a0,
        args=(r_fitted, theta_fitted, phi_fitted, Dn_fitted),
        maxfev=230000,
    )

    end = time.time()  # 現在時刻（処理完了後）を取得
    time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
    print(time_diff)  # 処理にかかった時間データを使用
    print(a1)
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

                fit_dense[z, y, x] = added_gaussinan_diffusion_function(
                    a1, r, theta, phi
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
