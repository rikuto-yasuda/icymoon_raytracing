import numpy as np
from scipy.optimize import curve_fit
from functools import partial
import math
import matplotlib.pyplot as plt

# scipy.specialから球面調和関数をインポート
from scipy.special import sph_harm


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


def spherical_function(r, theta, phi, m, l):
    """
    Radial function in the spherical coordinate system.
    Args:
    - r: radial distance
    - n, l: quantum numbers
    Returns:
    - value of the radial function
    """
    Kr = np.sqrt((4 * np.pi) / (2 * l + 1))
    # rn = r**l
    rn = r ** (-l - 1)
    sp = sph_harm(m, l, phi, theta)
    sph_func = np.abs(Kr * rn * sp)
    return sph_func


def added_spherical_function(
    X,
    a00,
    a01,
    a11,
    a02,
    a12,
    a22,
    a03,
    a13,
    a23,
    a33,
    a04,
    a14,
    a24,
    a34,
    a44,
    a05,
    a15,
    a25,
    a35,
    a45,
    a55,
):
    r, theta, phi = X
    total = (
        a00 * spherical_function(r, theta, phi, 0, 0)
        + a01 * spherical_function(r, theta, phi, 0, 1)
        + a11 * spherical_function(r, theta, phi, 1, 1)
        + a02 * spherical_function(r, theta, phi, 0, 2)
        + a12 * spherical_function(r, theta, phi, 1, 2)
        + a22 * spherical_function(r, theta, phi, 2, 2)
        + a03 * spherical_function(r, theta, phi, 0, 3)
        + a13 * spherical_function(r, theta, phi, 1, 3)
        + a23 * spherical_function(r, theta, phi, 2, 3)
        + a33 * spherical_function(r, theta, phi, 3, 3)
        + a04 * spherical_function(r, theta, phi, 0, 4)
        + a14 * spherical_function(r, theta, phi, 1, 4)
        + a24 * spherical_function(r, theta, phi, 2, 4)
        + a34 * spherical_function(r, theta, phi, 3, 4)
        + a44 * spherical_function(r, theta, phi, 4, 4)
        + a05 * spherical_function(r, theta, phi, 0, 5)
        + a15 * spherical_function(r, theta, phi, 1, 5)
        + a25 * spherical_function(r, theta, phi, 2, 5)
        + a35 * spherical_function(r, theta, phi, 3, 5)
        + a45 * spherical_function(r, theta, phi, 4, 5)
        + a55 * spherical_function(r, theta, phi, 5, 5)
    )

    return total


# ダミーデータの生成
n_points = 100000
x_data = np.random.uniform(-1, 1, n_points)  # xデータ (-1から1の一様分布)
y_data = np.random.uniform(-1, 1, n_points)  # yデータ (-1から1の一様分布)
z_data = np.random.uniform(-1, 1, n_points)  # zデータ (-1から1の一様分布)
potential_data = 2 - x_data**2 - y_data**2 + 0.2  # ポテンシャルデータ（ランダムな値）

r_modeled, theta_moedled, phi_modeled = cartesian_to_spherical(x_data, y_data, z_data)

popt, pcov = curve_fit(
    added_spherical_function,
    (r_modeled, theta_moedled, phi_modeled),
    potential_data,
    maxfev=1000000000,
)  # poptは最適推定値、pcovは共分散が出力される


plt.scatter(x_data, y_data, c=potential_data, s=1, vmin=0, vmax=2)
plt.colorbar()
plt.show()

plt.scatter(
    x_data,
    y_data,
    c=added_spherical_function(
        (np.sqrt(x_data * x_data + y_data * y_data), np.pi / 2, phi_modeled),
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
    ),
    s=1,
    vmin=0,
    vmax=2,
)
plt.colorbar()
plt.show()
