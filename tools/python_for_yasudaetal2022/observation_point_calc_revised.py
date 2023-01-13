import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
import math

# とりあえずガニメデ1フライバイの位置座標を計算しよう！

utctim = '1996-06-27 06:20:00'    # start date/time
et_ex = spice.str2et(utctim)      # seconds ?
#print('ET:', et_ex)
nd = 180  # number of data
dt = 60   # time interval [second]

et = et_ex + dt * np.arange(0, nd)

# reference frame
ref = 'IAU_GANYMEDE'
# light time correction
corr = 'LT+S'

# target
spacecraft = 'GALILEO'
sun = 'SUN'
jupiter = 'JUPITER'
earth = 'EARTH'

# origin
org = 'GANYMEDE'

# get KAGUYA orbit


def Calc_position(tar):
    x = np.zeros(nd)
    y = np.zeros(nd)
    z = np.zeros(nd)
    lat = np.zeros(nd)
    lon = np.zeros(nd)
    theta = np.zeros(nd)
    time = np.arange(0, nd)*dt
    for i in range(0, nd):
        # Get state vector: Moon to Kaguya
        [state, lttime] = spice.spkezr(tar, et[i], ref, corr, org)
        x[i] = state[0]
        y[i] = state[1]
        z[i] = state[2]
        r = math.sqrt(x[i]**2+y[i]**2+z[i]**2)
        theta[i] = math.acos(x[i]/r)*180.0/math.pi
        lat[i] = math.asin(z[i]/r)*180.0/math.pi
        lon[i] = math.atan2(y[i], x[i])*180.0/math.pi

    return time, x, y, z, lat, lon, theta


def Tangetial_point(x1, y1, z1, x2, y2, z2):
    def Flaot_128(hairetu):
        hairetu = np.array(hairetu, dtype=np.float128)

    x1 = Flaot_128(x1)
    y1 = Flaot_128(y1)
    z1 = Flaot_128(z1)

    x2 = Flaot_128(x2)
    y2 = Flaot_128(y2)
    z2 = Flaot_128(z2)

    t_upper = (x2*x2 + y2*y2 + z2*z2) - (x1*x2 + y1*y2 + z1*z2)
    t_lower = (x2*x2 + y2*y2 + z2*z2) + (x1*x1 + y1 *
                                         y1 + z1*z1) - 2*(x1*x2 + y1*y2 + z1*z2)
    t = t_upper / t_lower

    tangential_point_x = t * x1 + (1-t) * x2
    tangential_point_y = t * y1 + (1-t) * y2
    tangential_point_z = t * z1 + (1-t) * z2

    tangential_lat = np.zeros(nd)
    tangential_lon = np.zeros(nd)
    tangential_theta = np.zeros(nd)

    for i in range(0, nd):
        tangential_point_r = math.sqrt(
            tangential_point_x[i]**2+tangential_point_y[i]**2+tangential_point_z[i]**2)
        tangential_theta[i] = math.acos(
            tangential_point_x[i]/tangential_point_r)*180.0/math.pi
        tangential_lat[i] = math.asin(
            tangential_point_z[i]/tangential_point_r)*180.0/math.pi
        tangential_lon[i] = math.atan2(
            tangential_point_y[i], tangential_point_x[i])*180.0/math.pi

    return tangential_point_x, tangential_point_y, tangential_point_z, tangential_lat, tangential_lon


def main():
    spacecraft_x, spacecraft_y, spacecraft_z, spacecraft_lat, spacecraft_lon, spacecraft_theta = Calc_position(
        spacecraft)
    sun_x, sun_y, sun_z, sun_lat, sun_lon, sun_theta = Calc_position(sun)
    jupiter_x, jupiter_y, jupiter_z, jupiter_lat, jupiter_lon, jupiter_theta = Calc_position(
        jupiter)
    earth_x, earth_y, earth_z, earth_lat, earth_lon, earth_theta = Calc_position(
        earth)

    tangetial_craft_earth_x, tangetial_craft_earth_y, tangetial_craft_earth_z, tangetial_craft_earth_lat, tangetial_craft_earth_lon = Tangetial_point(
        spacecraft_x, spacecraft_y, spacecraft_z, earth_x, earth_y, earth_z)
    tangetial_craft_jupiter_x, tangetial_craft_jupiter_y, tangetial_craft_jupiter_z, tangetial_craft_jupiter_lat, tangetial_craft_jupiter_lon = Tangetial_point(
        spacecraft_x, spacecraft_y, spacecraft_z, jupiter_x, jupiter_y, jupiter_z)

    return 0


if __name__ == "__main__":
    main()
