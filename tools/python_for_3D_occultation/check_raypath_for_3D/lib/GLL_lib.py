import numpy as np
import math
import spiceypy as spice
import datetime
from planetary_coverage import MetaKernel

# ---------------------------------------------------------
# Load NAIF SPICE kernels for S/C
# ---------------------------------------------------------


def spice_ini(source_dir):

    # load spice kernel files
    # spice.furnsh(source_dir + "mk/juice_ops.tm")
    spice.furnsh(source_dir + "spk/gll_951120_021126_raj2007.bsp")
    spice.furnsh(source_dir + "sclk/mk00062a.tsc")
    spice.furnsh(source_dir + "ck/gll_plt_rec_1997_mav_v00.bc")

    spice.furnsh(source_dir + "lsk/naif0008.tls")
    spice.furnsh(source_dir + "pck/pck00007.tpc")

    """ 
    spice.furnsh(source_dir + "spk/jup310.bsp")
    spice.furnsh(source_dir + "spk/de440s.bsp")
    spice.furnsh(MetaKernel(source_dir + "mk/juice_plan.tm", kernels=source_dir))
    spice.furnsh(source_dir + "ck/juice_sc_crema_5_1_150lb_default_v01.bc")
    # spice.furnsh(source_dir + "mk/juice_plan.tm")
    spice.furnsh(source_dir + "spk/jup365_19900101_20500101.bsp")
    spice.furnsh(source_dir + "spk/de432s.bsp")
    spice.furnsh(source_dir + "lsk/naif0012.tls")
    spice.furnsh(source_dir + "pck/pck00010.tpc")
    """

    return


# ---------------------------------------------------------
#   Calculate GLL orbit
#   refernce target on the x-axis: x_ref
#   reference frame: ref
#   target: tar
#   origin: org
#   light time correction: corr
# ---------------------------------------------------------


def get_pos_xref(
    et, ref="IAU_EUROPA", tar="GLL", org="EUROPA", x_ref="JUPITER", corr="LT+S"
):
    """_return GLL orbit from EUROPA_

    Args:
        et (_np.array_): _time array_
        ref (str, optional): _referrence_. Defaults to 'IAU_EUROPA'.
        tar (str, optional): _target_. Defaults to 'GLL'.
        org (str, optional): _observer_. Defaults to 'EUROPA'.
        x_ref (str, optional): _x_axis definition (direction from org to x_ref in x-y plane is defied as x_axis)_. Defaults to 'JUPITER'.
        corr (str, optional): _light time correction_. Defaults to 'LT+S'.

    Returns:
        _np.ndarray?_: _[[x,y,z,r,lat,lon](t1), [x,y,z,r,lat,lon](t2), [..](t3),[] ... []]_
    """

    # number of data
    nd = len(et)

    # get S/C orbit
    x = np.zeros(nd)
    y = np.zeros(nd)
    z = np.zeros(nd)
    r = np.zeros(nd)
    lat = np.zeros(nd)
    lon = np.zeros(nd)

    # spice temporal variable
    vec_z = [0.0, 0.0, 1.0]

    for i in range(0, nd):

        # Get state vector of S/C
        [state, lttime] = spice.spkezr(tar, et[i], ref, corr, org)
        x_s = state[0]
        y_s = state[1]
        z_s = state[2]

        # Get state vector of reference target
        [state, lttime] = spice.spkezr(x_ref, et[i], ref, corr, org)
        x_r = state[0]
        y_r = state[1]
        z_r = state[2]

        vec_x = [x_r, y_r, 0.0]
        # x軸(1)をvec_x にz軸をvec_zにする座標変換行列を返す
        mout = spice.twovec(vec_x, 1, vec_z, 3)
        vec_in = [x_s, y_s, z_s]
        vec_out = spice.mxv(mout, vec_in)  # 座標変換のための行列を

        x[i] = vec_out[0]
        y[i] = vec_out[1]
        z[i] = vec_out[2]
        r[i] = math.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2)
        lat[i] = math.asin(z[i] / r[i])
        lon[i] = math.atan2(y[i], x[i])

    return [x, y, z, r, lat, lon]



def get_pos_yref(
    et, ref="IAU_EUROPA", tar="GLL", org="EUROPA", y_ref="JUPITER", corr="LT+S"
):
    """_return GLL orbit from EUROPA_

    Args:
        et (_np.array_): _time array_
        ref (str, optional): _referrence_. Defaults to 'IAU_EUROPA'.
        tar (str, optional): _target_. Defaults to 'GLL'.
        org (str, optional): _observer_. Defaults to 'EUROPA'.
        x_ref (str, optional): _x_axis definition (direction from org to x_ref in x-y plane is defied as x_axis)_. Defaults to 'JUPITER'.
        corr (str, optional): _light time correction_. Defaults to 'LT+S'.

    Returns:
        _np.ndarray?_: _[[x,y,z,r,lat,lon](t1), [x,y,z,r,lat,lon](t2), [..](t3),[] ... []]_
    """

    # number of data
    nd = len(et)

    # get S/C orbit
    x = np.zeros(nd)
    y = np.zeros(nd)
    z = np.zeros(nd)
    r = np.zeros(nd)
    lat = np.zeros(nd)
    lon = np.zeros(nd)

    # spice temporal variable
    vec_z = [0.0, 0.0, 1.0]

    for i in range(0, nd):

        # Get state vector of S/C
        [state, lttime] = spice.spkezr(tar, et[i], ref, corr, org)
        x_s = state[0]
        y_s = state[1]
        z_s = state[2]

        # Get state vector of reference target
        [state, lttime] = spice.spkezr(y_ref, et[i], ref, corr, org)
        x_r = state[0]
        y_r = state[1]
        z_r = state[2]

        vec_x = [x_r, y_r, 0.0]
        # x軸(1)をvec_x にz軸をvec_zにする座標変換行列を返す
        mout = spice.twovec(vec_x, 2, vec_z, 3)
        vec_in = [x_s, y_s, z_s]
        vec_out = spice.mxv(mout, vec_in)  # 座標変換のための行列を

        x[i] = vec_out[0]
        y[i] = vec_out[1]
        z[i] = vec_out[2]
        r[i] = math.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2)
        lat[i] = math.asin(z[i] / r[i])
        lon[i] = math.atan2(y[i], x[i])

    return [x, y, z, r, lat, lon]


# ---------------------------------------------------------
#   Calculate GLL orbit
#   reference frame: IAU_JUPITER
#   target: GLL
#   origin: JUPITER
#   refernce target on the x-axis: x_ref
# ---------------------------------------------------------


def get_GLL_pos_jup(et, x_ref="GANYMEDE"):

    x, y, z, r, lat, lon = get_pos_xref(
        et, ref="IAU_JUPITER", tar="GLL", org="JUPITER", x_ref=x_ref, corr="LT+S"
    )

    return [x, y, z, r, lat, lon]


# ---------------------------------------------------------
#   Calculate GLL orbit
#   reference frame: IAU_EUROPA
#   target: GLL
#   origin: MOON
#   refernce target on the x-axis: x_ref
# ---------------------------------------------------------


def get_GLL_pos_europa(et, x_ref="JUPITER"):

    x, y, z, r, lat, lon = get_pos_xref(
        et, ref="IAU_EUROPA", tar="GLL", org="EUROPA", x_ref=x_ref, corr="LT+S"
    )

    return [x, y, z, r, lat, lon]

# ---------------------------------------------------------
#   Calculate GLL orbit
#   reference frame: IAU_EUROPA
#   target: GLL
#   origin: MOON
#   refernce target on the x-axis: x_ref
# ---------------------------------------------------------


def get_GLL_pos_claire_definition(et, y_ref="JUPITER"):

    x, y, z, r, lat, lon = get_pos_yref(
        et, ref="IAU_EUROPA", tar="GLL", org="EUROPA", y_ref=y_ref, corr="LT+S"
    )

    return [x, y, z, r, lat, lon]

