# %%
from calendar import month
import pprint
import cdflib
import numpy as np
import pandas as pd
import re
import math

# %%

first_lat_low = 30.4
first_lat_high = 30.5  # deg表記
first_lon_low = 113.4
first_lon_high = 113.5

second_lat_low = -1.9
second_lat_high = -1.8
second_lon_low = 171.2
second_lon_high = 171.3

# %%


def calc_deg(naiseki_calced):
    Deg = np.degrees(np.arccos(naiseki_calced))

    return Deg


def naiseki_calc(lat_deg1, lon_deg1, lat_deg2, lon_deg2):
    lat1 = np.radians(lat_deg1)
    lon1 = np.radians(lon_deg1)
    lat2 = np.radians(lat_deg2)
    lon2 = np.radians(lon_deg2)
    first = np.cos(lat1) * np.cos(lat2) * np.cos(lon1) * np.cos(lon2)
    second = np.cos(lat1) * np.cos(lat2) * np.sin(lon1) * np.sin(lon2)
    third = np.sin(lat1) * np.sin(lat2)

    naiseki = first + second + third

    return naiseki


def angle_range(
    lat1_low, lat1_high, lon1_low, lon1_high, lat2_low, lat2_high, lon2_low, lon2_high
):
    min_angle = 180
    max_angle = 0

    lat1 = np.arange(lat1_low, lat1_high, 0.1)
    lon1 = np.arange(lon1_low, lon1_high, 0.1)
    lat2 = np.arange(lat2_low, lat2_high, 0.1)
    lon2 = np.arange(lon2_low, lon2_high, 0.1)

    for i in lat1:
        for j in lon1:
            for k in lat2:
                for l in lon2:
                    naiseki = naiseki_calc(i, j, k, l)
                    deg = calc_deg(naiseki)

                    if deg < min_angle:
                        min_angle = deg
                    if deg > max_angle:
                        max_angle = deg

    return max_angle, min_angle


# %%


def main():
    max, min = angle_range(
        first_lat_low,
        first_lat_high,
        first_lon_low,
        first_lon_high,
        second_lat_low,
        second_lat_high,
        second_lon_low,
        second_lon_high,
    )
    print(max, min)
    return 0


if __name__ == "__main__":
    main()


# %%
