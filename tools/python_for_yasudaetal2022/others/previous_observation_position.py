# %%
from calendar import month
import pprint
import cdflib
import numpy as np
import pandas as pd
import re
import math

# %%

first_lat = 0  # deg表記
first_lon = 82
second_lat = 3
second_lon = 266

# %%


def calc_deg(naiseki_calced):

    Deg = np.degrees(np.arccos(naiseki_calced))

    return Deg


def naiseki_calc(lat_deg1, lon_deg1, lat_deg2, lon_deg2):

    lat1 = np.radians(lat_deg1)
    lon1 = np.radians(lon_deg1)
    lat2 = np.radians(lat_deg2)
    lon2 = np.radians(lon_deg2)
    first = np.cos(lat1)*np.cos(lat2)*np.cos(lon1)*np.cos(lon2)
    second = np.cos(lat1)*np.cos(lat2)*np.sin(lon1)*np.sin(lon2)
    third = np.sin(lat1)*np.sin(lat2)

    naiseki = first+second+third

    return naiseki

# %%


def main():
    nai = naiseki_calc(first_lat, first_lon, second_lat, second_lon)
    print(calc_deg(nai))
    return 0


if __name__ == "__main__":
    main()


# %%
