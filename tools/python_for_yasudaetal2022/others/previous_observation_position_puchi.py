# %%
from calendar import month
import pprint
import cdflib
import numpy as np
import pandas as pd
import re
import math

# %%

object_name = "ganymede"  # europa/ganymde/callisto
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 8  # ..th flyby
information_list = ['year', 'month', 'start_day', 'end_day',
                    'start_hour', 'end_hour', 'start_min', 'end_min']


jupiter_csv_name = "WGC_StateVector_20221015155655.csv"
sun_csv_name = "WGC_StateVector_20221015155614.csv"
spacecraft_csv_name = "WGC_StateVector_20221015155522.csv"

# ヘッダーの位置と末端のデータの位置をheaderとskipfooterで設定
jupiter_csv = pd.read_csv(
    "../../result_for_yasudaetal2022/previous_study_ephemeris/jupiter_ephemeris/"+jupiter_csv_name, header=17, skipfooter=3)
sun_csv = pd.read_csv("../../result_for_yasudaetal2022/previous_study_ephemeris/sun_ephemeris/" +
                      sun_csv_name, header=17, skipfooter=3)
spacecraft_csv = pd.read_csv(
    "../../result_for_yasudaetal2022/previous_study_ephemeris/spacecraft_ephemeris/"+spacecraft_csv_name, header=17, skipfooter=3)

# %%


def check_time(csv_1, csv_2, csv_3):
    Csv_1 = np.array(csv_1['UTC calendar date'])
    Csv_2 = np.array(csv_2['UTC calendar date'])
    Csv_3 = np.array(csv_3['UTC calendar date'])

    if (Csv_1 == Csv_2).all() & (Csv_3 == Csv_2).all():
        print("good!")

    else:
        print("bad...")

    return 0


def calc_SZA(sun_ephemeris, spacecraft_ephemeris):

    sun_longitude_deg = np.array(sun_ephemeris['Longitude (deg)'])
    spacecraft_longitude_deg = np.array(
        spacecraft_ephemeris['Longitude (deg)'])

    sun_latitude_deg = np.array(sun_ephemeris['Latitude (deg)'])
    spacecraft_latitude_deg = np.array(spacecraft_ephemeris['Latitude (deg)'])

    sun_longitude_rad = np.radians(sun_longitude_deg)  # 経度Q1
    spacecraft_longitude_rad = np.radians(spacecraft_longitude_deg)  # 経度Q2

    sun_latitude_rad = np.radians(sun_latitude_deg)  # 緯度P1
    spacecraft_latitude_rad = np.radians(spacecraft_latitude_deg)  # 緯度P2

    naiseki_calced = naiseki_calc(
        sun_latitude_rad, sun_longitude_rad, spacecraft_latitude_rad, spacecraft_longitude_rad)

    SZA = np.degrees(np.arccos(naiseki_calced))

    # SZL = sun_longitude_deg - spacecraft_longitude_deg

    Time = np.array(sun_ephemeris['UTC calendar date'])

    return Time, SZA


def naiseki_calc(lat1, lon1, lat2, lon2):
    first = np.cos(lat1)*np.cos(lat2)*np.cos(lon1)*np.cos(lon2)
    second = np.cos(lat1)*np.cos(lat2)*np.sin(lon1)*np.sin(lon2)
    third = np.sin(lat1)*np.sin(lat2)

    naiseki = first+second+third

    return naiseki


def calc_Ram(jupiter_ephemeris, spacecraft_ephemeris):
    spacecraft_longitude_deg = np.array(
        spacecraft_ephemeris['Longitude (deg)'])
    jupiter_longitude_deg = np.array(
        jupiter_ephemeris['Longitude (deg)'])+90

    jupiter_latitude_deg = np.array(jupiter_ephemeris['Latitude (deg)'])
    spacecraft_latitude_deg = np.array(spacecraft_ephemeris['Latitude (deg)'])

    jupiter_longitude_rad = np.radians(jupiter_longitude_deg)  # 経度Q1
    spacecraft_longitude_rad = np.radians(spacecraft_longitude_deg)  # 経度Q2

    jupiter_latitude_rad = np.radians(jupiter_latitude_deg)  # 緯度P1
    spacecraft_latitude_rad = np.radians(spacecraft_latitude_deg)  # 緯度P2

    naiseki_calced = naiseki_calc(
        jupiter_latitude_rad, jupiter_longitude_rad, spacecraft_latitude_rad, spacecraft_longitude_rad)

    Ram = np.degrees(np.arccos(naiseki_calced))

    return Ram


def calc_Psi(jupiter_ephemeris, sun_ephemeris):
    sun_longitude_deg = np.array(sun_ephemeris['Longitude (deg)'])
    jupiter_longitude_deg = np.array(
        jupiter_ephemeris['Longitude (deg)']) + 90

    Psi = (jupiter_longitude_deg - sun_longitude_deg) % 360
    Psi = np.where(Psi > 180, 360 - Psi, Psi)

    return Psi


def save_all_result(time, SZA, Ram, Psi):
    df2 = pd.DataFrame(time, columns=['UTC date'])
    df2['SZA(deg)'] = SZA
    df2['Ram_angle(deg)'] = Ram
    df2['Psi(deg)'] = Psi
    print(df2)
    df2.to_csv('../../result_for_yasudaetal2022/previous_study_ephemeris'
               + '/results/' + object_name+'_'+str(time_of_flybies)+'_flyby.csv', index=False)

    return 0
# %%


def main():
    check_time(sun_csv, spacecraft_csv, jupiter_csv)
    Time, SZA = calc_SZA(sun_csv, spacecraft_csv)
    Ram = calc_Ram(jupiter_csv, spacecraft_csv)
    Psi = calc_Psi(jupiter_csv, sun_csv)
    save_all_result(Time, SZA, Ram, Psi)
    return 0


if __name__ == "__main__":
    main()


# %%
