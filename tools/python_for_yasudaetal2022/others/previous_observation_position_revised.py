# %%
from calendar import month
import pprint
import cdflib
import numpy as np
import pandas as pd
import re
import math

# %%

object_name = "callisto"  # europa/ganymde/callisto
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 23  # ..th flyby
information_list = ['year', 'month', 'start_day', 'end_day',
                    'start_hour', 'end_hour', 'start_min', 'end_min']


earth_csv_name = "WGC_StateVector_20230104104521.csv"
jupiter_csv_name = "WGC_StateVector_20230104104634.csv"
sun_csv_name = "WGC_StateVector_20230104104710.csv"
spacecraft_csv_name = "WGC_StateVector_20230104104318.csv"


# ヘッダーの位置と末端のデータの位置をheaderとskipfooterで設定
jupiter_csv = pd.read_csv(
    "../../result_for_yasudaetal2022/previous_study_ephemeris_revised/jupiter_ephemeris/"+jupiter_csv_name, header=17, skipfooter=3)
sun_csv = pd.read_csv("../../result_for_yasudaetal2022/previous_study_ephemeris_revised/sun_ephemeris/" +
                      sun_csv_name, header=17, skipfooter=3)
spacecraft_csv = pd.read_csv(
    "../../result_for_yasudaetal2022/previous_study_ephemeris_revised/spacecraft_ephemeris/"+spacecraft_csv_name, header=17, skipfooter=3)

earth_csv = pd.read_csv(
    "../../result_for_yasudaetal2022/previous_study_ephemeris_revised/earth_ephemeris/"+earth_csv_name, header=17, skipfooter=3)

###
"""
今回ダウンロードしている座標系はlongitudeの基準が西経が正、東経が負であるので直す必要あり
"""
# %%


def check_time(csv_1, csv_2, csv_3, csv_4):

    Csv_1 = np.array(csv_1['UTC calendar date'])  # Sun
    Csv_2 = np.array(csv_2['UTC calendar date'])  # Spacecraft
    Csv_3 = np.array(csv_3['UTC calendar date'])  # Jupiter
    Csv_4 = np.array(csv_4['UTC calendar date'])  # Earth
    print(Csv_1, Csv_2, Csv_3, Csv_4)  # エラー出た時用
    if (Csv_1 == Csv_2).all() & (Csv_3 == Csv_2).all() & (Csv_3 == Csv_4).all():
        print("good!")

    else:
        print("bad...")

    return 0


def calc_SZA(sun_ephemeris, spacecraft_latitude_deg, spacecraft_longitude_deg):
    """_summary_
    calculate solar zenith angle (between sun direction and spacecraft direction # latitude and longitude)
    Args:
        sun_ephemeris (_type_): _description_
        spacecraft_ephemeris (_type_): _description_

    Returns:
        _type_: _description_
    """

    sun_longitude_deg = np.array(sun_ephemeris['Longitude (deg)'])

    sun_latitude_deg = np.array(sun_ephemeris['Latitude (deg)'])

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


def calc_Ram(jupiter_ephemeris, spacecraft_latitude_deg, spacecraft_longitude_deg):
    """_summary_
    calculate ram angle (between trailing center longitude and spacecraft direction)
    trailing center longitude : jupiter longitude + 90 deg 
    Args:
        sun_ephemeris (_type_): _description_
        spacecraft_ephemeris (_type_): _description_

    Returns:
        _type_: _description_
    """

    jupiter_trailing_longitude_deg = np.array(
        jupiter_ephemeris['Longitude (deg)'])+90

    jupiter_trailing_latitude_deg = np.zeros(
        len(jupiter_ephemeris['Latitude (deg)']))  # trailing center latitude = 0

    jupiter_longitude_rad = np.radians(jupiter_trailing_longitude_deg)  # 経度Q1
    spacecraft_longitude_rad = np.radians(spacecraft_longitude_deg)  # 経度Q2

    jupiter_latitude_rad = np.radians(jupiter_trailing_latitude_deg)  # 緯度P1
    spacecraft_latitude_rad = np.radians(spacecraft_latitude_deg)  # 緯度P2

    naiseki_calced = naiseki_calc(
        jupiter_latitude_rad, jupiter_longitude_rad, spacecraft_latitude_rad, spacecraft_longitude_rad)

    Ram = np.degrees(np.arccos(naiseki_calced))

    return Ram


def calc_Psi(jupiter_ephemeris, sun_ephemeris):
    """_summary_
    calculate Psi (between trailing center longitude and spacecraft direction)
    trailing center longitude : jupiter longitude + 90 deg 
    Args:
        sun_ephemeris (_type_): _description_
        spacecraft_ephemeris (_type_): _description_

    Returns:
        _type_: _description_
    """

    sun_longitude_deg = np.array(sun_ephemeris['Longitude (deg)'])
    jupiter_trailing_longitude_deg = np.array(
        jupiter_ephemeris['Longitude (deg)']) + 90

    sun_latitude_deg = np.array(sun_ephemeris['Latitude (deg)'])

    jupiter_trailing_latitude_deg = np.zeros(
        len(jupiter_ephemeris['Latitude (deg)']))  # trailing center latitude = 0

    jupiter_trailing_longitude_rad = np.radians(
        jupiter_trailing_longitude_deg)  # 経度Q1
    sun_longitude_rad = np.radians(sun_longitude_deg)  # 経度Q2

    jupiter_latitude_rad = np.radians(jupiter_trailing_latitude_deg)  # 緯度P1
    sun_latitude_rad = np.radians(sun_latitude_deg)  # 緯度P2

    naiseki_calced = naiseki_calc(
        jupiter_latitude_rad, jupiter_trailing_longitude_rad, sun_latitude_rad, sun_longitude_rad)

    Psi = np.degrees(np.arccos(naiseki_calced))

    return Psi


def Earth_jupiter_tangetial_point(spacecraft_ephemeris, earth_ephemeris):

    def Float_64(hairetu):
        hairetu = np.array(hairetu, dtype=np.float64)
        return hairetu

    spacecraft_longitude_deg = Float_64(np.array(
        spacecraft_ephemeris['Longitude (deg)']))
    spacecraft_latitude_deg = Float_64(
        np.array(spacecraft_ephemeris['Latitude (deg)']))
    spacecraft_radius_km = Float_64(
        np.array(spacecraft_ephemeris['Radius (km)']))

    earth_longitude_deg = Float_64(
        np.array(earth_ephemeris['Longitude (deg)']))
    earth_latitude_deg = Float_64(np.array(earth_ephemeris['Latitude (deg)']))
    earth_radius_km = Float_64(np.array(earth_ephemeris['Radius (km)']))

    print(earth_radius_km)

    earth_longitude_rad = np.radians(earth_longitude_deg)  # 経度Q1
    spacecraft_longitude_rad = np.radians(spacecraft_longitude_deg)  # 経度Q2

    earth_latitude_rad = np.radians(earth_latitude_deg)  # 緯度P1
    spacecraft_latitude_rad = np.radians(spacecraft_latitude_deg)  # 緯度P2

    """
    def Kyoku_2_Choku(rad, lon, lat):
        nd = len(rad)
        x = Float_64(np.zeros(nd))
        y = Float_64(np.zeros(nd))
        z = Float_64(np.zeros(nd))

        for i in range(0, nd):
            x[i] = rad[i]*math.cos(lat[i])*math.cos(lon[i])
            y[i] = rad[i]*math.cos(lat[i])*math.sin(lon[i])
            z[i] = rad[i]*math.sin(lat[i])

        return x, y, z

    spacecraft_x, spacecraft_y, spacecraft_z = Kyoku_2_Choku(
        spacecraft_radius_km, spacecraft_longitude_rad, spacecraft_latitude_rad)

    earth_x, earth_y, earth_z = Kyoku_2_Choku(
        earth_radius_km, earth_longitude_rad, earth_latitude_rad)

    #print(spacecraft_x, spacecraft_y, spacecraft_z)
    #print(earth_x, earth_y, earth_z)
    print('{:.30g}'.format(earth_x[0]))

    t_upper = (earth_x*earth_x + earth_y*earth_y + earth_z*earth_z) - \
        (spacecraft_x*earth_x + spacecraft_y*earth_y + spacecraft_z*earth_z)
    t_lower = (earth_x*earth_x + earth_y*earth_y + earth_z*earth_z) + (spacecraft_x*spacecraft_x + spacecraft_y *
                                                                       spacecraft_y + spacecraft_z*spacecraft_z) - 2*(spacecraft_x*earth_x + spacecraft_y*earth_y + spacecraft_z*earth_z)
    t = t_upper / t_lower
    print('{:.30g}'.format(t[0]))
    tangential_point_x = t * spacecraft_x + (1-t) * earth_x
    tangential_point_y = t * spacecraft_y + (1-t) * earth_x
    tangential_point_z = t * spacecraft_z + (1-t) * earth_x

    tangential_lat = np.zeros(len(tangential_point_x))
    tangential_lon = np.zeros(len(tangential_point_y))
    tangential_point_r = np.zeros(len(tangential_point_y))

    for i in range(0, len(tangential_point_x)):
        tangential_point_r[i] = math.sqrt(
            tangential_point_x[i]**2+tangential_point_y[i]**2+tangential_point_z[i]**2)
        tangential_lat[i] = math.asin(
            tangential_point_z[i]/tangential_point_r[i])*180.0/math.pi
        tangential_lon[i] = math.atan2(
            tangential_point_y[i], tangential_point_x[i])*180.0/math.pi

    """

    def Kyoku_2_Choku(lon, lat):
        nd = len(lon)
        x = Float_64(np.zeros(nd))
        y = Float_64(np.zeros(nd))
        z = Float_64(np.zeros(nd))

        for i in range(0, nd):
            x[i] = math.cos(lat[i])*math.cos(lon[i])
            y[i] = math.cos(lat[i])*math.sin(lon[i])
            z[i] = math.sin(lat[i])

        return x, y, z

    spacecraft_x, spacecraft_y, spacecraft_z = Kyoku_2_Choku(
        spacecraft_longitude_rad, spacecraft_latitude_rad)
    earth_x, earth_y, earth_z = Kyoku_2_Choku(
        earth_longitude_rad, earth_latitude_rad)

    def Real_Kyoku_2_Choku(rad, lon, lat):
        nd = len(lon)
        x = Float_64(np.zeros(nd))
        y = Float_64(np.zeros(nd))
        z = Float_64(np.zeros(nd))

        for i in range(0, nd):
            x[i] = rad[i]*math.cos(lat[i])*math.cos(lon[i])
            y[i] = rad[i]*math.cos(lat[i])*math.sin(lon[i])
            z[i] = rad[i]*math.sin(lat[i])

        return x, y, z

    real_spacecraft_x, real_spacecraft_y, real_spacecraft_z = Real_Kyoku_2_Choku(
        spacecraft_radius_km, spacecraft_longitude_rad, spacecraft_latitude_rad)

    n = len(earth_x)
    tangential_lat = np.zeros(n)
    tangential_lon = np.zeros(n)
    tangential_rad = np.zeros(n)

    for j in range(0, n):
        spacecraft_vector = np.array(
            [spacecraft_x[j], spacecraft_y[j], spacecraft_z[j]])

        real_spacecraft_vector = np.array(
            [real_spacecraft_x[j], real_spacecraft_y[j], real_spacecraft_z[j]])

        earth_vector = np.array([earth_x[j], earth_y[j], earth_z[j]])
        vertical_vector = np.cross(spacecraft_vector, earth_vector)
        vertical_vector_length = np.linalg.norm(vertical_vector)
        vertical_unit_vector = vertical_vector / \
            vertical_vector_length  # 探査機方向と地球方向のベクトルに垂直な外積
        tangential_point = np.cross(
            earth_vector, vertical_unit_vector)  # 掩蔽地点の座標をxyz座標で

        tangential_point_length = np.linalg.norm(tangential_point)
        tangential_unit_vector = tangential_point / \
            tangential_point_length  # 掩蔽地点の方向を表す単位ベクトル　xyz

        tangential_lat[j] = math.asin(
            tangential_unit_vector[2])*180.0/math.pi
        tangential_lon[j] = math.atan2(
            tangential_unit_vector[1], tangential_unit_vector[0])*180.0/math.pi

        tangential_rad[j] = np.dot(real_spacecraft_vector, tangential_point)

    return tangential_lat, tangential_lon, tangential_rad


def save_all_result(time, sun_ephemeris, jupiter_ephemeris, spacecraft_ephemeris, occultaion_lat, occultaion_lon, occultation_rad, SZA_spacecraft, Ram_spacecraft, SZA_occultaion, Ram_occultaion, Psi):
    df2 = pd.DataFrame(time, columns=['UTC date'])

    def East_2_west(east_lon):
        west_long = (east_lon * -1.0) % 360
        return west_long

    df2['Sab_solar_longitude(deg)'] = East_2_west(
        np.array(sun_ephemeris['Longitude (deg)']))
    df2['Sab_solar_latitude(deg)'] = np.array(sun_ephemeris['Latitude (deg)'])
    trailing_center_wlong = East_2_west(
        np.array(jupiter_ephemeris['Longitude (deg)']) + 90.0)
    df2['Trailing_center_longitude(deg)'] = trailing_center_wlong
    df2['Leadling_center_longitude(deg)'] = trailing_center_wlong - 180
    df2['Spacecraft_longitude(deg)'] = East_2_west(
        np.array(spacecraft_ephemeris['Longitude (deg)']))
    df2['Spacecraft_latitude(deg)'] = np.array(
        spacecraft_ephemeris['Latitude (deg)'])
    df2['Occultation_latitude(deg)'] = occultaion_lat
    df2['Occultainon_longitude(deg'] = East_2_west(occultaion_lon)
    df2['Occultation_radius(km)'] = occultation_rad
    # ここに電波掩蔽のtangential pointでの緯度経度を書きたい

    df2['Spacecraft SZA(deg)'] = SZA_spacecraft
    df2['Spacecraft Ram_angle(deg)'] = Ram_spacecraft
    df2['Occultaion SZA(deg)'] = SZA_occultaion
    df2['Occultaion Ram_angle(deg)'] = Ram_occultaion

    df2['Psi(deg)'] = Psi
    print(df2)
    df2.to_csv('../../result_for_yasudaetal2022/previous_study_ephemeris_revised'
               + '/results/' + object_name+'_'+str(time_of_flybies)+'_flyby.csv', index=False)

    return 0
# %%


def main():
    check_time(sun_csv, spacecraft_csv, jupiter_csv, earth_csv)

    spacecraft_longitude_deg = np.array(spacecraft_csv['Longitude (deg)'])
    spacecraft_latitude_deg = np.array(spacecraft_csv['Latitude (deg)'])

    Time, SZA_spacecraft = calc_SZA(
        sun_csv, spacecraft_latitude_deg, spacecraft_longitude_deg)
    Ram_spacecraft = calc_Ram(
        jupiter_csv, spacecraft_latitude_deg, spacecraft_longitude_deg)

    Psi = calc_Psi(jupiter_csv, sun_csv)
    Occult_lat, Occult_lon, Occult_rad = Earth_jupiter_tangetial_point(
        spacecraft_csv, earth_csv)

    Time, SZA_occult = calc_SZA(sun_csv, Occult_lat, Occult_lon)
    Ram_occult = calc_Ram(jupiter_csv, Occult_lat, Occult_lon)

    save_all_result(Time, sun_csv, jupiter_csv, spacecraft_csv, Occult_lat,
                    Occult_lon, Occult_rad, SZA_spacecraft, Ram_spacecraft, SZA_occult, Ram_occult, Psi)
    return 0


if __name__ == "__main__":
    main()


# %%
