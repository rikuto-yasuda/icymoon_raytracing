# %%
from calendar import month
import pprint
import cdflib
import numpy as np
import pandas as pd
import re
import math

# %%

object_name = "europa"  # europa/ganymde/callisto
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 12  # ..th flyby
information_list = [
    "year",
    "month",
    "start_day",
    "end_day",
    "start_hour",
    "end_hour",
    "start_min",
    "end_min",
]

begining_ingress_hour = 12
begining_ingress_minute = 00

end_ingress_hour = 12
end_ingress_minute = 5

lowest_frequency_ingress = 0.4
highest_frequecy_ingress = 6.0

radio_type_ingress = "D"  # 複数選択可能にしたい

begining_egress_hour = 12
begining_egress_minute = 10

end_egress_hour = 12
end_egress_minute = 15

lowest_frequency_egress = 0.4
highest_frequecy_egress = 6.0
radio_type_egress = "D"  # 複数選択可能にしたい

# Callisto
if object_name == "callisto":
    Freq_str = [
        "3.612176179885864258e5",
        "3.984813988208770752e5",
        "4.395893216133117676e5",
        "4.849380254745483398e5",
        "5.349649786949157715e5",
        "5.901528000831604004e5",
        "6.510338783264160156e5",
        "7.181954979896545410e5",
        "7.922856807708740234e5",
        "8.740190267562866211e5",
        "9.641842246055603027e5",
        "1.063650846481323242e6",
        "1.173378825187683105e6",
        "1.294426321983337402e6",
        "1.427961349487304688e6",
        "1.575271964073181152e6",
        "1.737779378890991211e6",
        "1.917051434516906738e6",
        "2.114817380905151367e6",
        "2.332985162734985352e6",
        "2.573659420013427734e6",
        "2.839162111282348633e6",
        "3.132054328918457031e6",
        "3.455161809921264648e6",
        "3.811601638793945312e6",
        "4.204812526702880859e6",
        "4.638587474822998047e6",
        "5.117111206054687500e6",
        "5.644999980926513672e6",
    ]

# Ganymede
if object_name == "ganymede":
    Freq_str = [
        "3.984813988208770752e5",
        "4.395893216133117676e5",
        "4.849380254745483398e5",
        "5.349649786949157715e5",
        "5.901528000831604004e5",
        "6.510338783264160156e5",
        "7.181954979896545410e5",
        "7.922856807708740234e5",
        "8.740190267562866211e5",
        "9.641842246055603027e5",
        "1.063650846481323242e6",
        "1.173378825187683105e6",
        "1.294426321983337402e6",
        "1.427961349487304688e6",
        "1.575271964073181152e6",
        "1.737779378890991211e6",
        "1.917051434516906738e6",
        "2.114817380905151367e6",
        "2.332985162734985352e6",
        "2.573659420013427734e6",
        "2.839162111282348633e6",
        "3.132054328918457031e6",
        "3.455161809921264648e6",
        "3.811601638793945312e6",
        "4.204812526702880859e6",
        "4.638587474822998047e6",
        "5.117111206054687500e6",
        "5.644999980926513672e6",
    ]

# Europa
if object_name == "europa":
    Freq_str = [
        "3.984813988208770752e5",
        "4.395893216133117676e5",
        "4.849380254745483398e5",
        "5.349649786949157715e5",
        "5.901528000831604004e5",
        "6.510338783264160156e5",
        "7.181954979896545410e5",
        "7.922856807708740234e5",
        "8.740190267562866211e5",
        "9.641842246055603027e5",
        "1.063650846481323242e6",
        "1.173378825187683105e6",
        "1.294426321983337402e6",
        "1.427961349487304688e6",
        "1.575271964073181152e6",
        "1.737779378890991211e6",
        "1.917051434516906738e6",
        "2.114817380905151367e6",
        "2.332985162734985352e6",
        "2.573659420013427734e6",
        "2.839162111282348633e6",
        "3.132054328918457031e6",
        "3.455161809921264648e6",
        "3.811601638793945312e6",
        "4.204812526702880859e6",
        "4.638587474822998047e6",
        "5.117111206054687500e6",
        "5.644999980926513672e6",
    ]

Freq_num = []
for idx in Freq_str:
    Freq_num.append(float(idx) / 1000000)

# IAU_JUPITERで接点を求める用


def Pick_up_time():
    flyby_list_path = "../result_for_yasudaetal2022/occultation_flyby_list.csv"
    flyby_list = pd.read_csv(flyby_list_path)

    # csvファイルにフライバイごとで使う軌道データを記入しておく　上記のパラメータから必要なデータのファイル名が選ばれて読み込まれる
    # queryが数値非対応なのでまずはフライバイ数で絞り込み
    selected_flyby_list = flyby_list[flyby_list["flyby_time"] == time_of_flybies]
    complete_selecred_flyby_list = selected_flyby_list.query(
        'object == "' + object_name + '" & spacecraft == "' + spacecraft_name + '"'
    )  # queryでフライバイ数以外を絞り込み

    complete_selecred_flyby_list = complete_selecred_flyby_list.reset_index(drop=True)

    # complete_selecred_flyby_list = complete_selecred_flyby_list.index.tolist()

    # print(complete_selecred_flyby_list)

    # csvから時刻データを抽出['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']
    time_information = []
    for i in information_list:
        time_information.append(int(complete_selecred_flyby_list[i][0]))

    # csvの時刻データと電波データの名前をかえす
    return time_information


def Pick_up_spacecraft_csv():
    flyby_list_path = "../result_for_yasudaetal2022/occultation_flyby_list.csv"
    flyby_list = pd.read_csv(flyby_list_path)

    # csvファイルにフライバイごとで使う軌道データを記入しておく　上記のパラメータから必要なデータのファイル名が選ばれて読み込まれる
    # queryが数値非対応なのでまずはフライバイ数で絞り込み
    selected_flyby_list = flyby_list[flyby_list["flyby_time"] == time_of_flybies]
    complete_selecred_flyby_list = selected_flyby_list.query(
        'object == "' + object_name + '" & spacecraft == "' + spacecraft_name + '"'
    )  # queryでフライバイ数以外を絞り込み
    complete_selecred_flyby_list = complete_selecred_flyby_list.reset_index(
        drop=True
    )  # index振り直し
    # 使うcsvファイルの名前を取得
    csv_name = str(complete_selecred_flyby_list["spacecraft_ephemeris_csv"][0])

    # csvから時刻データを抽出
    time_information = []
    for i in information_list:
        time_information.append(str(complete_selecred_flyby_list[i][0]))

    spacecraft_csv_path = (
        "../result_for_yasudaetal2022/spacecraft_ephemeris/" + csv_name
    )
    spacecraft_ephemeris_csv = pd.read_csv(
        spacecraft_csv_path, header=13, skipfooter=7
    )  # ヘッダーの位置と末端のデータの位置をheaderとskipfooterで設定

    return (
        spacecraft_ephemeris_csv,
        time_information,
    )  # IAU_JUPITERで考えた木星から見た探査機の位置


# IAU_MOONに変換する用


def Pick_up_spacecraft_csv_coordinate():
    flyby_list_path = "../result_for_yasudaetal2022/occultation_flyby_list.csv"
    flyby_list = pd.read_csv(flyby_list_path)

    # csvファイルにフライバイごとで使う軌道データを記入しておく　上記のパラメータから必要なデータのファイル名が選ばれて読み込まれる
    # queryが数値非対応なのでまずはフライバイ数で絞り込み
    selected_flyby_list = flyby_list[flyby_list["flyby_time"] == time_of_flybies]
    complete_selecred_flyby_list = selected_flyby_list.query(
        'object == "' + object_name + '" & spacecraft == "' + spacecraft_name + '"'
    )  # queryでフライバイ数以外を絞り込み
    complete_selecred_flyby_list = complete_selecred_flyby_list.reset_index(
        drop=True
    )  # index振り直し
    # 使うcsvファイルの名前を取得
    csv_name = str(
        complete_selecred_flyby_list["spacecraft_ephemeris_coordinate_csv"][0]
    )

    # csvから時刻データを抽出
    time_information = []
    for i in information_list:
        time_information.append(str(complete_selecred_flyby_list[i][0]))

    spacecraft_csv_path = (
        "../result_for_yasudaetal2022/ephemeris_for_coordinate_transformation/spacecraft/"
        + csv_name
    )
    spacecraft_ephemeris_coordinate_csv = pd.read_csv(
        spacecraft_csv_path, header=15, skipfooter=5
    )  # ヘッダーの位置と末端のデータの位置をheaderとskipfooterで設定

    # IAU_"moon"で考えた木星から見た探査機の位置
    return spacecraft_ephemeris_coordinate_csv, time_information


# IAU_JUPITERで接点を求める用


def Pick_up_moon_csv():
    flyby_list_path = "../result_for_yasudaetal2022/occultation_flyby_list.csv"
    flyby_list = pd.read_csv(flyby_list_path)

    # csvファイルにフライバイごとで使う軌道データを記入しておく　上記のパラメータから必要なデータのファイル名が選ばれて読み込まれる
    # queryが数値非対応なのでまずはフライバイ数で絞り込み
    selected_flyby_list = flyby_list[flyby_list["flyby_time"] == time_of_flybies]
    complete_selecred_flyby_list = selected_flyby_list.query(
        'object == "' + object_name + '" & spacecraft == "' + spacecraft_name + '"'
    )  # queryでフライバイ数以外を絞り込み
    complete_selecred_flyby_list = complete_selecred_flyby_list.reset_index(
        drop=True
    )  # index振り直し
    # 使うcsvファイルの名前を取得
    csv_name = str(complete_selecred_flyby_list["moon_ephemeris_csv"][0])

    # csvから時刻データを抽出
    time_information = []
    for i in information_list:
        time_information.append(str(complete_selecred_flyby_list[i][0]))

    spacecraft_csv_path = "../result_for_yasudaetal2022/moon_ephemeris/" + csv_name
    spacecraft_ephemeris_csv = pd.read_csv(
        spacecraft_csv_path, header=15, skipfooter=4
    )  # ヘッダーの位置と末端のデータの位置をheaderとskipfooterで設定

    return (
        spacecraft_ephemeris_csv,
        time_information,
    )  # IAU_JUPITERで考えた木星から見た月の位置


# IAU_MOONに変換する用


def Pick_up_moon_csv_coordinate():
    flyby_list_path = "../result_for_yasudaetal2022/occultation_flyby_list.csv"
    flyby_list = pd.read_csv(flyby_list_path)

    # csvファイルにフライバイごとで使う軌道データを記入しておく　上記のパラメータから必要なデータのファイル名が選ばれて読み込まれる
    # queryが数値非対応なのでまずはフライバイ数で絞り込み
    selected_flyby_list = flyby_list[flyby_list["flyby_time"] == time_of_flybies]
    complete_selecred_flyby_list = selected_flyby_list.query(
        'object == "' + object_name + '" & spacecraft == "' + spacecraft_name + '"'
    )  # queryでフライバイ数以外を絞り込み
    complete_selecred_flyby_list = complete_selecred_flyby_list.reset_index(
        drop=True
    )  # index振り直し
    # 使うcsvファイルの名前を取得
    csv_name = str(complete_selecred_flyby_list["moon_ephemeris_coordinate_csv"][0])

    # csvから時刻データを抽出
    time_information = []
    for i in information_list:
        time_information.append(str(complete_selecred_flyby_list[i][0]))

    moon_csv_path = (
        "../result_for_yasudaetal2022/ephemeris_for_coordinate_transformation/moon/"
        + csv_name
    )
    moon_ephemeris_coordinate_csv = pd.read_csv(
        moon_csv_path, header=15, skipfooter=4
    )  # ヘッダーの位置と末端のデータの位置をheaderとskipfooterで設定

    return (
        moon_ephemeris_coordinate_csv,
        time_information,
    )  # IAU_"moon"で考えた木星から見た月の位置


def Check_time_validity_csv(time, spacecraft_csv_data, moon_csv_data):
    # 探査機の位置データの時間範囲・フライバイリストで指定している時刻データ範囲が一致するか確認
    Check_time_range_validity(time, spacecraft_csv_data)
    # 月の位置データの時間範囲・フライバイリストで指定している時刻データ範囲が一致するか確認
    Check_time_range_validity(time, moon_csv_data)
    # 探査機の位置データの時間ステップ数・フライバイリストで指定している時刻からか計算されるステップ数が一致するか確認
    Check_time_step_validity(time, spacecraft_csv_data)
    # 月の位置データの時間ステップ数・フライバイリストで指定している時刻からか計算されるステップ数が一致するか確認
    Check_time_step_validity(time, moon_csv_data)


def Check_time_range_validity(time, csv_data):
    first_str = str(
        csv_data["UTC calendar date"][0]
    )  # 位置データにおける初めの時刻の文字列
    last_str = str(
        csv_data["UTC calendar date"][-1:]
    )  # 位置データにおける最後の時刻の文字列

    # 位置データにおける初めの時刻の文字列から時刻データを抽出　フライバイリストの情報と一致しているか確認
    if (
        int(re.findall(r"\d+", first_str)[0]) != int(time[0])
        or int(re.findall(r"\d+", first_str)[1]) != int(time[1])
        or int(re.findall(r"\d+", first_str)[2]) != int(time[2])
        or int(re.findall(r"\d+", first_str)[3]) != int(time[4])
        or int(re.findall(r"\d+", first_str)[4]) != int(time[6])
    ):
        print("wrong time!!!!!")

    # 位置データにおける最後の時刻の文字列から時刻データを抽出　フライバイリストの情報と一致しているか確認
    if (
        int(re.findall(r"\d+", last_str)[3]) != int(time[3])
        or int(re.findall(r"\d+", last_str)[4]) != int(time[5])
        or int(re.findall(r"\d+", last_str)[5]) != int(time[7])
    ):
        print(int(re.findall(r"\d+", first_str)[0]))
        print("wrong time!!!!!")

    else:
        print("data range is correct")


def Check_time_step_validity(time, csv_data):
    day_range = int(time[3]) - int(time[2])
    hour_range = int(time[5]) - int(time[4])
    min_range = int(time[7]) - int(time[6])

    step_count = (
        day_range * 1440 + hour_range * 60 + min_range + 1
    )  # フライバイリストからステップ数を計算（今は1step1minを仮定してステップ数を計算）
    # フライバイリストのステップ数と位置データのステップ数が一致する確認（今は1step1minを仮定してステップ数を計算）
    if step_count == len(csv_data["UTC calendar date"][:]):
        print("data step is correct")
    else:
        print("wrong time step")


def Output_moon_radius(moon_name):
    moon_radius = None

    if moon_name == "io":
        moon_radius = 1821.6

    elif moon_name == "europa":
        moon_radius = 1560.8

    elif moon_name == "ganymede":
        moon_radius = 2634.1

    elif moon_name == "callisto":
        moon_radius = 2410.3

    else:
        print(
            "undefined object_name, please check the object_name (moon name) input and def Output_moon_radius function"
        )

    return moon_radius


def Coordinate_sysytem_transformation(polar_coordinate_data_csv):
    time_step_str = np.empty((0, 7), int)
    # 'UTC calendar date'の一行ごとに数字要素だけを抽出&挿入
    for i in range(len(polar_coordinate_data_csv["UTC calendar date"])):
        time_step_str = np.append(
            time_step_str,
            np.array(
                [re.findall(r"\d+", polar_coordinate_data_csv["UTC calendar date"][i])]
            ),
            axis=0,
        )

    # 各要素を文字データから数値データに [year,month,day,hour,min,sec,0]の7次元データの集まり
    time_step = time_step_str.astype(np.int32)

    # 経度・緯度・半径データを読み込み
    spacecraft_longitude_deg = np.array(polar_coordinate_data_csv["Longitude (deg)"])

    reshape_logitude_deg = spacecraft_longitude_deg.reshape(
        [len(spacecraft_longitude_deg), 1]
    )

    spacecraft_longitude_rad = np.radians(spacecraft_longitude_deg)

    spacecraft_latitude_rad = np.radians(polar_coordinate_data_csv["Latitude (deg)"])

    spacecraft_radius_km = np.array(polar_coordinate_data_csv["Radius (km)"])

    # x,y,z デカルト座標に変換
    x = np.array(
        spacecraft_radius_km
        * np.cos(spacecraft_longitude_rad)
        * np.cos(spacecraft_latitude_rad)
    )
    y = np.array(
        spacecraft_radius_km
        * np.sin(spacecraft_longitude_rad)
        * np.cos(spacecraft_latitude_rad)
    )

    z = np.array(spacecraft_radius_km * np.sin(spacecraft_latitude_rad))
    # reshapeして二次元配列に　この後に時刻データと結合するため
    reshape_x = x.reshape([len(x), 1])
    reshape_y = y.reshape([len(y), 1])
    reshape_z = z.reshape([len(z), 1])

    # 時刻データと位置データを結合[year,month,day,hour,min,sec,0,x,y,z,経度]の10次元データの集まり
    time_and_position = np.hstack(
        (time_step, reshape_x, reshape_y, reshape_z, reshape_logitude_deg)
    )

    return time_and_position


def Iau_jupiter_2_iau_moon(
    row_latitude,
    row_longitude,
    position1_moon_sys,
    position1_jup_sys,
    position2_moon_sys,
    position2_jup_sys,
):
    # row_latitude, row_longitude ともにradiunで入力
    # position dataはそれぞれxyzのベクトルで
    # 　以下の計算の考え方はredmeを参照（修論内）
    deff_position1 = position1_moon_sys - position1_jup_sys
    deff_position2 = position2_moon_sys - position2_jup_sys

    rotation_normal_axis = np.cross(deff_position1, deff_position2) / (
        np.linalg.norm(np.cross(deff_position1, deff_position2))
    )
    perp_position_jup = np.cross(position1_jup_sys, rotation_normal_axis) / (
        np.linalg.norm(np.cross(position1_jup_sys, rotation_normal_axis))
    )
    perp_position_moon = np.cross(position1_moon_sys, rotation_normal_axis) / (
        np.linalg.norm(np.cross(position1_moon_sys, rotation_normal_axis))
    )

    rotation_sin_theta = np.dot(
        np.cross(perp_position_jup, perp_position_moon), rotation_normal_axis
    )
    # rotation_cos_theta = np.cos(np.arcsin(rotation_sin_theta)
    rotation_cos_theta = np.dot(perp_position_jup, perp_position_moon)

    # print(rotation_cos_theta**2 + rotation_sin_theta**2)

    n1 = rotation_normal_axis[0]
    n2 = rotation_normal_axis[1]
    n3 = rotation_normal_axis[2]

    a11 = n1 * n1 * (1 - rotation_cos_theta) + rotation_cos_theta
    a12 = n1 * n2 * (1 - rotation_cos_theta) - n3 * rotation_sin_theta
    a13 = n1 * n3 * (1 - rotation_cos_theta) + n2 * rotation_sin_theta
    a21 = n1 * n2 * (1 - rotation_cos_theta) + n3 * rotation_sin_theta
    a22 = n2 * n2 * (1 - rotation_cos_theta) + rotation_cos_theta
    a23 = n2 * n3 * (1 - rotation_cos_theta) - n1 * rotation_sin_theta
    a31 = n1 * n3 * (1 - rotation_cos_theta) - n2 * rotation_sin_theta
    a32 = n2 * n3 * (1 - rotation_cos_theta) + n1 * rotation_sin_theta
    a33 = n3 * n3 * (1 - rotation_cos_theta) + rotation_cos_theta

    rotation_matrix = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

    tangential_point_row = np.array(
        [np.cos(row_longitude), np.sin(row_longitude), np.sin(row_latitude)]
    )

    tangential_point_revised = np.dot(rotation_matrix, tangential_point_row)

    # print(tangential_point_revised)

    revised_latitude = np.arcsin(tangential_point_revised[2])
    revised_longitude = np.arctan2(
        tangential_point_revised[1] * (-1), tangential_point_revised[0]
    )

    if revised_longitude < 0:
        revised_longitude += 2 * np.pi
    # latitude, longitude ともにradiunで出力
    return revised_latitude, revised_longitude


def Spacecraft_ephemeris_calc(
    spacecraft_ephemeris_csv,
    moon_ephemeris_csv,
    spacecraft_ephemeris_coordinate_csv,
    moon_ephemeris_coordinate_csv,
    time,
):
    # 電波源のデータファイルを読み込み　[0 年、1 月、2 日、3 時間、4 分、5 秒、◯、○、◯、9 周波数（MHz)、10 磁力線の経度(0~360)orイオの場合は(-1000)、11 極（北:1 南:-1)、12 x座標,13 y座標,14 z座標]を受かるパターン分だけ
    radio_source_position = np.loadtxt(
        "../result_for_yasudaetal2022/expres_detectable_radio_data_of_each_flyby/All_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_Radio_data.txt"
    )

    # 木星半径からkmに変換（System３座標）
    radio_source_position[:, 12:15] = radio_source_position[:, 12:15] * 71492

    # 直交座標から極座標へ
    # 探査機の時刻・位置データ[year,month,day,hour,min,sec,0,x,y,z]の10次元データ IAU_JUPITER
    spacecraft_time_position = Coordinate_sysytem_transformation(
        spacecraft_ephemeris_csv
    )

    # 月の時刻・位置データ[year,month,day,hour,min,sec,0,x,y,z]の10次元データ IAU_JUPITER
    moon_time_position = Coordinate_sysytem_transformation(moon_ephemeris_csv)

    # 探査機の時刻・位置データ[year,month,day,hour,min,sec,0,x,y,z]の10次元データ IAU_"moon"
    spacecraft_time_position_for_coordinate = Coordinate_sysytem_transformation(
        spacecraft_ephemeris_coordinate_csv
    )

    # 月の時刻・位置データ[year,month,day,hour,min,sec,0,x,y,z]の10次元データ IAU_"moon"
    moon_time_position_for_coordinate = Coordinate_sysytem_transformation(
        moon_ephemeris_coordinate_csv
    )

    radio_source_data_length = len(radio_source_position)
    res = np.empty((radio_source_data_length, 12))

    Xmax = 0
    Ymax = 0

    for i in range(radio_source_data_length):
        ex = np.array([0.0, 0.0, 0.0])
        ey = np.array([0.0, 0.0, 0.0])
        r1 = np.array([0.0, 0.0, 0.0])
        r2 = np.array([0.0, 0.0, 0.0])
        r3 = np.array([0.0, 0.0, 0.0])
        r4 = np.array([0.0, 0.0, 0.0])
        r5 = np.array([0.0, 0.0, 0.0])

        rc1 = np.array([0.0, 0.0, 0.0])
        rc2 = np.array([0.0, 0.0, 0.0])

        res[i][0] = radio_source_position[i][3].copy()  # 電波源データの時刻　hour
        res[i][1] = radio_source_position[i][4].copy()  # 電波源データの時刻　min
        res[i][2] = radio_source_position[i][9].copy()  # 電波源データの周波数　MHz
        # 電波源データの磁力線(根本)の経度  orイオの場合は(-1000)
        res[i][3] = radio_source_position[i][10].copy()
        res[i][4] = radio_source_position[i][
            11
        ].copy()  # 電波源データの南北（北:1 南:-1)
        res[i][9] = math.degrees(
            math.atan2(radio_source_position[i][13], radio_source_position[i][12])
        )  # 電波源の実際の経度 atan(y/x)を度数表記に変換

        Glow = np.intersect1d(
            np.where(spacecraft_time_position[:, 3] == radio_source_position[i][3]),
            np.where(spacecraft_time_position[:, 4] == radio_source_position[i][4]),
        )
        glow = int(Glow)  # 電波源の時刻データと同じ時刻の列を抽出

        # ある時間における各地点の座標xyzが 木星中心 & IGU_JUPITER で得られている
        r1[0:3] = radio_source_position[i][12:15]  # radio source IAU_JUPITER
        r2[0:3] = moon_time_position[glow, 7:10]  # moon position IAU_JUPITER
        # space craft position IAU_JUPITER
        r3[0:3] = spacecraft_time_position[glow, 7:10]

        # moon position IAU_"moon"
        r4[0:3] = moon_time_position_for_coordinate[glow, 7:10]
        # space craft position IAU_"moon"
        r5[0:3] = spacecraft_time_position_for_coordinate[glow, 7:10]

        ex = (r3 - r1) / np.linalg.norm(r3 - r1)  # 電波源→探査機ベクトル
        rc1 = np.cross((r2 - r1), (r3 - r1))  # 三点がなす平面と垂直なベクトル
        # ガニメデ表面に垂直なベクトル（タンジェンシャルポイントにおける）
        rc2 = np.cross((rc1 / np.linalg.norm(rc1)), ex)
        # ガニメデ表面に垂直な単位ベクトル（タンジェンシャルポイントにおける・IAU_JUIPTER）
        ey = rc2 / np.linalg.norm(rc2)

        res[i][11] = np.dot((r3 - r2), ey) - Output_moon_radius(object_name)  # 高度

        latitude_ey_rad = np.arcsin(ey[2])  # IAUJIPTERでのでの緯度radian
        longtitude_ey_rad = np.arctan2(ey[1], ey[0])  # IAUJIPTERでの経度radian

        # IAUJUPPITERからIAUmoonに変更
        latitude_ey_rad_IAUmoon, longtitude_ey_rad_IAUmoon = Iau_jupiter_2_iau_moon(
            latitude_ey_rad, longtitude_ey_rad, r4, r2, r5, r3
        )

        latitude_ey = np.degrees(latitude_ey_rad_IAUmoon)  # IAUmoonでのでの緯度deg
        longtitude_ey = np.degrees(longtitude_ey_rad_IAUmoon)  # IAUmoonでの経度deg

        res[i][5] = longtitude_ey
        res[i][6] = latitude_ey

        latitude_ex_rad = np.arcsin(ex[2])  # 緯度radiun
        longtitude_ex_rad = np.arctan2(ex[1], ex[0])  # 　経度radiun

        # IAUJUPPITERからIAUmoonに変更
        latitude_ex_rad_IAUmoon, longtitude_ex_rad_IAUmoon = Iau_jupiter_2_iau_moon(
            latitude_ex_rad, longtitude_ex_rad, r4, r2, r5, r3
        )

        latitude_ex = np.degrees(latitude_ex_rad_IAUmoon)  # IAUmoonでのでの緯度deg
        longtitude_ex = np.degrees(longtitude_ex_rad_IAUmoon)  # IAUmoonでの経度deg

        res[i][7] = longtitude_ex
        res[i][8] = latitude_ex

        res[i][10] = spacecraft_time_position[glow][10]  # 探査機の経度

    # res [0 hour, 1 min, 2 frequency(MHz), 3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000), 4 電波源の南北, 5 tangential pointのIAUmoon経度, 6 tangential pointnoのIAUmoon緯度, 7 tangential pointからの探査機方向（tangential から90度回転）のIAUmoon経度, 8 tangential pointからの探査機方向（tangential から90度回転）のIAUmoon緯度,9 座標変換した時のy(tangential pointからの高さ方向の距離),10 探査機の経度, 11 z]
    return res


def Save_data(data):
    np.savetxt(
        "../result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/calculated_all_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_tangential_point_revised.txt",
        data,
        fmt="%s",
    )


def Occultation_timing_select(row_data, time_data):
    # judgement [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 tangential pointでの衛星経度,6 tangential pointでの衛星緯度,7 tangential pointから探査機方向に伸ばした時の衛星経度,8 tangential pointから探査機方向に伸ばした時の衛星緯度, 9 電波源の実際の経度,10 探査機の経度, 11 z]

    def radio_source_select(judgement, time_information, radio_type):
        selected_data = np.zeros_like(judgement)

        for k in range(len(judgement)):
            Num = int(judgement[k][0] * 60 + judgement[k][1]) - (
                time_information[4] * 60 + time_information[6]
            )

            if np.abs(judgement[k][10] + 360 - judgement[k][9]) < np.abs(
                judgement[k][10] - judgement[k][9]
            ):
                Lon = judgement[k][10] + 360 - judgement[k][9]

            elif np.abs(judgement[k][9] + 360 - judgement[k][10]) < np.abs(
                judgement[k][9] - judgement[k][10]
            ):
                Lon = judgement[k][10] - 360 - judgement[k][9]

            else:
                Lon = judgement[k][10] - judgement[k][9]

            Lat = judgement[k][4]

            Fre = np.where(Freq_num == judgement[k][2])

            if "A" in radio_type:
                if Lon < 0 and Lat > 0:
                    selected_data[k, :] = judgement[k, :].copy()

            if "B" in radio_type:
                if Lon > 0 and Lat > 0:
                    selected_data[k, :] = judgement[k, :].copy()

            if "C" in radio_type:
                if Lon < 0 and Lat < 0:
                    selected_data[k, :] = judgement[k, :].copy()

            if "D" in radio_type:
                if Lon > 0 and Lat < 0:
                    selected_data[k, :] = judgement[k, :].copy()

        selected_data = selected_data[np.all(selected_data != 0, axis=1), :]
        return selected_data

    # np.savetxt('../result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/calculated_all_' +spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_'+radio_type+'_tangential_point.txt', selected_data, fmt="%s")

    ####
    def lon_and_lat(
        radio_data,
        begining_hour,
        begining_min,
        end_hour,
        end_min,
        lowest_freq,
        highest_freq,
    ):
        # radio_data[0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 tangential pointでの衛星経度,6 tangential pointでの衛星緯度,7 tangential pointから探査機方向に伸ばした時の衛星経度,8 tangential pointから探査機方向に伸ばした時の衛星緯度, 9 電波源の実際の経度,10 探査機の経度, 11 z]

        start_hour_select = np.where(radio_data[:, 0] == begining_hour)
        start_minute_select = np.where(radio_data[:, 1] == begining_min)
        start_select = np.intersect1d(start_hour_select, start_minute_select)[0]

        end_hour_select = np.where(radio_data[:, 0] == end_hour)
        end_minute_select = np.where(radio_data[:, 1] == end_min)
        end_select = np.intersect1d(end_hour_select, end_minute_select)[0]

        time_select = radio_data[start_select:end_select, :]

        freq_select = np.where(
            (time_select[:, 2] > lowest_freq) & (time_select[:, 2] < highest_freq)
        )[0]
        selected_freq_data = time_select[freq_select, :]

        detectable_select = np.where(selected_freq_data[:, 11] > 0)[0]
        selected_detectable_data = selected_freq_data[detectable_select, :]
        longitude = selected_detectable_data[:, 5]
        latitude = selected_detectable_data[:, 6]
        print(np.max(selected_detectable_data[:, 2]))
        print(longitude)
        print(latitude)
        print(
            "longitude max:"
            + str(np.max(longitude))
            + " min:"
            + str(np.min(longitude))
            + " average:"
            + str(np.average(longitude))
        )
        print(
            "latitude max:"
            + str(np.max(latitude))
            + " min:"
            + str(np.min(latitude))
            + " average:"
            + str(np.average(latitude))
        )

    print("ingress")
    ingress_data = radio_source_select(row_data, time_data, radio_type_ingress)
    lon_and_lat(
        ingress_data,
        begining_ingress_hour,
        begining_ingress_minute,
        end_ingress_hour,
        end_ingress_minute,
        lowest_frequency_ingress,
        highest_frequecy_ingress,
    )

    print("egress")
    egress_data = radio_source_select(row_data, time_data, radio_type_egress)
    lon_and_lat(
        egress_data,
        begining_egress_hour,
        begining_egress_minute,
        end_egress_hour,
        end_egress_minute,
        lowest_frequency_egress,
        highest_frequecy_egress,
    )
    return 0


def main():
    time_information = Pick_up_time()
    # 探査機の位置データとフライバイリストから持ってきた時刻データを出力(IAU_JUPITER)
    spacecraft_epemeris, time = Pick_up_spacecraft_csv()
    moon_epemeris, time = (
        Pick_up_moon_csv()
    )  # 月の位置データとフライバイリストから持ってきた時刻データを出力

    # 探査機の位置データとフライバイリストから持ってきた時刻データを出力(IAU_"moon")
    (
        spacecraft_epemeris_coordinate,
        time_coordinate,
    ) = Pick_up_spacecraft_csv_coordinate()
    # 月の位置データとフライバイリストから持ってきた時刻データを出力
    moon_epemeris_coordinate, time_coordinate = Pick_up_moon_csv_coordinate()

    # 探査機の位置データの時間・月の位置データの時間・フライバイリストで指定している時刻データが一致するか確認
    Check_time_validity_csv(time, spacecraft_epemeris, moon_epemeris)
    Check_time_validity_csv(
        time_coordinate, spacecraft_epemeris_coordinate, moon_epemeris_coordinate
    )

    res = Spacecraft_ephemeris_calc(
        spacecraft_epemeris,
        moon_epemeris,
        spacecraft_epemeris_coordinate,
        moon_epemeris_coordinate,
        time,
    )

    Save_data(res)  # 西経で出力される
    Occultation_timing_select(res, time_information)
    # print(res[0])

    return 0


if __name__ == "__main__":
    main()

# %%
