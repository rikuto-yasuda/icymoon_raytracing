# %%
from calendar import month
import pprint
import cdflib
import numpy as np
import pandas as pd
import re
import math
import requests
import os
import fnmatch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# %%

object_name = "titan"  # titan
spacecraft_name = "cassini"  # cassini

Source_latitude = 75
Source_beam_angle_range = [60, 62, 64, 66, 68, 70] # [65, 70, 75, 80, 85, 89]

Source_beam_angle = 75  # beam angle of the radio source
Source_latitude_range = [70, 75, 80]  # latitude range of the radio source #[60, 65, 70, 75, 80, 85, 89]


time_of_flybies = 15  # フライバイの時刻（フライバイリストの何番目か）を指定

hour = 9
minutes = 0
frequency = 0.66874999 # 0.090071, 0.1139, 0.17393 ,0.24168999, 0.27829, 0.32043001, 0.36875001, 0.43125001, 0.46875, 0.53125, 0.56875002, 0.63125002, 0.66874999, 0.73124999, 0.76875001, 0.83125001, 0.86874998 0.93124998  0.96875
Source_type = "D"

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

result_data_path = "/work1/rikutoyasuda/tools/result_titan/"


# 計算で使うcdfファイルを選定
def Pick_up_cdf(latitude, beam_angle):
    target_dir = result_data_path + "expres_cdf_for_sensitivity_check/"
    target_pattern = f"*{int(latitude)}r_spv_cst{int(beam_angle)}*"

    print("target_directory: " + target_dir)

    picked_up_path = None  # ← ここで初期化

    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if fnmatch.fnmatch(file, target_pattern):
                picked_up_path = os.path.join(root, file)
                print("picked up:", picked_up_path)
                break
        if picked_up_path:
            break

    if picked_up_path is None:
        raise FileNotFoundError("該当するCDFファイルが見つかりませんでした")

    cdf_file = cdflib.CDF(picked_up_path)
    return cdf_file


# 受信可能な時刻・位置・周波数・磁力線のリストを作成
# 出力は[0 年、1 月、2 日、3 時間、4 分、5 秒、◯、○、◯、9 周波数（MHz)、10 磁力線の経度(0~360)orイオの場合は(-1000)、11 極（北:1 南:-1)、12 x座標,13 y座標,14 z座標]を受かるパターン分だけ

def Pick_up_time_data():
    flyby_list_path = result_data_path + "occultation_flyby_list.csv"
    flyby_list = pd.read_csv(flyby_list_path)
    # csvファイルにフライバイごとで使う軌道データを記入しておく　上記のパラメータから必要なデータのファイル名が選ばれて読み込まれる
    # queryが数値非対応なのでまずはフライバイ数で絞り込み
    selected_flyby_list = flyby_list[flyby_list["flyby_time"] == time_of_flybies]
    complete_selecred_flyby_list = selected_flyby_list.query(
        'object == "' + object_name + '" & spacecraft == "' + spacecraft_name + '"'
    )  # queryでフライバイ数以外を絞り込み

    index_number = complete_selecred_flyby_list.index.tolist()[0]

    complete_selecred_flyby_list = complete_selecred_flyby_list.reset_index(
        drop=True
    )  # index振り直し

    # csvから時刻データを抽出
    time_information = []
    for i in information_list:
        time_information.append(str(complete_selecred_flyby_list[i][0]))

    return time_information, index_number



def Detectable_time_position_fre_long_list(cdf_file):
    # epoch frequency longtitude source position
    # (↓) This command will return all data inside of the variable Variable1, from records 0 to 180.
    # xは外側から　時間181ステップ（3時間分）・周波数42種(0.1MHz~5.6MHz)・南北の磁力線722(南北360✖️2＋(io_north & io_south))・受かっている場合はその位置(xyz座標三次元)/受かってなければ-9.9999998e+30 の四次元配列
    x = cdf_file.varget("SrcPosition", startrec=0, endrec=180)

    # time (need to check galireo spacecraft position as time) の具体的な値が入っている
    time = cdf_file.varget("Epoch")


    # 使える形に変換　年・月・日・時・分・秒.. にわけられ(181✖️9 の配列に)
    TIME2 = cdflib.cdfepoch.breakdown(time[:])

    fre = cdf_file.varget("Frequency")  # frequency (important for altitude)

    long = cdf_file.varget("Src_ID_Label")

    # galireo spacecraft can catch the radio or not (if can, where the radio is emitted)
    # y = cdf_file.varget("Src_Pos_Coord")

    # 電波が受信可能なもの（座標が書かれているもの）の四次元配列番号を取得[時刻座標、周波数座標、磁力線座標、位置座標（０、１、２）]
    idx = np.where(x > -1.0e31)
    print(idx)
    
    timeindex = idx[0]  # 受かってるものの時刻座標
    times = time[timeindex]  # 受かってるものの時刻

    freindex = idx[1]  # 受かってるものの周波数座標
    fres = fre[freindex]  # 受かってるものの周波数

    longs = np.array(long[idx[2]], dtype=object)

    # 位置座標がxyzの３つ分あるのでその分をまとめる
    n = int(times.shape[0] / 3)  # 受かってるものの全パターンの数
    position = x[idx].reshape(
        n, 3
    )  # 受かってるものの全座標のリスト完成([x,y,z],[x,y,z]...)

    #print(idx)

    # 受かってるものの時間のリスト作成([year,month,day,hour,mim,sec,..,..,..],[year,month,day,hour,mim,sec,..,..,..]..)
    TIME = np.array(cdflib.cdfepoch.breakdown(times.reshape(n, 3)[:, 0]))

    # 受かってるものの周波数のリスト完成 ex.0.3984813988208771
    FRE = fres.reshape(n, 3)[:, 0]
    FRES = np.reshape(FRE, [FRE.shape[0], 1])

    # 受かっているものの磁力線(経度）のリスト完成 ex.'24d-30R NORTH '
    LONG = longs.reshape(n, 3)[:, 0]
    LONG2 = np.reshape(LONG, [LONG.shape[0], 1])

    # 受かっているものの磁力線(経度）のリストの編集 ex.'24d-30R NORTH '→ 24 (磁力線の経度) 'Io NORTH '→ -1000 (イオにつながる磁力線)
    LON = np.zeros(len(LONG2))  # 空配列

    for i in range(len(LONG2)):
        # Io SOUTH / Io NORTHを例外処理
        if "Io" in str(LONG2[i]):
            LON[i] = -1000

        # それぞれの文字列において　はじめに現れる　\d+：一文字以上の数字列　を検索　（search)
        # group()メソッドでマッチした部分を文字列として取得
        else:
            LON[i] = re.search(r"\d+", str(LONG2[i].copy())).group()

    LONGS = np.reshape(LON, [LON.shape[0], 1])  # 配列を整理

    # 受かっているものの南北判定 ex.'24d-30R NORTH '→ 1  'Io  SOUTH '→ -1
    POL = np.zeros(len(LONG2))  # 空配列

    for i in range(len(LONG2)):
        # .find()検索に引っ掛かればその文字の位置を・引っ掛からなければ-1
        POL[i] = str(LONG2[i].copy()).find("NORTH")

    POLSS = np.where(POL < 0, POL, 1)  # 真の時POLの値(-1)を偽の時1を返す
    POLS = np.reshape(POLSS, [POLSS.shape[0], 1])

    DATA = np.hstack((TIME, FRES, LONGS, POLS, position))
    # 受かってるものの時間のリスト作成([0 年、1 月、2 日、3 時間、4 分、5 秒、◯、○、◯、9 周波数（MHz)、10 磁力線の経度(0~360)orイオの場合は(-1000)、11 極（北:1 南:-1)、12 x座標,13 y座標,14 z座標]を受かるパターン分だけ


    return DATA


# result_for_yasudaetal2022の下に保存


def Check_time_validity_cdf(time, cdf_data):
    for i in range(3):
        if float(time[i]) != cdf_data[0][i]:
            print(
                "wrong time!!!!! you need to check the time in cdf_file and the time in csv"
            )

    if (
        float(time[3]) != cdf_data[-1][2]
        or float(time[4]) != cdf_data[0][3]
        or float(time[5]) != cdf_data[-1][3]
        or float(time[6]) != cdf_data[0][4]
        or float(time[7]) != cdf_data[-1][4]
    ):
        print("wrong time!!!!!")
        print(
            "cdf_end_day:"
            + str(cdf_data[-1][2])
            + "cdf_start_hour:"
            + str(cdf_data[0][3])
            + "cdf_end_hour:"
            + str(cdf_data[-1][3])
            + "cdf_start_min"
            + str(cdf_data[0][4])
            + "cdf_end_min"
            + str(cdf_data[-1][4])
        )



def Pick_up_spacecraft_csv():
    flyby_list_path = result_data_path + "/occultation_flyby_list.csv"
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
        result_data_path + "spacecraft_ephemeris/" + csv_name
    )
    spacecraft_ephemeris_csv = pd.read_csv(
        spacecraft_csv_path, header=15, skipfooter=3
    )  # ヘッダーの位置と末端のデータの位置をheaderとskipfooterで設定

    return spacecraft_ephemeris_csv, time_information


def Pick_up_moon_csv():
    flyby_list_path = result_data_path + "occultation_flyby_list.csv"
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

    spacecraft_csv_path = result_data_path + "moon_ephemeris/" + csv_name
    spacecraft_ephemeris_csv = pd.read_csv(
        spacecraft_csv_path, header=15, skipfooter=4
    )  # ヘッダーの位置と末端のデータの位置をheaderとskipfooterで設定

    return spacecraft_ephemeris_csv, time_information


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

    #print(time[0])

    # 位置データにおける初めの時刻の文字列から時刻データを抽出　フライバイリストの情報と一致しているか確認
    if (
        int(re.findall(r"\d+", first_str)[0]) != int(float(time[0]))
        or int(re.findall(r"\d+", first_str)[1]) != int(float(time[1]))
        or int(re.findall(r"\d+", first_str)[2]) != int(float(time[2]))
        or int(re.findall(r"\d+", first_str)[3]) != int(float(time[4]))
        or int(re.findall(r"\d+", first_str)[4]) != int(float(time[6]))
    ):
        print("wrong time!!")

    # 位置データにおける最後の時刻の文字列から時刻データを抽出　フライバイリストの情報と一致しているか確認
    if (
        int(re.findall(r"\d+", last_str)[3]) != int(float(time[3]))
        or int(re.findall(r"\d+", last_str)[4]) != int(float(time[5]))
        or int(re.findall(r"\d+", last_str)[5]) != int(float(time[7]))
    ):
        print(int(re.findall(r"\d+", first_str)[0]))
        print("wrong time!!!")

    else:
        print("data range is correct")


def Check_time_step_validity(time, csv_data):
    day_range = int(float(time[3])) - int(float(time[2]))
    hour_range = int(float(time[5])) - int(float(time[4]))
    min_range = int(float(time[7])) - int(float(time[6]))

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
    
    elif moon_name == "titan":
        moon_radius = 2574.7

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



def Spacecraft_ephemeris_calc(spacecraft_ephemeris_data, moon_ephemeris_data, radio_source_data, hour, minutes, frequency, source_type):
    # spacecraft_ephemeris_data & moon_ephemeris_data ... [0year,1month,2day,3hour,4min,5sec,6 0,7x,8y,9z,10lon] の11次元データ
    # rasdio_source_data ... [0 年、1 月、2 日、3 時間、4 分、5 秒、◯、○、◯、9 周波数（MHz)、10 磁力線の経度(0~360)orイオの場合は(-1000)、11 極（北:1 南:-1)、12 x座標,13 y座標,14 z座標] 

    spacecraft_hour_index = np.where(spacecraft_ephemeris_data[:, 3] == hour)[0]
    spacecraft_min_index = np.where(spacecraft_ephemeris_data[:, 4] == minutes)[0]
    spacecraft_index = np.intersect1d(spacecraft_hour_index, spacecraft_min_index)  # 探査機の位置データを時刻で制限

    moon_hour_index = np.where(moon_ephemeris_data[:, 3] == hour)[0]
    moon_min_index = np.where(moon_ephemeris_data[:, 4] == minutes)[0]
    moon_index = np.intersect1d(moon_hour_index, moon_min_index)  # 月の位置データを時刻で制限

    radio_source_hour_index = np.where(radio_source_data[:, 3] == hour)[0]
    radio_source_min_index = np.where(radio_source_data[:, 4] == minutes)[0]
    radio_source_time_index = np.intersect1d(radio_source_hour_index, radio_source_min_index)  # 電波源の位置データを時刻で制限

    radio_source_frequency_index_lower = np.where(radio_source_data[:, 9] > frequency-0.0001)[0]
    radio_source_frequency_index_upper = np.where(radio_source_data[:, 9] < frequency+0.0001)[0]
    radio_source_frequency_index = np.intersect1d(radio_source_frequency_index_lower, radio_source_frequency_index_upper)  # 電波源の位置データを周波数で制限

    radio_source_index = np.intersect1d(radio_source_time_index, radio_source_frequency_index)  # 電波源の位置データを時刻と周波数で制限
    

    radio_source_selected = []  # 選択された電波源のデータを格納するリスト

    for i in radio_source_index:
        theta_spacecraft = np.arctan2(spacecraft_ephemeris_data[spacecraft_index, 8],spacecraft_ephemeris_data[spacecraft_index, 7]) 
        theta_radio_source = np.arctan2(radio_source_data[i, 13],radio_source_data[i, 12])

        theta_def = theta_radio_source - theta_spacecraft
        if source_type == "A":
            if ((theta_def > 0 and theta_def < np.pi) or (theta_def > -2 * np.pi and theta_def < -np.pi)) and radio_source_data[i][11] == 1:
                radio_source_selected.append(radio_source_data[i, :])
        
        elif source_type == "B":
            if ((theta_def > -np.pi and theta_def < 0) or (theta_def > np.pi and theta_def < 2 * np.pi)) and radio_source_data[i][11] == 1:
                radio_source_selected.append(radio_source_data[i,:])
        
        elif source_type == "C":
            if ((theta_def > 0 and theta_def < np.pi) or (theta_def > -2 * np.pi and theta_def < -np.pi)) and radio_source_data[i][11] == -1:
                radio_source_selected.append(radio_source_data[i,:])

        elif source_type == "D":
            if ((theta_def > -np.pi and theta_def < 0) or (theta_def > np.pi and theta_def < 2 * np.pi)) and radio_source_data[i][11] == -1:
                radio_source_selected.append(radio_source_data[i,:])
    
    if radio_source_selected == []:
       r_list, longitude, latitude, diff_lon, diff_lat = [], [], [], [], []

    else: 
        radio_source_selected = np.vstack(radio_source_selected)
        r_list = np.sqrt(radio_source_selected[:, 12]**2 + radio_source_selected[:, 13]**2 + radio_source_selected[:, 14]**2) # 電波源の位置ベクトルの大きさを計算
        
        longitude = np.arctan2(radio_source_selected[:, 13], radio_source_selected[:, 12]) * 180 / np.pi # 電波源の経度
        latitude = np.arcsin(radio_source_selected[:, 14] / r_list) * 180 / np.pi # 電波源の緯度

        r_list = r_list * 58232 # Rsからkmに変換

        # 探査機から土星までのベクトルと探査機から電波源までのベクトルを計算
        dir_space2source = radio_source_selected[:, 12:15] * 58232 - spacecraft_ephemeris_data[spacecraft_index, 7:10]
        dir_space2saturn = -1 * spacecraft_ephemeris_data[spacecraft_index, 7:10][0]


        lon_space2saturn = np.arctan2(dir_space2saturn[1], dir_space2saturn[0]) * 180 / np.pi
        lat_space2saturn = np.arcsin(dir_space2saturn[2] / (np.sqrt(dir_space2saturn[0]**2 + dir_space2saturn[1]**2 + dir_space2saturn[2]**2))) * 180 / np.pi

        lon_space2source = np.arctan2(dir_space2source[:, 1], dir_space2source[:, 0]) * 180 / np.pi
        lat_space2source = np.arcsin(dir_space2source[:, 2] / (np.sqrt(dir_space2source[:, 0]**2 + dir_space2source[:, 1]**2 + dir_space2source[:, 2]**2))) * 180 / np.pi


        # 電波源と土星の経度・緯度の差を計算
        diff_lon = lon_space2source - lon_space2saturn
        diff_lat = lat_space2source - lat_space2saturn

        print(diff_lon)
        print(diff_lat)
    
    return r_list, longitude, latitude, diff_lon, diff_lat


def Edit_csv(index, x_farthest, z_farthest):
    flyby_list_path = result_data_path + "occultation_flyby_list.csv"
    df = pd.read_csv(flyby_list_path, index_col=0)
    x_colum = df.columns.get_loc("x_farthest")
    df.iat[index, x_colum] = x_farthest
    z_colum = df.columns.get_loc("z_farthest")
    df.iat[index, z_colum] = z_farthest

    df.to_csv(result_data_path + "occultation_flyby_list.csv")

def lineNotify(message):
    line_notify_token = "MFCL4nEMoT0m9IyjUXLeVsoePNXCfbAInnBs7tZeGts"
    line_notify_api = "https://notify-api.line.me/api/notify"
    payload = {"message": message}
    headers = {"Authorization": "Bearer " + line_notify_token}
    requests.post(line_notify_api, data=payload, headers=headers)

def plot_source_latitude_dev(source_beam_angle, source_type, frequency):
    fig, axs = plt.subplots(4, 1)

    beaming_angle = pd.read_csv(result_data_path+"expres_detectable_radio_data_of_each_flyby/source_beam_angle_"+str(source_beam_angle)+"_radio_type_"+source_type+"_frequency_"+str(frequency)+"MHz.csv")
    print(beaming_angle.columns)


    # 仮のデータ（例として5個分）
    x1 = beaming_angle["source_lat"]  # source_lat（上3つで共通のx軸）
    y1 = beaming_angle["r_list"] # 電波源の土星からの動径距離
    y2 = beaming_angle["longitude"]  # 電波源の経度
    y3 = beaming_angle["latitude"] # 電波源の緯度
    x4 = beaming_angle["diff_lon"]  # 探査機から見た電波源の土星中心からの角度（経度方向）
    y4 = beaming_angle["diff_lat"]  # 探査機から見た電波源の土星中心からの角度（経度方向） 

    Rs = 58232
    Rt2s = 1257543 


    # サブプロット作成（5つ）
    fig, axs = plt.subplots(4, 1, figsize=(6, 12))
    fig.suptitle("Latitude check (Beaming: "+str(source_beam_angle)+"deg, Type: "+source_type+", Freq: "+str(np.round(frequency, 3))+"MHz)", fontsize=14)


    # 上の3つ：x軸共有
    axs[0].scatter(x1, y1)
    axs[0].set_ylim(30000, 150000)
    axs[1].scatter(x1, y2)
    axs[2].scatter(x1, y3)
    axs[2].set_ylim(0, 90)

    # 下の2つ：個別のx軸
    sc = axs[3].scatter(x4, y4, c=x1, cmap='viridis')
    cbar = fig.colorbar(sc, ax=axs[3])
    cbar.set_label("Source Latitude (deg)")
 

    # ラベルとタイトルの設定
    labels = ['R (km)', 'Longitude (deg)', 'Latitude (deg)', 'viewing angle for Latitude (deg)']
    xlabels = ['Source Latitude (deg)', 'Source Latitude (deg)', 'Source Latitude (deg)', 'viewing angle for Longitude (deg)']
    titles = ['Source Latitude vs source radius',
            'Source Latitude vs source longitude',
            'Source Latitude vs source latitude',
            'Viewing angle']

    for i in range(4):
        axs[i].set_ylabel(labels[i])
        axs[i].set_xlabel(xlabels[i])
        axs[i].set_title(titles[i])
        axs[i].grid(True)
    axs[3].set_xlim(-20, 20)
    axs[3].set_ylim(-10, 10)

    axs[3].set_xticks(np.arange(-20, 20, 5))
    axs[3].set_yticks(np.arange(-10, 10, 5))
    axs[3].invert_xaxis()

    circle = patches.Circle((0, 0), np.arctan2(Rs, Rt2s) *180/np.pi, edgecolor='red', facecolor='none', linestyle='--', linewidth=1.5)
    axs[-1].tick_params(labelbottom=True)
    axs[3].add_patch(circle)
    axs[3].set_aspect('equal')
    plt.tight_layout()
    plt.show()

    plt.savefig("/work1/rikutoyasuda/tools/result_titan/ExPRES_parameter_check/source_beam_angle_"+str(source_beam_angle)+"_radio_type_"+source_type+"_frequency_"+str(frequency)+"2MHz.png")

    max_theta = 0
    for i in range(len(x4)):
        for j in range(len(x4)):
            x_i = np.cos(y4[i] * np.pi / 180) * np.cos(x4[i] * np.pi / 180)
            y_i = np.cos(y4[i] * np.pi / 180) * np.sin(x4[i] * np.pi / 180)
            z_i = np.sin(y4[i] * np.pi / 180) 

            i_vector = np.array([x_i, y_i, z_i])
            
            x_j = np.cos(y4[j] * np.pi / 180) * np.cos(x4[j] * np.pi / 180)
            y_j = np.cos(y4[j] * np.pi / 180) * np.sin(x4[j] * np.pi / 180)
            z_j = np.sin(y4[j] * np.pi / 180)

            j_vector = np.array([x_j, y_j, z_j])

            # 角度を計算
            theta = np.arccos(np.dot(i_vector, j_vector) / (np.linalg.norm(i_vector) * np.linalg.norm(j_vector))) * 180 / np.pi
            
            if theta > max_theta:
                max_theta = theta
            

    print("Max theta difference in source latitude change: ", max_theta)

    return 0


def plot_beaming_angle_dev(source_latitude, source_type, frequency):
    fig, axs = plt.subplots(4, 1)

    beaming_angle = pd.read_csv(result_data_path+"expres_detectable_radio_data_of_each_flyby/source_latitude_"+str(source_latitude)+"_radio_type_"+source_type+"_frequency_"+str(frequency)+"MHz.csv")
    print(beaming_angle.columns)


    # 仮のデータ（例として5個分）
    x1 = beaming_angle["beaming_angle"]  # source_lat（上3つで共通のx軸）
    y1 = beaming_angle["r_list"] # 電波源の土星からの動径距離
    y2 = beaming_angle["longitude"]  # 電波源の経度
    y3 = beaming_angle["latitude"] # 電波源の緯度
    x4 = beaming_angle["diff_lon"]  # 探査機から見た電波源の土星中心からの角度（経度方向）
    y4 = beaming_angle["diff_lat"]  # 探査機から見た電波源の土星中心からの角度（経度方向） 

    Rs = 58232
    Rt2s = 1257543 


    # サブプロット作成（5つ）
    fig, axs = plt.subplots(4, 1, figsize=(6, 12))
    fig.suptitle("Beaming angle check (Latitude: "+str(source_latitude)+"deg, Type: "+source_type+", Freq: "+str(np.round(frequency, 3))+"MHz)", fontsize=12)


    # 上の3つ：x軸共有
    axs[0].scatter(x1, y1)
    axs[0].set_ylim(30000, 150000)
    axs[1].scatter(x1, y2)
    axs[2].scatter(x1, y3)
    axs[2].set_ylim(0, 90)

    # 下の2つ：個別のx軸
    sc = axs[3].scatter(x4, y4, c=x1, cmap='viridis')
    cbar = fig.colorbar(sc, ax=axs[3])
    cbar.set_label("Beaming angle (deg)")
 

    # ラベルとタイトルの設定
    labels = ['R (km)', 'Longitude (deg)', 'Latitude(deg)', 'viewing angle for Latitude (deg)']
    xlabels = ['Beaming angle (deg)', 'Beaming angle (deg)', 'Beaming angle (deg)',
            'viewing angle for Longitude (deg)']
    titles = ['Beaming angle vs source radius',
            'Beaming angle vs source longitude',
            'Beaming angle vs source latitude',
            'Viewing angle']

    for i in range(4):
        axs[i].set_ylabel(labels[i])
        axs[i].set_xlabel(xlabels[i])
        axs[i].set_title(titles[i])
        axs[i].grid(True)
    axs[3].set_xlim(-20, 20)
    axs[3].set_ylim(-10, 10)

    axs[3].set_xticks(np.arange(-20, 20, 5))
    axs[3].set_yticks(np.arange(-10, 10, 5))
    axs[3].invert_xaxis()

    circle = patches.Circle((0, 0), np.arctan2(Rs, Rt2s) *180/np.pi, edgecolor='red', facecolor='none', linestyle='--', linewidth=1.5)
    axs[-1].tick_params(labelbottom=True)
    axs[3].add_patch(circle)
    axs[3].set_aspect('equal')
    plt.tight_layout()
    plt.show()

    plt.savefig("/work1/rikutoyasuda/tools/result_titan/ExPRES_parameter_check/source_latitude_"+str(source_latitude)+"_radio_type_"+source_type+"_frequency_"+str(frequency)+"2MHz.png")

    max_theta = 0
    for i in range(len(x4)):
        for j in range(len(x4)):
            x_i = np.cos(y4[i] * np.pi / 180) * np.cos(x4[i] * np.pi / 180)
            y_i = np.cos(y4[i] * np.pi / 180) * np.sin(x4[i] * np.pi / 180)
            z_i = np.sin(y4[i] * np.pi / 180) 

            i_vector = np.array([x_i, y_i, z_i])
            
            x_j = np.cos(y4[j] * np.pi / 180) * np.cos(x4[j] * np.pi / 180)
            y_j = np.cos(y4[j] * np.pi / 180) * np.sin(x4[j] * np.pi / 180)
            z_j = np.sin(y4[j] * np.pi / 180)

            j_vector = np.array([x_j, y_j, z_j])

            # 角度を計算
            theta = np.arccos(np.dot(i_vector, j_vector) / (np.linalg.norm(i_vector) * np.linalg.norm(j_vector))) * 180 / np.pi

            if theta > max_theta:
                max_theta = theta

    print("Max theta difference in beaming angle change: ", max_theta)

    return 0

def main():

    stacked_num = 0
    for source_beam_angle in Source_beam_angle_range:

        source_latitude = Source_latitude  # 電波源の緯度
        # まずはcdfのデータを取得
        cdf_data = Pick_up_cdf(source_latitude, source_beam_angle)
        detectable_list = Detectable_time_position_fre_long_list(cdf_data)
        #[0 年、1 月、2 日、3 時間、4 分、5 秒、◯、○、◯、9 周波数（MHz)、10 磁力線の経度(0~360)orイオの場合は(-1000)、11 極（北:1 南:-1)、12 x座標,13 y座標,14 z座標 

        time,index_number = Pick_up_time_data()
        
        if len(detectable_list) != 0:


            Check_time_validity_cdf(
                time, detectable_list
            )  # cdfの時刻の頭と尻がcsv一致してるかを確認

            # 探査機の位置データとフライバイリストから持ってきた時刻データを出力
            spacecraft_epemeris, time = Pick_up_spacecraft_csv()
            
            # 探査機の位置データを座標系変換する
            # 出力は時刻データと位置データを結合[year,month,day,hour,min,sec,0,x,y,z,経度]の10次元データの集まり
            spacecraft_ephemeris_transformed = Coordinate_sysytem_transformation(spacecraft_epemeris)


            # 月の位置データとフライバイリストから持ってきた時刻データを出力
            moon_epemeris, time = Pick_up_moon_csv()  

            # 月の位置データを座標系変換する
            # 出力は時刻データと位置データを結合[year,month,day,hour,min,sec,0,x,y,z,経度]の10次元データの集まり
            moon_ephemeris_transformed = Coordinate_sysytem_transformation(moon_epemeris)

            # 探査機の位置データの時間・月の位置データの時間・フライバイリストで指定している時刻データが一致するか確認
            Check_time_validity_csv(time, spacecraft_epemeris, moon_epemeris)

            
            r_list, longitude, latitude, diff_lon, diff_lat = Spacecraft_ephemeris_calc(
                spacecraft_ephemeris_transformed, moon_ephemeris_transformed, detectable_list, hour, minutes, frequency, Source_type
            )
            source_latitude_list = np.ones(len(r_list)) * source_latitude  # 電波源の緯度を配列に変換
            beam_angle_list = np.ones(len(r_list)) * source_beam_angle  # 電波源のビーム角を配列に変換
            
            if stacked_num == 0:
                stacked_results = np.vstack([source_latitude_list, beam_angle_list, r_list, longitude, latitude, diff_lon, diff_lat])
                stacked_results = stacked_results.T  # 転置して列にする 

            
            else:
                additional_results = np.vstack([source_latitude_list, beam_angle_list, r_list, longitude, latitude, diff_lon, diff_lat])
                additional_results = additional_results.T  # 転置して列にする
                stacked_results = np.vstack(
                    [stacked_results, additional_results]
                )
            stacked_num += 1


    df = pd.DataFrame(stacked_results, columns=['source_lat', 'beaming_angle', 'r_list', 'longitude', 'latitude', 'diff_lon', 'diff_lat'])

    filename = f"{result_data_path}expres_detectable_radio_data_of_each_flyby/source_latitude_{source_latitude}_radio_type_{Source_type}_frequency_{frequency}MHz.csv"
    df.to_csv(filename, index=False)

    stacked_num = 0
    for source_latitude in Source_latitude_range:

        source_beam_angle = Source_beam_angle  # 電波源のビーム角
        # まずはcdfのデータを取得
        cdf_data = Pick_up_cdf(source_latitude, source_beam_angle)
        detectable_list = Detectable_time_position_fre_long_list(cdf_data)
        #[0 年、1 月、2 日、3 時間、4 分、5 秒、◯、○、◯、9 周波数（MHz)、10 磁力線の経度(0~360)orイオの場合は(-1000)、11 極（北:1 南:-1)、12 x座標,13 y座標,14 z座標 

        time,index_number = Pick_up_time_data()
        
        if len(detectable_list) != 0:


            Check_time_validity_cdf(
                time, detectable_list
            )  # cdfの時刻の頭と尻がcsv一致してるかを確認

            # 探査機の位置データとフライバイリストから持ってきた時刻データを出力
            spacecraft_epemeris, time = Pick_up_spacecraft_csv()
            
            # 探査機の位置データを座標系変換する
            # 出力は時刻データと位置データを結合[year,month,day,hour,min,sec,0,x,y,z,経度]の10次元データの集まり
            spacecraft_ephemeris_transformed = Coordinate_sysytem_transformation(spacecraft_epemeris)


            # 月の位置データとフライバイリストから持ってきた時刻データを出力
            moon_epemeris, time = Pick_up_moon_csv()  

            # 月の位置データを座標系変換する
            # 出力は時刻データと位置データを結合[year,month,day,hour,min,sec,0,x,y,z,経度]の10次元データの集まり
            moon_ephemeris_transformed = Coordinate_sysytem_transformation(moon_epemeris)

            # 探査機の位置データの時間・月の位置データの時間・フライバイリストで指定している時刻データが一致するか確認
            Check_time_validity_csv(time, spacecraft_epemeris, moon_epemeris)

            
            r_list, longitude, latitude, diff_lon, diff_lat = Spacecraft_ephemeris_calc(
                spacecraft_ephemeris_transformed, moon_ephemeris_transformed, detectable_list, hour, minutes, frequency, Source_type
            )
            source_latitude_list = np.ones(len(r_list)) * source_latitude  # 電波源の緯度を配列に変換
            beam_angle_list = np.ones(len(r_list)) * source_beam_angle  # 電波源のビーム角を配列に変換

            if stacked_num == 0:
                stacked_results = np.vstack([source_latitude_list, beam_angle_list, r_list, longitude, latitude, diff_lon, diff_lat])
                stacked_results = stacked_results.T  # 転置して列にする 

            
            else:
                additional_results = np.vstack([source_latitude_list, beam_angle_list, r_list, longitude, latitude, diff_lon, diff_lat])
                additional_results = additional_results.T  # 転置して列にする
                stacked_results = np.vstack(
                    [stacked_results, additional_results]
                )
            stacked_num += 1

    df = pd.DataFrame(stacked_results, columns=['source_lat', 'beaming_angle', 'r_list', 'longitude', 'latitude', 'diff_lon', 'diff_lat'])

    filename = f"{result_data_path}expres_detectable_radio_data_of_each_flyby/source_beam_angle_{source_beam_angle}_radio_type_{Source_type}_frequency_{frequency}MHz.csv"
    df.to_csv(filename, index=False)
    ###ここまでに必要なデータは計算して保存済み
    ###次はわかりやすくプロットする
    plot_source_latitude_dev(Source_beam_angle, Source_type, frequency)
    plot_beaming_angle_dev(Source_latitude, Source_type, frequency)

    return 0


if __name__ == "__main__":
    main()

# %%
