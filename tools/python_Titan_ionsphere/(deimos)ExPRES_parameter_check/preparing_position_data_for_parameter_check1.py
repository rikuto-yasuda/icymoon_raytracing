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

# %%

object_name = "titan"  # titan
spacecraft_name = "cassini"  # cassini
time_of_flybies = 15  # ..th flyby
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


Source_latitude = 80
Source_beam_angle_range = [66, 67, 68, 69, 70, 71, 72, 73, 74, 75] # [65, 70, 75, 80, 85, 89]

Source_beam_angle = 75  # beam angle of the radio source
Source_latitude_range = [75]  # latitude range of the radio source #[60, 65, 70, 75, 80, 85, 89]

result_data_path = "/work1/rikutoyasuda/tools/result_titan/"


def slackNotify(message):
    TOKEN = 'xoxb-8730854466197-8744040363857-W9I5SjtaONRvymhoYjCTnq8a'
    CHANNEL = 'pythonからの通知'

    url = "https://slack.com/api/chat.postMessage"
    headers = {"Authorization": "Bearer "+TOKEN}
    data  = {
    'channel': CHANNEL,
    'text': message
    }
    requests.post(url, headers=headers, data=data)

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
    """
    print(times)
    print(times.shape)

    print(fres)
    print(fres.shape)
    """
    # 位置座標がxyzの３つ分あるのでその分をまとめる
    n = int(times.shape[0] / 3)  # 受かってるものの全パターンの数
    position = x[idx].reshape(
        n, 3
    )  # 受かってるものの全座標のリスト完成([x,y,z],[x,y,z]...)

    print(idx)

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

    print(time[0])

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


def Interpolate_radio_source_and_save(
    spacecraft_ephemeris_csv, moon_ephemeris_csv, source_data, time
):
    # 電波源のデータファイルを読み込み　[0 年、1 月、2 日、3 時間、4 分、5 秒、◯、○、◯、9 周波数（MHz)、10 磁力線の経度(0~360)orイオの場合は(-1000)、11 極（北:1 南:-1)、12 x座標,13 y座標,14 z座標]を受かるパターン分だけ
    radio_source_position = source_data

    # 探査機の時刻・位置データ[year,month,day,hour,min,sec,0,x,y,z,lon]の10次元データ
    spacecraft_time_position = Coordinate_sysytem_transformation(
        spacecraft_ephemeris_csv
    )

    # 月の時刻・位置データ[year,month,day,hour,min,sec,0,x,y,z,lon]の10次元データ
    moon_time_position = Coordinate_sysytem_transformation(moon_ephemeris_csv)

    radio_source_data_length = len(radio_source_position)
    interpolate_radio_source_position = np.empty((radio_source_data_length, 12))

    # 補間用の配列の作成　基本はradio source position の配列の要素のコピー
    # interpolate_radio_source_position [0時, 1分, 2秒(0), 3周波数, 4磁力線経度, 5南北, 6x, 7y, 8z, 9電波源経度, 10探査機経度, 11電波源タイプ右左]
    for i in range(radio_source_data_length):
        print(
            "radio source interpolation :"
            + str(i * 100 / radio_source_data_length)
            + " %"
        )
        interpolate_radio_source_position[i][0] = radio_source_position[i][
            3
        ].copy()  # 電波源データの時刻　hour

        interpolate_radio_source_position[i][1] = radio_source_position[i][
            4
        ].copy()  # 電波源データの時刻　min

        interpolate_radio_source_position[i][2] = 0  # 電波源データの時刻　sec

        interpolate_radio_source_position[i][3] = radio_source_position[i][
            9
        ].copy()  # 電波源データの周波数　MHz

        # 電波源データの磁力線(根本)の経度  orイオの場合は(-1000)
        interpolate_radio_source_position[i][4] = radio_source_position[i][10].copy()
        interpolate_radio_source_position[i][5] = radio_source_position[i][
            11
        ].copy()  # 電波源データの南北（北:1 南:-1)

        interpolate_radio_source_position[i][6] = radio_source_position[i][
            12
        ].copy()  # x
        interpolate_radio_source_position[i][7] = radio_source_position[i][
            13
        ].copy()  # y
        interpolate_radio_source_position[i][8] = radio_source_position[i][
            14
        ].copy()  # z

        interpolate_radio_source_position[i][9] = math.degrees(
            math.atan2(radio_source_position[i][13], radio_source_position[i][12])
        )  # 電波源の実際の経度 atan(y/x)を度数表記に変換

        Glow = np.intersect1d(
            np.where(spacecraft_time_position[:, 3] == radio_source_position[i][3]),
            np.where(spacecraft_time_position[:, 4] == radio_source_position[i][4]),
        )
        glow = int(Glow)

        interpolate_radio_source_position[i][10] = spacecraft_time_position[glow][
            10
        ]  # 探査機の実際の経度

        if np.abs(
            interpolate_radio_source_position[i][10]
            + 360
            - interpolate_radio_source_position[i][9]
        ) < np.abs(
            interpolate_radio_source_position[i][10]
            - interpolate_radio_source_position[i][9]
        ):
            Lon = (
                interpolate_radio_source_position[i][10]
                + 360
                - interpolate_radio_source_position[i][9]
            )

        elif np.abs(
            interpolate_radio_source_position[i][9]
            + 360
            - interpolate_radio_source_position[i][10]
        ) < np.abs(
            interpolate_radio_source_position[i][9]
            - interpolate_radio_source_position[i][10]
        ):
            Lon = (
                interpolate_radio_source_position[i][10]
                - 360
                - interpolate_radio_source_position[i][9]
            )

        else:
            Lon = (
                interpolate_radio_source_position[i][10]
                - interpolate_radio_source_position[i][9]
            )

        if Lon < 0:
            interpolate_radio_source_position[i][11] = 1  # 電波源タイプAorC

        else:
            interpolate_radio_source_position[i][11] = -1  # 　電波源タイプBorD

        # interpolate_radio_source_position [0時, 1分, 2秒(0), 3周波数, 4磁力線経度, 5南北, 6x, 7y, 8z, 9電波源経度, 10探査機経度, 11電波源タイプ右左]

    interpolate_radio_source_position_ref = (
        interpolate_radio_source_position.copy()
    )  # interpolate_radio_source_positionの参照用配列をコピー

    # 探査機の位置データ補間用ループ
    for i in range(radio_source_data_length):
        print(i / radio_source_data_length)

        if (
            interpolate_radio_source_position_ref[i][0]
            == interpolate_radio_source_position_ref[i - 1][0]
            and interpolate_radio_source_position_ref[i][1]
            == interpolate_radio_source_position_ref[i - 1][1]
            and interpolate_radio_source_position_ref[i][3]
            == interpolate_radio_source_position_ref[i - 1][3]
            and interpolate_radio_source_position_ref[i][5]
            == interpolate_radio_source_position_ref[i - 1][5]
            and interpolate_radio_source_position_ref[i][11]
            == interpolate_radio_source_position_ref[i - 1][11]
        ):
            continue  # 同じ時刻・周波数・電波源タイプの電波源は先頭のものを保管に使う

        else:
            # 参照用データ内の補間に使う電波と同じ電波が書き換えよう配列内のどこにあるかを調べる
            row_array = np.intersect1d(
                np.intersect1d(
                    np.intersect1d(
                        np.where(
                            interpolate_radio_source_position[:, 0]
                            == interpolate_radio_source_position_ref[i][0]
                        ),
                        np.where(
                            interpolate_radio_source_position[:, 1]
                            == interpolate_radio_source_position_ref[i][1]
                        ),
                    ),
                    np.intersect1d(
                        np.where(
                            interpolate_radio_source_position[:, 4]
                            == interpolate_radio_source_position_ref[i][4]
                        ),
                        np.where(
                            interpolate_radio_source_position[:, 3]
                            == interpolate_radio_source_position_ref[i][3]
                        ),
                    ),
                ),
                np.intersect1d(
                    np.where(
                        interpolate_radio_source_position[:, 5]
                        == interpolate_radio_source_position_ref[i][5]
                    ),
                    np.where(
                        interpolate_radio_source_position[:, 11]
                        == interpolate_radio_source_position_ref[i][11]
                    ),
                ),
            )

            row_number = int(row_array[0])

            # 補間用の次時刻のデータが、参照用配列内のどこにあるかを調べる
            # xx:59のようなときは例外的な処理をする
            if interpolate_radio_source_position_ref[i][1] == 59:
                i_next_array = np.intersect1d(
                    np.intersect1d(
                        np.intersect1d(
                            np.where(
                                interpolate_radio_source_position_ref[:, 0]
                                == interpolate_radio_source_position_ref[i][0] + 1
                            ),
                            np.where(interpolate_radio_source_position_ref[:, 1] == 0),
                        ),
                        np.intersect1d(
                            np.where(
                                interpolate_radio_source_position_ref[:, 3]
                                == interpolate_radio_source_position_ref[i][3]
                            ),
                            np.where(
                                interpolate_radio_source_position_ref[:, 5]
                                == interpolate_radio_source_position_ref[i][5]
                            ),
                        ),
                    ),
                    np.where(
                        interpolate_radio_source_position_ref[:, 11]
                        == interpolate_radio_source_position_ref[i][11]
                    ),
                )

            else:
                i_next_array = np.intersect1d(
                    np.intersect1d(
                        np.intersect1d(
                            np.where(
                                interpolate_radio_source_position_ref[:, 0]
                                == interpolate_radio_source_position_ref[i][0]
                            ),
                            np.where(
                                interpolate_radio_source_position_ref[:, 1]
                                == interpolate_radio_source_position_ref[i][1] + 1
                            ),
                        ),
                        np.intersect1d(
                            np.where(
                                interpolate_radio_source_position_ref[:, 3]
                                == interpolate_radio_source_position_ref[i][3]
                            ),
                            np.where(
                                interpolate_radio_source_position_ref[:, 5]
                                == interpolate_radio_source_position_ref[i][5]
                            ),
                        ),
                    ),
                    np.where(
                        interpolate_radio_source_position_ref[:, 11]
                        == interpolate_radio_source_position_ref[i][11]
                    ),
                )

            # 補間先のデータがないときは補間しない
            if len(i_next_array) == 0:
                continue
            i_next = int(i_next_array[0])

            # 補間する配列を作成
            # [0時, 1分, 2秒(0), 3周波数, 4磁力線経度, 5南北, 6x, 7y, 8z, 9電波源経度, 10探査機経度, 11電波源タイプ右左] x 59秒分
            # 基本は分頭のデータをブロードキャスト　保管が必要なのは２秒、４磁力線経度、６x、７y、８z、９電波源経度、１０探査機経度だけ

            time_interpolate_array = np.empty((59, 12))

            time_interpolate_array[:, 0] = int(
                interpolate_radio_source_position_ref[i][0]
            )  # 時

            time_interpolate_array[:, 1] = int(
                interpolate_radio_source_position_ref[i][1]
            )  # 分
            time_interpolate_array[:, 2] = np.arange(1, 60, 1)  # 秒

            time_interpolate_array[:, 3] = interpolate_radio_source_position_ref[i][
                3
            ]  # 周波数

            time_interpolate_array[:, 4] = (
                interpolate_radio_source_position_ref[i][4]
                + (
                    interpolate_radio_source_position_ref[i_next][4]
                    - interpolate_radio_source_position_ref[i][4]
                )
                * np.arange(1, 60, 1)
                / 60
            )  # 磁力線経度

            time_interpolate_array[:, 5] = int(
                interpolate_radio_source_position_ref[i][5]
            )  # 南北

            time_interpolate_array[:, 6] = (
                interpolate_radio_source_position_ref[i][6]
                + (
                    interpolate_radio_source_position_ref[i_next][6]
                    - interpolate_radio_source_position_ref[i][6]
                )
                * np.arange(1, 60, 1)
                / 60
            )  # x

            time_interpolate_array[:, 7] = (
                interpolate_radio_source_position_ref[i][7]
                + (
                    interpolate_radio_source_position_ref[i_next][7]
                    - interpolate_radio_source_position_ref[i][7]
                )
                * np.arange(1, 60, 1)
                / 60
            )  # y

            time_interpolate_array[:, 8] = (
                interpolate_radio_source_position_ref[i][8]
                + (
                    interpolate_radio_source_position_ref[i_next][8]
                    - interpolate_radio_source_position_ref[i][8]
                )
                * np.arange(1, 60, 1)
                / 60
            )  # z

            time_interpolate_array[:, 9] = (
                interpolate_radio_source_position_ref[i][9]
                + (
                    interpolate_radio_source_position_ref[i_next][9]
                    - interpolate_radio_source_position_ref[i][9]
                )
                * np.arange(1, 60, 1)
                / 60
            )  # 電波源経度

            time_interpolate_array[:, 10] = (
                interpolate_radio_source_position_ref[i][10]
                + (
                    interpolate_radio_source_position_ref[i_next][10]
                    - interpolate_radio_source_position_ref[i][10]
                )
                * np.arange(1, 60, 1)
                / 60
            )  # 探査機経度

            time_interpolate_array[:, 11] = int(
                interpolate_radio_source_position_ref[i][11]
            )  # 　電波源たいぷ

            interpolate_radio_source_position = np.insert(
                interpolate_radio_source_position,
                row_number + 1,
                time_interpolate_array,
                axis=0,
            )

        # interpolate_radio_source_position [時, 分, 秒, 周波数, 磁力線経度, 南北, x, y, z, 電波源経度, 探査機経度, 電波源タイプ]補間済み

    return interpolate_radio_source_position


def Interpolate_csv(row_csv):
    # 時刻・位置データ[year,month,day,hour,min,sec,0,x,y,z,lon]の10次元データ

    time_position = Coordinate_sysytem_transformation(row_csv)  # 補間するデータ
    time_position_ref = time_position.copy()  # 補間するデータ参照用
    time_position_data_length = len(time_position_ref)

    for i in range(time_position_data_length - 1):
        print(
            "spacecraft or moon position interpolation :"
            + str(i * 100 / time_position_data_length)
            + " %"
        )
        # 参照用データでfor文を回す

        # 補間用データの保管位置を探す for文で見ているデータと一致するデータを補間用データから探す
        row_array = np.intersect1d(
            np.intersect1d(
                np.where(time_position[:, 2] == time_position_ref[i][2]),
                np.where(time_position[:, 3] == time_position_ref[i][3]),
            ),
            np.intersect1d(
                np.where(time_position[:, 4] == time_position_ref[i][4]),
                np.where(time_position[:, 5] == time_position_ref[i][5]),
            ),
        )
        row_number = int(row_array[0])

        # 補間する配列を作成
        # [0year,1month,2day,3hour,4min,5sec,6 0,7x,8y,9z,10lon] x 59秒分
        # 基本は分頭のデータをブロードキャスト　保管が必要なのは５秒、７x、８y、９z、１０lonだけ

        time_interpolate_array = np.empty((59, 11))

        time_interpolate_array[:, 0] = int(time_position_ref[i][0])  # 年
        time_interpolate_array[:, 1] = int(time_position_ref[i][1])  # 月
        time_interpolate_array[:, 2] = int(time_position_ref[i][2])  # 日
        time_interpolate_array[:, 3] = int(time_position_ref[i][3])  # 時
        time_interpolate_array[:, 4] = int(time_position_ref[i][4])  # 分
        time_interpolate_array[:, 5] = np.arange(1, 60, 1)  # 秒
        time_interpolate_array[:, 6] = int(time_position_ref[i][6])  # 0
        time_interpolate_array[:, 7] = (
            time_position_ref[i][7]
            + (time_position_ref[i + 1][7] - time_position_ref[i][7])
            * np.arange(1, 60, 1)
            / 60
        )
        time_interpolate_array[:, 8] = (
            time_position_ref[i][8]
            + (time_position_ref[i + 1][8] - time_position_ref[i][8])
            * np.arange(1, 60, 1)
            / 60
        )
        time_interpolate_array[:, 9] = (
            time_position_ref[i][9]
            + (time_position_ref[i + 1][9] - time_position_ref[i][9])
            * np.arange(1, 60, 1)
            / 60
        )
        time_interpolate_array[:, 10] = (
            time_position_ref[i][10]
            + (time_position_ref[i + 1][10] - time_position_ref[i][10])
            * np.arange(1, 60, 1)
            / 60
        )

        # 対応する位置にデータを挿入
        time_position = np.insert(
            time_position,
            row_number + 1,
            time_interpolate_array,
            axis=0,
        )

    return time_position


def Spacecraft_ephemeris_calc(spacecraft_ephemeris_data, moon_ephemeris_data, source_data, time):
    # spacecraft_ephemeris_data & moon_ephemeris_data ... [0year,1month,2day,3hour,4min,5sec,6 0,7x,8y,9z,10lon] の11次元データ

    # 電波源のデータファイルを読み込み [0時, 1分, 2秒, 3周波数, 4磁力線経度, 5南北, 6x, 7y, 8z, 9電波源経度, 10探査機経度, 11電波源タイプ]補間済み
    radio_source_position =source_data
    """
    np.loadtxt(
        result_data_path + "expres_detectable_radio_data_of_each_flyby/Interpolated_all_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_Radio_data.txt"
    )
    """ 
    # 木星半径からkmに変換（System３座標）
    radio_source_position[:, 6:9] = radio_source_position[:, 6:9] * 71492

    radio_source_data_length = len(radio_source_position)
    res = np.empty((radio_source_data_length, 11))

    Xmax = 0
    Xmin = 1e10
    Ymax = 0
    Ymin = 1e10


    for i in range(radio_source_data_length):
        print(
            " Coordinate transformation:"
            + str(i * 100 / radio_source_data_length)
            + " %"
        )

        ex = np.array([0.0, 0.0, 0.0])
        ey = np.array([0.0, 0.0, 0.0])
        r1 = np.array([0.0, 0.0, 0.0])
        r2 = np.array([0.0, 0.0, 0.0])
        r3 = np.array([0.0, 0.0, 0.0])
        rc1 = np.array([0.0, 0.0, 0.0])
        rc2 = np.array([0.0, 0.0, 0.0])

        res[i][0] = radio_source_position[i][0].copy()  # 電波源データの時刻　hour
        res[i][1] = radio_source_position[i][1].copy()  # 電波源データの時刻　min
        res[i][2] = radio_source_position[i][2].copy()  # 電波源データの時刻　sec
        res[i][3] = radio_source_position[i][3].copy()  # 電波源データの周波数　MHz
        # 電波源データの磁力線(根本)の経度  orイオの場合は(-1000)
        res[i][4] = radio_source_position[i][4].copy()
        res[i][5] = radio_source_position[i][
            5
        ].copy()  # 電波源データの南北（北:1 南:-1)
        res[i][8] = radio_source_position[i][
            9
        ].copy()  # 電波源の実際の経度 atan(y/x)を度数表記に変換
        res[i][9] = radio_source_position[i][10].copy()  # 探査機の経度
        res[i][10] = radio_source_position[i][11].copy()  # 電波源のたいぷ

        Glow = np.intersect1d(
            np.intersect1d(
                np.where(
                    spacecraft_ephemeris_data[:, 3] == radio_source_position[i][0]
                ),
                np.where(
                    spacecraft_ephemeris_data[:, 4] == radio_source_position[i][1]
                ),
            ),
            np.where(spacecraft_ephemeris_data[:, 5] == radio_source_position[i][2]),
        )
        glow = int(Glow)

        #
        r1[0:3] = radio_source_position[i][6:9]
        r2[0:3] = moon_ephemeris_data[glow, 7:10]  # moon position
        r3[0:3] = spacecraft_ephemeris_data[glow, 7:10]  # space craft position
        ex = (r3 - r1) / np.linalg.norm(r3 - r1)
        rc1 = np.cross((r2 - r1), (r3 - r1))
        rc2 = np.cross((rc1 / np.linalg.norm(rc1)), ex)
        ey = rc2 / np.linalg.norm(rc2)

        res[i][7] = np.dot((r3 - r2), ey) - Output_moon_radius(object_name)
        res[i][6] = np.dot((r3 - r2), ex)

        if res[i][6] > 0:
            if res[i][6] > Xmax:
                Xmax = res[i][6]

            if res[i][7] > Ymax:
                Ymax = res[i][7]
            
            if res[i][6] < Xmin:
                Xmin = res[i][6]
            
            if res[i][7] < Ymin:
                Ymin = res[i][7]
    # res [0hour,1min, 2sec, 3frequency(MHz), 4電波源データの磁力線(根本)の経度  orイオの場合は(-1000), 5電波源の南北,
    # 6座標変換した時のx(tangential point との水平方向の距離), 7座標変換した時のy(tangential pointからの高さ方向の距離), 8電波源の実際の経度, 9探査機の経度, 10 電波源左右(A&C..1 B&D..-1)]
    
    return res, Xmax, Xmin, Ymax, Ymin 

def Save_data(data,source_latitude, source_beam_angle):
    np.savetxt(
        result_data_path + "ExPRES_parameter_check/Interpolated_all_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_sourcelat_"
        + str(source_latitude)
        + "_sourcebeamangle_"
        + str(source_beam_angle)
        + "_Radio_data.txt",
        data,
        fmt="%s",
    )


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


def main():
    lineNotify("start")
    source_latitude = Source_latitude
    

    for source_latitude in Source_latitude_range:
        source_beam_angle = Source_beam_angle

        selected_cdf_file = Pick_up_cdf(source_latitude, source_beam_angle)
        detectable_list = Detectable_time_position_fre_long_list(
            selected_cdf_file
        )  # cdfを整理→電波源データに

        time,index_number = Pick_up_time_data()


        # 探査機の位置データとフライバイリストから持ってきた時刻データを出力
        spacecraft_epemeris, time = Pick_up_spacecraft_csv()
        moon_epemeris, time = (
            Pick_up_moon_csv()
        )  # 月の位置データとフライバイリストから持ってきた時刻データを出力



        # 探査機の位置データを補間する
        Interpolated_source_data = Interpolate_radio_source_and_save(spacecraft_epemeris, moon_epemeris, detectable_list, time)

        interpolated_spacecraft_ephemeris = Interpolate_csv(spacecraft_epemeris)
        interpolated_moon_ephemeris = Interpolate_csv(moon_epemeris)

        res, Xmax, Xmin, Zmax, Zmin  = Spacecraft_ephemeris_calc(
            interpolated_spacecraft_ephemeris, interpolated_moon_ephemeris, Interpolated_source_data, time
        )
        Save_data(res,source_latitude, source_beam_angle)


    for source_beam_angle in Source_beam_angle_range:

        source_latitude = Source_latitude

        selected_cdf_file = Pick_up_cdf(source_latitude, source_beam_angle)
        detectable_list = Detectable_time_position_fre_long_list(
            selected_cdf_file
        )  # cdfを整理→電波源データに

        time,index_number = Pick_up_time_data()


        # 探査機の位置データとフライバイリストから持ってきた時刻データを出力
        spacecraft_epemeris, time = Pick_up_spacecraft_csv()
        moon_epemeris, time = (
            Pick_up_moon_csv()
        )  # 月の位置データとフライバイリストから持ってきた時刻データを出力



        # 探査機の位置データを補間する
        Interpolated_source_data = Interpolate_radio_source_and_save(spacecraft_epemeris, moon_epemeris, detectable_list, time)

        interpolated_spacecraft_ephemeris = Interpolate_csv(spacecraft_epemeris)
        interpolated_moon_ephemeris = Interpolate_csv(moon_epemeris)

        res, Xmax, Xmin, Zmax, Zmin  = Spacecraft_ephemeris_calc(
            interpolated_spacecraft_ephemeris, interpolated_moon_ephemeris, Interpolated_source_data, time
        )
        Save_data(res,source_latitude, source_beam_angle)
        slackNotify("ExPRSS check 1 end")

    return 0


if __name__ == "__main__":
    main()

# %%
