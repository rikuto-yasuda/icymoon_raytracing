# %%
import numpy as np
import math
import pandas as pd
import re

object_name = "europa"  # europa/ganymde/callisto/io
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 14  # ..th flyby
information_list = ['year', 'month', 'start_day', 'end_day',
                    'start_hour', 'end_hour', 'start_min', 'end_min']

# %%


def Pick_up_spacecraft_csv():

    flyby_list_path = '../result_for_yasudaetal2022/occultation_flyby_list.csv'
    flyby_list = pd.read_csv(flyby_list_path)

    # csvファイルにフライバイごとで使う軌道データを記入しておく　上記のパラメータから必要なデータのファイル名が選ばれて読み込まれる
    # queryが数値非対応なのでまずはフライバイ数で絞り込み
    selected_flyby_list = flyby_list[flyby_list['flyby_time']
                                     == time_of_flybies]
    complete_selecred_flyby_list = selected_flyby_list.query(
        'object == "'+object_name+'" & spacecraft == "'+spacecraft_name+'"')  # queryでフライバイ数以外を絞り込み
    complete_selecred_flyby_list = complete_selecred_flyby_list.reset_index(
        drop=True)  # index振り直し
    # 使うcsvファイルの名前を取得
    csv_name = str(complete_selecred_flyby_list['spacecraft_ephemeris_csv'][0])

    # csvから時刻データを抽出
    time_information = []
    for i in information_list:
        time_information.append(str(complete_selecred_flyby_list[i][0]))

    spacecraft_csv_path = '../result_for_yasudaetal2022/spacecraft_ephemeris/' + csv_name
    spacecraft_ephemeris_csv = pd.read_csv(
        spacecraft_csv_path, header=13, skipfooter=7)  # ヘッダーの位置と末端のデータの位置をheaderとskipfooterで設定

    return spacecraft_ephemeris_csv, time_information


def Pick_up_moon_csv():

    flyby_list_path = '../result_for_yasudaetal2022/occultation_flyby_list.csv'
    flyby_list = pd.read_csv(flyby_list_path)

    # csvファイルにフライバイごとで使う軌道データを記入しておく　上記のパラメータから必要なデータのファイル名が選ばれて読み込まれる
    # queryが数値非対応なのでまずはフライバイ数で絞り込み
    selected_flyby_list = flyby_list[flyby_list['flyby_time']
                                     == time_of_flybies]
    complete_selecred_flyby_list = selected_flyby_list.query(
        'object == "'+object_name+'" & spacecraft == "'+spacecraft_name+'"')  # queryでフライバイ数以外を絞り込み
    complete_selecred_flyby_list = complete_selecred_flyby_list.reset_index(
        drop=True)  # index振り直し
    # 使うcsvファイルの名前を取得
    csv_name = str(complete_selecred_flyby_list['moon_ephemeris_csv'][0])

    # csvから時刻データを抽出
    time_information = []
    for i in information_list:
        time_information.append(str(complete_selecred_flyby_list[i][0]))

    spacecraft_csv_path = '../result_for_yasudaetal2022/moon_ephemeris/' + csv_name
    spacecraft_ephemeris_csv = pd.read_csv(
        spacecraft_csv_path, header=15, skipfooter=4)  # ヘッダーの位置と末端のデータの位置をheaderとskipfooterで設定

    return spacecraft_ephemeris_csv, time_information


def Check_time_validity(time, spacecraft_csv_data, moon_csv_data):

    # 探査機の位置データの時間範囲・フライバイリストで指定している時刻データ範囲が一致するか確認
    Check_time_range_validity(time, spacecraft_csv_data)
    # 月の位置データの時間範囲・フライバイリストで指定している時刻データ範囲が一致するか確認
    Check_time_range_validity(time, moon_csv_data)
    # 探査機の位置データの時間ステップ数・フライバイリストで指定している時刻からか計算されるステップ数が一致するか確認
    Check_time_step_validity(time, spacecraft_csv_data)
    # 月の位置データの時間ステップ数・フライバイリストで指定している時刻からか計算されるステップ数が一致するか確認
    Check_time_step_validity(time, moon_csv_data)


def Check_time_range_validity(time, csv_data):

    first_str = str(csv_data['UTC calendar date'][0])  # 位置データにおける初めの時刻の文字列
    last_str = str(csv_data['UTC calendar date'][-1:])  # 位置データにおける最後の時刻の文字列

    # 位置データにおける初めの時刻の文字列から時刻データを抽出　フライバイリストの情報と一致しているか確認
    if int(re.findall(r'\d+', first_str)[0]) != int(time[0]) or int(re.findall(r'\d+', first_str)[1]) != int(time[1]) or int(re.findall(r'\d+', first_str)[2]) != int(time[2]) or int(re.findall(r'\d+', first_str)[3]) != int(time[4]) or int(re.findall(r'\d+', first_str)[4]) != int(time[6]):
        print("wrong time!!!!!")

    # 位置データにおける最後の時刻の文字列から時刻データを抽出　フライバイリストの情報と一致しているか確認
    if int(re.findall(r'\d+', last_str)[3]) != int(time[3]) or int(re.findall(r'\d+', last_str)[4]) != int(time[5]) or int(re.findall(r'\d+', last_str)[5]) != int(time[7]):
        print(int(re.findall(r'\d+', first_str)[0]))
        print("wrong time!!!!!")

    else:
        print("data range is correct")


def Check_time_step_validity(time, csv_data):
    day_range = int(time[3])-int(time[2])
    hour_range = int(time[5])-int(time[4])
    min_range = int(time[7])-int(time[6])

    step_count = day_range*1440 + hour_range*60 + min_range + \
        1  # フライバイリストからステップ数を計算（今は1step1minを仮定してステップ数を計算）
    # フライバイリストのステップ数と位置データのステップ数が一致する確認（今は1step1minを仮定してステップ数を計算）
    if step_count == len(csv_data['UTC calendar date'][:]):
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
        print("undefined object_name, please check the object_name (moon name) input and def Output_moon_radius function")

    return moon_radius


def Coordinate_sysytem_transformation(polar_coordinate_data_csv):

    time_step_str = np.empty((0, 7), int)
    # 'UTC calendar date'の一行ごとに数字要素だけを抽出&挿入
    for i in range(len(polar_coordinate_data_csv['UTC calendar date'])):
        time_step_str = np.append(time_step_str, np.array(
            [re.findall(r'\d+', polar_coordinate_data_csv['UTC calendar date'][i])]), axis=0)

    # 各要素を文字データから数値データに [year,month,day,hour,min,sec,0]の7次元データの集まり
    time_step = time_step_str.astype(np.int32)

    # 経度・緯度・半径データを読み込み
    spacecraft_longitude_rad = np.radians(
        polar_coordinate_data_csv['Longitude (deg)'])

    spacecraft_latitude_rad = np.radians(
        polar_coordinate_data_csv['Latitude (deg)'])

    spacecraft_radius_km = np.array(polar_coordinate_data_csv['Radius (km)'])

    # x,y,z デカルト座標に変換
    x = np.array(spacecraft_radius_km *
                 np.cos(spacecraft_longitude_rad)*np.cos(spacecraft_latitude_rad))
    y = np.array(spacecraft_radius_km *
                 np.sin(spacecraft_longitude_rad)*np.cos(spacecraft_latitude_rad))

    z = np.array(spacecraft_radius_km*np.sin(spacecraft_latitude_rad))
    # reshapeして二次元配列に　この後に時刻データと結合するため
    reshape_x = x.reshape([len(x), 1])
    reshape_y = y.reshape([len(y), 1])
    reshape_z = z.reshape([len(z), 1])

    # 時刻データと位置データを結合[year,month,day,hour,min,sec,0,x,y,z]の10次元データの集まり
    time_and_position = np.hstack((time_step, reshape_x, reshape_y, reshape_z))

    return time_and_position


def Spacecraft_ephemeris_calc(spacecraft_ephemeris_csv, moon_ephemeris_csv, time):

    # 電波源のデータファイルを読み込み　[0 年、1 月、2 日、3 時間、4 分、5 秒、◯、○、◯、9 周波数（MHz)、10 磁力線の経度(0~360)orイオの場合は(-1000)、11 極（北:1 南:-1)、12 x座標,13 y座標,14 z座標]を受かるパターン分だけ
    radio_source_position = np.loadtxt("../result_for_yasudaetal2022/expres_detectable_radio_data_of_each_flyby/All_" +
                                       spacecraft_name+"_"+object_name+"_"+str(time_of_flybies)+"_Radio_data.txt")
    # 木星半径からkmに変換（System３座標）
    radio_source_position[:, 12:15] = radio_source_position[:, 12:15]*71492

    # 探査機の時刻・位置データ[year,month,day,hour,min,sec,0,x,y,z]の10次元データ
    spacecraft_time_position = Coordinate_sysytem_transformation(
        spacecraft_ephemeris_csv)

    # 月の時刻・位置データ[year,month,day,hour,min,sec,0,x,y,z]の10次元データ
    moon_time_position = Coordinate_sysytem_transformation(moon_ephemeris_csv)

    radio_source_data_length = len(radio_source_position)
    res = np.empty((radio_source_data_length, 8))

    Xmax = 0
    Ymax = 0

    for i in range(radio_source_data_length):
        ex = np.array([0.0, 0.0, 0.0])
        ey = np.array([0.0, 0.0, 0.0])
        r1 = np.array([0.0, 0.0, 0.0])
        r2 = np.array([0.0, 0.0, 0.0])
        r3 = np.array([0.0, 0.0, 0.0])
        rc1 = np.array([0.0, 0.0, 0.0])
        rc2 = np.array([0.0, 0.0, 0.0])

        res[i][0] = radio_source_position[i][3].copy()  # 電波源データの時刻　hour
        res[i][1] = radio_source_position[i][4].copy()  # 電波源データの時刻　min
        res[i][2] = radio_source_position[i][9].copy()  # 電波源データの周波数　MHz
        # 電波源データの磁力線(根本)の経度  orイオの場合は(-1000)
        res[i][3] = radio_source_position[i][10].copy()
        res[i][4] = radio_source_position[i][11].copy()  # 電波源データの南北（北:1 南:-1)
        res[i][7] = math.degrees(math.atan2(
            radio_source_position[i][13], radio_source_position[i][12]))  # 電波源の実際の経度 atan(y/x)を度数表記に変換

        Glow = np.intersect1d(
            np.where(spacecraft_time_position[:, 3] == radio_source_position[i][3]), np.where(spacecraft_time_position[:, 4] == radio_source_position[i][4]))
        glow = int(Glow)

        #
        r1[0:3] = radio_source_position[i][12:15]
        r2[0:3] = moon_time_position[glow, 7:10]  # moon position
        r3[0:3] = spacecraft_time_position[glow, 7:10]  # space craft position
        ex = (r3-r1) / np.linalg.norm(r3-r1)
        rc1 = np.cross((r2-r1), (r3-r1))
        rc2 = np.cross((rc1 / np.linalg.norm(rc1)), ex)
        ey = rc2 / np.linalg.norm(rc2)

        res[i][6] = np.dot((r3-r2), ey) - Output_moon_radius(object_name)
        res[i][5] = np.dot((r3-r2), ex)

        if res[i][5] > 0:
            if res[i][5] > Xmax:
                Xmax = res[i][5]

            if res[i][6] > Ymax:
                Ymax = res[i][6]
    # res [hour,min,frequency(MHz),電波源データの磁力線(根本)の経度  orイオの場合は(-1000),電波源の南北,座標変換した時のx(tangential point との水平方向の距離),座標変換した時のy(tangential pointからの高さ方向の距離),電波源の実際の経度]
    # Xmax 座標変換した時のx(tangential point との水平方向の距離)の最大値
    # Ymax 座標変換した時のy(tangential pointからの高さ方向の距離) の最大値
    return res, Xmax, Ymax


def Save_data(data):
    np.savetxt('../result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/calculated_all_' +
               spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_Radio_data.txt', data, fmt="%s")


def main():
    # 探査機の位置データとフライバイリストから持ってきた時刻データを出力
    spacecraft_epemeris, time = Pick_up_spacecraft_csv()
    moon_epemeris, time = Pick_up_moon_csv()  # 月の位置データとフライバイリストから持ってきた時刻データを出力
    # 探査機の位置データの時間・月の位置データの時間・フライバイリストで指定している時刻データが一致するか確認
    Check_time_validity(time, spacecraft_epemeris, moon_epemeris)
    res, Xmax, Ymax = Spacecraft_ephemeris_calc(
        spacecraft_epemeris, moon_epemeris, time)
    Save_data(res)
    print(Xmax, Ymax)

    return 0


if __name__ == "__main__":
    main()
