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
time_of_flybies = 30  # ..th flyby
information_list = ['year', 'month', 'start_day', 'end_day',
                    'start_hour', 'end_hour', 'start_min', 'end_min']

# 計算で使うcdfファイルを選定


def Pick_up_cdf():
    flyby_list_path = '../result_for_yasudaetal2022/occultation_flyby_list.csv'
    flyby_list = pd.read_csv(flyby_list_path)

    # csvファイルにフライバイごとで使う軌道データを記入しておく　上記のパラメータから必要なデータのファイル名が選ばれて読み込まれる
    # queryが数値非対応なのでまずはフライバイ数で絞り込み
    selected_flyby_list = flyby_list[flyby_list['flyby_time']
                                     == time_of_flybies]
    complete_selecred_flyby_list = selected_flyby_list.query(
        'object == "'+object_name+'" & spacecraft == "'+spacecraft_name+'"')  # queryでフライバイ数以外を絞り込み

    index_number = complete_selecred_flyby_list.index.tolist()[0]

    complete_selecred_flyby_list = complete_selecred_flyby_list.reset_index(
        drop=True)  # index振り直し
    # 使うcdfファイルの名前を取得
    cdf_name = str(complete_selecred_flyby_list['cdf_name'][0])

    # csvから時刻データを抽出
    time_information = []
    for i in information_list:
        time_information.append(str(complete_selecred_flyby_list[i][0]))

    cdf_file = cdflib.CDF(
        '../result_for_yasudaetal2022/expres_cdf_data/'+cdf_name)

    return cdf_file, time_information, index_number

# 受信可能な時刻・位置・周波数・磁力線のリストを作成
# 出力は[0 年、1 月、2 日、3 時間、4 分、5 秒、◯、○、◯、9 周波数（MHz)、10 磁力線の経度(0~360)orイオの場合は(-1000)、11 極（北:1 南:-1)、12 x座標,13 y座標,14 z座標]を受かるパターン分だけ


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
    print('{:.19g}'.format(fre[12]))

    long = cdf_file.varget("Src_ID_Label")

    # galireo spacecraft can catch the radio or not (if can, where the radio is emitted)
    # y = cdf_file.varget("Src_Pos_Coord")

    # 電波が受信可能なもの（座標が書かれているもの）の四次元配列番号を取得[時刻座標、周波数座標、磁力線座標、位置座標（０、１、２）]
    idx = np.where(x > -1.0e+31)

    """
    print(idx)
    """

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
    n = int(times.shape[0]/3)  # 受かってるものの全パターンの数
    position = x[idx].reshape(n, 3)  # 受かってるものの全座標のリスト完成([x,y,z],[x,y,z]...)

    # 受かってるものの時間のリスト作成([year,month,day,hour,mim,sec,..,..,..],[year,month,day,hour,mim,sec,..,..,..]..)
    TIME = np.array(cdflib.cdfepoch.breakdown(times.reshape(n, 3)[:, 0]))

    # 受かってるものの周波数のリスト完成 ex.0.3984813988208771
    FRE = fres.reshape(n, 3)[:, 0]
    FRES = np.reshape(FRE, [FRE.shape[0], 1])

    # 受かっているものの磁力線(経度）のリスト完成 ex.'24d-30R NORTH '
    LONG = longs.reshape(n, 3)[:, 0]
    LONG2 = np.reshape(LONG, [LONG.shape[0], 1])
    print(LONG2)

    # 受かっているものの磁力線(経度）のリストの編集 ex.'24d-30R NORTH '→ 24 (磁力線の経度) 'Io NORTH '→ -1000 (イオにつながる磁力線)
    LON = np.zeros(len(LONG2))  # 空配列

    for i in range(len(LONG2)):
        # Io SOUTH / Io NORTHを例外処理
        if 'Io' in str(LONG2[i]):
            LON[i] = -1000

        # それぞれの文字列において　はじめに現れる　\d+：一文字以上の数字列　を検索　（search)
        # group()メソッドでマッチした部分を文字列として取得
        else:
            LON[i] = re.search(r'\d+', str(LONG2[i].copy())).group()

    LONGS = np.reshape(LON, [LON.shape[0], 1])  # 配列を整理

    # 受かっているものの南北判定 ex.'24d-30R NORTH '→ 1  'Io  SOUTH '→ -1
    POL = np.zeros(len(LONG2))  # 空配列

    for i in range(len(LONG2)):
        # .find()検索に引っ掛かればその文字の位置を・引っ掛からなければ-1
        POL[i] = str(LONG2[i].copy()).find('NORTH')

    POLSS = np.where(POL < 0, POL, 1)  # 真の時POLの値(-1)を偽の時1を返す
    POLS = np.reshape(POLSS, [POLSS.shape[0], 1])

    DATA = np.hstack((TIME, FRES, LONGS, POLS, position))

    return DATA

# result_for_yasudaetal2022の下に保存


def Check_time_validity_cdf(time, cdf_data):
    for i in range(3):
        if float(time[i]) != cdf_data[0][i]:
            print(
                "wrong time!!!!! you need to check the time in cdf_file and the time in csv")

    if float(time[3]) != cdf_data[-1][2] or float(time[4]) != cdf_data[0][3] or float(time[5]) != cdf_data[-1][3] or float(time[6]) != cdf_data[0][4] or float(time[7]) != cdf_data[-1][4]:
        print("wrong time!!!!!")


def Save_detectable_data(data):
    np.savetxt('../result_for_yasudaetal2022/expres_detectable_radio_data_of_each_flyby/All_' +
               spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_Radio_data.txt', data, fmt="%s")


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
    spacecraft_longitude_deg = np.array(
        polar_coordinate_data_csv['Longitude (deg)'])

    reshape_logitude_deg = spacecraft_longitude_deg.reshape(
        [len(spacecraft_longitude_deg), 1])

    spacecraft_longitude_rad = np.radians(spacecraft_longitude_deg)

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

    # 時刻データと位置データを結合[year,month,day,hour,min,sec,0,x,y,z,経度]の10次元データの集まり
    time_and_position = np.hstack(
        (time_step, reshape_x, reshape_y, reshape_z, reshape_logitude_deg))

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
    res = np.empty((radio_source_data_length, 9))

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
        res[i][8] = spacecraft_time_position[glow][10]  # 探査機の経度

        if res[i][5] > 0:
            if res[i][5] > Xmax:
                Xmax = res[i][5]

            if res[i][6] > Ymax:
                Ymax = res[i][6]

        """
        if res[i][5] > 0:
            if res[i][0] > 12:
                pass

            elif res[i][1] < 15:
                if res[i][5] > Xmax:
                    Xmax = res[i][5]

                if res[i][6] > Ymax:
                    Ymax = res[i][6]
        """

    # res [hour,min,frequency(MHz),電波源データの磁力線(根本)の経度  orイオの場合は(-1000),電波源の南北,座標変換した時のx(tangential point との水平方向の距離),座標変換した時のy(tangential pointからの高さ方向の距離),電波源の実際の経度,探査機の経度]
    # Xmax 座標変換した時のx(tangential point との水平方向の距離)の最大値
    # Ymax 座標変換した時のy(tangential pointからの高さ方向の距離) の最大値
    return res, Xmax, Ymax


def Save_data(data):
    np.savetxt('../result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/calculated_all_' +
               spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_Radio_data.txt', data, fmt="%s")


def Edit_csv(index, x_farthest, z_farthest):
    flyby_list_path = '../result_for_yasudaetal2022/occultation_flyby_list.csv'
    df = pd.read_csv(flyby_list_path, index_col=0)
    x_colum = df.columns.get_loc("x_farthest")
    df.iat[index, x_colum] = x_farthest
    z_colum = df.columns.get_loc("z_farthest")
    df.iat[index, z_colum] = z_farthest

    df.to_csv('../result_for_yasudaetal2022/occultation_flyby_list.csv')


def main():
    selected_cdf_file, time, index_number = Pick_up_cdf()
    detectable_list = Detectable_time_position_fre_long_list(selected_cdf_file)

    Check_time_validity_cdf(time, detectable_list)
    Save_detectable_data(detectable_list)

    # 探査機の位置データとフライバイリストから持ってきた時刻データを出力
    spacecraft_epemeris, time = Pick_up_spacecraft_csv()
    moon_epemeris, time = Pick_up_moon_csv()  # 月の位置データとフライバイリストから持ってきた時刻データを出力
    # 探査機の位置データの時間・月の位置データの時間・フライバイリストで指定している時刻データが一致するか確認
    Check_time_validity_csv(time, spacecraft_epemeris, moon_epemeris)
    res, Xmax, Zmax = Spacecraft_ephemeris_calc(
        spacecraft_epemeris, moon_epemeris, time)

    Save_data(res)
    Edit_csv(index_number, Xmax, Zmax)

    return 0


if __name__ == "__main__":
    main()

# %%
