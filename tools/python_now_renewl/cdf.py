# %%
from calendar import month
import pprint
import cdflib
import numpy as np
import pandas as pd
import re

# %%

object_name = "europa"  # europa/ganymde/callisto
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 14  # ..th flyby
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

    return cdf_file, time_information

# 受信可能な時刻・位置・周波数・磁力線のリストを作成
#　出力は[0 年、1 月、2 日、3 時間、4 分、5 秒、◯、○、◯、9 周波数（MHz)、10 磁力線の経度(0~360)orイオの場合は(-1000)、11 極（北:1 南:-1)、12 x座標,13 y座標,14 z座標]を受かるパターン分だけ


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
    # longtitude from which magnetic field line (north 360 and south 360)
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


def Check_time_validity(time, cdf_data):
    for i in range(3):
        if float(time[i]) != cdf_data[0][i]:
            print(
                "wrong time!!!!! you need to check the time in cdf_file and the time in csv")

    if float(time[3]) != cdf_data[-1][2] or float(time[4]) != cdf_data[0][3] or float(time[5]) != cdf_data[-1][3] or float(time[6]) != cdf_data[0][4] or float(time[7]) != cdf_data[-1][4]:
        print("wrong time!!!!!")


def Save_detectable_data(data):
    np.savetxt('../result_for_yasudaetal2022/expres_detectable_radio_data_of_each_flyby/All_' +
               spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_Radio_data.txt', data, fmt="%s")


def main():
    selected_cdf_file, time = Pick_up_cdf()
    detectable_list = Detectable_time_position_fre_long_list(selected_cdf_file)
    Check_time_validity(time, detectable_list)
    Save_detectable_data(detectable_list)
    return 0


if __name__ == "__main__":
    main()
