# %%
from calendar import month
import pprint
import cdflib
import numpy as np
import pandas as pd
import re
import math
import matplotlib.pyplot as plt

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

result_data_path = "/Users/yasudarikuto/research/icymoon_raytracing/tools/result_titan/"

# 計算で使うcdfファイルを選定

def Pick_up_cdf():
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
    # 使うcdfファイルの名前を取得
    cdf_name = str(complete_selecred_flyby_list["cdf_name"][0])

    # csvから時刻データを抽出
    time_information = []
    for i in information_list:
        time_information.append(str(complete_selecred_flyby_list[i][0]))

    cdf_file = cdflib.CDF(result_data_path + "expres_cdf_data/" + cdf_name)
    print(cdf_name)

    print(cdf_file)

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


    long = cdf_file.varget("Src_ID_Label")

    theta = cdf_file.varget("Theta")    


    # galireo spacecraft can catch the radio or not (if can, where the radio is emitted)
    # y = cdf_file.varget("Src_Pos_Coord")

    # 電波が受信可能なもの（座標が書かれているもの）の四次元配列番号を取得[時刻座標、周波数座標、磁力線座標、位置座標（０、１、２）]
    id_theta = np.where(theta > -1.0e31)
    
    timeindex = id_theta[0]  # 受かってるものの時刻座標
    times = time[timeindex]  # 受かってるものの時刻

    freindex = id_theta[1]  # 受かってるものの周波数座標
    fres = fre[freindex]  # 受かってるものの周波数

    longs = np.array(long[id_theta[2]], dtype=object)
    """
    print(times)
    print(times.shape)

    print(fres)
    print(fres.shape)
    """
    # 位置座標がxyzの３つ分あるのでその分をまとめる
    n = int(times.shape[0])  # 受かってるものの全パターンの数
    beaming_angle = theta[id_theta].reshape(
        n, 1
    )  # 受かってるものの全座標のリスト完成([x,y,z],[x,y,z]...)


    # 受かってるものの時間のリスト作成([year,month,day,hour,mim,sec,..,..,..],[year,month,day,hour,mim,sec,..,..,..]..)
    TIME = np.array(cdflib.cdfepoch.breakdown(times.reshape(n, 1)[:, 0]))

    # 受かってるものの周波数のリスト完成 ex.0.3984813988208771
    FRE = fres.reshape(n, 1)[:, 0]
    FRES = np.reshape(FRE, [FRE.shape[0], 1])

    # 受かっているものの磁力線(経度）のリスト完成 ex.'24d-30R NORTH '
    LONG = longs.reshape(n, 1)[:, 0]
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

    # [year,month,day,hour,mim,sec,..,..,..,freq,long,pole,beamangle]
    DATA = np.hstack((TIME, FRES, LONGS, POLS, beaming_angle))

    print(DATA.shape)

    return DATA



def main():
    selected_cdf_file, time, index_number = Pick_up_cdf()

     # [year,month,day,hour,mim,sec,..,..,..,freq,long,pole,beamangle]
    detectable_list = Detectable_time_position_fre_long_list(
        selected_cdf_file
    )  # cdfを整理→電波源データに

    plt.scatter(detectable_list[:, 9], detectable_list[:, 12])
    plt.show()

    return 0



if __name__ == "__main__":
    main()

# %%
