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
def Pick_up_cdf():

    picked_up_path = "/work1/rikutoyasuda/tools/result_titan/expres_cdf_for_sensitivity_check/ab7f0f_expres_cassini_saturn_0d-74r_spv_lossc-wid1deg_9kev_rx_20060702_v14.cdf"
    cdf_file = cdflib.CDF(picked_up_path)
    return cdf_file

def Detectable_time_position_fre_long_list(cdf_file):
    # epoch frequency longtitude source position
    # (↓) This command will return all data inside of the variable Variable1, from records 0 to 180.
    # xは外側から　時間181ステップ（3時間分）・周波数42種(0.1MHz~5.6MHz)・南北の磁力線722(南北360✖️2＋(io_north & io_south))・受かっている場合はその位置(xyz座標三次元)/受かってなければ-9.9999998e+30 の四次元配列

    cdf_info = cdf_file.cdf_info()
    
    # 変数名の正しい取得方法
    variable_names = cdf_info.zVariables 
    
    print(f"変数数: {len(variable_names)}")
    print(f"変数名: {variable_names}")
    x = cdf_file.varget("SrcPosition", startrec=0, endrec=180)

    # time (need to check galireo spacecraft position as time) の具体的な値が入っている
    time = cdf_file.varget("Epoch")


    # 使える形に変換　年・月・日・時・分・秒.. にわけられ(181✖️9 の配列に)
    TIME2 = cdflib.cdfepoch.breakdown(time[:])

    fre = cdf_file.varget("Frequency")  # frequency (important for altitude)

    long = cdf_file.varget("Src_ID_Label")

    #beaming_angle = cdf_file.varget("Src_Beam_Angle")  # beam angle of the radio source


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



def main():

    selected_cdf_file = Pick_up_cdf()
    detectable_list = Detectable_time_position_fre_long_list(
        selected_cdf_file
    )  # cdfを整理→電波源データに


    return 0


if __name__ == "__main__":
    main()

# %%
