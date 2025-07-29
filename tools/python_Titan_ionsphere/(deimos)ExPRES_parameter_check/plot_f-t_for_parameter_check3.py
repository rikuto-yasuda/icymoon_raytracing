# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
import os
import time
import glob
import sys
import requests
from datetime import datetime
import math
from maser.data import Data
import re
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# %%
# あらかじめ ../result_sgepss2021/~/~ に必要なレイトレーシング結果とパラメータセットを入れること
cassini_data_path = "/work1/rikutoyasuda/tools/python_Titan_ionsphere/Cassini_radio_data/"
plot_result_path = "/work1/rikutoyasuda/tools/result_titan/radio_plot/"
result_output_path = "/work1/rikutoyasuda/tools/result_titan/"
args = sys.argv

object_name = "titan"  # ganydeme/europa/calisto
spacecraft_name = "cassini"  # galileo/JUICE(?)
time_of_flybies = 15  # ..th flyby

Source_latitude = 74
Source_beam_angle_range = [58, 59, 60, 61, 62, 63, 64] # [65, 70, 75, 80, 85, 89]

Source_beam_angle = 75  # beam angle of the radio source
Source_latitude_range = [75, 80] #[70, 75, 80]  # latitude range of the radio source #[60, 65, 70, 75, 80, 85, 89]


boundary_average_str = "10"  # boundary_intensity_str = '10'⇨ノイズフロアの10倍強度まで
Source_type = "D"
vertical_line_freq = np.array([0])  # MHz

if spacecraft_name == "cassini":
    if time_of_flybies==15:
        Cassini_flyby = "T15"
        start_year = 2006
        start_date = 183
        start_hour = 8
        duration = 3
        save_path = plot_result_path + Cassini_flyby + "/"
        os.makedirs(plot_result_path + Cassini_flyby, exist_ok=True)  # フォルダが存在しない場合は作成


        plot_time_step_sec = [0, 1800, 3600, 5400, 7200, 9000, 10800]
        plot_time_step_label = [
            "8:00",
            "8:30",
            "9:00",
            "9:30",
            "10:00",
            "10:30",
            "11:00",
        ]
        exclave_exception = np.array([[0.10870189666748047,0.11393309783935547,0.11941609954833984,0.1311864013671875,0.15832159423828124,0.1659407958984375,0.17392660522460937,0.1822967071533203,0.22000390625,0.2305915069580078,0.2416885986328125,0.25331979370117186,0.26551071166992185,0.27828829956054685,0.3204302978515625,0.36875,0.38125,0.41875,0.45625,0.46875,0.48125,0.51875,0.53125,0.54375,0.55625,0.56875,0.58125,0.61875,0.63125,0.66875,0.68125,0.71875,0.73125],[1,0,0,0,0,0,2,2,1,1,1,3,2,0,0,0,0,0,0,0,0,2,1,0,3,3,2,1,2,3,1,1,1],[0,0,0,0,10,9,2,9,0,0,10,35,24,8,6,1,1,1,1,4,2,5,3,4,2,2,2,0,0,0,0,0,0]])  # [周波数, ingress, egress] 例: [0.10870189666748047,0.17392660522460937,0.1822967071533203,0.22000390625,0.2305915069580078,0.2416885986328125,0.25331979370117186,0.26551071166992185],[1,3,3,3,1,1,1,1],[0,0,0,0,0,0,0,0]
        ingress_egress_time_window = 2000

        # MHz
        Freq_num = np.array([0.090071, 0.113930, 0.173930, 0.241690, 0.278290, 0.320430, 0.368750, 0.431250, 0.468750, 0.531250, 0.568750, 0.631250, 0.668750, 0.731250, 0.768750, 0.831250, 0.868750, 0.931250, 0.968750])
        Freq_underline = 0.0859355 


    else:
        Cassini_flyby = "T15"
        start_year = 2006
        start_date = 183
        start_hour = 8
        duration = 3
        save_path = plot_result_path + Cassini_flyby + "/"
        os.makedirs(plot_result_path + Cassini_flyby, exist_ok=True)  # フォルダが存在しない場合は作成


        plot_time_step_sec = [0, 1800, 3600, 5400, 7200, 9000, 10800]
        plot_time_step_label = [
            "8:00",
            "8:30",
            "9:00",
            "9:30",
            "10:00",
            "10:30",
            "11:00",
        ]
        exclave_exception = np.array([])
        ingress_egress_time_window = 2500

        # MHz
        Freq_num = np.array([0.090071, 0.113930, 0.173930, 0.241690, 0.278290, 0.320430, 0.368750, 0.431250, 0.468750, 0.531250, 0.568750, 0.631250, 0.668750, 0.731250, 0.768750, 0.831250, 0.868750, 0.931250, 0.968750])
        Freq_underline = 0.0859355 

information_list = [
    "year",
    "month",
    "start_day",
    "end_day",
    "start_hour",
    "end_hour",
    "start_min",
    "end_min",
    "occultaton_center_day",
    "occultaton_center_hour",
    "occultaton_center_min",
]

# [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 座標変換した時のx(tangential point との水平方向の距離),6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度,8 探査機の経度]




# %%


def Pick_up_timeinfo():
    flyby_list_path = result_output_path + "occultation_flyby_list.csv"
    flyby_list = pd.read_csv(flyby_list_path, engine="python")

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


def Time_step(time_data):
    """_csvファイルの時効データからレイトレーシングで計算した総秒数を出力する_

    Args:
        time_data (_type_): _pick_up_cdfでcsvファイルから取ってきた時刻情報をそのまま入れる_

    Returns:
        _type_: _レイトレーシングで計算した総秒数_
    """
    day_range = int(time_data[3]) - int(time_data[2])
    hour_range = int(time_data[5]) - int(time_data[4])
    min_range = int(time_data[7]) - int(time_data[6])

    step_count = (
        day_range * 1440 * 60 + hour_range * 60 * 60 + min_range * 60 + 1
    )  # フライバイリストからステップ数を計算（今は1step1secondを仮定してステップ数を計算）

    return step_count


def Prepare_Figure(judgement, time_information):
    time_step = Time_step(time_information)
    time_step_list = np.arange(0, time_step, 1)
    # (周波数の数 +1) ×(時間数（正確には開始時刻からの秒数の数) ）の0配列を４つ用意
    DataA = np.zeros(len(time_step_list) * (len(Freq_num) + 1)).reshape(
        len(Freq_num) + 1, len(time_step_list)
    )
    DataB = np.zeros(len(time_step_list) * (len(Freq_num) + 1)).reshape(
        len(Freq_num) + 1, len(time_step_list)
    )
    DataC = np.zeros(len(time_step_list) * (len(Freq_num) + 1)).reshape(
        len(Freq_num) + 1, len(time_step_list)
    )
    DataD = np.zeros(len(time_step_list) * (len(Freq_num) + 1)).reshape(
        len(Freq_num) + 1, len(time_step_list)
    )

    # judgement [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]


    for k in range(len(judgement)):
        Num = int(
            judgement[k][0] * 60 * 60 + judgement[k][1] * 60 + judgement[k][2]
        ) - (time_information[4] * 60 * 60 + time_information[6] * 60)

        Fre = np.where((Freq_num >= math.floor(judgement[k][3]*10000000)/10000000) & (Freq_num <= math.ceil(judgement[k][3]*10000000)/10000000))

        if Fre[0]!=Fre[-1]:
            raise Exception('Error!')

        # int(Fre[0])+1になっているのは、のちのコンタープロットのために一個多い周波数で作ってあるのでミスではない
        if judgement[k][10] == 1 and judgement[k][5] == 1:
            DataA[int(Fre[0]) + 1][Num] = 1

        if judgement[k][10] == -1 and judgement[k][5] == 1:
            DataB[int(Fre[0]) + 1][Num] = 1

        if judgement[k][10] == 1 and judgement[k][5] == -1:
            DataC[int(Fre[0]) + 1][Num] = 1

        if judgement[k][10] == -1 and judgement[k][5] == -1:
            DataD[int(Fre[0]) + 1][Num] = 1


    return DataA, DataB, DataC, DataD


# ガリレオ探査機の周波数一覧（Hz)とダウンロードした電波強度電波を代入（das2をcsvに変換）
def download_data(url, path, file_name):

    # 保存先のフォルダ
    os.makedirs(path, exist_ok=True)  # フォルダが存在しない場合は作成

    # 保存するファイル名
    save_path = os.path.join(path, file_name)

    # データをダウンロード
    response = requests.get(url)
    response.raise_for_status()  # エラーチェック

    # ダウンロードしたデータをファイルに保存
    with open(save_path, "wb") as file:
        file.write(response.content)

def load_intensity_data(path, year, date, hour, duration):

    n2_path = path + "n2/"
    n3_path = path + "n3e/"


    for i in range(duration):
        n3_data_name = "N3e_dsq" + str(year) + "{:03}".format(date+((hour + i)//24)) + "." + "{:02}".format((hour + i)%24)
        n2_data_name = "P" + str(year) + "{:03}".format(date+((hour + i)//24)) + "." + "{:02}".format((hour + i)%24)

        if (0 < date) and (date < 91):
            download_folder_str = "001_090"
        
        elif (90 < date) and (date < 181):
            download_folder_str = "091_180"
        
        elif (180 < date) and (date < 271):
            download_folder_str = "181_270"
        
        else:
            download_folder_str = "271_366"
        
        n2_data_download_url = "https://lesia.obspm.fr/kronos/data/" + str(year) + "_" + download_folder_str + "/n2/" + n2_data_name
        n3_data_download_url = "https://lesia.obspm.fr/kronos/data/" + str(year) + "_" + download_folder_str + "/n3e/" + n3_data_name


        if os.path.exists(n3_path + n3_data_name)==False:
            download_data(n3_data_download_url, n3_path, n3_data_name)
            download_data(n2_data_download_url, n2_path, n2_data_name)

        # データの読み込み and 強度や周波数、時間配列をインプット
        data_rpwi = Data(n3_path + n3_data_name)
        xr_st = data_rpwi.as_xarray()

        if i == 0:
            Radio_intensity = xr_st["s"].values # ?
            Epoch = xr_st["time"].values
        
        else:
            Radio_intensity = np.concatenate((Radio_intensity, xr_st["s"].values), axis=1)
            Epoch = np.concatenate([Epoch, xr_st["time"].values])


    # 周波数と時間を単純化した一次元配列を生成（周波数 Hz, 時間 0:00からの秒）
    # Epoch = [parse(iso_str) for iso_str in Epoch_row]
    Frequency_1d = xr_st["frequency"].values[0]/1000  # MHz

    day_array = (
        np.array([np.datetime64(dt, "D").astype(int) for dt in Epoch])
        - np.array([np.datetime64(dt, "D").astype(int) for dt in Epoch])[0]
    )
    hour_array = np.array([np.datetime64(dt, "h").astype(int) % 24 for dt in Epoch])
    minute_array = np.array([np.datetime64(dt, "m").astype(int) % 60 for dt in Epoch])
    second_array = np.array([np.datetime64(dt, "s").astype(int) % 60 for dt in Epoch])
    mili_second_array = np.array(
        [np.datetime64(dt, "ms").astype(int) % 1000 for dt in Epoch]

    )

    Radio_epoch_from_0min_1d = (
        day_array * 60 * 60 * 24
        + hour_array * 60 * 60
        + minute_array * 60
        + second_array
        + mili_second_array / 1000
        - day_array[10] * 60 * 60 * 24
        - hour * 60 * 60
    )
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True)
   
    np.set_printoptions(threshold=1)

    return hour_array, minute_array, second_array, Radio_epoch_from_0min_1d, Frequency_1d, Radio_intensity

def load_polarizaiton_data(path, year, date, hour, duration):

    n3_path = path + "n3e/"

    for i in range(duration):
        n3_data_name = "N3e_dsq" + str(year) + "{:03}".format(date+((hour + i)//24)) + "." + "{:02}".format((hour + i)%24)

        # データの読み込み and 強度や周波数、時間配列をインプット
        data_rpwi = Data(n3_path + n3_data_name)
        xr_st = data_rpwi.as_xarray()

        if i == 0:
            Radio_polarization = xr_st["v"].values # ?
        
        else:
            Radio_polarization = np.concatenate((Radio_polarization, xr_st["v"].values), axis=1)

    return Radio_polarization


def calculate_day_of_year(year, month, day):
    date = datetime(year, month, day)
    day_of_year = date.timetuple().tm_yday
    return day_of_year

def Make_intensity_FT_full(
    Data, raytrace_time_information, beam_or_lat
):
    time_step = Time_step(raytrace_time_information)

    time_list = np.arange(
        0, time_step, 1
    )  # エクスプレスコードで計算している時間幅（sec)を60で割る

    boundary_average = float(boundary_average_str)

    DOY = calculate_day_of_year(raytrace_time_information[0],raytrace_time_information[1],raytrace_time_information[2])

        # ガリレオ探査機のデータ取得開始時刻からの経過時間（sec) , ガリレオ探査機のデータ取得周波数（MHz), ガリレオ探査機の取得した電波強度（代入したデータと同じ単位）
    (
        hour_array, 
        minute_array, 
        second_array, 
        Radio_epoch_from_0min_1d,
        Radio_frequency_1d, 
        Radio_intensity
    ) = load_intensity_data(cassini_data_path, raytrace_time_information[0], DOY , raytrace_time_information[4], raytrace_time_information[5] -raytrace_time_information[4])


    Radio_intensity_row = Radio_intensity.copy()
    boundary_intensity_array = np.zeros(len(Radio_frequency_1d))

    
    # [[周波数一覧][各周波数でのノイズフロアの電波強度平均][角周波数でのノイズフロアの電波強度標準偏差][各周波数でのノイズフロアの電波強度の中央値]]
    noise_data = np.genfromtxt(
        plot_result_path
        + "T"
        + str(time_of_flybies)
        + "/average_noise_floor_T"
        + str(time_of_flybies)
        + ".csv",
        delimiter=",",
    )
    
    # noise_data = np.full(len(Radio_frequency_1d),10**-16)


    for i in range(len(Radio_frequency_1d)):
        certain_freq_data = Radio_intensity[i]  # i番目の周波数の全データ

        boundary_intensity = (
            boundary_average * noise_data[1, i]
        )  # i番目の周波数の強度平均値×boundary_average
        """
        boundary_intensity = (
            boundary_average * noise_data[i]
        )  # i番目の周波数の強度平均値×boundary_average
        """
        boundary_intensity_array[i] = boundary_intensity
        # print(certain_freq_data.shape)
        detectable_position_array = np.where(certain_freq_data > boundary_intensity)[0]
        undetectable_position_array = np.where(certain_freq_data <= boundary_intensity)[
            0
        ]
        # print(certain_freq_data)
        Radio_intensity[i, detectable_position_array] = 1
        Radio_intensity[i, undetectable_position_array] = 0

    # raytrace_time_information ['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']

    # start時刻から掩蔽中心時刻までの差分（秒）を計算
    middle_time = (
        (raytrace_time_information[8] - raytrace_time_information[2]) * 86400
        + (raytrace_time_information[9] - raytrace_time_information[4]) * 3600
        + (raytrace_time_information[10] - raytrace_time_information[6]) * 60
    )

    # ガリレオ電波データの掩蔽中心時刻を初めに超える時間ステップの配列番号を取得

    occulted_time = int(np.where(Radio_epoch_from_0min_1d > middle_time)[0][0])
    ingress_time_list = Radio_frequency_1d.copy()
    egress_time_list = Radio_frequency_1d.copy()

    # それぞれの周波数に対して..
    for k in range(len(Radio_frequency_1d)):
        # k番目の周波数の強度で閾値を超える電波強度を捉えている電波データの配列番号のリストを取得
        over_judge_time_list = np.array(np.where((Radio_intensity[k][:] > 0.5)))

        # print(over_judge_time_list)
        # 閾値を超える電波強度を観測しているものの中でも掩蔽中心時刻前のもののリストをA、後のもののリストをBとしてわける
        A = over_judge_time_list[over_judge_time_list < occulted_time]
        B = over_judge_time_list[over_judge_time_list > occulted_time]

        ingress_exception = 0
        egress_exception = 0

        
        # 閾値を超える電波強度を
        if exclave_exception.size > 0:
            if np.any(exclave_exception[0] == Radio_frequency_1d[k]):
                #print("exclave_exception")
                exception_position = np.where(exclave_exception[0] == Radio_frequency_1d[k])[
                    0
                ][0]

                ingress_exception = int(exclave_exception[1][exception_position])
                egress_exception = int(exclave_exception[2][exception_position])

        # 電波強度が十分であればAもBも複数ヒットするはず...
        if len(B) > 0:
            b = B[egress_exception]
            # 初めて閾値を超える強度が観測される時刻（正確にはstart時刻からの差分（秒））がRadio_epoch_from_0min_1d[b]
            # その１つ前の時刻 Radio_epoch_from_0min_1d[b-1]なので、その真ん中の時間がegress_time_listに加わる（要するに終了タイミング）
            egress_time_list = np.append(
                egress_time_list,
                ((Radio_epoch_from_0min_1d[b] + Radio_epoch_from_0min_1d[b - 1]) / 2),
            )

        else:
            egress_time_list = np.append(
                egress_time_list, -1e8
            )  # 閾値を適宜できない周波数においては掩蔽タイミングを-10^8としておく

        if len(A) > 0:
            a = A[-1 - ingress_exception]
            # 最後に閾値を超える強度が観測される時刻（正確にはstart時刻からの差分（秒））がRadio_epoch_from_0min_1d[a]
            # その１つ次の時刻 Radio_epoch_from_0min_1d[a+1]なので、その真ん中の時間がingress_time_listに加わる（要するに開始タイミング）
            ingress_time_list = np.append(
                ingress_time_list,
                ((Radio_epoch_from_0min_1d[a] + Radio_epoch_from_0min_1d[a + 1]) / 2),
            )

        else:
            ingress_time_list = np.append(
                ingress_time_list, -1e8
            )  # 閾値を適宜できない周波数においては掩蔽タイミングを-10^8としておく

    ingress_time_list = ingress_time_list.reshape(2, int(len(ingress_time_list) / 2))
    # print(ingress_time_list)
    egress_time_list = egress_time_list.reshape(2, int(len(egress_time_list) / 2))
    # print("array=", np.array(Freq_num))
    # print("underline=", Freq_underline)
    FREQ = np.insert(np.array(Freq_num), 0, Freq_underline)

    """
    ここまでで出来上がっている材料たち
    time_list レイトレーシング(1分間隔)の時刻データ(観測開始隊ミンングからの秒) ex. [0,1,2, ...]
    FREQ レイトレーシングの周波数リスト
    (contour plotをする関係で配列の初めに本来はない周波数を挿入しているので周波数の数的には多くなているので注意)
    DataA-DataD 電波源A-Dの電波データ 受かる場合は1 受からない場合は0 (周波数の数+1)✖️(時刻)の二次元データ

    Radio_epoch_from_0min_1d 探査機の観測結果の時刻データ(観測開始隊ミンングからの秒) ex. [0,8.434,26.56 ...]
    Radio_frequency_1d 探査機の観測結果の周波数データ(MHz) ex. [5.620e-06 1.000e-05 ...]
    Radio_intensity_row 探査機の電波強度データ　周波数の数✖️時刻の数の二次元データ
    Radio_intensity 探査機の電波強度データで閾値を超えてる部分を1、それ以外を0としているデータ 周波数の数✖️時刻の数の二次元データ
    ingress_time_list 電波観測データにおける掩蔽開始時刻 [[周波数一覧][掩蔽開始時刻一覧]]の二次元データ
    egress_time_list ingress_time_listの掩蔽終了ver

    """

    def plot_and_save(start_time, end_time, name, beam_or_lat):
        # ガリレオ探査機の電波データの時刻・周波数でメッシュ作成
        xx, yy = np.meshgrid(Radio_epoch_from_0min_1d, Radio_frequency_1d)


        fig, ax = plt.subplots(1, 1)

        # ガリレオ探査機の電波強度をカラーマップへ
        pcm = ax.pcolormesh(
            xx,
            yy,
            Radio_intensity_row,
            norm=mpl.colors.LogNorm(vmin=1e-16, vmax=1e-12),
            cmap="Spectral_r",
            alpha=0.9,
        )
        # print(xx)
        fig.colorbar(
            pcm,
            extend="max",
            label="GLL/PWS Electric Power spectral density (V2/m2/Hz)",
        )

        # ガリレオ探査機の電波強度の閾値を赤線

        #print(Data)
        legend_lines = []       
        
        def find_contour_frequencies_at_time(contour, target_time_seconds, contour_levels, freq_range=None):
            """
            指定した時間でのコンター周波数を抽出
            
            Args:
                contour: ax.contour()の戻り値
                target_time_seconds: 対象時間（秒）
                contour_levels: コンターレベルのリスト（例：[0.5]）
                freq_range: 周波数範囲のタプル（例：(0.1, 1.0)）
            
            Returns:
                dict: {レベル値: 周波数配列} の辞書
            """
            results = {}
            
            # 各コンターレベルについて処理
            for level_idx, level_value in enumerate(contour_levels):
                frequencies_at_time = []
                
                # そのレベルのコンター線を取得
                if level_idx < len(contour.collections):
                    level_contour = contour.collections[level_idx]
                    
                    # 各パス（線分）について処理
                    for path in level_contour.get_paths():
                        vertices = path.vertices
                        
                        if len(vertices) > 0:
                            # 指定時間に近い点を探す（±30秒の範囲）
                            time_mask = np.abs(vertices[:, 0] - target_time_seconds) < 30
                            
                            if np.any(time_mask):
                                nearby_frequencies = vertices[time_mask, 1]
                                #print(f"レベル {level_value}: 見つかった周波数 {nearby_frequencies}")
                                
                                # 周波数範囲でフィルタリング
                                if freq_range is not None:
                                    freq_mask = (nearby_frequencies >= freq_range[0]) & (nearby_frequencies <= freq_range[1])
                                    nearby_frequencies = nearby_frequencies[freq_mask]
                                    #print(f"フィルタ後: {nearby_frequencies}")
                                
                                frequencies_at_time.extend(nearby_frequencies)
                
                # 結果を辞書に格納
                if frequencies_at_time:
                    results[level_value] = np.array(frequencies_at_time)
                else:
                    results[level_value] = np.array([])
    
            return results

        
        # レイトレーシングの結果をコンタープロットで表示
        min_num = 90
        max_num = 0

        for label, data in Data.items():
            number = int(re.search(r'\d+', str(label)).group())
            if number < min_num:
                min_num = number
            if number > max_num:
                max_num = number

        parameters =np.empty(len(Data))
        ingress_BDs = np.empty(len(Data))
        egress_BDs = np.empty(len(Data))

        # レイトレーシングの結果をコンタープロットで表示
        for label, data in Data.items():
            number = int(re.search(r'\d+', str(label)).group())
            cmap = cm.get_cmap('cividis')
            color = cmap((number - min_num) / (max_num - min_num))  # 色を正規化して取得

            cs = ax.contour(
                time_list,
                FREQ,
                data,
                levels=[0.5],
                colors= [color])

            
            legend_lines.append(Line2D([], [], color=color, label=beam_or_lat + ":" + str(number)))

            # ingress時間範囲でのコンター交点を抽出・プロット
            ingress_time_array = np.arange(1800, 3600, 60)  # 1分間隔
            boundary_freq = 0
            count = 0

            for ingress_idx, ingress_time in enumerate(ingress_time_array):
                
                ingress_contour = find_contour_frequencies_at_time(
                    cs,
                    ingress_time, 
                    [0.5], 
                    freq_range=(0.1,1)
                )

                # 0.05レベルの周波数を確認（安全にアクセス）
                frequencies_bound = ingress_contour.get(0.5, np.array([]))   
                

                if len(frequencies_bound) > 0:
                    # 全ての周波数をプロット
                    A =frequencies_bound[0]
                    count += 1
                    boundary_freq += A

            if count != 0: 
                ingress_average_boundary_freq = boundary_freq / count
                print(beam_or_lat + str(number) + " ingress average boundary frequency:", ingress_average_boundary_freq)

                if Source_type == "A" or Source_type == "B":
                    observed_freq = 0.7850639198097141
                
                elif Source_type == "C" or Source_type == "D":
                    observed_freq = 0.535843606746542

                print("difference from observed frequency:", abs(ingress_average_boundary_freq - observed_freq))
                



            egress_time_array = np.arange(9000, 10800, 60)  # 1分間隔 
            boundary_freq = 0
            count = 0
            for egress_idx, egress_time in enumerate(egress_time_array):

                egress_contour = find_contour_frequencies_at_time(
                    cs,
                    egress_time, 
                    [0.5], 
                    freq_range=(0.1, 1)
                )

                # 0.05レベルの周波数を確認（安全にアクセス）
                frequencies_bound = egress_contour.get(0.5, np.array([]))   

                if len(frequencies_bound) > 0:
                    # 全ての周波数をプロット
                    A =frequencies_bound[0]
                    count += 1
                    boundary_freq += A
                

            if count != 0: 
                egress_average_boundary_freq = boundary_freq / count
                print(beam_or_lat + str(number) + " egress average boundary frequency:", egress_average_boundary_freq)

                if Source_type == "A" or Source_type == "B":
                    observed_freq = 0.7246157924392409
                
                elif Source_type == "C" or Source_type == "D":
                    observed_freq = 0.405005143415984

                print("difference from observed frequency:", abs(ingress_average_boundary_freq - observed_freq))

            

        ax.legend(handles=legend_lines,loc='lower right')


        ax.scatter(
            ingress_time_list[1],
            ingress_time_list[0],
            c="gray",
            marker=".",
            s=5,
            zorder=2,
        )

        ax.scatter(
            egress_time_list[1],
            egress_time_list[0],
            c="gray",
            marker=".",
            s=5,
            zorder=2,
        )

        extracted_ingress_times = []
        extracted_egress_times = []



        for target_freq in Freq_num:
            # egress_time_list[0]の中でfreqに最も近い値のインデックスを取得
            closest_index = np.argmin(np.abs(ingress_time_list[0] - target_freq))
            # 対応するegress_time_list[1]の値を抽出
            extracted_ingress_times.append(ingress_time_list[1][closest_index])
            extracted_egress_times.append(egress_time_list[1][closest_index])

        # 抽出したegress_time_list[1]の値をNumPy配列に変換
        extracted_ingress_times = np.array(extracted_ingress_times) 
        extracted_egress_times = np.array(extracted_egress_times)

        if time_of_flybies < 10000:
            ax.scatter(
                extracted_ingress_times,
                Freq_num,
                c="red",
                marker="*",#s=10,
                s=35,
                zorder=2,
            )

            ax.scatter(
                extracted_egress_times,
                Freq_num,
                c="red",
                marker="*",#s=10,
                s=35,
                zorder=2,
            )


        ax.contour(
            xx,
            yy,
            Radio_intensity,
            levels=[0.5],
            colors="gray",
            zorder=1,
            linewidths=0.4,
        )

        ax.set_yscale("log")
        ax.set_ylim(0.1, 0.9)

        ax.set_ylabel("Frequency (MHz)")

        # raytrace_time_information ['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']
        # ax.set_xlabel("Time of 27 June 1996")
        # 日時はcsvファイルの情報から記入される
        ax.set_xlabel(
            "Time of "
            + str(raytrace_time_information[1])
            + "/"
            + str(raytrace_time_information[2])
            + "/"
            + str(raytrace_time_information[0])
        )

        # 論文中で使われている横軸の幅とそれに対応する計算開始時刻からの秒数はグローバル変数で指定しておく
        ax.set_xticks(plot_time_step_sec)
        ax.set_xticklabels(plot_time_step_label)

        # 横軸の幅は作りたい図によって変わるので引数用いる
        ax.set_xlim(start_time, end_time)


        for hline in vertical_line_freq:
            plt.hlines(
                hline,
                start_time,
                end_time,
                colors="black",
                linestyle="dashed",
                linewidths=1,
            )

        if beam_or_lat == "Beam":
            ax.set_title("Latitude:" + str(Source_latitude) + " check for " + Source_type)

        elif beam_or_lat == "Latitude":
            ax.set_title("Beam:" + str(Source_beam_angle) + " check for " + Source_type)

        # plt.show()

        if beam_or_lat == "Beam":

            fig.savefig(
                os.path.join(
                    result_output_path
                    + "ExPRES_parameter_check/Beam_check/",
                    "check_ExPRES_parameter"
                    + spacecraft_name
                    + "-"
                    + object_name
                    + "-"
                    + str(time_of_flybies)
                    + "-"
                    + name
                    + "_f-t_"+beam_or_lat+"_"+Source_type+".png",
                ),
                dpi=1000,
            )
        
        elif beam_or_lat == "Latitude":

            fig.savefig(
                os.path.join(
                    result_output_path
                    + "ExPRES_parameter_check/Latitude_check/",
                    "check_ExPRES_parameter"
                    + spacecraft_name
                    + "-"
                    + object_name
                    + "-"
                    + str(time_of_flybies)
                    + "-"
                    + name
                    + "_f-t_"+beam_or_lat+"_"+Source_type+".png",
                ),
                dpi=1000,
            )

    plot_and_save(int(plot_time_step_sec[0]), int(plot_time_step_sec[-1]), "full", beam_or_lat)
    plot_and_save(middle_time+600, middle_time + 2400, "egress", beam_or_lat)
    plot_and_save(middle_time - 2400, middle_time - 600, "ingress", beam_or_lat)



    return ingress_time_list, egress_time_list


def ingress(data, raytrace_time_information):
    """_電波データから掩蔽開始時刻を出力する関数 結果は[[周波数一覧][掩蔽開始時刻一覧]]の二次元データ 掩蔽開始時刻…レイトレーシング開始時刻からの秒数で出力_

    Args:
        data (_type_): _電波源ごとの電波データ 受かる場合は1 受からない場合は0 (周波数の数+1)✖️(時刻)の二次元データ_
        # csvから抽出された時刻データ['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']_
        raytrace_time_information (_type_): _

    Returns:
        _type_: _レイトレーシングデータにおける掩蔽開始時刻け[[周波数一覧][掩蔽開始時刻一覧]]の二次元データ_
    """
    time_step = Time_step(raytrace_time_information)
    raytrace_time = np.arange(
        0, time_step, 1
    )  # レイトレーシング計算した間隔での時間リスト（秒）

    # csvファイルから読み取ったstart時刻から掩蔽中心時刻までの差分（秒）を計算
    middle_time = (
        (raytrace_time_information[8] - raytrace_time_information[2]) * 86400
        + (raytrace_time_information[9] - raytrace_time_information[4]) * 3600
        + (raytrace_time_information[10] - raytrace_time_information[6]) * 60
    )

    # レイトレーシングの周波数リスト(contour plotをする関係で配列の初めに本来はない周波数を挿入しているので周波数の数的には多くなっているので注意)
    raytrace_freq = np.insert(np.array(Freq_num), 0, Freq_underline)

    # レイトレーシングの時間間隔の中で初めて掩蔽中時刻を超えるタイミングの配列番号を取得
    occulted_time = int(np.where(raytrace_time > middle_time)[0][0])
    ingress_time_list = raytrace_freq.copy()
    Data = data

    # 各周波数で..
    for k in range(len(raytrace_freq)):
        # 電波が受かっている配列番号の配列（リスト）を取得
        over_judge_time_list = np.array(np.where((Data[k][:] == 1)))
        # print(over_judge_time_list）

        # その中でも掩蔽中心時刻より手前にある配列番号を取得
        A = over_judge_time_list[over_judge_time_list < occulted_time]

        if len(A) > 0:
            a = A[
                len(A) - 1
            ]  # 電波がうかっている且つ掩蔽中心時刻より早い且つその中で一番遅いものの配列番号
            ingress_time_list = np.append(
                ingress_time_list, ((raytrace_time[a] + raytrace_time[a + 1]) / 2)
            )  # その時刻とその次の時刻の中心時刻を配列に追加

        else:
            ingress_time_list = np.append(
                ingress_time_list, -1e12
            )  # レイトレーシングで掩蔽タイミングを定義できない周波数においては掩蔽タイミングを-10^12としておく

    ingress_time_list = ingress_time_list.reshape(2, int(len(ingress_time_list) / 2))
    # レイトレーシングデータにおける掩蔽開始時刻 [[周波数一覧][掩蔽開始時刻一覧]]の二次元データ　⇨保存（電子密度を変えるたびに異なる値になる・閾値は関係ない）
    return ingress_time_list


def egress(data, raytrace_time_information):
    """_summary_

    Args:
        data (_type_): _電波源ごとの電波データ 受かる場合は1 受からない場合は0 (周波数の数+1)✖️(時刻)の二次元データ_
        # csvから抽出された時刻データ['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']_
        raytrace_time_information (_type_): _


    Returns:
        _type_: _レイトレーシングデータにおける掩蔽終了時刻 [[周波数一覧][掩蔽開始時刻一覧]]の二次元データ_
    """

    time_step = Time_step(raytrace_time_information)
    raytrace_time = np.arange(
        0, time_step, 1
    )  # レイトレーシング計算した間隔での時間リスト（秒）

    # csvファイルから読み取ったstart時刻から掩蔽中心時刻までの差分（秒）を計算
    middle_time = (
        (raytrace_time_information[8] - raytrace_time_information[2]) * 86400
        + (raytrace_time_information[9] - raytrace_time_information[4]) * 3600
        + (raytrace_time_information[10] - raytrace_time_information[6]) * 60
    )

    # レイトレーシングの周波数リスト(contour plotをする関係で配列の初めに本来はない周波数を挿入しているので周波数の数的には多くなているので注意)
    raytrace_freq = np.insert(np.array(Freq_num), 0, Freq_underline)

    # レイトレーシングの時間間隔の中で初めて掩蔽中時刻を超えるタイミングの配列番号を取得
    occulted_time = int(np.where(raytrace_time > middle_time)[0][0])
    egress_time_list = raytrace_freq.copy()
    Data = data

    for k in range(len(raytrace_freq)):
        # 電波が受かっている配列番号の配列（リスト）を取得
        over_judge_time_list = np.array(np.where((Data[k][:] == 1)))
        # print(over_judge_time_list)
        # その中でも掩蔽中心時刻より後ろにある配列番号を取得
        B = over_judge_time_list[over_judge_time_list > occulted_time]

        if len(B) > 0:
            b = B[
                0
            ]  # 電波がうかっている且つ掩蔽中心時刻より遅い且つその中で一番早いものの配列番号
            egress_time_list = np.append(
                egress_time_list, ((raytrace_time[b] + raytrace_time[b - 1]) / 2)
            )  # その時刻とその前の時刻の中心時刻を配列に追加

        else:
            egress_time_list = np.append(
                egress_time_list, -1e12
            )  # レイトレーシングで掩蔽タイミングを定義できない周波数においては掩蔽タイミングを-10^12としておく

    egress_time_list = egress_time_list.reshape(2, int(len(egress_time_list) / 2))

    # レイトレーシングデータにおける掩蔽終了時刻 [[周波数一覧][掩蔽終了時刻一覧]]の二次元データ　⇨保存（電子密度を変えるたびに異なる値になる・閾値は関係ない）
    return egress_time_list


def Make_FT_polarization_full(
    Data, raytrace_time_information, beam_or_lat):
    time_step = Time_step(raytrace_time_information)

    time_list = np.arange(
        0, time_step, 1
    )  # エクスプレスコードで計算している時間幅（sec)を60で割る

    DOY = calculate_day_of_year(raytrace_time_information[0],raytrace_time_information[1],raytrace_time_information[2])

    # 偏波データ取得
    Radio_polarization = load_polarizaiton_data(cassini_data_path, raytrace_time_information[0], DOY , raytrace_time_information[4], raytrace_time_information[5] -raytrace_time_information[4])

    # 電波データ取得（ここでは時間と周波数のデータしか使わない）
    (
        hour_array, 
        minute_array, 
        second_array, 
        Radio_epoch_from_0min_1d,
        Radio_frequency_1d, 
        Radio_intensity
    ) = load_intensity_data (cassini_data_path, raytrace_time_information[0], DOY , raytrace_time_information[4], raytrace_time_information[5] -raytrace_time_information[4])

    Radio_polarization_row = Radio_polarization.copy()

    FREQ = np.insert(np.array(Freq_num), 0, Freq_underline)

    # csvファイルから読み取ったstart時刻から掩蔽中心時刻までの差分（秒）を計算
    middle_time = (
        (raytrace_time_information[8] - raytrace_time_information[2]) * 86400
        + (raytrace_time_information[9] - raytrace_time_information[4]) * 3600
        + (raytrace_time_information[10] - raytrace_time_information[6]) * 60
    )


    """
    ここまでで出来上がっている材料たち
    time_list レイトレーシング(1分間隔)の時刻データ(観測開始隊ミンングからの秒) ex. [0,1,2, ...]
    FREQ レイトレーシングの周波数リスト
    (contour plotをする関係で配列の初めに本来はない周波数を挿入しているので周波数の数的には多くなているので注意)
    DataA-DataD 電波源A-Dの電波データ 受かる場合は1 受からない場合は0 (周波数の数+1)✖️(時刻)の二次元データ

    Radio_epoch_from_0min_1d 探査機の観測結果の時刻データ(観測開始隊ミンングからの秒) ex. [0,8.434,26.56 ...]
    Radio_frequency_1d 探査機の観測結果の周波数データ(MHz) ex. [5.620e-06 1.000e-05 ...]
    Radio_polarization_row 探査機の電波強度データ　周波数の数✖️時刻の数の二次元データ
    Radio_polarization 探査機の電波強度データで閾値を超えてる部分を1、それ以外を0としているデータ 周波数の数✖️時刻の数の二次元データ
    ingress_time_list 電波観測データにおける掩蔽開始時刻 [[周波数一覧][掩蔽開始時刻一覧]]の二次元データ
    egress_time_list ingress_time_listの掩蔽終了ver

    """


    def plot_and_save(start_time, end_time, name, beam_or_lat):
        # ガリレオ探査機の電波データの時刻・周波数でメッシュ作成
        xx, yy = np.meshgrid(Radio_epoch_from_0min_1d, Radio_frequency_1d)


        fig, ax = plt.subplots(1, 1)

        # ガリレオ探査機の電波強度をカラーマップへ
        pcm = ax.pcolormesh(
            xx,
            yy,
            Radio_polarization_row,
            vmin=-1,
            vmax=1,
            cmap='seismic'
        )    
        fig.colorbar(
            pcm,
            extend="max",
            label="Polarization",
        )
        #print(Data)
        contours = {}
        legend_lines = []

        min_num = 90
        max_num = 0
        for label, data in Data.items():
            number = int(re.search(r'\d+', str(label)).group())
            if number < min_num:
                min_num = number
            if number > max_num:
                max_num = number

        # レイトレーシングの結果をコンタープロットで表示
        for label, data in Data.items():
            number = int(re.search(r'\d+', str(label)).group())
            
            cmap = cm.get_cmap('cividis')
            color = cmap((number - min_num) / (max_num - min_num))  # 色を正規化して取得

            cs = ax.contour(
                time_list,
                FREQ,
                data,
                levels=[0.5],
                colors= [color])
            
            legend_lines.append(Line2D([], [], color=color, label=beam_or_lat + ":" + str(number)))

        ax.legend(handles=legend_lines,loc='lower right')

        
        # ax.contour(time_list, FREQ, DataC, levels=[0.5], colors="orange")

        # 緑点で強度の閾値を超えた部分を表示


        """
        ax.scatter(
            extracted_ingress_times,
            Freq_num,
            cmap="bwr",
            edgecolors="palegreen",
            marker="*",
            s=35,
            linewidths=0.5,
            zorder=2,
        )

        ax.scatter(
            extracted_egress_times,
            Freq_num,
            cmap="bwr",
            edgecolors="palegreen",
            marker="*",
            s=35,
            linewidths=0.5,
            zorder=2,
        )
        """

        ax.set_yscale("log")
        ax.set_ylim(0.1, 0.9)

        ax.set_ylabel("Frequency (MHz)")

        # raytrace_time_information ['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']
        # ax.set_xlabel("Time of 27 June 1996")
        # 日時はcsvファイルの情報から記入される
        ax.set_xlabel(
            "Time of "
            + str(raytrace_time_information[1])
            + "/"
            + str(raytrace_time_information[2])
            + "/"
            + str(raytrace_time_information[0])
        )

        # 論文中で使われている横軸の幅とそれに対応する計算開始時刻からの秒数はグローバル変数で指定しておく
        ax.set_xticks(plot_time_step_sec)
        ax.set_xticklabels(plot_time_step_label)

        # 横軸の幅は作りたい図によって変わるので引数用いる
        ax.set_xlim(start_time, end_time)

        for hline in vertical_line_freq:
            plt.hlines(
                hline,
                start_time,
                end_time,
                colors="black",
                linestyle="dashed",
                linewidths=1,
            )

        if beam_or_lat == "Beam":
            ax.set_title("Latitude:" + str(Source_latitude) + " check for " + Source_type)

        elif beam_or_lat == "Latitude":
            ax.set_title("Beam:" + str(Source_beam_angle) + " check for " + Source_type)

        # plt.show()

        if beam_or_lat == "Beam":

            fig.savefig(
                os.path.join(
                    result_output_path
                    + "ExPRES_parameter_check/Beam_check/",
                    "check_ExPRES_parameter"
                    + spacecraft_name
                    + "-"
                    + object_name
                    + "-"
                    + str(time_of_flybies)
                    + "-"
                    + name
                    + "_lat-"
                    + str(Source_latitude)
                    + "_f-t_polarization_"+beam_or_lat+"_"+Source_type+".png",
                ),
                dpi=1000,
            )

        elif beam_or_lat == "Latitude":
            fig.savefig(
                os.path.join(
                    result_output_path
                    + "ExPRES_parameter_check/Latitude_check/",
                    "check_ExPRES_parameter"
                    + spacecraft_name
                    + "-"
                    + object_name
                    + "-"
                    + str(time_of_flybies)
                    + "-"
                    + name
                    + "_f-t_polarization_"+beam_or_lat+"_"+Source_type+".png",
                ),
                dpi=1000,
            )

    plot_and_save(int(plot_time_step_sec[0]), int(plot_time_step_sec[-1]), "full", beam_or_lat)
    plot_and_save(middle_time+600, middle_time + 2400, "egress", beam_or_lat)
    plot_and_save(middle_time - 2400, middle_time-600, "ingress", beam_or_lat)
    return 0



class Evaluate_raytrace_data:
    def __init__(self, Data, time_data):
        self.data = Data
        self.ingress = ingress(Data, time_data)
        self.egress = egress(Data, time_data)


def nearest_raytrace_time(arr, target_frequency):
    """_レイトレーシングの掩蔽データから指定した周波数における掩蔽タイミングを出力。周波数が一致しない場合は周波数を対数線形補間した場合の掩蔽タイミングを出力_

    Args:
        arr (_numpy.array_): _レイトレーシングにおける掩蔽タイミング [[周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ_
        target_frequency (_float_): _掩蔽タイミングを知りたい周波数_

    Returns:
        _type
    """

    left_index = (
        np.searchsorted(arr[0][:], target_frequency, side="right") - 1
    )  # 左側のインデックスを取得
    right_index = np.searchsorted(arr[0][:], target_frequency, side="right")

    if right_index == len((arr[0][:])) or (
        np.searchsorted(arr[0][:], target_frequency, side="left") == 0
    ):
        raytrace_timing = (
            -1e10
        )  # ガリレオ探査機の周波数ステップのうち、レイトレーシングの周波数範囲で掩蔽タイミングを定義できない周波数においては、レイトレーシングの掩蔽タイミングを-10^10とおく

    else:
        small_raytrace_freq = arr[0][left_index]
        large_raytrace_freq = arr[0][right_index]

        small_raytrace_freq_log = np.log(small_raytrace_freq)
        large_raytrace_freq_log = np.log(large_raytrace_freq)
        target_freq_log = np.log(target_frequency)

        if large_raytrace_freq_log == small_raytrace_freq_log:
            t = 0

        else:
            t = (target_freq_log - small_raytrace_freq_log) / (
                large_raytrace_freq_log - small_raytrace_freq_log
            )

        small_raytrace_timing = arr[1][left_index]
        large_raytrace_timing = arr[1][right_index]

        raytrace_timing = (
            small_raytrace_timing + (large_raytrace_timing - small_raytrace_timing) * t
        )

    return raytrace_timing


def Evaluate_ionosphere_density(raytrace_data, radio_data, polarization_data):
    """_レイトレーシングによる掩蔽タイミングとガリレオデータにおける掩蔽タイミングの差分を計算_

    Args:
        raytrace_data (_type_): _レイトレーシングにおける掩蔽タイミング [[周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ_
        radio_data (_type_): _ 電波観測データにおける掩蔽タイミング [[周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ_
        polarization_data (_type_): _電波観測データの偏波データ [掩蔽開始時刻直前1分間の偏波 for 電波データのの周波数数 （0以下..-1, 0以上..1）]_

    Returns:
        _type_: _電波データの各周波数における掩蔽タイミングの差分  [[周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ_
    """

    
    time_defference_index = radio_data.copy()  # ガリレオデータの周波数リストをコピー
    num = radio_data.shape[
        1
    ]  # 2（周波数・時間）✖️(ガリレオ周波数の種類分)の２つ目の要素　⇨ ガリレオ周波数の種類分

    # レイトレーシングの周波数種類ごとに
    for i in range(num):
        raytrace_occulatation_timing = nearest_raytrace_time(
            raytrace_data, radio_data[0][i]
        )  # レイトレーシングの周波数とガリレオ探査機の電波データの周波数リストで一番その差が小さい部分のインデックスを取得

        time_defference = abs(
            radio_data[1][i] - raytrace_occulatation_timing
        )  # 取得した周波数の電波データでの掩蔽開始・終了時間とレイトレーシングの時間の差を取る
        time_defference_index[1][i] = time_defference

    # 掩蔽開始時刻直前1分間の偏波データを追加
    time_defference_index = np.vstack(
        (time_defference_index, polarization_data)
    )  # ガリレオデータの周波数リストをコピー

    return time_defference_index


def Evaluate_data_coutour(time_data):

    DOY = calculate_day_of_year(time_data[0],time_data[1],time_data[2])

    (hour_array, 
    minute_array, 
    second_array, 
    Radio_epoch_from_0min_1d,
    Radio_frequency_1d, 
    Radio_intensity)= load_intensity_data(cassini_data_path, time_data[0], DOY , time_data[4], time_data[5] -time_data[4])

    using_radio_data = Radio_intensity[np.where(Radio_frequency_1d > 1e-1)][
        :
    ].flatten()

    fig, ax = plt.subplots(1, 1)
    # ax.hist(using_radio_data, range=(1e-18, 1e-12),bins=np.logspace(-18, -12, 30))
    ax.hist(using_radio_data, bins=np.logspace(-17, -12, 50))
    ax.set_xscale("log")
    fig.savefig("A")

    return 0


# %%


def main():
    time_information = Pick_up_timeinfo()
    # print(time_information, radio_data)
    # old [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 座標変換した時のx(tangential point との水平方向の距離),6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度,8 探査機の経度]
    # new [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]

    # latitude 変化用の図を作る
    arrays = {}

    for source_latitude in Source_latitude_range:
        source_beam_angle = Source_beam_angle

        file_name = (result_output_path
            + "ExPRES_parameter_check/interpolated_"
            + object_name
            + "_"
            + spacecraft_name
            + "_"
            + str(time_of_flybies)
            + "_lat"
            + str(source_latitude)
            + "_beamangle"
            + str(source_beam_angle)
            + "_dectable_radio_data.txt")

        if os.path.exists(file_name):

            detectable_radio = np.loadtxt(file_name)
            
            # レイトレーシングの結果をコンタープロットで表示
            if detectable_radio.shape[1] == 11:
                detectable_A, detectable_B, detectable_C, detectable_D = Prepare_Figure(
                    detectable_radio, time_information)
        
                if Source_type == "A":
                    arrays["Latitude" + str(source_latitude)] = detectable_A
                
                elif Source_type == "B":
                    arrays["Latitude" + str(source_latitude)] = detectable_B
                
                elif Source_type == "C":
                    arrays["Latitude" + str(source_latitude)] = detectable_C
                
                elif Source_type == "D":
                    arrays["Latitude" + str(source_latitude)] = detectable_D

    
    Hour, Minute, Second, Radio_epoch_from_0min_1d, Frequency_1d, Radio_intensity = load_intensity_data(cassini_data_path, start_year, start_date, start_hour, duration)

    ingress_time, egress_time = Make_intensity_FT_full(
        arrays,
        time_information,
        "Latitude"

    )

    Make_FT_polarization_full(
        arrays,
        time_information,
        "Latitude"
    )


    # beaming angle の範囲を指定する
    arrays = {}

    for source_beam_angle in Source_beam_angle_range:
        source_latitude = Source_latitude

        file_name = (result_output_path
            + "ExPRES_parameter_check/interpolated_"
            + object_name
            + "_"
            + spacecraft_name
            + "_"
            + str(time_of_flybies)
            + "_lat"
            + str(source_latitude)
            + "_beamangle"
            + str(source_beam_angle)
            + "_dectable_radio_data.txt")

        if os.path.exists(file_name):

            detectable_radio = np.loadtxt(file_name)
            
            # レイトレーシングの結果をコンタープロットで表示
            if detectable_radio.shape[1] == 11:
                detectable_A, detectable_B, detectable_C, detectable_D = Prepare_Figure(
                    detectable_radio, time_information)
        
                if Source_type == "A":
                    arrays["Beam" + str(source_beam_angle)] = detectable_A
                
                elif Source_type == "B":
                    arrays["Beam" + str(source_beam_angle)] = detectable_B
                
                elif Source_type == "C":
                    arrays["Beam" + str(source_beam_angle)] = detectable_C
                
                elif Source_type == "D":
                    arrays["Beam" + str(source_beam_angle)] = detectable_D

    
    Hour, Minute, Second, Radio_epoch_from_0min_1d, Frequency_1d, Radio_intensity = load_intensity_data(cassini_data_path, start_year, start_date, start_hour, duration)

    ingress_time, egress_time = Make_intensity_FT_full(
        arrays,
        time_information,
        "Beam"

    )

    Make_FT_polarization_full(
        arrays,
        time_information,
        "Beam"
    )



    return 0


if __name__ == "__main__":
    main()


# %%
