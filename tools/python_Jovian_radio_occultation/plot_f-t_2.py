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

# %%
# あらかじめ ../result_sgepss2021/~/~ に必要なレイトレーシング結果とパラメータセットを入れること
args = sys.argv

object_name = "ganymede"  # ganydeme/europa/calisto``

spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 1  # ..th flyby
highest_plasma = "1e2"  # 単位は(/cc) 2e2/4e2/16e22
plasma_scaleheight = "0.25e2"  # 単位は(km) 1.5e2/3e2/6e2

# object_name = args[1]  # ganydeme/europa/calisto``
# time_of_flybies = int(args[2])  # ..th flyby
# highest_plasma = args[1]  # 単位は(/cc) 2e2/4e2/16e22 #12.5 13.5
# plasma_scaleheight = args[2]  # 単位は(km) 1.5e2/3e2/6e2
# boundary_intensity_str = "7e-16"  # boundary_intensity_str = '1e-15'
boundary_average_str = "10"  # boundary_intensity_str = '10'⇨ノイズフロアの10倍強度まで

vertical_line_freq = np.array([0.65, 4.5])  # MHz
# vertical_line_freq = np.array([0])  # MHz
# print(args[1] + args[2] + " max:" + args[3] + " scale:" + args[4])

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

Radio_name_csv = (
    "../result_for_yasudaetal2022/tracing_range_"
    + spacecraft_name
    + "_"
    + object_name
    + "_"
    + str(time_of_flybies)
    + "_flybys/para_"
    + highest_plasma
    + "_"
    + plasma_scaleheight
    + ".csv"
)
Radio_Range = pd.read_csv(Radio_name_csv, header=0, engine="python")
# [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 座標変換した時のx(tangential point との水平方向の距離),6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度,8 探査機の経度]


if object_name == "ganymede" and time_of_flybies == 1:
    # G1 flyby
    plot_time_step_sec = [0, 900, 1800, 2700, 3600, 4500, 5400]
    plot_time_step_label = [
        "05:30",
        "05:45",
        "06:00",
        "06:15",
        "06:30",
        "06:45",
        "07;00",
    ]
    exclave_exception = np.array([[4.298], [1], [0]])
    ingress_egress_time_window = 1800


if object_name == "europa" and time_of_flybies == 12:
    # E12 flyby
    plot_time_step_sec = [6300, 6600, 6900, 7200, 7500, 7800, 8100, 8400, 8700]
    plot_time_step_label = [
        "11:45",
        "11:50",
        "11:55",
        "12:00",
        "12:05",
        "12:10",
        "12:15",
        "12:20",
        "12:25",
    ]

if object_name == "callisto" and time_of_flybies == 30:
    # C30 flyby
    plot_time_step_sec = [0, 1800, 3600, 4200, 4800, 5400, 7200, 9000, 10800]
    plot_time_step_label = [
        "10:00",
        "10:30",
        "11:00",
        "11:10",
        "11:20",
        "11:30",
        "12:00",
        "12:30",
        "13:00",
    ]
    exclave_exception = np.array(
        [
            [
                0.8099,
                0.9856,
                1.087,
                1.199,
                1.323,
                1.460,
                1.610,
                1.776,
                1.960,
                2.162,
                2.385,
                2.631,
                4.741,
            ],
            [1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 1, 2, 9, 12, 10, 3, 7, 1, 1, 1],
        ]
    )
    ingress_egress_time_window = 600

if object_name == "callisto" and time_of_flybies == 9:
    # C9 flyby
    plot_time_step_sec = [0, 1800, 3600, 5400, 6000, 6600, 7200, 9000, 10800]
    plot_time_step_label = [
        "12:00",
        "12:30",
        "13:00",
        "13:30",
        "13:40",
        "13:50",
        "14:00",
        "14:30",
        "15:00",
    ]

    exclave_exception = np.array([[1.960, 2.902, 3.201], [0, 0, 0], [1, 1, 1]])
    ingress_egress_time_window = 900

# europa & ganymede
if object_name == "ganymede" or object_name == "europa":
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

    Freq_underline = 0.36122


# callisto
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

    Freq_underline = 0.32744

Freq_num = []
for idx in Freq_str:
    Freq_num.append(float(idx) / 1000000)

Highest = Radio_Range.highest
Lowest = Radio_Range.lowest
Except = Radio_Range.exc


# ガリレオ探査機によって取得される周波数・探査機が変わったらこの周波数も変わってくるはず
gal_fleq_tag_row = [
    5.620e00,
    1.000e01,
    1.780e01,
    3.110e01,
    4.213e01,
    4.538e01,
    4.888e01,
    5.265e01,
    5.671e01,
    6.109e01,
    6.580e01,
    7.087e01,
    7.634e01,
    8.223e01,
    8.857e01,
    9.541e01,
    1.028e02,
    1.107e02,
    1.192e02,
    1.284e02,
    1.383e02,
    1.490e02,
    1.605e02,
    1.729e02,
    1.862e02,
    2.006e02,
    2.160e02,
    2.327e02,
    2.507e02,
    2.700e02,
    2.908e02,
    3.133e02,
    3.374e02,
    3.634e02,
    3.915e02,
    4.217e02,
    4.542e02,
    4.892e02,
    5.270e02,
    5.676e02,
    6.114e02,
    6.586e02,
    7.094e02,
    7.641e02,
    8.230e02,
    8.865e02,
    9.549e02,
    1.029e03,
    1.108e03,
    1.193e03,
    1.285e03,
    1.385e03,
    1.491e03,
    1.606e03,
    1.730e03,
    1.864e03,
    2.008e03,
    2.162e03,
    2.329e03,
    2.509e03,
    2.702e03,
    2.911e03,
    3.135e03,
    3.377e03,
    3.638e03,
    3.918e03,
    4.221e03,
    4.546e03,
    4.897e03,
    5.275e03,
    5.681e03,
    6.120e03,
    6.592e03,
    7.100e03,
    7.648e03,
    8.238e03,
    8.873e03,
    9.558e03,
    1.029e04,
    1.109e04,
    1.194e04,
    1.287e04,
    1.386e04,
    1.493e04,
    1.608e04,
    1.732e04,
    1.865e04,
    2.009e04,
    2.164e04,
    2.331e04,
    2.511e04,
    2.705e04,
    2.913e04,
    3.138e04,
    3.380e04,
    3.641e04,
    3.922e04,
    4.224e04,
    4.550e04,
    4.901e04,
    5.279e04,
    5.686e04,
    6.125e04,
    6.598e04,
    7.106e04,
    7.655e04,
    8.245e04,
    8.881e04,
    9.566e04,
    1.030e05,
    1.030e05,
    1.137e05,
    1.254e05,
    1.383e05,
    1.526e05,
    1.683e05,
    1.857e05,
    2.049e05,
    2.260e05,
    2.493e05,
    2.750e05,
    3.034e05,
    3.347e05,
    3.692e05,
    4.073e05,
    4.493e05,
    4.957e05,
    5.468e05,
    6.033e05,
    6.655e05,
    7.341e05,
    8.099e05,
    8.934e05,
    9.856e05,
    1.087e06,
    1.199e06,
    1.323e06,
    1.460e06,
    1.610e06,
    1.776e06,
    1.960e06,
    2.162e06,
    2.385e06,
    2.631e06,
    2.902e06,
    3.201e06,
    3.532e06,
    3.896e06,
    4.298e06,
    4.741e06,
    5.231e06,
    5.770e06,
]


# %%


def Pick_up_cdf():
    flyby_list_path = "../result_for_yasudaetal2022/occultation_flyby_list.csv"
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

    # csvから対象の電波データを有するcsvの名前を取得
    radio_data_name = str(complete_selecred_flyby_list["radio_data_txt"][0])

    # csvの時刻データと電波データの名前をかえす
    return time_information, radio_data_name


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

        Fre = np.where(Freq_num == judgement[k][3])
        # int(Fre[0])+1になっているのは、のちのコンタープロットのために一個多い周波数で作ってあるのでミスではない
        if judgement[k][10] == 1 and judgement[k][5] == 1:
            DataA[int(Fre[0]) + 1][Num] = 1

        if judgement[k][10] == -1 and judgement[k][5] == 1:
            DataB[int(Fre[0]) + 1][Num] = 1

        if judgement[k][10] == 1 and judgement[k][5] == -1:
            DataC[int(Fre[0]) + 1][Num] = 1

        if judgement[k][10] == -1 and judgement[k][5] == -1:
            DataD[int(Fre[0]) + 1][Num] = 1

    print("complete")

    return DataA, DataB, DataC, DataD


# ガリレオ探査機の周波数一覧（Hz)とダウンロードした電波強度電波を代入（das2をcsvに変換）


def Prepare_Galileo_data(time_info, data_name):
    """_探査機による電波データのファイル名から電波データの時刻(電波データから読み取れる時刻とcsvファイルの時刻の差・周波数(ソースははじめ電波リストから)・電波強度を出力する_

    Args:
        data_name (_str_): _用いる電波データのファイル名を入力_

    Returns:
        _type_: _電波データの時刻の配列・周波数の配列・電波強度の配列_
    """
    # 電波強度のデータを取得（一列目は時刻データになってる）
    # 初めの数行は読み取らないよう設定・時刻データを読み取って時刻をプロットするためここがずれても影響はないが、データがない行を読むと怒られるのでその時はd2sファイルを確認

    rad_row_data = pd.read_csv(
        "../result_for_yasudaetal2022/galileo_radio_data/" + data_name,
        header=None,
        skiprows=24,
        delimiter="  ",
        engine="python",
    )

    # 電波データの周波数の単位をHzからMHzに変換する
    gal_fleq_tag = np.array(gal_fleq_tag_row, dtype="float64") / 1000000

    # 一列目の時刻データを文字列で取得（例; :10:1996-06-27T05:30:08.695） ・同じ長さの０配列を準備・
    gal_time_tag_prepare = np.array(rad_row_data.iloc[:, 0])
    gal_time_tag_prepare = gal_time_tag_prepare.astype(str)
    gal_time_tag = np.zeros(len(gal_time_tag_prepare))

    # 文字列のデータから開始時刻からの経過時間（秒）に変換
    # Tで分けた[1]　例 :10:1996-06-27T05:30:08.695 ⇨ 05:30:08.695
    # :で分ける　例;05:30:08.695 ⇨ 05 30 08.695
    for i in range(len(gal_time_tag)):
        hour_min_sec = np.char.split(
            np.char.split(gal_time_tag_prepare[:], sep="T")[i][1], sep=[":"]
        )[0]

        hour_min_sec_list = [float(vle) for vle in hour_min_sec]

        # Tで分けた[0]　例; :10:1996-06-27T05:30:08.695 ⇨ 1996-06-27
        # :で分けた最後の部分　例; :10:1996-06-27 ⇨ 10 1996-06-27
        year_month_day_pre = np.char.split(
            np.char.split(gal_time_tag_prepare[:], sep="T")[i][0], sep=[":"]
        )[0][-1]

        year_month_day = np.char.split(year_month_day_pre, sep=["-"])[0]

        year_month_day_list = [float(vle) for vle in year_month_day]

        # 秒に変換 27✖️86400 + 05✖️3600 + 30✖️60 ＋ 08.695

        gal_time_tag[i] = (
            hour_min_sec_list[2]
            + hour_min_sec_list[1] * 60
            + hour_min_sec_list[0] * 3600
            + year_month_day_list[2] * 86400
        )  # 経過時間(sec)に変換

    # time_info['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']
    # csvファイルからの開始時刻を秒に変換
    # startday(2)*86400+start_hour(4)*3600+ start_min(6)*60
    start_time = time_info[2] * 86400 + time_info[4] * 3600 + time_info[6] * 60
    gal_time_tag = np.array(gal_time_tag - start_time)
    df = pd.DataFrame(rad_row_data.iloc[:, 1:])

    DDF = np.array(df).astype(np.float64).T
    # print(DDF)
    # print(len(gal_fleq_tag), len(gal_time_tag), DDF.shape)

    # ガリレオ探査機のデータ取得開始時刻からの経過時間（sec) , ガリレオ探査機のデータ取得周波数（MHz), ガリレオ探査機の取得した電波強度（代入したデータと同じ単位）
    return gal_time_tag, gal_fleq_tag, DDF


def Make_FT_full(
    DataA, DataB, DataC, DataD, raytrace_time_information, radio_data_name
):
    time_step = Time_step(raytrace_time_information)

    time_list = np.arange(0, time_step, 1)  # エクスプレスコードで計算している時間幅（sec)を60で割る

    # ガリレオ探査機のデータ取得開始時刻からの経過時間（sec) , ガリレオ探査機のデータ取得周波数（MHz), ガリレオ探査機の取得した電波強度（代入したデータと同じ単位）
    (
        galileo_data_time,
        galileo_data_freq,
        galileo_radio_intensity,
    ) = Prepare_Galileo_data(raytrace_time_information, radio_data_name)

    galileo_radio_intensity_row = galileo_radio_intensity.copy()
    boundary_intensity_array = np.zeros(len(galileo_data_freq))

    # [[周波数一覧][各周波数でのノイズフロアの電波強度平均][角周波数でのノイズフロアの電波強度標準偏差][各周波数でのノイズフロアの電波強度の中央値]]
    noise_data = np.genfromtxt(
        "../result_for_yasudaetal2022/radio_plot/"
        + object_name
        + str(time_of_flybies)
        + "/"
        + spacecraft_name
        + "_noise_floor_excepted_two_sigma_"
        + object_name
        + str(time_of_flybies)
        + ".csv",
        delimiter=",",
    )
    boundary_average = float(boundary_average_str)

    for i in range(len(galileo_data_freq)):
        certain_freq_data = galileo_radio_intensity[i]  # i番目の周波数の全データ
        boundary_intensity = (
            boundary_average * noise_data[1, i]
        )  # i番目の周波数の強度平均値×boundary_average
        boundary_intensity_array[i] = boundary_intensity
        # print(certain_freq_data.shape)
        detectable_position_array = np.where(certain_freq_data > boundary_intensity)[0]
        undetectable_position_array = np.where(certain_freq_data <= boundary_intensity)[
            0
        ]
        # print(certain_freq_data)
        galileo_radio_intensity[i, detectable_position_array] = 1
        galileo_radio_intensity[i, undetectable_position_array] = 0

    # raytrace_time_information ['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']

    # start時刻から掩蔽中心時刻までの差分（秒）を計算
    middle_time = (
        (raytrace_time_information[8] - raytrace_time_information[2]) * 86400
        + (raytrace_time_information[9] - raytrace_time_information[4]) * 3600
        + (raytrace_time_information[10] - raytrace_time_information[6]) * 60
    )

    # ガリレオ電波データの掩蔽中心時刻を初めに超える時間ステップの配列番号を取得
    occulted_time = int(np.where(galileo_data_time > middle_time)[0][0])
    ingress_time_list = galileo_data_freq.copy()
    egress_time_list = galileo_data_freq.copy()

    # それぞれの周波数に対して..
    for k in range(len(galileo_data_freq)):
        # k番目の周波数の強度で閾値を超える電波強度を捉えている電波データの配列番号のリストを取得
        over_judge_time_list = np.array(np.where((galileo_radio_intensity[k][:] > 0.5)))

        # print(over_judge_time_list)
        # 閾値を超える電波強度を観測しているものの中でも掩蔽中心時刻前のもののリストをA、後のもののリストをBとしてわける
        A = over_judge_time_list[over_judge_time_list < occulted_time]
        B = over_judge_time_list[over_judge_time_list > occulted_time]

        ingress_exception = 0
        egress_exception = 0

        # 閾値を超える電波強度を
        if np.any(exclave_exception[0] == galileo_data_freq[k]):
            exception_position = np.where(exclave_exception[0] == galileo_data_freq[k])[
                0
            ][0]

            ingress_exception = int(exclave_exception[1][exception_position])
            egress_exception = int(exclave_exception[2][exception_position])

        # 電波強度が十分であればAもBも複数ヒットするはず...
        if len(B) > 0:
            b = B[egress_exception]
            # 初めて閾値を超える強度が観測される時刻（正確にはstart時刻からの差分（秒））がgalileo_data_time[b]
            # その１つ前の時刻 galileo_data_time[b-1]なので、その真ん中の時間がegress_time_listに加わる（要するに終了タイミング）
            egress_time_list = np.append(
                egress_time_list,
                ((galileo_data_time[b] + galileo_data_time[b - 1]) / 2),
            )

        else:
            egress_time_list = np.append(
                egress_time_list, -1e8
            )  # 閾値を適宜できない周波数においては掩蔽タイミングを-10^8としておく

        if len(A) > 0:
            a = A[-1 - ingress_exception]
            # 最後に閾値を超える強度が観測される時刻（正確にはstart時刻からの差分（秒））がgalileo_data_time[a]
            # その１つ次の時刻 galileo_data_time[a+1]なので、その真ん中の時間がingress_time_listに加わる（要するに開始タイミング）
            ingress_time_list = np.append(
                ingress_time_list,
                ((galileo_data_time[a] + galileo_data_time[a + 1]) / 2),
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

    galileo_data_time 探査機の観測結果の時刻データ(観測開始隊ミンングからの秒) ex. [0,8.434,26.56 ...]
    galileo_data_freq 探査機の観測結果の周波数データ(MHz) ex. [5.620e-06 1.000e-05 ...]
    galileo_radio_intensity_row 探査機の電波強度データ　周波数の数✖️時刻の数の二次元データ
    galileo_radio_intensity 探査機の電波強度データで閾値を超えてる部分を1、それ以外を0としているデータ 周波数の数✖️時刻の数の二次元データ
    ingress_time_list 電波観測データにおける掩蔽開始時刻 [[周波数一覧][掩蔽開始時刻一覧]]の二次元データ
    egress_time_list ingress_time_listの掩蔽終了ver

    """

    def plot_and_save(start_time, end_time, name):
        # ガリレオ探査機の電波データの時刻・周波数でメッシュ作成
        xx, yy = np.meshgrid(galileo_data_time, galileo_data_freq)

        fig, ax = plt.subplots(1, 1)

        # ガリレオ探査機の電波強度をカラーマップへ
        pcm = ax.pcolormesh(
            xx,
            yy,
            galileo_radio_intensity_row,
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

        # ガリレオ探査機の電波強度の閾値を赤線＃

        # レイトレーシングの結果をコンタープロットで表示
        ax.contour(time_list, FREQ, DataA, levels=[0.5], colors="1.0")
        ax.contour(time_list, FREQ, DataB, levels=[0.5], colors="0.6")
        # ax.contour(time_list, FREQ, DataC, levels=[0.5], colors="0.3")
        ax.contour(time_list, FREQ, DataD, levels=[0.5], colors="0")
        ax.contour(time_list, FREQ, DataC, levels=[0.5], colors="orange")
        ax.scatter(
            ingress_time_list[1],
            ingress_time_list[0],
            c="red",
            marker=".",
            s=8,
            zorder=2,
        )
        ax.scatter(
            egress_time_list[1],
            egress_time_list[0],
            c="red",
            marker=".",
            s=8,
            zorder=3,
        )
        ax.contour(
            xx,
            yy,
            galileo_radio_intensity,
            levels=[0.5],
            colors="red",
            zorder=1,
            linewidths=0.8,
        )

        ax.set_yscale("log")
        ax.set_ylim(0.1, 6.0)

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

        # 最大電子密度の動画作るときはこっち
        """
        ax.text(
            3700,
            0.13,
            str(float(highest_plasma)) + " (cm⁻³)",
            fontsize=30,
            fontname="Helvetica",
            color="white",
            ha="center",
            va="center",
        )
        """

        # スケールハイトの動画作るときはこっち
        ax.text(
            3700,
            0.13,
            str(float(plasma_scaleheight)) + " (km)",
            fontsize=30,
            fontname="Helvetica",
            color="white",
            ha="center",
            va="center",
        )

        for hline in vertical_line_freq:
            plt.hlines(
                hline,
                start_time,
                end_time,
                colors="black",
                linestyle="dashed",
                linewidths=1,
            )
            """
            plt.annotate(
                str(hline) + "MHz",
                (start_time + 20, hline + 0.05),
                color="hotpink",
            )
            """
        ax.set_title(
            "Maximum density "
            + highest_plasma
            + "(cm-3) & scale height "
            + plasma_scaleheight
            + "(km)"
        )
        # ax.set_title("No ionosphere")

        # plt.show()

        fig.savefig(
            os.path.join(
                "../result_for_yasudaetal2022/raytracing_"
                + object_name
                + "_results/"
                + object_name
                + "_"
                + highest_plasma
                + "_"
                + plasma_scaleheight
                + "/",
                "interpolated_"
                + spacecraft_name
                + "_"
                + object_name
                + "_"
                + str(time_of_flybies)
                + "_"
                + highest_plasma
                + "_"
                + plasma_scaleheight
                + "_boundary_int="
                + boundary_average_str
                + "dB_"
                + name
                + "_f-t.png",
            ),
            dpi=1000,
        )

        fig.savefig(
            os.path.join(
                "../result_for_yasudaetal2022/f-t_plot_"
                + spacecraft_name
                + "_"
                + object_name
                + "_"
                + str(time_of_flybies)
                + "_flyby/radio_boundary_intensity_"
                + boundary_average_str
                + "dB",
                "interpolated_"
                + spacecraft_name
                + "_"
                + object_name
                + "_"
                + str(time_of_flybies)
                + "_"
                + highest_plasma
                + "_"
                + plasma_scaleheight
                + "_boundary_int="
                + boundary_average_str
                + "dB_"
                + name
                + "_f-t.png",
            ),
            dpi=1000,
        )

    plot_and_save(int(plot_time_step_sec[0]), int(plot_time_step_sec[-1]), "full")
    plot_and_save(middle_time, middle_time + ingress_egress_time_window, "egress")
    plot_and_save(middle_time - ingress_egress_time_window, middle_time, "ingress")

    # 以下は電波データにおける掩蔽タイミングを決めるものなので、閾値を買えない限りは毎回やる必要はない
    # print(ingress_time_list)

    np.savetxt(
        "../result_for_yasudaetal2022/radio_data_occultation_timing_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_flyby/"
        + object_name
        + "_"
        + boundary_average_str
        + "dB_ingress_time_data.txt",
        ingress_time_list,
    )
    np.savetxt(
        "../result_for_yasudaetal2022/radio_data_occultation_timing_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_flyby/"
        + object_name
        + "_"
        + boundary_average_str
        + "dB_egress_time_data.txt",
        egress_time_list,
    )

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
    raytrace_time = np.arange(0, time_step, 1)  # レイトレーシング計算した間隔での時間リスト（秒）

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
            a = A[len(A) - 1]  # 電波がうかっている且つ掩蔽中心時刻より早い且つその中で一番遅いものの配列番号
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
    raytrace_time = np.arange(0, time_step, 1)  # レイトレーシング計算した間隔での時間リスト（秒）

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
            b = B[0]  # 電波がうかっている且つ掩蔽中心時刻より遅い且つその中で一番早いものの配列番号
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


def Evaluate_ionosphere_density(raytrace_data, galileo_data):
    """_レイトレーシングによる掩蔽タイミングとガリレオデータにおける掩蔽タイミングの差分を計算_

    Args:
        raytrace_data (_type_): _レイトレーシングにおける掩蔽タイミング [[周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ_
        galileo_data (_type_): _ 電波観測データにおける掩蔽タイミング [[周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ_

    Returns:
        _type_: _電波データの各周波数における掩蔽タイミングの差分  [[周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ_
    """

    time_defference_index = galileo_data.copy()
    num = galileo_data.shape[1]  # 2（周波数・時間）✖️(ガリレオ周波数の種類分)の２つ目の要素　⇨ ガリレオ周波数の種類分

    # レイトレーシングの周波数種類ごとに
    for i in range(num):
        raytrace_occulatation_timing = nearest_raytrace_time(
            raytrace_data, galileo_data[0][i]
        )  # レイトレーシングの周波数とガリレオ探査機の電波データの周波数リストで一番その差が小さい部分のインデックスを取得

        time_defference = abs(
            galileo_data[1][i] - raytrace_occulatation_timing
        )  # 取得した周波数の電波データでの掩蔽開始・終了時間とレイトレーシングの時間の差を取る
        time_defference_index[1][i] = time_defference

    return time_defference_index


def Evaluate_data_coutour(time_data, radio_data_name):
    (
        galileo_data_time,
        galileo_data_freq,
        galileo_radio_intensity,
    ) = Prepare_Galileo_data(time_data, radio_data_name)
    using_galileo_data = galileo_radio_intensity[np.where(galileo_data_freq > 1e-1)][
        :
    ].flatten()

    fig, ax = plt.subplots(1, 1)
    # ax.hist(using_galileo_data, range=(1e-18, 1e-12),bins=np.logspace(-18, -12, 30))
    ax.hist(using_galileo_data, bins=np.logspace(-17, -12, 50))
    ax.set_xscale("log")
    fig.savefig("A")

    return 0


# %%


def main():
    time_information, radio_data = Pick_up_cdf()
    # print(time_information, radio_data)
    # old [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 座標変換した時のx(tangential point との水平方向の距離),6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度,8 探査機の経度]
    # new [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]

    detectable_radio = np.loadtxt(
        "../result_for_yasudaetal2022/raytracing_"
        + object_name
        + "_results/"
        + object_name
        + "_"
        + highest_plasma
        + "_"
        + plasma_scaleheight
        + "/interpolated_"
        + object_name
        + "_"
        + spacecraft_name
        + "_"
        + str(time_of_flybies)
        + "_"
        + highest_plasma
        + "_"
        + plasma_scaleheight
        + "_dectable_radio_data.txt"
    )

    detectable_A, detectable_B, detectable_C, detectable_D = Prepare_Figure(
        detectable_radio, time_information
    )

    ingress_time, egress_time = Make_FT_full(
        detectable_A,
        detectable_B,
        detectable_C,
        detectable_D,
        time_information,
        radio_data,
    )

    Evaluate_data_coutour(time_information, radio_data)

    # PLOT 用

    detectable_data = [detectable_A, detectable_B, detectable_C, detectable_D]

    detectable_data_str = ["dataA", "dataB", "dataC", "dataD"]

    for i in range(4):
        evaluated_data = Evaluate_raytrace_data(
            detectable_data[i], time_information
        )  # モデルとしての掩蔽タイミングを出力　[[周波数一覧][掩蔽時刻一覧]]

        # レイトレーシングデータにおける掩蔽開始時刻 [[周波数一覧][掩蔽開始時刻一覧]]の二次元データ　⇨保存（電子密度を変えるたびに異なる値になる・閾値は関係ない）

        np.savetxt(
            "../result_for_yasudaetal2022/raytracing_occultation_timing_"
            + spacecraft_name
            + "_"
            + object_name
            + "_"
            + str(time_of_flybies)
            + "_flyby/interpolated_"
            + spacecraft_name
            + "_"
            + object_name
            + "_"
            + str(time_of_flybies)
            + "_"
            + highest_plasma
            + "_"
            + plasma_scaleheight
            + "_ingress_"
            + detectable_data_str[i]
            + "_time_list.txt",
            evaluated_data.ingress,
        )
        # レイトレーシングデータにおける掩蔽終了時刻 [[周波数一覧][掩蔽終了時刻一覧]]の二次元データ　⇨保存（電子密度を変えるたびに異なる値になる・閾値は関係ない）
        np.savetxt(
            "../result_for_yasudaetal2022/raytracing_occultation_timing_"
            + spacecraft_name
            + "_"
            + object_name
            + "_"
            + str(time_of_flybies)
            + "_flyby/interpolated_"
            + spacecraft_name
            + "_"
            + object_name
            + "_"
            + str(time_of_flybies)
            + "_"
            + highest_plasma
            + "_"
            + plasma_scaleheight
            + "_egress_"
            + detectable_data_str[i]
            + "_time_list.txt",
            evaluated_data.egress,
        )

        time_defference_ingress = Evaluate_ionosphere_density(
            evaluated_data.ingress, ingress_time
        )  # （モデルでの掩蔽タイミング、電波データの掩蔽タイミング）⇨ 掩蔽開始タイミングの差分  [[電波データの周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ

        time_defference_egress = Evaluate_ionosphere_density(
            evaluated_data.egress, egress_time
        )  # （モデルでの掩蔽タイミング、電波データの掩蔽タイミング）⇨ 掩蔽開始タイミングの差分  [[電波データの周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ

        np.savetxt(
            "../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_"
            + spacecraft_name
            + "_"
            + object_name
            + "_"
            + str(time_of_flybies)
            + "_flyby_radioint_"
            + boundary_average_str
            + "dB/interpolated_"
            + object_name
            + "_"
            + highest_plasma
            + "_"
            + plasma_scaleheight
            + "_ingress_defference_time_"
            + detectable_data_str[i]
            + "_"
            + boundary_average_str
            + "dB.txt",
            time_defference_ingress,
        )
        np.savetxt(
            "../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_"
            + spacecraft_name
            + "_"
            + object_name
            + "_"
            + str(time_of_flybies)
            + "_flyby_radioint_"
            + boundary_average_str
            + "dB/interpolated_"
            + object_name
            + "_"
            + highest_plasma
            + "_"
            + plasma_scaleheight
            + "_egress_defference_time_"
            + detectable_data_str[i]
            + "_"
            + boundary_average_str
            + "dB.txt",
            time_defference_egress,
        )

    return 0


if __name__ == "__main__":
    main()


# %%
