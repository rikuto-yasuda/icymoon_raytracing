# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
import os
import time
import glob
import requests
import sys
import math
from scipy.interpolate import interp1d

# %%
# あらかじめ ../result_sgepss2021/~/~ に必要なレイトレーシング結果とパラメータセットを入れること

object_name = "callisto"  # ganydeme/europa/calisto``
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 30  # ..th flyby
occultaion_type = "ingress"  # 'ingress' or 'egress
radio_type_A2D = "C"  # 'A' or 'B' or 'C' or 'D'

Radio_observer_position = np.loadtxt(
    "/Users/yasudarikuto/research/icymoon_raytracing/tools/result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/interpolated_calculated_all_"
    + spacecraft_name
    + "_"
    + object_name
    + "_"
    + str(time_of_flybies)
    + "_Radio_data.txt"
)  # 電波源の経度を含む


# europa & ganymede
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

    uncertatinty = np.array([1.7, 1.4, 1.0, 0.6, 0.4])

    if occultaion_type == "ingress" and time_of_flybies == 1:
        using_frequency_range = [8.0e-1, 4.5]  # G1 ingress

    if occultaion_type == "egress" and time_of_flybies == 1:
        using_frequency_range = [6.5e-1, 4.5]  # G1 egress

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

    uncertatinty = np.array([1.0, 0.8, 0.6, 0.3, 0.2])

    if occultaion_type == "egress" and time_of_flybies == 9:
        using_frequency_range = [6.5e-1, 5.0]  # C9 egres

    if occultaion_type == "ingress" and time_of_flybies == 30:
        using_frequency_range = [8.0e-1, 4.5]

Freq_num = []
for idx in Freq_str:
    Freq_num.append(float(idx) / 1000000)

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
gal_fleq_tag = np.array(gal_fleq_tag_row, dtype="float64") / 1000000

# %%

Frequency = np.array([0.7, 1.0, 2.0, 5.0, 10.0])
Frequency_log = np.log(Frequency)
int_xarray = Freq_num
int_xarray_log = np.log(int_xarray)
f1 = interp1d(Frequency_log, uncertatinty, kind="linear", fill_value="extrapolate")
int_yarray = f1(int_xarray_log)
"""
plt.scatter(Frequency, uncertatinty)
plt.scatter(int_xarray, int_yarray)
# plt.xscale("log")
plt.show()
"""


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

    # csvの時刻データと電波データの名前をかえす
    return time_information


def Judge_occultation_lower(i):
    # Radio_observation_position
    # new [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]
    aa = 0
    detectable_obsever_position_x = Radio_observer_position[i][6]
    detectable_obsever_position_z = Radio_observer_position[i][7]
    detectable_frequency = Radio_observer_position[i][3]  # 使うレイの周波数を取得
    # レイの周波数と周波数リスト（Freq＿num）の値が一致する場所を取得　周波数リスト（Freq＿num）とcsvファイルの週数リストが一致しているのでそこからその周波数における電波源の幅を取得

    Fre = np.where(Freq_num == detectable_frequency)
    lowest_deg = -1 * int_yarray[Fre] / 2

    tangent = detectable_obsever_position_z / detectable_obsever_position_x

    # タンジェントから角度に変換
    degree_value = math.degrees(math.atan(tangent))
    if detectable_obsever_position_x > 0:
        if degree_value > lowest_deg:
            aa = 1

    # 条件を満たす場合 aa=1 満たさない場合 aa=0
    return aa


def Judge_occultation_higher(i):
    # Radio_observation_position
    # new [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]
    aa = 0
    detectable_obsever_position_x = Radio_observer_position[i][6]
    detectable_obsever_position_z = Radio_observer_position[i][7]
    detectable_frequency = Radio_observer_position[i][3]  # 使うレイの周波数を取得
    # レイの周波数と周波数リスト（Freq＿num）の値が一致する場所を取得　周波数リスト（Freq＿num）とcsvファイルの週数リストが一致しているのでそこからその周波数における電波源の幅を取得

    Fre = np.where(Freq_num == detectable_frequency)
    highest_deg = int_yarray[Fre] / 2

    tangent = detectable_obsever_position_z / detectable_obsever_position_x

    # タンジェントから角度に変換
    degree_value = math.degrees(math.atan(tangent))
    if detectable_obsever_position_x > 0:
        if degree_value > highest_deg:
            aa = 1

    # 条件を満たす場合 aa=1 満たさない場合 aa=0
    return aa


def Remain_error_range(
    judgement,
    befrore_judged_array,
):
    # 想定した電子密度分布で観測可能な電波のリスト
    # [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]
    # を想定した電子密度でのレイトレーシング結果が保存されているフォルダに保存
    occultaion_aray = np.array(judgement)
    judge_array = befrore_judged_array[np.where(occultaion_aray[:] == 1)[0]]

    return judge_array


def Calculate_error_time(lower_judgement, higher_judgement, time_information):
    # lower_judgement and higher_judgement [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]
    # time_information['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']
    Center_sec = (time_information[9] * 60 * 60 + time_information[10] * 60) - (
        time_information[4] * 60 * 60 + time_information[6] * 60
    )

    if occultaion_type == "ingress":
        error_first = np.full(len(Freq_num), -1 * (10**10))
        error_last = np.full(len(Freq_num), -1 * (10**10))
        for k in range(len(lower_judgement)):
            Sec = (
                lower_judgement[k][0] * 60 * 60
                + lower_judgement[k][1] * 60
                + lower_judgement[k][2]
            ) - (time_information[4] * 60 * 60 + time_information[6] * 60)
            Fre = np.where(Freq_num == lower_judgement[k][3])

            if Sec > Center_sec:
                continue
            else:
                if (
                    radio_type_A2D == "A"
                    and lower_judgement[k][10] == 1
                    and lower_judgement[k][5] == 1
                    and error_last[Fre] < Sec
                ):
                    error_last[Fre] = Sec

                if (
                    radio_type_A2D == "B"
                    and lower_judgement[k][10] == -1
                    and lower_judgement[k][5] == 1
                    and error_last[Fre] < Sec
                ):
                    error_last[Fre] = Sec

                if (
                    radio_type_A2D == "C"
                    and lower_judgement[k][10] == 1
                    and lower_judgement[k][5] == -1
                    and error_last[Fre] < Sec
                ):
                    error_last[Fre] = Sec

                if (
                    radio_type_A2D == "D"
                    and lower_judgement[k][10] == -1
                    and lower_judgement[k][5] == -1
                    and error_last[Fre] < Sec
                ):
                    error_last[Fre] = Sec

        for k in range(len(higher_judgement)):
            Sec = (
                higher_judgement[k][0] * 60 * 60
                + higher_judgement[k][1] * 60
                + higher_judgement[k][2]
            ) - (time_information[4] * 60 * 60 + time_information[6] * 60)
            Fre = np.where(Freq_num == higher_judgement[k][3])
            if Sec > Center_sec:
                continue
            else:
                if (
                    radio_type_A2D == "A"
                    and higher_judgement[k][10] == 1
                    and higher_judgement[k][5] == 1
                    and error_first[Fre] < Sec
                ):
                    error_first[Fre] = Sec

                if (
                    radio_type_A2D == "B"
                    and higher_judgement[k][10] == -1
                    and higher_judgement[k][5] == 1
                    and error_first[Fre] < Sec
                ):
                    error_first[Fre] = Sec

                if (
                    radio_type_A2D == "C"
                    and higher_judgement[k][10] == 1
                    and higher_judgement[k][5] == -1
                    and error_first[Fre] < Sec
                ):
                    error_first[Fre] = Sec

                if (
                    radio_type_A2D == "D"
                    and higher_judgement[k][10] == -1
                    and higher_judgement[k][5] == -1
                    and error_first[Fre] < Sec
                ):
                    error_first[Fre] = Sec

    if occultaion_type == "egress":
        error_first = np.full(len(Freq_num), 10**10)
        error_last = np.full(len(Freq_num), 10**10)
        for k in range(len(lower_judgement)):
            Sec = (
                lower_judgement[k][0] * 60 * 60
                + lower_judgement[k][1] * 60
                + lower_judgement[k][2]
            ) - (time_information[4] * 60 * 60 + time_information[6] * 60)
            Fre = np.where(Freq_num == lower_judgement[k][3])
            if Sec < Center_sec:
                continue
            else:
                if (
                    radio_type_A2D == "A"
                    and lower_judgement[k][10] == 1
                    and lower_judgement[k][5] == 1
                    and error_first[Fre] > Sec
                ):
                    error_first[Fre] = Sec

                if (
                    radio_type_A2D == "B"
                    and lower_judgement[k][10] == -1
                    and lower_judgement[k][5] == 1
                    and error_first[Fre] > Sec
                ):
                    error_first[Fre] = Sec

                if (
                    radio_type_A2D == "C"
                    and lower_judgement[k][10] == 1
                    and lower_judgement[k][5] == -1
                    and error_first[Fre] > Sec
                ):
                    error_first[Fre] = Sec

                if (
                    radio_type_A2D == "D"
                    and lower_judgement[k][10] == -1
                    and lower_judgement[k][5] == -1
                    and error_first[Fre] > Sec
                ):
                    error_first[Fre] = Sec

        for k in range(len(higher_judgement)):
            Sec = (
                higher_judgement[k][0] * 60 * 60
                + higher_judgement[k][1] * 60
                + higher_judgement[k][2]
            ) - (time_information[4] * 60 * 60 + time_information[6] * 60)
            Fre = np.where(Freq_num == higher_judgement[k][3])
            if Sec < Center_sec:
                continue
            else:
                if (
                    radio_type_A2D == "A"
                    and higher_judgement[k][10] == 1
                    and higher_judgement[k][5] == 1
                    and error_last[Fre] > Sec
                ):
                    error_last[Fre] = Sec

                if (
                    radio_type_A2D == "B"
                    and higher_judgement[k][10] == -1
                    and higher_judgement[k][5] == 1
                    and error_last[Fre] > Sec
                ):
                    error_last[Fre] = Sec

                if (
                    radio_type_A2D == "C"
                    and higher_judgement[k][10] == 1
                    and higher_judgement[k][5] == -1
                    and error_last[Fre] > Sec
                ):
                    error_last[Fre] = Sec

                if (
                    radio_type_A2D == "D"
                    and higher_judgement[k][10] == -1
                    and higher_judgement[k][5] == -1
                    and error_last[Fre] > Sec
                ):
                    error_last[Fre] = Sec

    error_dif = error_last - error_first

    express_fre_log = np.log(int_xarray)
    GLL_fre = np.array(gal_fleq_tag)
    GLL_fre_log = np.log(GLL_fre)

    f2 = interp1d(express_fre_log, error_dif, kind="linear", fill_value="extrapolate")
    GLL_error_dif = f2(GLL_fre_log)
    plt.scatter(Freq_num, error_dif, label="ExPRES")
    plt.scatter(GLL_fre, GLL_error_dif, label="GLL")
    plt.xlim(using_frequency_range[0] * 0.95, using_frequency_range[1] * 1.05)
    plt.ylim(0, 400)
    plt.legend()
    plt.show()

    error_by_express = GLL_error_dif / (2 * np.sqrt(3))  # np.array() ~ [GLL freq ch]
    error_by_time_step = 18.67 / (2 * np.sqrt(3))  # float

    using_freq_position = np.intersect1d(
        np.where(GLL_fre > np.array(using_frequency_range)[0])[0],
        np.where(GLL_fre < np.array(using_frequency_range)[1])[0],
    )

    total_error = np.average(
        np.sqrt(
            (error_by_express * error_by_express)
            + (error_by_time_step * error_by_time_step)
        )[using_freq_position]
    )

    print("0.8 MHz dt:" + str(f2(np.log(0.8))))
    print("1 MHz dt:" + str(f2(np.log(1))))
    print("2 MHz dt:" + str(f2(np.log(2))))
    print("4 MHz dt:" + str(f2(np.log(4))))
    print("time step dt:" + str(18.67))

    print("0.8 MHz u:" + str(f2(np.log(0.8)) / (2 * np.sqrt(3))))
    print("1 MHz u:" + str(f2(np.log(1)) / (2 * np.sqrt(3))))
    print("2 MHz u:" + str(f2(np.log(2)) / (2 * np.sqrt(3))))
    print("4 MHz u:" + str(f2(np.log(4)) / (2 * np.sqrt(3))))
    print("time step u:" + str(18.67 / (2 * np.sqrt(3))))

    print("total:" + str(total_error))
    return 0


def main():
    time_information = Pick_up_cdf()
    total_radio_number = list(np.arange(0, len(Radio_observer_position)))

    with Pool(processes=20) as pool:
        lower_detectable_list = list(
            pool.map(Judge_occultation_lower, total_radio_number)
        )

    with Pool(processes=20) as pool:
        higher_detectable_list = list(
            pool.map(Judge_occultation_higher, total_radio_number)
        )

    # 誤差範囲の電波のみを保存
    lower_detectable_data = Remain_error_range(
        lower_detectable_list,
        Radio_observer_position,
    )

    # 誤差範囲の電波のみを保存
    higher_detectable_data = Remain_error_range(
        higher_detectable_list,
        Radio_observer_position,
    )
    Calculate_error_time(
        lower_detectable_data, higher_detectable_data, time_information
    )

    return 0


if __name__ == "__main__":
    main()

# %%
