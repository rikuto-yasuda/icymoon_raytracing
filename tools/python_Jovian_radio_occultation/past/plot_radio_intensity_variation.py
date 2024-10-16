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


args = sys.argv
## プロットしたいフライバイを指定
spacecraft_name = "galileo"  # galileo/JUICE(?)

object_name = "callisto"  # ganydeme/europa/callisto`
time_of_flybies = 30  # ..th flyby
plot_kinds = "egress"  # ingress/egress/noise_floor/manulal
"""
object_name = args[1]  # ganydeme/europa/callisto`
time_of_flybies = int(args[2])  # ..th flyby
"""
plot_kinds = args[1]

## 詳細設定

# plot_kindsをプロットしたい時間範囲を指定(秒)
plot_first_time = 2280  # 06:08
plot_last_time = 2700  # 06:15


# カラーマップの強度範囲（）
max_intensity = 1e-12  # カラーマップの最大強度
min_intensity = 5e-17  # カラーマップの最小強度

# 平均値を折れ線でプロットするときの範囲
averaged_max_intensity = 3e-16
averaged_min_intensity = 1e-17


# ftダイヤグラムに等高線をひく強度を指定
boundary_intensity_str = "7e-16"  # boundary_intensity_str = '1e-15'
boundary_intensity = float(boundary_intensity_str)

# プロットする周波数範囲 (MHz)
max_frequency = 6
min_frequency = 0.1

# 強度変化の比較を行う時間は幅
variablility_minutes = int(args[2])
print(variablility_minutes)
noise_floor_sigma = int(args[3])
print(noise_floor_sigma)

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


## フライバイの共通データを指定
def basic_data(object, spacecraft, flyby):
    if (object == "callisto") and (spacecraft == "galileo") and (flyby == 30):
        # プロットしたい電波データのパスを指定
        radio_data_name = (
            "Survey_Electric_2001-05-25T10-00_2001-05-25T13-00.d2s"  # C30 flyby
        )

        # 読み込んだデータの開始日時(実際の観測時刻の切り下げ値を代入)
        start_day = 25  # 電波データの開始日
        start_hour = 10  # 電波データの開始時刻
        start_min = 0  # 電波データの開始分

        # 電波データの時刻ラベルを作成（列番号と時刻の対応を示すもの）/
        plot_time_step_sec = [0, 1800, 3600, 5400, 7200, 9000, 10800]
        plot_time_step_label = [
            "10:00",
            "10:30",
            "11:00",
            "11:30",
            "12:00",
            "12:30",
            "13:00",
        ]

    if (object == "callisto") and (spacecraft == "galileo") and (flyby == 9):
        # プロットしたい電波データのパスを指定
        radio_data_name = (
            "Survey_Electric_1997-06-25T12-00_1997-06-25T15-00.d2s"  # C09 flyby
        )

        # 読み込んだデータの開始日時(実際の観測時刻の切り下げ値を代入)
        start_day = 25  # 電波データの開始日
        start_hour = 12  # 電波データの開始時刻
        start_min = 0  # 電波データの開始分

        # 電波データの時刻ラベルを作成（列番号と時刻の対応を示すもの）/
        plot_time_step_sec = [0, 1800, 3600, 5400, 7200, 9000, 10800]
        plot_time_step_label = [
            "12:00",
            "12:30",
            "13:00",
            "13:30",
            "14:00",
            "14:30",
            "15:00",
        ]

    if (object == "ganymede") and (spacecraft == "galileo") and (flyby == 1):
        # プロットしたい電波データのパスを指定
        radio_data_name = (
            "Survey_Electric_1996-06-27T05-30_1996-06-27T07-00.d2s"  # C30 flyby
        )

        # 読み込んだデータの開始日時(実際の観測時刻の切り下げ値を代入)
        start_day = 27  # 電波データの開始日
        start_hour = 5  # 電波データの開始時刻
        start_min = 30  # 電波データの開始分

        # 電波データの時刻ラベルを作成（列番号と時刻の対応を示すもの）/
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

    return (
        radio_data_name,
        start_day,
        start_hour,
        start_min,
        plot_time_step_sec,
        plot_time_step_label,
    )


def time_range_data(object, spacecraft, flyby, timing):
    if (object == "callisto") and (spacecraft == "galileo") and (flyby == 30):
        if timing == "ingress":
            plot_first_time = 4200  # 11:10
            plot_last_time = 6000  # 11:40

        elif timing == "egress":
            plot_first_time = 6000  # 11:40
            plot_last_time = 7800  # 12:10

        elif timing == "noise_floor":
            plot_first_time = 5400  # 11:30
            plot_last_time = 6000  # 11:40

        elif timing == "full":
            plot_first_time = 0  # 10:00
            plot_last_time = 10800  # 10800

    if (object == "callisto") and (spacecraft == "galileo") and (flyby == 9):
        if timing == "ingress":
            plot_first_time = 0
            plot_last_time = 0

        elif timing == "egress":
            plot_first_time = 5400  # 13:30
            plot_last_time = 7200  # 14:00

        elif timing == "noise_floor":
            plot_first_time = 5700  # 13:35
            plot_last_time = 6000  # 13:40

        elif timing == "full":
            plot_first_time = 0  # 12:00
            plot_last_time = 10800  # 15:00

    if (object == "ganymede") and (spacecraft == "galileo") and (flyby == 1):
        if timing == "ingress":
            plot_first_time = 900  # 05:45
            plot_last_time = 2700  # 06:15

        elif timing == "egress":
            plot_first_time = 2400  # 06:10
            plot_last_time = 3600  # 07:30

        elif timing == "noise_floor":
            # plot_first_time = 2100  # 06:05
            plot_first_time = 2280  # 06:08
            plot_last_time = 2700  # 06:15

        elif timing == "full":
            plot_first_time = 0  # 05:30
            plot_last_time = 5400  # 07:00

    return plot_first_time, plot_last_time


# 電波強度のデータを取得（一列目は時刻データになってる）
# 初めの数行は読み取らないよう設定・時刻データを読み取って時刻をプロットするためここがずれても影響はないが、データがない行を読むと怒られるのでその時はd2sファイルを確認


"""ここから下は基本いじらない"""


def Prepare_Galileo_data(rad_row_data):
    """_探査機による電波データのファイル名から電波データの時刻(電波データから読み取れる時刻とcsvファイルの時刻の差・周波数(ソースははじめ電波リストから)・電波強度を出力する_

    Args:
        data_name (_str_): _用いる電波データのファイル名を入力_

    Returns:
        _type_: _電波データの時刻の配列・周波数の配列・電波強度の配列_
    """

    (
        radio_data_name,
        start_day,
        start_hour,
        start_min,
        plot_time_step_sec,
        plot_time_step_label,
    ) = basic_data(object_name, spacecraft_name, time_of_flybies)

    # 電波データの周波数の単位をHzからMHzに変換する
    gal_fleq_tag = np.array(gal_fleq_tag_row, dtype="float64") / 1000000

    # 一列目の時刻データを文字列で取得（例; :10:1996-06-27T05:30:08.695） ・同じ長さの０配列を準備・
    gal_time_tag_prepare = np.array(rad_row_data.iloc[:, 0])
    gal_time_tag_prepare = gal_time_tag_prepare.astype(np.str_)
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
    start_time = start_day * 86400 + start_hour * 3600 + start_min * 60
    gal_time_tag = np.array(gal_time_tag - start_time)
    df = pd.DataFrame(rad_row_data.iloc[:, 1:])

    DDF = np.array(df).astype(np.float64).T
    # print(DDF)
    # print(len(gal_fleq_tag), len(gal_time_tag), DDF.shape)

    # ガリレオ探査機のデータ取得開始時刻からの経過時間（sec) , ガリレオ探査機のデータ取得周波数（MHz), ガリレオ探査機の取得した電波強度（代入したデータと同じ単位）
    return gal_time_tag, gal_fleq_tag, DDF


def output_noisefoor(frequency_channel):
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

    mean = noise_data[1][frequency_channel]
    sigma = noise_data[2][frequency_channel]
    median = noise_data[3][frequency_channel]

    return mean, sigma, median


def Make_FT_full():
    (
        radio_data_name,
        start_day,
        start_hour,
        start_min,
        plot_time_step_sec,
        plot_time_step_label,
    ) = basic_data(object_name, spacecraft_name, time_of_flybies)

    plot_first_time, plot_last_time = time_range_data(
        object_name, spacecraft_name, time_of_flybies, plot_kinds
    )

    radio_row_data = pd.read_csv(
        "../result_for_yasudaetal2022/"
        + spacecraft_name
        + "_radio_data/"
        + radio_data_name,
        header=None,
        skiprows=24,
        delimiter="  ",
        engine="python",
    )

    # ガリレオ探査機のデータ取得開始時刻からの経過時間（sec) , ガリレオ探査機のデータ取得周波数（MHz), ガリレオ探査機の取得した電波強度（代入したデータと同じ単位）
    (
        galileo_data_time,
        galileo_data_freq,
        galileo_radio_intensity,
    ) = Prepare_Galileo_data(radio_row_data)

    galileo_radio_intensity_row = galileo_radio_intensity.copy()

    def plot_average_intensity_vs_time_and_save():
        ingress_timing_array = np.zeros(len(galileo_data_freq))
        egress_timing_array = np.zeros(len(galileo_data_freq))

        # 各々の周波数での掩蔽タイミングを計算するfor文
        # 計算結果はingress_timing_array と egress_timing_arrayに格納される

        for j in range(len(galileo_data_freq)):
            ######################## 周波数、電波強度・時間配列（電波強度０を抜いた）、ノイズフロア平均＋nσ #######################
            usable_data = galileo_radio_intensity_row[j, :]
            selected_usable_data = usable_data[usable_data > 0]
            selected_galileo_data_time = galileo_data_time[usable_data > 0]

            noise_floor = (
                output_noisefoor(j)[0] + noise_floor_sigma * output_noisefoor(j)[1]
            )
            ####################### 電波強度変化配列・時間配列 #######################
            galileo_intermediate_time = (
                selected_galileo_data_time[:-1] + selected_galileo_data_time[1:]
            ) / 2
            intensity_ratio_array = np.zeros(len(galileo_intermediate_time))

            for i in range(len(galileo_intermediate_time)):
                max_sec = galileo_intermediate_time[i] + variablility_minutes * 60
                min_sec = galileo_intermediate_time[i] - variablility_minutes * 60
                min_position = np.where(selected_galileo_data_time > min_sec)[0][0]
                max_position = np.where(selected_galileo_data_time < max_sec)[0][-1]

                if i + 1 == max_position or min_position == i + 1:
                    intensity_ratio = 0

                else:
                    before_data = selected_usable_data[min_position : i + 1]
                    before_average_intensity = np.average(before_data)

                    after_data = selected_usable_data[i + 1 : max_position]
                    after_average_intensity = np.average(after_data)
                    intensity_ratio = 10 * np.log10(
                        after_average_intensity / before_average_intensity
                    )

                intensity_ratio_array[i] = intensity_ratio

            ####################### 掩蔽開始時における減少量最大になる位置#######################
            # 掩蔽開始時間帯の調査開始時刻・終了時刻を出力
            ingress_begin_time, ingress_end_time = time_range_data(
                object_name, spacecraft_name, time_of_flybies, "ingress"
            )

            # 掩蔽開始時間帯に該当するデータ位置を出力（galileo_intermediate_timeにおける）
            ingress_range_position_in_intermediate_time = np.where(
                (galileo_intermediate_time > ingress_begin_time)
                & (galileo_intermediate_time < ingress_end_time)
            )[0]

            # 掩蔽開始時間帯に該当するデータ位置を出力（selected_galileo_data_timeにおける）
            ingress_range_position_in_selected_galileo_time = np.where(
                (selected_galileo_data_time > ingress_begin_time)
                & (selected_galileo_data_time < ingress_end_time)
            )[0]

            # 掩蔽開始時間帯に該当するデータの電波強度一覧
            selected_usable_ingress_data = selected_usable_data[
                ingress_range_position_in_selected_galileo_time
            ]

            if len(np.where(selected_usable_ingress_data < noise_floor)[0]) > 0:
                # 掩蔽開始時間帯で最初にノイズフロアを下回ったデータの位置（selected_galileo_data_timeにおける）
                ingress_first_noisefloor_position = (
                    ingress_range_position_in_selected_galileo_time[
                        np.where(selected_usable_ingress_data < noise_floor)[0][0]
                    ]
                )
            else:
                ingress_first_noisefloor_position = 0

            # 掩蔽開始時間帯のノイズフロア以外のデータの位置（galileo_intermediate_timeにおける）
            selected_ingress_position_range = (
                ingress_range_position_in_intermediate_time[
                    ingress_range_position_in_intermediate_time
                    < ingress_first_noisefloor_position
                ]
            )

            if len(intensity_ratio_array[selected_ingress_position_range]) == 0:
                min_index = 0

            else:
                # 掩蔽開始時間帯のノイズフロア以外のデータの位置で減少量が最大の位置（galileo_intermediate_timeにおける）
                min_index = selected_ingress_position_range[
                    np.argmin(intensity_ratio_array[selected_ingress_position_range])
                ]

            # 掩蔽開始時間帯のノイズフロア以外のデータの位置で減少量が最大の時間（秒）
            ingress_timing = galileo_intermediate_time[min_index]

            ingress_timing_array[j] = ingress_timing

            ####################### 掩蔽終了時における増加量最大になる位置#######################
            # 掩蔽終了時間帯の調査開始時刻・終了時刻を出力
            egress_begin_time, egress_end_time = time_range_data(
                object_name, spacecraft_name, time_of_flybies, "egress"
            )

            # 掩蔽終了時間帯に該当するデータ位置を出力（galileo_intermediate_timeにおける）
            egress_range_position_in_intermediate_time = np.where(
                (galileo_intermediate_time > egress_begin_time)
                & (galileo_intermediate_time < egress_end_time)
            )[0]

            # 掩蔽終了時間帯に該当するデータ位置を出力（selected_galileo_data_timeにおける）
            egress_range_position_in_selected_galileo_time = np.where(
                (selected_galileo_data_time > egress_begin_time)
                & (selected_galileo_data_time < egress_end_time)
            )[0]

            # 掩蔽終了時間帯に該当するデータの電波強度一覧
            selected_usable_egress_data = selected_usable_data[
                egress_range_position_in_selected_galileo_time
            ]

            # 掩蔽終了時間帯で最後にノイズフロアを下回ったデータの位置（selected_galileo_data_timeにおける）
            egress_last_noisefloor_position = (
                egress_range_position_in_selected_galileo_time[
                    np.where(selected_usable_egress_data < noise_floor)[0][-1]
                ]
            )

            # 掩蔽開始時間帯のノイズフロア以外のデータの位置（galileo_intermediate_timeにおける）
            selected_egress_position_range = egress_range_position_in_intermediate_time[
                egress_range_position_in_intermediate_time
                >= egress_last_noisefloor_position
            ]

            if len(intensity_ratio_array[selected_egress_position_range]) == 0:
                max_index = 0

            else:
                # 掩蔽開始時間帯のノイズフロア以外のデータの位置で減少量が最大の位置（galileo_intermediate_timeにおける）
                max_index = selected_egress_position_range[
                    np.argmax(intensity_ratio_array[selected_egress_position_range])
                ]

            # 掩蔽開始時間帯のノイズフロア以外のデータの位置で減少量が最大の時間（秒）
            egress_timing = galileo_intermediate_time[max_index]
            egress_timing_array[j] = egress_timing

        # 計算結果を保存する
        # [ガリレオの周波数][掩蔽開始時刻(基準時刻からの秒数)][掩蔽開始時刻(記順時刻からの秒数)]
        np.savetxt(
            "../result_for_yasudaetal2022/radio_plot/"
            + object_name
            + str(time_of_flybies)
            + "/intensity_change/"
            + plot_kinds
            + "/"
            + str(noise_floor_sigma)
            + "sigma_"
            + str(variablility_minutes)
            + "min/occultation_timing.txt",
            np.array([galileo_data_freq, ingress_timing_array, egress_timing_array]),
        )

        xx, yy = np.meshgrid(galileo_data_time, galileo_data_freq)
        pcm = plt.pcolor(
            xx,
            yy,
            galileo_radio_intensity_row,
            norm=mpl.colors.LogNorm(vmin=min_intensity, vmax=max_intensity),
            cmap="Spectral_r",
        )
        plt.colorbar(pcm, extend="max", orientation="horizontal", pad=0.15)
        plt.yscale("log")
        plt.ylim(min_frequency, max_frequency)
        plt.ylabel("Frequency (MHz)")

        # raytrace_time_information ['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']
        # ax.set_xlabel("Time of 27 June 1996")
        # 日時はcsvファイルの情報から記入される
        plt.xlabel("time")

        # 論文中で使われている横軸の幅とそれに対応する計算開始時刻からの秒数はグローバル変数で指定しておく
        plt.xticks(plot_time_step_sec, plot_time_step_label)

        # 横軸の幅は作りたい図によって変わるので引数用いる
        plt.xlim(plot_first_time, plot_last_time)

        plt.title(
            "intensity variability in "
            + str(variablility_minutes)
            + "min & noise floor "
            + str(noise_floor_sigma)
            + " sigma"
        )

        plt.plot(ingress_timing_array, galileo_data_freq, c="red")
        plt.plot(egress_timing_array, galileo_data_freq, c="red")

        plt.savefig(
            os.path.join(
                "../result_for_yasudaetal2022/radio_plot/"
                + object_name
                + str(time_of_flybies)
                + "/intensity_change/"
                + plot_kinds
                + "/"
                + str(noise_floor_sigma)
                + "sigma_"
                + str(variablility_minutes)
                + "min/occultation_timing.png",
            ),
            dpi=900,
        )

        selsected_frecuency_first_position = np.where(
            min_frequency < galileo_data_freq
        )[0][0]
        selsected_frecuency_last_position = np.where(galileo_data_freq < max_frequency)[
            0
        ][-1]

        print(selsected_frecuency_first_position, selsected_frecuency_last_position)

        for k in range(
            selsected_frecuency_first_position, selsected_frecuency_last_position
        ):
            print(galileo_data_freq[k])
            fig, ax = plt.subplots(3, 1, figsize=(10, 15))

            ########### 図0 freqency-time plot　################

            xx, yy = np.meshgrid(galileo_data_time, galileo_data_freq)

            # ガリレオ探査機の電波強度をカラーマップへ
            pcm = ax[0].pcolor(
                xx,
                yy,
                galileo_radio_intensity_row,
                norm=mpl.colors.LogNorm(vmin=min_intensity, vmax=max_intensity),
                cmap="Spectral_r",
            )
            fig.colorbar(pcm, extend="max", orientation="horizontal", pad=0.15)
            ax[0].set_yscale("log")
            ax[0].set_ylim(min_frequency, max_frequency)
            ax[0].set_ylabel("Frequency (MHz)")

            # raytrace_time_information ['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']
            # ax.set_xlabel("Time of 27 June 1996")
            # 日時はcsvファイルの情報から記入される
            ax[0].set_xlabel("time")

            # 論文中で使われている横軸の幅とそれに対応する計算開始時刻からの秒数はグローバル変数で指定しておく
            ax[0].set_xticks(plot_time_step_sec)
            ax[0].set_xticklabels(plot_time_step_label)

            # 横軸の幅は作りたい図によって変わるので引数用いる
            ax[0].set_xlim(plot_first_time, plot_last_time)

            ax[0].set_title(
                "intensity variability in "
                + str(variablility_minutes)
                + "min & noise floor "
                + str(noise_floor_sigma)
                + " sigma"
            )

            ax[0].axhline(
                y=galileo_data_freq[k] * (20 / 19),
                color="black",
                linestyle="dotted",
            )
            ax[0].axhline(
                y=galileo_data_freq[k] * (19 / 20),
                color="black",
                linestyle="dotted",
            )

            # 掩蔽タイミングをプロットする
            ax[0].plot(ingress_timing_array, galileo_data_freq, c="red")
            ax[0].plot(egress_timing_array, galileo_data_freq, c="red")

            ######################## 周波数、電波強度・時間配列（電波強度０を抜いた）、ノイズフロア平均＋nσ #######################
            usable_data = galileo_radio_intensity_row[k, :]
            selected_usable_data = usable_data[usable_data > 0]
            selected_galileo_data_time = galileo_data_time[usable_data > 0]

            noise_floor = (
                output_noisefoor(k)[0] + noise_floor_sigma * output_noisefoor(k)[1]
            )
            ####################### 電波強度変化配列・時間配列 #######################
            galileo_intermediate_time = (
                selected_galileo_data_time[:-1] + selected_galileo_data_time[1:]
            ) / 2
            intensity_ratio_array = np.zeros(len(galileo_intermediate_time))

            for i in range(len(galileo_intermediate_time)):
                max_sec = galileo_intermediate_time[i] + variablility_minutes * 60
                min_sec = galileo_intermediate_time[i] - variablility_minutes * 60
                min_position = np.where(selected_galileo_data_time > min_sec)[0][0]
                max_position = np.where(selected_galileo_data_time < max_sec)[0][-1]

                if i + 1 == max_position or min_position == i + 1:
                    intensity_ratio = 0

                else:
                    before_data = selected_usable_data[min_position : i + 1]
                    before_average_intensity = np.average(before_data)

                    after_data = selected_usable_data[i + 1 : max_position]
                    after_average_intensity = np.average(after_data)
                    intensity_ratio = 10 * np.log10(
                        after_average_intensity / before_average_intensity
                    )

                intensity_ratio_array[i] = intensity_ratio

            # 横軸時間・縦軸電波強度のプロットを作成
            # ノイズフロアの強度を付与
            ax[1].plot(selected_galileo_data_time, selected_usable_data)
            ax[1].set_xticks(plot_time_step_sec)
            ax[1].set_xticklabels(plot_time_step_label)
            ax[1].set_xlim(plot_first_time, plot_last_time)
            ax[1].set_yscale("log")
            ax[1].set_xlabel("time")
            ax[1].set_ylabel("intensity (V2/m2/Hz)")
            ax[1].set_title(
                object_name
                + str(time_of_flybies)
                + "_"
                + str(galileo_data_freq[k])
                + "MHz_"
                + str(plot_kinds)
            )
            ax[1].axhline(y=noise_floor, color="red", label="noise floor (5 sigma)")
            #### 掩蔽開始時間帯・掩蔽終了時間帯の変動量のみを調査＋掩蔽前・後にノイズフロアを下回らない時刻を選択

            ax[1].scatter(
                ingress_timing_array[k],
                np.interp(
                    ingress_timing_array[k],
                    selected_galileo_data_time,
                    selected_usable_data,
                ),
                marker="x",
                s=50,
                label="dec max " + str(variablility_minutes) + "min",
            )
            ax[1].scatter(
                egress_timing_array[k],
                np.interp(
                    egress_timing_array[k],
                    selected_galileo_data_time,
                    selected_usable_data,
                ),
                marker="x",
                s=50,
                label="inc max " + str(variablility_minutes) + "min",
            )

            ax[1].legend()

            ########### 図2　電波強度変化　################
            # 横軸時間・縦軸電波強度の変化率のプロットを作成
            # galileoの時間データの中間値でできた新たな時間配列を生成（前後）

            ax[2].plot(
                galileo_intermediate_time,
                intensity_ratio_array,
                label=str(variablility_minutes) + " min after ave/ before ave ",
            )
            ax[2].set_xticks(plot_time_step_sec)
            ax[2].set_xticklabels(plot_time_step_label)
            ax[2].set_xlim(plot_first_time, plot_last_time)
            # ax.set_ylim(min_intensity, max_intensity)
            ax[2].set_xlabel("time")
            ax[2].set_ylabel("intensity_change (dB)")
            ax[2].set_title("intensity change")
            ax[2].legend()

            # plt.show()

            fig.savefig(
                os.path.join(
                    "../result_for_yasudaetal2022/radio_plot/"
                    + object_name
                    + str(time_of_flybies)
                    + "/intensity_change/"
                    + plot_kinds
                    + "/"
                    + str(noise_floor_sigma)
                    + "sigma_"
                    + str(variablility_minutes)
                    + "min/freq_"
                    + str(galileo_data_freq[k])
                    + ".png"
                )
            )

    plot_average_intensity_vs_time_and_save()

    return 0


def main():
    Make_FT_full()

    return 0


if __name__ == "__main__":
    main()


# %%
