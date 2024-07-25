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

args = sys.argv
## プロットしたいフライバイを指定
spacecraft_name = "galileo"  # galileo/JUICE(?)
"""
object_name = args[1]  # ganydeme/europa/calisto
time_of_flybies = int(args[2])  # ..th flyby
plot_timing = args[3]  # ingress/egress/full/occultation/manual
"""
object_name = "europa"  # ganydeme/europa/calisto`
time_of_flybies = 12  # ..th flyby
plot_timing = "egress"  # ingress/egress/full/occultation/manual


## 詳細設定
# 自由にプロットしたい時間範囲を指定する場合には plot_timing = "manual"にして以下で開始・終了時間（秒）を指定


# カラーマップの強度範囲（）
max_intensity = 1e-12  # カラーマップの最大強度
min_intensity = 1e-16  # カラーマップの最小強度


# ftダイヤグラムに等高線をひく強度を指定
# 電波データから掩蔽・非掩蔽を判定する際に全周波数チャンネルに対して単一の電波強度を閾値とする場合には boundary="V^2/m2/Hz"
# 周波数ごとに観測装置のノイズの大きさ（平均値）とばらつき（標準偏差）を調べ、平均値＋（n x 標準偏差）を閾値とする場合には boundary="sigma"
boundary = "average"  # "V^2/m2/Hz" or "sigma" or "average"
boundary_intensity_str = (
    "7e-16"  # (V^2/m2/Hzの場合)boundary_intensity_str = '1e-15'[V^2/m2/Hz]
)
boundary_sigma_str = "20"  # (sigmaの場合)boundary_sigma_str = '1'[sigma]
boundary_average_str = "10"  # (average場合)boundary_intensity_str = '1'[sigma]

print(object_name + str(time_of_flybies) + plot_timing + boundary_sigma_str)

# プロットする周波数範囲 (MHz)
max_frequency = 6
min_frequency = 0.1


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

if (
    (object_name == "callisto")
    and (spacecraft_name == "galileo")
    and (time_of_flybies == 30)
):
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

    if plot_timing == "ingress":
        plot_first_time = 4800
        plot_last_time = 6000

    elif plot_timing == "egress":
        plot_first_time = 6000
        plot_last_time = 7800

    elif plot_timing == "full":
        plot_first_time = 3600
        plot_last_time = 9000

    elif plot_timing == "occultation":
        plot_first_time = 5400
        plot_last_time = 6000


if (
    (object_name == "callisto")
    and (spacecraft_name == "galileo")
    and (time_of_flybies == 9)
):
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

    if plot_timing == "ingress":
        plot_first_time = 0
        plot_last_time = 6000

    elif plot_timing == "egress":
        plot_first_time = 6000
        plot_last_time = 7200

    elif plot_timing == "full":
        plot_first_time = 0
        plot_last_time = 10800

    elif plot_timing == "occultation":
        plot_first_time = 5700
        plot_last_time = 6000


if (
    (object_name == "ganymede")
    and (spacecraft_name == "galileo")
    and (time_of_flybies == 1)
):
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

    if plot_timing == "ingress":
        plot_first_time = 900
        plot_last_time = 2400

    elif plot_timing == "egress":
        plot_first_time = 2400
        plot_last_time = 4200

    elif plot_timing == "full":
        plot_first_time = 0
        plot_last_time = 5400

    elif plot_timing == "occultation":
        plot_first_time = 2280
        plot_last_time = 2700

if (
    (object_name == "europa")
    and (spacecraft_name == "galileo")
    and (time_of_flybies == 12)
):
    # プロットしたい電波データのパスを指定
    radio_data_name = (
        "Survey_Electric_1997-12-16T10-00_1997-12-16T13-00.d2s"  # C30 flyby
    )

    # 読み込んだデータの開始日時(実際の観測時刻の切り下げ値を代入)
    start_day = 16  # 電波データの開始日
    start_hour = 10  # 電波データの開始時刻
    start_min = 0  # 電波データの開始分

    # 電波データの時刻ラベルを作成（列番号と時刻の対応を示すもの）/
    plot_time_step_sec = [6300, 6600, 6900, 7200, 7500, 7800, 8100, 8400, 8700]
    plot_time_step_label = [
        "11:45",
        "11:50",
        "11:55",
        "12:00",
        "12:05",
        "12:10",
        "12;15",
        "12:20",
        "12;25",
    ]

    if plot_timing == "ingress":
        plot_first_time = 6600
        plot_last_time = 7500

    elif plot_timing == "egress":
        plot_first_time = 7500
        plot_last_time = 8400

    elif plot_timing == "full":
        plot_first_time = 6300
        plot_last_time = 8700

    elif plot_timing == "occultation":
        plot_first_time = 7440
        plot_last_time = 7680

# 電波強度のデータを取得（一列目は時刻データになってる）
# 初めの数行は読み取らないよう設定・時刻データを読み取って時刻をプロットするためここがずれても影響はないが、データがない行を読むと怒られるのでその時はd2sファイルを確認

radio_row_data = pd.read_csv(
    "../result_for_yasudaetal2022/galileo_radio_data/" + radio_data_name,
    header=None,
    skiprows=24,
    delimiter="  ",
    engine="python",
)


"""ここから下は基本いじらない"""
# %%


def Prepare_Galileo_data(rad_row_data):
    """_探査機による電波データのファイル名から電波データの時刻(電波データから読み取れる時刻とcsvファイルの時刻の差・周波数(ソースははじめ電波リストから)・電波強度を出力する_

    Args:
        data_name (_str_): _用いる電波データのファイル名を入力_

    Returns:
        _type_: _電波データの時刻の配列・周波数の配列・電波強度の配列_
    """

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


def Make_FT_full():
    # ガリレオ探査機のデータ取得開始時刻からの経過時間（sec) , ガリレオ探査機のデータ取得周波数（MHz), ガリレオ探査機の取得した電波強度（代入したデータと同じ単位）
    (
        galileo_data_time,
        galileo_data_freq,
        galileo_radio_intensity,
    ) = Prepare_Galileo_data(radio_row_data)

    # (周波数チャンネル数)x(時間ステップ数)の配列　ex. [[]]
    galileo_radio_intensity_row = galileo_radio_intensity.copy()

    # ガリレオ電波データが閾値より大きいとこは1 それ以外0
    if boundary == "V^2/m2/Hz":
        boundary_intensity = float(boundary_intensity_str)
        galileo_radio_intensity[boundary_intensity < galileo_radio_intensity] = 1
        galileo_radio_intensity[galileo_radio_intensity <= boundary_intensity] = 0

    elif boundary == "sigma":
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
        boundary_sigma = float(boundary_sigma_str)
        boundary_intensity_array = np.zeros(len(galileo_data_freq))

        for i in range(len(galileo_data_freq)):
            certain_freq_data = galileo_radio_intensity[i]  # i番目の周波数の全データ
            boundary_intensity = (
                noise_data[1, i] + boundary_sigma * noise_data[2, i]
            )  # i番目の周波数の強度平均値＋標準偏差×σ値
            boundary_intensity_array[i] = boundary_intensity
            # print(certain_freq_data.shape)
            detectable_position_array = np.where(
                certain_freq_data > boundary_intensity
            )[0]
            undetectable_position_array = np.where(
                certain_freq_data <= boundary_intensity
            )[0]
            print(certain_freq_data)
            galileo_radio_intensity[i, detectable_position_array] = 1
            galileo_radio_intensity[i, undetectable_position_array] = 0

    elif boundary == "average":
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
        boundary_intensity_array = np.zeros(len(galileo_data_freq))

        for i in range(len(galileo_data_freq)):
            certain_freq_data = galileo_radio_intensity[i]  # i番目の周波数の全データ
            boundary_intensity = (
                boundary_average * noise_data[1, i]
            )  # i番目の周波数の強度平均値×σ値
            boundary_intensity_array[i] = boundary_intensity
            # print(certain_freq_data.shape)
            detectable_position_array = np.where(
                certain_freq_data > boundary_intensity
            )[0]
            undetectable_position_array = np.where(
                certain_freq_data <= boundary_intensity
            )[0]
            print(certain_freq_data)
            galileo_radio_intensity[i, detectable_position_array] = 1
            galileo_radio_intensity[i, undetectable_position_array] = 0

        # print(noise_data)
        # print(len(noise_data[0]))  # 周波数(行)のみ一次元
        # print(noise_data[0][0])  # 左上のみを抽出
        # print(noise_data[:, 0]) #周波数、平均、標準偏差、中央値抽出

        # print(galileo_radio_intensity)
        # print(galileo_radio_intensity[:, 0])  # 時間方向
        # print(galileo_radio_intensity[0])  # 周波数方向方向

    else:
        print("invalid boudary type check boudary")

    def plot_and_save(first_time, last_time):
        # ガリレオ探査機の電波データの時刻・周波数でメッシュ作成
        xx, yy = np.meshgrid(galileo_data_time, galileo_data_freq)

        fig, ax = plt.subplots(2, 1, figsize=(8, 12))

        # ガリレオ探査機の電波強度をカラーマップへ
        pcm = ax[0].pcolor(
            xx,
            yy,
            galileo_radio_intensity_row,
            norm=mpl.colors.LogNorm(vmin=min_intensity, vmax=max_intensity),
            cmap="Spectral_r",
        )
        fig.colorbar(pcm, extend="max")

        # ガリレオ探査機の電波強度の閾値を赤線で
        ax[0].contour(xx, yy, galileo_radio_intensity, levels=[0.5], colors="red")

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
        ax[0].set_xlim(first_time, last_time)

        if boundary == "V^2/m2/Hz":
            ax[0].set_title(
                " Radio intensity boundary: "
                + boundary_intensity_str
                + " "
                + object_name
                + str(time_of_flybies)
            )

            ax[1].plot(
                galileo_data_freq,
                np.full(len(galileo_data_freq), boundary_intensity),
                marker="x",
                linestyle="-",
                color="b",
                label=boundary_intensity_str,
            )
            ax[1].set_xlim(min_frequency, max_frequency)
            ax[1].set_xscale("log")
            ax[1].set_ylim(min_intensity, min_intensity * 100)
            ax[1].set_yscale("log")
            ax[1].set_xlabel("Frequency (MHz)")
            ax[1].set_ylabel("Threshold intensity (V^2/m2/Hz)")
            # plt.show()

            fig.savefig(
                os.path.join(
                    "../result_for_yasudaetal2022/radio_plot/"
                    + object_name
                    + str(time_of_flybies)
                    + "/radio_ft_plot_time_"
                    + str(plot_first_time)
                    + "-"
                    + str(plot_last_time)
                    + "_intensity_boundary_"
                    + boundary_intensity_str
                    + ".png"
                )
            )

        elif boundary == "sigma":
            ax[0].set_title(
                "Radio intensity boundary : noise floor average x "
                + boundary_sigma_str
                + object_name
                + str(time_of_flybies)
            )

            ax[1].plot(
                galileo_data_freq,
                boundary_intensity_array,
                marker="x",
                linestyle="-",
                color="b",
                label=boundary_intensity_str,
            )
            ax[1].set_xlim(min_frequency, max_frequency)
            ax[1].set_xscale("log")
            ax[1].set_ylim(min_intensity / 5, min_intensity * 100)
            ax[1].set_yscale("log")
            ax[1].set_xlabel("Frequency [MHz]")
            ax[1].set_ylabel("Threshold intensity (V^2/m2/Hz)")
            # plt.show()

            fig.savefig(
                os.path.join(
                    "../result_for_yasudaetal2022/radio_plot/"
                    + object_name
                    + str(time_of_flybies)
                    + "/sigma_boundary/radio_ft_plot_time_"
                    + str(plot_first_time)
                    + "-"
                    + str(plot_last_time)
                    + "_intensity_boundary_sigma"
                    + boundary_sigma_str
                    + "sigma.png"
                )
            )

        elif boundary == "average":
            ax[0].set_title(
                "Radio intensity boundary : noise floor average + "
                + boundary_average_str
                + " x average "
                + object_name
                + str(time_of_flybies)
            )

            ax[1].plot(
                galileo_data_freq,
                boundary_intensity_array,
                marker="x",
                linestyle="-",
                color="b",
                label=boundary_intensity_str,
            )
            ax[1].set_xlim(min_frequency, max_frequency)
            ax[1].set_xscale("log")
            ax[1].set_ylim(min_intensity, min_intensity * 100)
            ax[1].set_yscale("log")
            ax[1].set_xlabel("Frequency [MHz]")
            ax[1].set_ylabel("Threshold intensity (V^2/m2/Hz)")
            # plt.show()

            fig.savefig(
                os.path.join(
                    "../result_for_yasudaetal2022/radio_plot/"
                    + object_name
                    + str(time_of_flybies)
                    + "/average_boundary/radio_ft_plot_time_"
                    + str(plot_first_time)
                    + "-"
                    + str(plot_last_time)
                    + "_intensity_boundary_average"
                    + boundary_sigma_str
                    + ".png"
                )
            )

    plot_and_save(plot_first_time, plot_last_time)

    return 0


# %%


def main():
    Make_FT_full()

    return 0


if __name__ == "__main__":
    main()


# %%
