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
object_name = "ganymede"  # ganydeme/europa/calisto`
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 1  # ..th flyby


## 詳細設定

# プロットしたい時間範囲を指定(秒)

# G1

plot_first_time = 0
plot_last_time = 5400
"""
plot_first_time = 1800  # 06;00
# plot_first_time = 2100  # 06:05
# plot_first_time = 2280  # 06:08
plot_last_time = 2700
"""

"""
# C09
plot_first_time = 0
plot_last_time = 10800

# C09
plot_first_time = 5700
plot_last_time = 6000
"""
"""
# C30
plot_first_time = 3600
plot_last_time = 9000

# C30
plot_first_time = 5400
plot_last_time = 6000
"""

# カラーマップの強度範囲（）
max_intensity = 3e-15  # カラーマップの最大強度
min_intensity = 3e-17  # カラーマップの最小強度


# ftダイヤグラムに等高線をひく強度を指定
# 電波データから掩蔽・非掩蔽を判定する際に全周波数チャンネルに対して単一の電波強度を閾値とする場合には boundary="V^2/m2/Hz"
# 周波数ごとに観測装置のノイズの大きさ（平均値）とばらつき（標準偏差）を調べ、平均値＋（n x 標準偏差）を閾値とする場合には boundary="sigma"
boundary = "V^2/m2/Hz"  # "" or "sigma"
boundary_intensity_str = "7e-16"  # (V^2/m2/Hzの場合)boundary_intensity_str = '1e-15'[V^2/m2/Hz] (sigmaの場合)boundary_intensity_str = '1'[sigma]
boundary_intensity = float(boundary_intensity_str)

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
        "Survey_Electric_2001-05-25T10-00_2001-05-25T13-00_for_examine.d2s"  # C30 flyby
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

    galileo_radio_intensity_row = galileo_radio_intensity.copy()

    # ガリレオ電波データが閾値より大きいとこは1 それ以外0
    if boundary == "V^2/m2/Hz":
        galileo_radio_intensity[boundary_intensity < galileo_radio_intensity] = 1
        galileo_radio_intensity[galileo_radio_intensity < boundary_intensity] = 0

    elif boundary == "sigma":
        noise_data = pd.read_csv(
            "../result_for_yasudaetal2022/radio_plot/"
            + object_name
            + str(time_of_flybies)
            + "/"
            + spacecraft_name
            + "_noise_floor_excepted_two_sigma_"
            + object_name
            + str(time_of_flybies)
            + ".csv",
            header=None,
            delimiter="  ",
            engine="python",
        )
        print(noise_data)

    else:
        print("invalid boudary type check boudary")

    def plot_and_save(first_time, last_time):
        # ガリレオ探査機の電波データの時刻・周波数でメッシュ作成
        xx, yy = np.meshgrid(galileo_data_time, galileo_data_freq)

        fig, ax = plt.subplots(3, 1, figsize=(6, 12))

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
        # ax[0].set_xlabel("Time of 27 June 1996")
        # 日時はcsvファイルの情報から記入さsれる
        ax[0].set_xlabel("time")

        # 論文中で使われている横軸の幅とそれに対応する計算開始時刻からの秒数はグローバル変数で指定しておく
        ax[0].set_xticks(plot_time_step_sec)
        ax[0].set_xticklabels(plot_time_step_label)

        # 横軸の幅は作りたい図によって変わるので引数用いる
        ax[0].set_xlim(first_time, last_time)
        ax[0].set_title("Radio intensity boundary: " + boundary_intensity_str)

        # ガリレオ探査機のデータ取得タイミングステップをプロット
        # galileo_data_timeは電波データ開始時刻からの経過時間の配列
        galileo_time_step_array = (
            np.roll(galileo_data_time, -1) - galileo_data_time
        )  # 2番目の要素と1番目の要素の差分（時間差）を１番目の要素に　最後の配列は１番目と最後の要素の差分に
        galileo_time_step_array[-1] = np.nan

        ax[1].scatter(galileo_data_time, galileo_time_step_array, s=4)
        for i in range(len(galileo_data_time)):
            ax[1].text(
                galileo_data_time[i],
                galileo_time_step_array[i],
                str(int(galileo_data_time[i])),
                fontsize=2,
            )

        # 論文中で使われている横軸の幅とそれに対応する計算開始時刻からの秒数はグローバル変数で指定しておく
        ax[1].set_xticks(plot_time_step_sec)
        ax[1].set_xticklabels(plot_time_step_label)
        ax[1].set_title("Time step")
        ax[1].set_xlabel("time")
        ax[1].set_ylabel("time step (sec)")
        ax[1].set_yscale("log")
        ax[1].set_xlim(first_time, last_time)

        # ガリレオ探査機の電波強度変化をプロット
        # galileo_radio_intensity_row.copy()で周波数配列x電波データの時間が入ってる
        galileo_radio_intensity_vary = galileo_radio_intensity_row.copy()
        galileo_radio_intensity_vary = np.abs(
            (
                np.roll(galileo_radio_intensity_vary, -1, axis=1)
                - galileo_radio_intensity_vary
            )
        )

        # ガリレオ探査機の電波強度をカラーマップへ
        pcm = ax[2].pcolor(
            xx,
            yy,
            galileo_radio_intensity_vary,
            norm=mpl.colors.LogNorm(vmin=min_intensity, vmax=max_intensity),
            cmap="Spectral_r",
        )
        fig.colorbar(pcm, extend="max")

        ax[2].set_yscale("log")
        ax[2].set_ylim(min_frequency, max_frequency)
        ax[2].set_ylabel("Frequency (MHz)")

        # 日時はcsvファイルの情報から記入さsれる
        ax[2].set_xlabel("time")

        # 論文中で使われている横軸の幅とそれに対応する計算開始時刻からの秒数はグローバル変数で指定しておく
        ax[2].set_xticks(plot_time_step_sec)
        ax[2].set_xticklabels(plot_time_step_label)

        # 横軸の幅は作りたい図によって変わるので引数用いる
        ax[2].set_xlim(first_time, last_time)
        ax[2].set_title("Radio intensity variation")

        plt.show()

        fig.savefig(
            os.path.join(
                "../result_for_yasudaetal2022/radio_plot/"
                + object_name
                + str(time_of_flybies)
                + "/radio_ft_plot_time_"
                + str(plot_first_time)
                + "-"
                + str(plot_last_time)
                + ".png",
            ),
            dpi=300,
        )

        return 0

    plot_and_save(plot_first_time, plot_last_time)

    return 0


# %%


def main():
    Make_FT_full()

    return 0


if __name__ == "__main__":
    main()


# %%
