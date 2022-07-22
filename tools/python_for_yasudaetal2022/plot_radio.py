# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
import os
import time
import glob

# %%
# あらかじめ ../result_sgepss2021/~/~ に必要なレイトレーシング結果とパラメータセットを入れること

plot_time_step_sec = [0, 1800, 3600, 5400, 7200, 9000, 10800]
plot_time_step_label = ["10:00", "10:30",
                        "11:00", "11:30", "12:00", "12:30", "13:00"]


information_list = ['year', 'month', 'start_day', 'end_day',
                    'start_hour', 'end_hour', 'start_min', 'end_min', 'occultaton_center_day', 'occultaton_center_hour', 'occultaton_center_min']

boundary_intensity_str = '7e-16'  # boundary_intensity_str = '1e-15'

# [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 座標変換した時のx(tangential point との水平方向の距離),6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度]
"""
Freq_str = ['3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]
"""
Freq_str = ['3.612176179885864258e5', '3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]

Freq_underline = 0.32744

Freq_num = []
for idx in Freq_str:
    Freq_num.append(float(idx)/1000000)

boundary_intensity = float(boundary_intensity_str)

# ガリレオ探査機によって取得される周波数・探査機が変わったらこの周波数も変わってくるはず
gal_fleq_tag_row = [5.620e+00, 1.000e+01, 1.780e+01, 3.110e+01, 4.213e+01, 4.538e+01, 4.888e+01, 5.265e+01, 5.671e+01, 6.109e+01, 6.580e+01, 7.087e+01, 7.634e+01, 8.223e+01, 8.857e+01,
                    9.541e+01, 1.028e+02, 1.107e+02, 1.192e+02, 1.284e+02, 1.383e+02, 1.490e+02, 1.605e+02, 1.729e+02, 1.862e+02, 2.006e+02, 2.160e+02, 2.327e+02, 2.507e+02, 2.700e+02, 2.908e+02,
                    3.133e+02, 3.374e+02, 3.634e+02, 3.915e+02, 4.217e+02, 4.542e+02, 4.892e+02, 5.270e+02, 5.676e+02, 6.114e+02, 6.586e+02, 7.094e+02, 7.641e+02, 8.230e+02, 8.865e+02, 9.549e+02,
                    1.029e+03, 1.108e+03, 1.193e+03, 1.285e+03, 1.385e+03, 1.491e+03, 1.606e+03, 1.730e+03, 1.864e+03, 2.008e+03, 2.162e+03, 2.329e+03, 2.509e+03, 2.702e+03, 2.911e+03, 3.135e+03,
                    3.377e+03, 3.638e+03, 3.918e+03, 4.221e+03, 4.546e+03, 4.897e+03, 5.275e+03, 5.681e+03, 6.120e+03, 6.592e+03, 7.100e+03, 7.648e+03, 8.238e+03, 8.873e+03, 9.558e+03, 1.029e+04,
                    1.109e+04, 1.194e+04, 1.287e+04, 1.386e+04, 1.493e+04, 1.608e+04, 1.732e+04, 1.865e+04, 2.009e+04, 2.164e+04, 2.331e+04, 2.511e+04, 2.705e+04, 2.913e+04, 3.138e+04, 3.380e+04,
                    3.641e+04, 3.922e+04, 4.224e+04, 4.550e+04, 4.901e+04, 5.279e+04, 5.686e+04, 6.125e+04, 6.598e+04, 7.106e+04, 7.655e+04, 8.245e+04, 8.881e+04, 9.566e+04, 1.030e+05, 1.030e+05,
                    1.137e+05, 1.254e+05, 1.383e+05, 1.526e+05, 1.683e+05, 1.857e+05, 2.049e+05, 2.260e+05, 2.493e+05, 2.750e+05, 3.034e+05, 3.347e+05, 3.692e+05, 4.073e+05, 4.493e+05, 4.957e+05,
                    5.468e+05, 6.033e+05, 6.655e+05, 7.341e+05, 8.099e+05, 8.934e+05, 9.856e+05, 1.087e+06, 1.199e+06, 1.323e+06, 1.460e+06, 1.610e+06,
                    1.776e+06, 1.960e+06, 2.162e+06, 2.385e+06, 2.631e+06, 2.902e+06, 3.201e+06, 3.532e+06, 3.896e+06, 4.298e+06, 4.741e+06, 5.231e+06, 5.770e+06]

# 電波強度のデータを取得（一列目は時刻データになってる）
# 初めの数行は読み取らないよう設定・時刻データを読み取って時刻をプロットするためここがずれても影響はないが、データがない行を読むと怒られるのでその時はd2sファイルを確認


radio_data_name = "Survey_Electric_2001-05-25T10-00_2001-05-25T13-00_for_examine.d2s"
rad_row_data = pd.read_csv('../result_for_yasudaetal2022/galileo_radio_data/' +
                           radio_data_name, header=None, skiprows=24, delimiter='  ')

start_day = 25
start_hour = 10
start_min = 0

# %%


def Prepare_Galileo_data():
    """_探査機による電波データのファイル名から電波データの時刻(電波データから読み取れる時刻とcsvファイルの時刻の差・周波数(ソースははじめ電波リストから)・電波強度を出力する_

    Args:
        data_name (_str_): _用いる電波データのファイル名を入力_

    Returns:
        _type_: _電波データの時刻の配列・周波数の配列・電波強度の配列_
    """

    # 電波データの周波数の単位をHzからMHzに変換する
    gal_fleq_tag = np.array(gal_fleq_tag_row, dtype='float64')/1000000

    # 一列目の時刻データを文字列で取得（例; :10:1996-06-27T05:30:08.695） ・同じ長さの０配列を準備・
    gal_time_tag_prepare = np.array(rad_row_data.iloc[:, 0])
    gal_time_tag_prepare = gal_time_tag_prepare.astype(np.str)
    gal_time_tag = np.zeros(len(gal_time_tag_prepare))

    # 文字列のデータから開始時刻からの経過時間（秒）に変換
    # Tで分けた[1]　例 :10:1996-06-27T05:30:08.695 ⇨ 05:30:08.695
    # :で分ける　例;05:30:08.695 ⇨ 05 30 08.695
    for i in range(len(gal_time_tag)):
        hour_min_sec = np.char.split(np.char.split(
            gal_time_tag_prepare[:], sep='T')[i][1], sep=[':'])[0]

        hour_min_sec_list = [float(vle) for vle in hour_min_sec]

    # Tで分けた[0]　例; :10:1996-06-27T05:30:08.695 ⇨ 1996-06-27
    # :で分けた最後の部分　例; :10:1996-06-27 ⇨ 10 1996-06-27
        year_month_day_pre = np.char.split(np.char.split(
            gal_time_tag_prepare[:], sep='T')[i][0], sep=[':'])[0][-1]

        year_month_day = np.char.split(year_month_day_pre, sep=['-'])[0]

        year_month_day_list = [float(vle) for vle in year_month_day]

    # 秒に変換 27✖️86400 + 05✖️3600 + 30✖️60 ＋ 08.695

        gal_time_tag[i] = hour_min_sec_list[2] + hour_min_sec_list[1] * \
            60 + hour_min_sec_list[0]*3600 + \
            year_month_day_list[2]*86400  # 経過時間(sec)に変換

    # time_info['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']
    # csvファイルからの開始時刻を秒に変換
    # startday(2)*86400+start_hour(4)*3600+ start_min(6)*60
    start_time = start_day*86400 + start_hour*3600 + start_min*60
    gal_time_tag = np.array(gal_time_tag-start_time)
    df = pd.DataFrame(rad_row_data.iloc[:, 1:])

    DDF = np.array(df).astype(np.float64).T
    print(DDF)
    print(len(gal_fleq_tag), len(gal_time_tag), DDF.shape)

    # ガリレオ探査機のデータ取得開始時刻からの経過時間（sec) , ガリレオ探査機のデータ取得周波数（MHz), ガリレオ探査機の取得した電波強度（代入したデータと同じ単位）
    return gal_time_tag, gal_fleq_tag, DDF


def Make_FT_full():

    # ガリレオ探査機のデータ取得開始時刻からの経過時間（sec) , ガリレオ探査機のデータ取得周波数（MHz), ガリレオ探査機の取得した電波強度（代入したデータと同じ単位）
    galileo_data_time, galileo_data_freq, galileo_radio_intensity = Prepare_Galileo_data()

    galileo_radio_intensity_row = galileo_radio_intensity.copy()

    # ガリレオ電波データが閾値より大きいとこは1 それ以外0
    galileo_radio_intensity[boundary_intensity
                            < galileo_radio_intensity] = 1
    galileo_radio_intensity[galileo_radio_intensity <
                            boundary_intensity] = 0

    def plot_and_save(start_time, end_time):
        # ガリレオ探査機の電波データの時刻・周波数でメッシュ作成
        xx, yy = np.meshgrid(galileo_data_time, galileo_data_freq)

        fig, ax = plt.subplots(1, 1)

        # ガリレオ探査機の電波強度をカラーマップへ
        pcm = ax.pcolor(xx, yy, galileo_radio_intensity_row, norm=mpl.colors.LogNorm(
            vmin=1e-16, vmax=1e-12), cmap='Spectral_r')
        fig.colorbar(pcm, extend='max')

        # ガリレオ探査機の電波強度の閾値を赤線で
        ax.contour(xx, yy, galileo_radio_intensity,
                   levels=[0.5], colors='red')

        ax.set_yscale("log")
        ax.set_ylim(1.0, 3.0)
        ax.set_ylabel("Frequency (MHz)")

        # raytrace_time_information ['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']
        # ax.set_xlabel("Time of 27 June 1996")
        # 日時はcsvファイルの情報から記入される
        ax.set_xlabel("")

        # 論文中で使われている横軸の幅とそれに対応する計算開始時刻からの秒数はグローバル変数で指定しておく
        ax.set_xticks(plot_time_step_sec)
        ax.set_xticklabels(plot_time_step_label)

        # 横軸の幅は作りたい図によって変わるので引数用いる
        ax.set_xlim(start_time, end_time)
        ax.set_title("")
        plt.show()
        fig.savefig(os.path.join(
            '../result_for_yasudaetal2022/radio_plot/radio_plot.png'))

    plot_and_save(6000, 6900)

    return 0

# %%


def main():
    Make_FT_full()

    return 0


if __name__ == "__main__":
    main()


# %%
