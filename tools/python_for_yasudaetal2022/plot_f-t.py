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

object_name = 'ganymede'  # ganydeme/europa/calisto``
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 1  # ..th flyby
highest_plasma = '0e2'  # 単位は(/cc) 2e2/4e2/16e22
plasma_scaleheight = '0.25e2'  # 単位は(km) 1.5e2/3e2/6e2
boundary_intensity_str = '7e-16'  # boundary_intensity_str = '1e-15'
# boundary_intensity_str = '0'  # boundary_intensity_str = '1e-15'
vertical_line_freq = 0.65  # MHz

# G1 flyby
plot_time_step_sec = [0, 900, 1800, 2700, 3600, 4500, 5400]
plot_time_step_label = ["05:30", "05:45",
                        "06:00", "06:15", "06:30", "06:45", "07;00"]

# E12 flyby
# plot_time_step_sec = [6300, 6600, 6900, 7200, 7500, 7800, 8100, 8400, 8700]
# plot_time_step_label = ["11:45", "11:50", "11:55","12:00", "12:05", "12:10", "12:15", "12:20", "12:25"]

# C30 flyby
# plot_time_step_sec = [0, 1800, 3600, 5400, 7200, 9000, 10800]
# plot_time_step_label = ["10:00", "10:30","11:00", "11:30", "12:00", "12:30", "13:00"]

# C9 flyby
# plot_time_step_sec = [0, 1800, 3600, 5400, 7200, 9000, 10800]
# plot_time_step_label = ["12:00", "12:30","13:00", "13:30", "14:00", "14:30", "15:00"]

information_list = ['year', 'month', 'start_day', 'end_day',
                    'start_hour', 'end_hour', 'start_min', 'end_min', 'occultaton_center_day', 'occultaton_center_hour', 'occultaton_center_min']

Radio_name_csv = '../result_for_yasudaetal2022/tracing_range_'+spacecraft_name+'_'+object_name + \
    '_'+str(time_of_flybies)+'_flybys/para_' + \
    highest_plasma+'_'+plasma_scaleheight+'.csv'
Radio_Range = pd.read_csv(Radio_name_csv, header=0, engine="python")
# [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 座標変換した時のx(tangential point との水平方向の距離),6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度,8 探査機の経度]


# europa & ganymede

Freq_str = ['3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]

Freq_underline = 0.36122

####

# callisto
"""
Freq_str = ['3.612176179885864258e5', '3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]

Freq_underline = 0.32744
"""
####

Freq_num = []
for idx in Freq_str:
    Freq_num.append(float(idx)/1000000)

Highest = Radio_Range.highest
Lowest = Radio_Range.lowest
Except = Radio_Range.exc


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


# %%

def Pick_up_cdf():
    flyby_list_path = '../result_for_yasudaetal2022/occultation_flyby_list.csv'
    flyby_list = pd.read_csv(flyby_list_path, engine="python")

    # csvファイルにフライバイごとで使う軌道データを記入しておく　上記のパラメータから必要なデータのファイル名が選ばれて読み込まれる
    # queryが数値非対応なのでまずはフライバイ数で絞り込み
    selected_flyby_list = flyby_list[flyby_list['flyby_time']
                                     == time_of_flybies]
    complete_selecred_flyby_list = selected_flyby_list.query(
        'object == "'+object_name+'" & spacecraft == "'+spacecraft_name+'"')  # queryでフライバイ数以外を絞り込み

    complete_selecred_flyby_list = complete_selecred_flyby_list.reset_index(
        drop=True)

    # complete_selecred_flyby_list = complete_selecred_flyby_list.index.tolist()

    # print(complete_selecred_flyby_list)

    # csvから時刻データを抽出['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']
    time_information = []
    for i in information_list:
        time_information.append(int(complete_selecred_flyby_list[i][0]))

    # csvから対象の電波データを有するcsvの名前を取得
    radio_data_name = str(complete_selecred_flyby_list['radio_data_txt'][0])

    # csvの時刻データと電波データの名前をかえす
    return time_information, radio_data_name


def Time_step(time_data):
    """_csvファイルの時効データからレイトレーシングで計算した総秒数を出力する_

    Args:
        time_data (_type_): _pick_up_cdfでcsvファイルから取ってきた時刻情報をそのまま入れる_

    Returns:
        _type_: _レイトレーシングで計算した総秒数_
    """
    day_range = int(time_data[3])-int(time_data[2])
    hour_range = int(time_data[5])-int(time_data[4])
    min_range = int(time_data[7])-int(time_data[6])

    step_count = day_range*1440*60 + hour_range*60*60 + min_range*60 + \
        1  # フライバイリストからステップ数を計算（今は1step1minを仮定してステップ数を計算）
    # フライバイリストのステップ数と位置データのステップ数が一致する確認（今は1step1minを仮定してステップ数を計算）

    return step_count


def Prepare_Figure(judgement, time_information):

    time_step = Time_step(time_information)
    time_step_list = np.arange(0, time_step, 60)
    # (周波数の数 +1) ×(時間数（正確には開始時刻からの秒数の数) ）の0配列を４つ用意
    DataA = np.zeros(len(time_step_list)*(len(Freq_num)+1)
                     ).reshape(len(Freq_num)+1, len(time_step_list))
    DataB = np.zeros(len(time_step_list)*(len(Freq_num)+1)
                     ).reshape(len(Freq_num)+1, len(time_step_list))
    DataC = np.zeros(len(time_step_list)*(len(Freq_num)+1)
                     ).reshape(len(Freq_num)+1, len(time_step_list))
    DataD = np.zeros(len(time_step_list)*(len(Freq_num)+1)
                     ).reshape(len(Freq_num)+1, len(time_step_list))

    # judgement [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 座標変換した時のx(tangential point との水平方向の距離),6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度,8 探査機の経度]

    for k in range(len(judgement)):

        Num = int(judgement[k][0]*60+judgement[k][1]) - \
            (time_information[4]*60+time_information[6])

        if np.abs(judgement[k][8]+360-judgement[k][7]) < np.abs(judgement[k][8]-judgement[k][7]):
            Lon = judgement[k][8]+360 - judgement[k][7]

        elif np.abs(judgement[k][7]+360-judgement[k][8]) < np.abs(judgement[k][7]-judgement[k][8]):
            Lon = judgement[k][8]-360 - judgement[k][7]

        else:
            Lon = judgement[k][8] - judgement[k][7]

        Lat = judgement[k][4]

        Fre = np.where(Freq_num == judgement[k][2])
        # int(Fre[0])+1になっているのは、のちのコンタープロットのために一個多い周波数で作ってあるのでミスではない
        if Lon < 0 and Lat > 0:
            DataA[int(Fre[0])+1][Num] = 1
        if Lon > 0 and Lat > 0:
            DataB[int(Fre[0])+1][Num] = 1

        if Lon < 0 and Lat < 0:
            DataC[int(Fre[0])+1][Num] = 1

        if Lon > 0 and Lat < 0:
            DataD[int(Fre[0])+1][Num] = 1

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
        '../result_for_yasudaetal2022/galileo_radio_data/'+data_name, header=None, skiprows=24, delimiter='  ', engine="python")

    # 電波データの周波数の単位をHzからMHzに変換する
    gal_fleq_tag = np.array(gal_fleq_tag_row, dtype='float64')/1000000

    # 一列目の時刻データを文字列で取得（例; :10:1996-06-27T05:30:08.695） ・同じ長さの０配列を準備・
    gal_time_tag_prepare = np.array(rad_row_data.iloc[:, 0])
    gal_time_tag_prepare = gal_time_tag_prepare.astype(str)
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
    start_time = time_info[2]*86400 + time_info[4]*3600 + time_info[6]*60
    gal_time_tag = np.array(gal_time_tag-start_time)
    df = pd.DataFrame(rad_row_data.iloc[:, 1:])

    DDF = np.array(df).astype(np.float64).T
    # print(DDF)
    # print(len(gal_fleq_tag), len(gal_time_tag), DDF.shape)

    # ガリレオ探査機のデータ取得開始時刻からの経過時間（sec) , ガリレオ探査機のデータ取得周波数（MHz), ガリレオ探査機の取得した電波強度（代入したデータと同じ単位）
    return gal_time_tag, gal_fleq_tag, DDF


"""
# ガリレオ探査機の周波数一覧（Hz)とダウンロードした電波強度電波を代入（das2をcsvに変換）
def Prepare_Galileo_data2(time_info, data_name):
    """
"""_探査機による電波データのファイル名から電波データの時刻(電波データから読み取れる時刻とcsvファイルの時刻の差・周波数(ソースははじめ電波リストから)・電波強度を出力する_

    Args:
        data_name(_str_): _用いる電波データのファイル名を入力_

    Returns:
        _type_: _電波データの時刻の配列・周波数の配列・電波強度の配列_
    """
"""
    # 電波強度のデータを取得（一行目は時刻データになってる）
    rad_row_data = pd.read_csv(
        '../result_for_yasudaetal2022/galileo_radio_data/'+data_name, header=None)

    # 電波データの周波数の単位をHzからMHzに変換する
    gal_fleq_tag = np.array(gal_fleq_tag_row, dtype='float64')/1000000

    # 一行目の時刻データを文字列で取得（例;1996-06-27T05:30:08.695） ・同じ長さの０配列を準備・
    gal_time_tag_prepare = np.array(rad_row_data.iloc[0])
    gal_time_tag_prepare = gal_time_tag_prepare.astype(np.str)
    gal_time_tag = np.zeros(len(gal_time_tag_prepare))

    # 文字列のデータから開始時刻からの経過時間（秒）に変換
    # Tで分けた[1]　例;1996-06-27T05:30:08.695 ⇨ 05:30:08.695
    # :で分ける　例;05:30:08.695 ⇨ 05 30 08.695
    for i in range(len(gal_time_tag)):
        hour_min_sec = np.char.split(np.char.split(
            gal_time_tag_prepare[:], sep='T')[i][1], sep=[':'])[0]

        hour_min_sec_list = [float(vle) for vle in hour_min_sec]

    # Tで分けた[0]　例;1996-06-27T05:30:08.695 ⇨ 1996-06-27
    # -で分ける　例;1996-06-27 ⇨ 1996 06 27

        year_month_day = np.char.split(np.char.split(
            gal_time_tag_prepare[:], sep='T')[i][0], sep=['-'])[0]

        year_month_day_list = [float(vle) for vle in year_month_day]

    # 秒に変換 27✖️86400 + 05✖️3600 + 30✖️60 ＋ 08.695

        gal_time_tag[i] = hour_min_sec_list[2] + hour_min_sec_list[1] * \
            60 + hour_min_sec_list[0]*3600 + \
            year_month_day_list[2]*86400  # 経過時間(sec)に変換

    # time_info['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']
    # csvファイルからの開始時刻を秒に変換
    # startday(2)*86400+start_hour(4)*3600+ start_min(6)*60
    start_time = time_info[2]*86400 + time_info[4]*3600 + time_info[6]*60
    gal_time_tag = np.array(gal_time_tag-start_time)
    df = pd.DataFrame(rad_row_data[1:])
    DDF = np.array(df).astype(np.float64)

    print(DDF)
    print(len(gal_fleq_tag), len(gal_time_tag), DDF.shape)

    # ガリレオ探査機のデータ取得開始時刻からの経過時間（sec) , ガリレオ探査機のデータ取得周波数（MHz), ガリレオ探査機の取得した電波強度（代入したデータと同じ単位）
    return gal_time_tag, gal_fleq_tag, DDF
"""


def Make_FT_full(DataA, DataB, DataC, DataD, raytrace_time_information, radio_data_name):

    time_step = Time_step(raytrace_time_information)

    time_list = np.arange(0, time_step, 60)  # エクスプレスコードで計算している時間幅（sec)を60で割る

    # ガリレオ探査機のデータ取得開始時刻からの経過時間（sec) , ガリレオ探査機のデータ取得周波数（MHz), ガリレオ探査機の取得した電波強度（代入したデータと同じ単位）
    galileo_data_time, galileo_data_freq, galileo_radio_intensity = Prepare_Galileo_data(raytrace_time_information,
                                                                                         radio_data_name)

    galileo_radio_intensity_row = galileo_radio_intensity.copy()

    # ガリレオ電波データが閾値より大きいとこは1 それ以外0
    galileo_radio_intensity[boundary_intensity
                            < galileo_radio_intensity] = 1
    galileo_radio_intensity[galileo_radio_intensity <
                            boundary_intensity] = 0
    # print(galileo_data_time.shape)
    # print(galileo_radio_intensity.shape)

    # raytrace_time_information ['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']

    # start時刻から掩蔽中心時刻までの差分（秒）を計算
    middle_time = (raytrace_time_information[8]-raytrace_time_information[2])*86400 + (raytrace_time_information[9] -
                                                                                       raytrace_time_information[4])*3600 + (raytrace_time_information[10]-raytrace_time_information[6]) * 60

    # ガリレオ電波データの掩蔽中心時刻を初めに超える時間ステップの配列番号を取得
    occulted_time = int(np.where(galileo_data_time > middle_time)[0][0])
    ingress_time_list = galileo_data_freq.copy()
    egress_time_list = galileo_data_freq.copy()

    # それぞれの周波数に対して..
    for k in range(len(galileo_data_freq)):

        # k番目の周波数の強度で閾値を超える電波強度を捉えている電波データの配列番号のリストを取得
        over_judge_time_list = np.array(np.where(
            (galileo_radio_intensity[k][:] > boundary_intensity)))

        # print(over_judge_time_list)
        # 閾値を超える電波強度を観測しているものの中でも掩蔽中心時刻前のもののリストをA、後のもののリストをBとしてわける
        A = over_judge_time_list[over_judge_time_list < occulted_time]
        B = over_judge_time_list[over_judge_time_list > occulted_time]

        # 電波強度が十分であればAもBも複数ヒットするはず...
        if len(B) > 0:
            b = B[0]
            # 初めて閾値を超える強度が観測される時刻（正確にはstart時刻からの差分（秒））がgalileo_data_time[b]
            # その１つ前の時刻 galileo_data_time[b-1]なので、その真ん中の時間がegress_time_listに加わる（要するに終了タイミング）
            egress_time_list = np.append(
                egress_time_list, ((galileo_data_time[b] + galileo_data_time[b-1])/2))

        else:
            egress_time_list = np.append(egress_time_list, -100000)

        if len(A) > 0:
            a = A[len(A)-1]
            # 最後に閾値を超える強度が観測される時刻（正確にはstart時刻からの差分（秒））がgalileo_data_time[a]
            # その１つ次の時刻 galileo_data_time[a+1]なので、その真ん中の時間がingress_time_listに加わる（要するに開始タイミング）
            ingress_time_list = np.append(ingress_time_list, ((
                galileo_data_time[a] + galileo_data_time[a+1])/2))

        else:
            ingress_time_list = np.append(ingress_time_list, 100000)

    ingress_time_list = ingress_time_list.reshape(
        2, int(len(ingress_time_list)/2))
    print(ingress_time_list)
    egress_time_list = egress_time_list.reshape(
        2, int(len(egress_time_list)/2))
    print("array=", np.array(Freq_num))
    print("underline=", Freq_underline)
    FREQ = np.insert(np.array(Freq_num), 0, Freq_underline)

    """
    ここまでで出来上がっている材料たち
    time_list レイトレーシング(1分間隔)の時刻データ(観測開始隊ミンングからの秒) ex. [0,60,120 ...]
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
        pcm = ax.pcolormesh(xx, yy, galileo_radio_intensity_row, norm=mpl.colors.LogNorm(
            vmin=1e-16, vmax=1e-12), cmap='Spectral_r')
        print(xx)
        fig.colorbar(pcm, extend='max',
                     label='GLL/PWS Electric Power spectral density (V2/m2/Hz)')

        # ガリレオ探査機の電波強度の閾値を赤線＃

        # レイトレーシングの結果をコンタープロットで表示
        ax.contour(time_list, FREQ, DataA, levels=[0.5], colors='white')
        ax.contour(time_list, FREQ, DataB, levels=[0.5], colors='lightgray')
        ax.contour(time_list, FREQ, DataC, levels=[0.5], colors='darkgray')
        ax.contour(time_list, FREQ, DataD, levels=[0.5], colors='black')
        ax.contour(xx, yy, galileo_radio_intensity, levels=[0.5], colors='red')
        ax.set_yscale("log")
        ax.set_ylim(0.1, 6.0)

        ax.set_ylabel("Frequency (MHz)")

        # raytrace_time_information ['year', 'month', 'start_day', 'end_day','start_hour', 'end_hour', 'start_min', 'end_min','occultaton_center_day','occultaton_center_hour','occultaton_center_min']
        # ax.set_xlabel("Time of 27 June 1996")
        # 日時はcsvファイルの情報から記入される
        ax.set_xlabel("Time of "+str(raytrace_time_information[1])+"/"+str(
            raytrace_time_information[2])+"/"+str(raytrace_time_information[0]))

        # 論文中で使われている横軸の幅とそれに対応する計算開始時刻からの秒数はグローバル変数で指定しておく
        ax.set_xticks(plot_time_step_sec)
        ax.set_xticklabels(plot_time_step_label)

        # 横軸の幅は作りたい図によって変わるので引数用いる
        ax.set_xlim(start_time, end_time)
        plt.hlines(vertical_line_freq, start_time, end_time,
                   colors='hotpink', linestyle='dashed')
        plt.annotate(str(vertical_line_freq)+"MHz",
                     (start_time+20, vertical_line_freq+0.05), color="hotpink")
        ax.set_title("max density "+highest_plasma +
                     "(cm-3) scale height "+plasma_scaleheight+"(km)")

        fig.savefig(os.path.join('../result_for_yasudaetal2022/raytracing_'+object_name+'_results/' + object_name+'_'+highest_plasma +
                                 '_'+plasma_scaleheight+'/', spacecraft_name+'_' + object_name+'_'+str(time_of_flybies)+'_'+highest_plasma+'_'+plasma_scaleheight+'_boundary_int='+boundary_intensity_str+'_'+name+'_f-t.png'), dpi=1000)

        fig.savefig(os.path.join('../result_for_yasudaetal2022/f-t_plot_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby/radio_boundary_intensity_'+boundary_intensity_str,
                                 spacecraft_name+'_' + object_name+'_'+str(time_of_flybies)+'_'+highest_plasma+'_'+plasma_scaleheight+'_boundary_int='+boundary_intensity_str+'_'+name+'_f-t.png'), dpi=1000)

    plot_and_save(int(plot_time_step_sec[0]), int(
        plot_time_step_sec[-1]), "full")
    # plot_and_save(middle_time, int(plot_time_step_sec[-1]), "egress")
    # plot_and_save(int(plot_time_step_sec[0]), middle_time, "ingress")
    plot_and_save(middle_time, middle_time+1800, "egress")
    plot_and_save(middle_time-1800, middle_time, "ingress")

    # 以下は電波データにおける掩蔽タイミングを決めるものなので、閾値を買えない限りは毎回やる必要はない
    # print(ingress_time_list)
    np.savetxt('../result_for_yasudaetal2022/radio_data_occultation_timing_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby/'+object_name +
               '_'+boundary_intensity_str+'_ingress_time_data.txt', ingress_time_list)
    np.savetxt('../result_for_yasudaetal2022/radio_data_occultation_timing_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby/'+object_name +
               '_'+boundary_intensity_str+'_egress_time_data.txt', egress_time_list)

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
    raytrace_time = np.arange(0, time_step, 60)  # レイトレーシング計算した間隔での時間リスト（秒）

    # csvファイルから読み取ったstart時刻から掩蔽中心時刻までの差分（秒）を計算
    middle_time = (raytrace_time_information[8]-raytrace_time_information[2])*86400 + (raytrace_time_information[9] -
                                                                                       raytrace_time_information[4])*3600 + (raytrace_time_information[10]-raytrace_time_information[6]) * 60

    # レイトレーシングの周波数リスト(contour plotをする関係で配列の初めに本来はない周波数を挿入しているので周波数の数的には多くなっているので注意)
    raytrace_freq = np.insert(np.array(Freq_num), 0, Freq_underline)

    # レイトレーシングの時間間隔の中で初めて掩蔽中時刻を超えるタイミングの配列番号を取得
    occulted_time = int(np.where(raytrace_time > middle_time)[0][0])
    ingress_time_list = raytrace_freq.copy()
    Data = data

    # 各周波数で..
    for k in range(len(raytrace_freq)):
        # 電波が受かっている配列番号の配列（リスト）を取得
        over_judge_time_list = np.array(np.where(
            (Data[k][:] == 1)))
        # print(over_judge_time_list）

        # その中でも掩蔽中心時刻より手前にある配列番号を取得
        A = over_judge_time_list[over_judge_time_list < occulted_time]

        if len(A) > 0:
            a = A[len(A)-1]  # 電波がうかっている且つ掩蔽中心時刻より早い且つその中で一番遅いものの配列番号
            ingress_time_list = np.append(ingress_time_list, ((
                raytrace_time[a] + raytrace_time[a+1])/2))  # その時刻とその次の時刻の中心時刻を配列に追加

        else:
            ingress_time_list = np.append(ingress_time_list, 100000)  # 例外処理

    ingress_time_list = ingress_time_list.reshape(
        2, int(len(ingress_time_list)/2))
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
    raytrace_time = np.arange(0, time_step, 60)  # レイトレーシング計算した間隔での時間リスト（秒）

    # csvファイルから読み取ったstart時刻から掩蔽中心時刻までの差分（秒）を計算
    middle_time = (raytrace_time_information[8]-raytrace_time_information[2])*86400 + (raytrace_time_information[9] -
                                                                                       raytrace_time_information[4])*3600 + (raytrace_time_information[10]-raytrace_time_information[6]) * 60

    # レイトレーシングの周波数リスト(contour plotをする関係で配列の初めに本来はない周波数を挿入しているので周波数の数的には多くなているので注意)
    raytrace_freq = np.insert(np.array(Freq_num), 0, Freq_underline)

    # レイトレーシングの時間間隔の中で初めて掩蔽中時刻を超えるタイミングの配列番号を取得
    occulted_time = int(np.where(raytrace_time > middle_time)[0][0])
    egress_time_list = raytrace_freq.copy()
    Data = data

    for k in range(len(raytrace_freq)):
        # 電波が受かっている配列番号の配列（リスト）を取得
        over_judge_time_list = np.array(np.where(
            (Data[k][:] == 1)))
        # print(over_judge_time_list)
        # その中でも掩蔽中心時刻より後ろにある配列番号を取得
        B = over_judge_time_list[over_judge_time_list > occulted_time]

        if len(B) > 0:
            b = B[0]  # 電波がうかっている且つ掩蔽中心時刻より遅い且つその中で一番早いものの配列番号
            egress_time_list = np.append(
                egress_time_list, ((raytrace_time[b] + raytrace_time[b-1])/2))  # その時刻とその前の時刻の中心時刻を配列に追加

        else:
            egress_time_list = np.append(egress_time_list, -100000)

    egress_time_list = egress_time_list.reshape(
        2, int(len(egress_time_list)/2))

    # レイトレーシングデータにおける掩蔽終了時刻 [[周波数一覧][掩蔽終了時刻一覧]]の二次元データ　⇨保存（電子密度を変えるたびに異なる値になる・閾値は関係ない）
    return egress_time_list


class Evaluate_raytrace_data:

    def __init__(self, Data, time_data):
        self.data = Data
        self.ingress = ingress(Data, time_data)
        self.egress = egress(Data, time_data)


def idx_of_the_nearest(data, value):
    idx = np.argmin(np.abs(np.array(data) - value))
    return idx


def Evaluate_ionosphere_density(raytrace_data, galileo_data):
    """_レイトレーシングによる掩蔽タイミングとガリレオデータにおける掩蔽タイミングの差分を計算_

    Args:
        raytrace_data (_type_): _レイトレーシングにおける掩蔽タイミング [[周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ_
        galileo_data (_type_): _ 電波観測データにおける掩蔽タイミング [[周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ_

    Returns:
        _type_: _掩蔽タイミングの差分  [[周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ_
    """

    time_defference_index = raytrace_data.copy()
    num = raytrace_data.shape[1]  # 2（周波数・時間）✖️29(周波数の種類分)の２つ目の要素　⇨ 29

    # レイトレーシングの周波数種類ごとに
    for i in range(num):
        nearest_galileo_freq = idx_of_the_nearest(
            galileo_data[0][:], raytrace_data[0][i])  # レイトレーシングの周波数とガリレオ探査機の電波データの周波数リストで一番その差が小さい部分のインデックスを取得

        time_defference = abs(
            galileo_data[1][nearest_galileo_freq] - raytrace_data[1][i])  # 取得した周波数の電波データでの掩蔽開始・終了時間とレイトレーシングの時間の差を取る
        time_defference_index[1][i] = time_defference

    return time_defference_index


def Evaluate_data_coutour(time_data, radio_data_name):

    galileo_data_time, galileo_data_freq, galileo_radio_intensity = Prepare_Galileo_data(
        time_data, radio_data_name)
    using_galileo_data = galileo_radio_intensity[np.where(
        galileo_data_freq > 1e-1)][:].flatten()

    fig, ax = plt.subplots(1, 1)
    # ax.hist(using_galileo_data, range=(1e-18, 1e-12),bins=np.logspace(-18, -12, 30))
    ax.hist(using_galileo_data, bins=np.logspace(-17, -12, 50))
    ax.set_xscale('log')
    fig.savefig('A')

    return 0


# %%


def main():
    time_information, radio_data = Pick_up_cdf()
    # print(time_information, radio_data)
    # [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 座標変換した時のx(tangential point との水平方向の距離),6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度,8 探査機の経度]

    detectable_radio = np.loadtxt('../result_for_yasudaetal2022/raytracing_'+object_name+'_results/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight +
                                  '/' + object_name+'_'+spacecraft_name+'_'+str(time_of_flybies)+'_'+highest_plasma+'_'+plasma_scaleheight+'_dectable_radio_data.txt')

    detectable_A, detectable_B, detectable_C, detectable_D = Prepare_Figure(
        detectable_radio, time_information)

    ingress_time, egress_time = Make_FT_full(detectable_A, detectable_B,
                                             detectable_C, detectable_D, time_information, radio_data)

    Evaluate_data_coutour(time_information, radio_data)

    # PLOT 用

    detectable_data = [detectable_A, detectable_B,
                       detectable_C, detectable_D]

    detectable_data_str = ['dataA', 'dataB', 'dataC', 'dataD']

    for i in range(4):
        evaluated_data = Evaluate_raytrace_data(
            detectable_data[i], time_information)

        # レイトレーシングデータにおける掩蔽開始時刻 [[周波数一覧][掩蔽開始時刻一覧]]の二次元データ　⇨保存（電子密度を変えるたびに異なる値になる・閾値は関係ない）
        np.savetxt('../result_for_yasudaetal2022/raytracing_occultation_timing_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby/' + spacecraft_name +
                   '_' + object_name+'_'+str(time_of_flybies)+'_'+highest_plasma+'_'+plasma_scaleheight+'_ingress_'+detectable_data_str[i]+'_time_list.txt', evaluated_data.ingress)

        np.savetxt('../result_for_yasudaetal2022/raytracing_occultation_timing_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby/' + spacecraft_name +
                   '_' + object_name+'_'+str(time_of_flybies)+'_'+highest_plasma+'_'+plasma_scaleheight+'_egress_'+detectable_data_str[i]+'_time_list.txt', evaluated_data.egress)

        time_defference_ingress = Evaluate_ionosphere_density(
            evaluated_data.ingress, ingress_time)  # 掩蔽開始タイミングの差分  [[周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ

        time_defference_egress = Evaluate_ionosphere_density(
            evaluated_data.egress, egress_time)  # 掩蔽終了タイミングの差分  [[周波数一覧][掩蔽開始時刻一覧(秒)]]の二次元データ

        np.savetxt('../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_'+boundary_intensity_str+'/'+object_name+'_' +
                   highest_plasma+'_'+plasma_scaleheight+'_ingress_defference_time_'+detectable_data_str[i]+'_'+boundary_intensity_str+'.txt', time_defference_ingress)
        np.savetxt('../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_'+boundary_intensity_str+'/'+object_name+'_' +
                   highest_plasma+'_'+plasma_scaleheight+'_egress_defference_time_'+detectable_data_str[i]+'_'+boundary_intensity_str+'.txt', time_defference_egress)

    return 0


if __name__ == "__main__":
    main()


# %%
