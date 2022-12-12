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
# 下の方に使う周波数と時刻を入れるとこがあるので注意

object_name = 'callisto'  # ganydeme/europa/calisto``
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 9  # ..th flyby

radio_type = "D"  # 'A' or 'B' or 'C' or 'D'

maximum_subsolar_long = 83.93602662
minimum_subsolar_long = 81.22316967

information_list = ['year', 'month', 'start_day', 'end_day',
                    'start_hour', 'end_hour', 'start_min', 'end_min', 'occultaton_center_day', 'occultaton_center_hour', 'occultaton_center_min']


Tangential_point_txt = '../result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/calculated_all_' + \
    spacecraft_name+'_'+object_name+'_' + \
    str(time_of_flybies)+'_tangential_point.txt'

Tangential_point = np.genfromtxt(Tangential_point_txt)

# [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 tangential pointでの衛星経度,6 tangential pointでの衛星緯度,7 tangential pointから探査機方向に伸ばした時の衛星経度,8 tangential pointから探査機方向に伸ばした時の衛星緯度, 9 電波源の実際の経度,10 探査機の経度, 11 z座標]

"""
Freq_str = ['3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]

# europa & ganymede
"""

Freq_str = ['3.612176179885864258e5', '3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]
# callisto

Freq_num = []
for idx in Freq_str:
    Freq_num.append(float(idx)/1000000)


# %%

def Pick_up_cdf():
    flyby_list_path = '../result_for_yasudaetal2022/occultation_flyby_list.csv'
    flyby_list = pd.read_csv(flyby_list_path)

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
    # judgement [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 tangential pointでの衛星経度,6 tangential pointでの衛星緯度,7 tangential pointから探査機方向に伸ばした時の衛星経度,8 tangential pointから探査機方向に伸ばした時の衛星緯度, 9 電波源の実際の経度,10 探査機の経度, 11 z]

    selected_data = np.zeros_like(judgement)
    for k in range(len(judgement)):

        Num = int(judgement[k][0]*60+judgement[k][1]) - \
            (time_information[4]*60+time_information[6])

        if np.abs(judgement[k][10]+360-judgement[k][9]) < np.abs(judgement[k][10]-judgement[k][9]):
            Lon = judgement[k][10]+360 - judgement[k][9]

        elif np.abs(judgement[k][9]+360-judgement[k][10]) < np.abs(judgement[k][9]-judgement[k][10]):
            Lon = judgement[k][10]-360 - judgement[k][9]

        else:
            Lon = judgement[k][10] - judgement[k][9]

        Lat = judgement[k][4]

        Fre = np.where(Freq_num == judgement[k][2])

        if radio_type == 'A':
            if Lon < 0 and Lat > 0:
                selected_data[k, :] = judgement[k, :].copy()

        if radio_type == 'B':
            if Lon > 0 and Lat > 0:
                selected_data[k, :] = judgement[k, :].copy()

        if radio_type == 'C':
            if Lon < 0 and Lat < 0:
                selected_data[k, :] = judgement[k, :].copy()

        if radio_type == 'D':
            if Lon > 0 and Lat < 0:
                selected_data[k, :] = judgement[k, :].copy()

    print("complete")
    selected_data = selected_data[np.all(selected_data != 0, axis=1), :]
    print(selected_data)

    np.savetxt('../result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/calculated_all_' +
               spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_'+radio_type+'_tangential_point.txt', selected_data, fmt="%s")

    selected_data = selected_data[np.all(selected_data != 0, axis=1), :]

    ##############

    hour_select = np.where(selected_data[:, 0] == 13)

    selected_hour_data = (selected_data[hour_select, :])[0]

    minute_select = np.where(
        (40 < selected_hour_data[:, 1]) & (selected_hour_data[:, 1] < 46))

    selected_time_data = (selected_hour_data[minute_select, :])[0]
    print(hour_select)

    # using_frequency_range
    # C6 egress D [0.45, 6]

    freq_select = np.where(
        (5.3e-1 < selected_time_data[:, 2]) & (selected_time_data[:, 2] < 5.5))

    selected_freq_data = (selected_time_data[freq_select, :])[0]

    detectable_select = np.where(selected_freq_data[:, 11] > 0)
    selected_detectable_data = (selected_freq_data[detectable_select, :])[0]

    ax = plt.subplot(111, projection="polar")
    x = np.deg2rad(selected_detectable_data[:, 5])
    y = selected_detectable_data[:, 6]*-1
    plt.ylim([0, 90])
    ax.scatter(x, y, s=1)
    print(x, y)
    ax.invert_yaxis()
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_rgrids(np.arange(0, 90, 10), labels=[
                  "", "", "", "", "", "", "", "", ""])
    ax.set_thetagrids(np.arange(0, 360, 30),
                      labels=["0 (Jovian)", "", "", "90 \n      (Leading)", "", "",
                              "180  (Anti-Jovian)", "", "", "270 \n(Trailing)     ", "", ""])

    ax.set_title('Egress (North)')
    ax.vlines(np.deg2rad(maximum_subsolar_long+90),
              0, 90, color='0.5', linestyles='solid')
    ax.vlines(np.deg2rad(maximum_subsolar_long-90),
              0, 90, color='0.5', linestyles='solid')
    ax.vlines(np.deg2rad(minimum_subsolar_long+90),
              0, 90, color='0.5', linestyles='solid')
    ax.vlines(np.deg2rad(minimum_subsolar_long-90),
              0, 90, color='0.5', linestyles='solid')

    ax.text(np.deg2rad(maximum_subsolar_long), 80, "day side", color="orangered",
            fontfamily="serif", fontweight="bold", fontstyle="italic", fontsize=10, horizontalalignment="center")
    ax.text(np.deg2rad(maximum_subsolar_long+180), 80, "night side", color="navy",
            fontfamily="serif", fontweight="bold", fontstyle="italic", fontsize=10, horizontalalignment="center")

    plt.savefig(os.path.join('../result_for_yasudaetal2022/plot_tangential_point/', spacecraft_name +
                             '_' + object_name+'_'+str(time_of_flybies)+'_'+radio_type+'_Ingress_tangential.png'))

    plt.show()

    return selected_data

# ガリレオ探査機の周波数一覧（Hz)とダウンロードした電波強度電波を代入（das2をcsvに変換）


def Make_FT_full(Data, raytrace_time_information):

    return


# %%
def main():
    time_information, radio_data = Pick_up_cdf()
    print(time_information, radio_data)

    Tangential_point = np.genfromtxt(Tangential_point_txt)

    selected_data = Prepare_Figure(
        Tangential_point, time_information)

    Make_FT_full(selected_data, time_information)

    return 0


if __name__ == "__main__":
    main()


# %%
