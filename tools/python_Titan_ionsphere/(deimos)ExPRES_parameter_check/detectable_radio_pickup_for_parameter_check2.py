## 2025/06/08 二股に分かれるようなレイパスが生じる際に正しく、掩蔽境界線を計算できていないことが判明
## 2025/06/09 掩蔽境界線の計算を修正
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

# %%
# あらかじめ ../result_sgepss2021/~/~ に必要なレイトレーシング結果とパラメータセットを入れること
args = sys.argv
rng = np.random.default_rng()

object_name = "titan"  # ganydeme/europa/calisto
spacecraft_name = "cassini"  # galileo/JUICE(?)
time_of_flybies = 15  # ..th flyby


Source_latitude = 80
Source_beam_angle_range = [66, 67, 68, 69, 70, 71, 72, 73, 74, 75] # [65, 70, 75, 80, 85, 89]

Source_beam_angle = 75  # beam angle of the radio source
Source_latitude_range = [75]  # latitude range of the radio source #[60, 65, 70, 75, 80, 85, 89]

# highest_plasma = args[1]  # 単位は(/cc) 2e2/4e2/16e22 #12.5 13.5
# peak_altitude = args[2]  # 単位は(km) 1.5e2/3e2/6e2
ignore = 0  # 1のときは既存のファイルを無視して新たに計算


result_path = "/work1/rikutoyasuda/tools/result_titan/"


# europa & ganymede
if object_name == "titan":
    if time_of_flybies == 15:
        #Freq_str = ['51242.6', '53708.6', '56293.3', '59002.4', '61841.8', '64817.9', '67937.3', '71206.7', '74633.5', '78225.2', '81989.8', '85935.5', '90071.1', '94405.7', '98949', '103710.9', '108701.9', '113933.1', '119416.1', '125162.9', '131186.4', '137499.6', '144116.7', '151052.31', '158321.59', '165940.8', '173926.61', '182296.71', '191069.7', '200264.8', '209902.5', '220003.91', '230591.51', '241688.6', '253319.79', '265510.71', '278288.3', '291680.79', '305717.8', '320430.3', '318750', '331250', '343750', '356250', '368750', '381250', '393750', '406250', '418750', '431250', '443750', '456250', '468750', '481250', '493750', '506250', '518750', '531250', '543750', '556250', '568750', '581250', '593750', '606250', '618750', '631250', '643750', '656250', '668750', '681250', '693750', '706250', '718750', '731250', '743750', '756250', '768750', '781250', '793750', '806250', '818750', '831250', '843750', '856250', '868750', '881250', '893750', '906250', '918750', '931250', '943750', '956250', '968750', '981250', '993750']
        Freq_str = ['90071', '113930', '173930', '241690', '278290', '320430', '368750', '431250', '468750', '531250', '568750', '631250', '668750', '731250', '768750', '831250',  '868750',  '931250', '968750']
#
        judge_start_time = np.array([9.0, 1.0])  # 9:01 前は全て受信可能判定
        judge_end_time = np.array([10.0, 30.0])  # 10:30 後は全て受信可能判定
    

Freq_num = []
for idx in Freq_str:
    Freq_num.append(float(idx) / 1000000)



# %%

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

def Judgemet_array(data):
    before_nonjudge_array = np.intersect1d(
        np.where(data[:, 0] <= judge_start_time[0]),
        np.where(data[:, 1] < judge_start_time[1]),
    )

    after_nonjudge_array = np.intersect1d(
        np.where(data[:, 0] >= judge_end_time[0]),
        np.where(data[:, 1] > judge_end_time[1]),
    )

    if len(before_nonjudge_array) == 0:
        before_nonjudge_number = 0

    else:
        before_nonjudge_number = before_nonjudge_array[-1]

    if len(after_nonjudge_array) == 0:
        after_nonjudge_number = len(data)

    else:
        after_nonjudge_number = after_nonjudge_array[0]

    judge_time_array = np.intersect1d(
        np.arange(before_nonjudge_number, after_nonjudge_number, 1),
        np.where(data[:, 7] >= 0)[0],
    )

    no_judge_detectable_time_array = np.concatenate(
        [
            np.arange(0, before_nonjudge_number, 1),
            np.arange(after_nonjudge_number, len(data), 1),
        ],
        0,
    )

    no_judge_undetectable_time_array = np.intersect1d(
        np.arange(before_nonjudge_number, after_nonjudge_number, 1),
        np.where(data[:, 7] < 0)[0],
    )

    return (
        judge_time_array,
        no_judge_detectable_time_array,
        no_judge_undetectable_time_array,
    )



def Judge_occultation(data):
    # Radio_observation_position
    # new [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]
    # aa  受かるときは1 受からないとき0を出力
    i, Radio_observer_position = data
    aa = 0
    #detectable_obsever_position_x = Radio_observer_position[i][6]
    detectable_obsever_position_z = Radio_observer_position[i][7]
    #detectable_frequency = Radio_observer_position[i][3]  # 使うレイの周波数を取得
    # レイの周波数と周波数リスト（Freq＿num）の値が一致する場所を取得　周波数リスト（Freq＿num）とcsvファイルの週数リストが一致しているのでそこからその周波数における電波源の幅を取得

    if detectable_obsever_position_z > 0:
        aa = 1

    return aa


def Replace_Save(
    judgement, befrore_judged_array, all_radio_data, expected_detectable_array, source_latitude, source_beam_angle
):
    # 想定した電子密度分布で観測可能な電波のリスト
    # [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]
    # を想定した電子密度でのレイトレーシング結果が保存されているフォルダに保存
    occultaion_aray = np.array(judgement)
    judge_array = befrore_judged_array[np.where(occultaion_aray[:] == 1)[0]]

    detectable_array = np.sort(
        np.concatenate(
            [judge_array, expected_detectable_array],
            0,
        )
    )
    # np.savetxt("./detectable_array.txt", detectable_array)
    all_detectable_radio = all_radio_data[detectable_array][:]

    np.savetxt(
        result_path
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
        + "_dectable_radio_data.txt",
        all_detectable_radio,
    )

    return all_detectable_radio


def lineNotify(message):
    line_notify_token = "MFCL4nEMoT0m9IyjUXLeVsoePNXCfbAInnBs7tZeGts"
    line_notify_api = "https://notify-api.line.me/api/notify"
    payload = {"message": message}
    headers = {"Authorization": "Bearer " + line_notify_token}
    requests.post(line_notify_api, data=payload, headers=headers)


def main():

    for source_latitude in Source_latitude_range:
        source_beam_angle = Source_beam_angle

        Radio_observer_position = np.loadtxt(
        result_path + "ExPRES_parameter_check/Interpolated_all_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_sourcelat_"
        + str(source_latitude)
        + "_sourcebeamangle_"
        + str(source_beam_angle)
        + "_Radio_data.txt"
        )  # 電波源の経度を含む

        freq_num_list = list(np.arange(len(Freq_num)))
        
        if Radio_observer_position.shape[0] != 0:
            (
                judge_array,
                no_judge_detectable_array,
                no_judge_undetectable_array,
            ) = Judgemet_array(Radio_observer_position)

            total_radio_number = list(judge_array)
            input_data =[(x,Radio_observer_position) for x in total_radio_number]

            with Pool(processes=30) as pool:
                result_list = list(pool.map(Judge_occultation, input_data))

            # 受かっている電波のみを保存
            Replace_Save(
                result_list,
                judge_array,
                Radio_observer_position,
                no_judge_detectable_array,
                source_latitude, 
                source_beam_angle 
            )

    for source_beam_angle in Source_beam_angle_range:
        source_latitude = Source_latitude

        Radio_observer_position = np.loadtxt(
        result_path + "ExPRES_parameter_check/Interpolated_all_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_sourcelat_"
        + str(source_latitude)
        + "_sourcebeamangle_"
        + str(source_beam_angle)
        + "_Radio_data.txt"
        )  # 電波源の経度を含む

        freq_num_list = list(np.arange(len(Freq_num)))
        
        if Radio_observer_position.shape[0] != 0:

            (
                judge_array,
                no_judge_detectable_array,
                no_judge_undetectable_array,
            ) = Judgemet_array(Radio_observer_position)

            total_radio_number = list(judge_array)
            input_data =[(x,Radio_observer_position) for x in total_radio_number]

            with Pool(processes=30) as pool:
                result_list = list(pool.map(Judge_occultation, input_data))

            # 受かっている電波のみを保存
            Replace_Save(
                result_list,
                judge_array,
                Radio_observer_position,
                no_judge_detectable_array,
                source_latitude, 
                source_beam_angle 
            )
        slackNotify("ExPRSS check 2 end")

    return 0


if __name__ == "__main__":
    main()

# %%
