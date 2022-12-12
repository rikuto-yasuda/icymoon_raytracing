from calendar import month
import pprint
import cdflib
import numpy as np
import pandas as pd
import re
import math

# %%

object_name = "callisto"  # europa/ganymde/callisto
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 9  # ..th flyby
information_list = ['year', 'month', 'start_day', 'end_day',
                    'start_hour', 'end_hour', 'start_min', 'end_min']
"""
sun_csv = "../result_for_yasudaetal2022/ephemeris_data_for_dayside_plot/WGC_StateVector_20220914193052.csv"  # c30 flyby
jupiter_csv = "../result_for_yasudaetal2022/ephemeris_data_for_dayside_plot/WGC_StateVector_20220914193145.csv"  # c30 flyby

sun_csv = "../result_for_yasudaetal2022/ephemeris_data_for_dayside_plot/WGC_StateVector_20220914201541.csv"  # g1 flyby
jupiter_csv = "../result_for_yasudaetal2022/ephemeris_data_for_dayside_plot/WGC_StateVector_20220914201856.csv"  # g1 flyby
"""

sun_csv = "../result_for_yasudaetal2022/ephemeris_data_for_dayside_plot/WGC_StateVector_20221130183102.csv"  # c9 flyby
jupiter_csv = "../result_for_yasudaetal2022/ephemeris_data_for_dayside_plot/WGC_StateVector_20221130182522.csv"  # c9 flyby

# 計算で使うcdfファイルを選定


def Pick_up_csv():

    flyby_list_path = '../result_for_yasudaetal2022/occultation_flyby_list.csv'
    flyby_list = pd.read_csv(flyby_list_path)

    # csvファイルにフライバイごとで使う軌道データを記入しておく　上記のパラメータから必要なデータのファイル名が選ばれて読み込まれる
    # queryが数値非対応なのでまずはフライバイ数で絞り込み
    selected_flyby_list = flyby_list[flyby_list['flyby_time']
                                     == time_of_flybies]
    complete_selecred_flyby_list = selected_flyby_list.query(
        'object == "'+object_name+'" & spacecraft == "'+spacecraft_name+'"')  # queryでフライバイ数以外を絞り込み
    complete_selecred_flyby_list = complete_selecred_flyby_list.reset_index(
        drop=True)  # index振り直し
    # 使うcsvファイルの名前を取得
    csv_name = str(complete_selecred_flyby_list['moon_ephemeris_csv'][0])

    # csvから時刻データを抽出
    time_information = []
    for i in information_list:
        time_information.append(str(complete_selecred_flyby_list[i][0]))

    # ヘッダーの位置と末端のデータの位置をheaderとskipfooterで設定
    sun_ephemeris_csv = pd.read_csv(sun_csv, header=17, skipfooter=4)
    # ヘッダーの位置と末端のデータの位置をheaderとskipfooterで設定
    jupiter_ephemeris_csv = pd.read_csv(jupiter_csv, header=17, skipfooter=4)

    return sun_ephemeris_csv, jupiter_ephemeris_csv, time_information


def Check_time_validity_csv(time, sun_csv_data, jov_csv_data):

    # 太陽の位置データの時間範囲・フライバイリストで指定している時刻データ範囲が一致するか確認
    Check_time_range_validity(time, sun_csv_data)
    # 木星の位置データの時間範囲・フライバイリストで指定している時刻データ範囲が一致するか確認
    Check_time_range_validity(time, jov_csv_data)
    # 太陽の位置データの時間ステップ数・フライバイリストで指定している時刻からか計算されるステップ数が一致するか確認
    Check_time_step_validity(time, sun_csv_data)
    # 木星の位置データの時間ステップ数・フライバイリストで指定している時刻からか計算されるステップ数が一致するか確認
    Check_time_step_validity(time, jov_csv_data)


def Check_time_range_validity(time, csv_data):

    first_str = str(csv_data['UTC calendar date'][0])  # 位置データにおける初めの時刻の文字列
    last_str = str(csv_data['UTC calendar date'][-1:])  # 位置データにおける最後の時刻の文字列

    # 位置データにおける初めの時刻の文字列から時刻データを抽出　フライバイリストの情報と一致しているか確認
    if int(re.findall(r'\d+', first_str)[0]) != int(time[0]) or int(re.findall(r'\d+', first_str)[1]) != int(time[1]) or int(re.findall(r'\d+', first_str)[2]) != int(time[2]) or int(re.findall(r'\d+', first_str)[3]) != int(time[4]) or int(re.findall(r'\d+', first_str)[4]) != int(time[6]):
        print("wrong time!!!!!")

    # 位置データにおける最後の時刻の文字列から時刻データを抽出　フライバイリストの情報と一致しているか確認
    if int(re.findall(r'\d+', last_str)[3]) != int(time[3]) or int(re.findall(r'\d+', last_str)[4]) != int(time[5]) or int(re.findall(r'\d+', last_str)[5]) != int(time[7]):
        print(int(re.findall(r'\d+', first_str)[0]))
        print("wrong time!!!!!")

    else:
        print("data range is correct")


def Check_time_step_validity(time, csv_data):
    day_range = int(time[3])-int(time[2])
    hour_range = int(time[5])-int(time[4])
    min_range = int(time[7])-int(time[6])

    step_count = day_range*1440 + hour_range*60 + min_range + \
        1  # フライバイリストからステップ数を計算（今は1step1minを仮定してステップ数を計算）
    # フライバイリストのステップ数と位置データのステップ数が一致する確認（今は1step1minを仮定してステップ数を計算）
    if step_count == len(csv_data['UTC calendar date'][:]):
        print("data step is correct")
    else:
        print("wrong time step")


def calc_logitude(sun_csv, jupiter_csv):

    time_step_str = np.empty((0, 7), int)
    # 'UTC calendar date'の一行ごとに数字要素だけを抽出&挿入
    for i in range(len(sun_csv['UTC calendar date'])):
        time_step_str = np.append(time_step_str, np.array(
            [re.findall(r'\d+', sun_csv['UTC calendar date'][i])]), axis=0)

    # 各要素を文字データから数値データに [year,month,day,hour,min,sec,0]の7次元データの集まり
    time_step = time_step_str.astype(np.int32)

    # 経度を読み込み 座標系の+xから＋yになる方向を経度が増える方向と定義（太陽と木星の衛星から見た経度を計算）
    sun_longitude_deg = np.array(
        sun_csv['Longitude (deg)'])

    reshape_sun_logitude_deg = sun_longitude_deg.reshape(
        [len(sun_longitude_deg), 1])

    jupiter_longitude_deg = np.array(
        jupiter_csv['Longitude (deg)'])

    reshape_jupiter_logitude_deg = jupiter_longitude_deg.reshape(
        [len(jupiter_longitude_deg), 1])

    # 軽度の違いから木星方向を0度、leadingを90度とした経度を計算
    def_longitude = (360 - (reshape_sun_logitude_deg -
                            reshape_jupiter_logitude_deg)) % 360

    # 時刻データと位置データを結合[year,month,day,hour,min,sec,0,経度]の8次元データの集まり
    time_and_position = np.hstack(
        (time_step, def_longitude))

    return time_and_position


def main():

    sun_ephemeris, jupiter_ephemeris, time = Pick_up_csv()
    # 探査機の位置データの時間・月の位置データの時間・フライバイリストで指定している時刻データが一致するか確認
    Check_time_validity_csv(time, sun_ephemeris, jupiter_ephemeris)
    longitude_result = calc_logitude(sun_ephemeris, jupiter_ephemeris)

    print(longitude_result.shape)
    print(np.max(longitude_result, axis=0)[7])
    print(np.min(longitude_result, axis=0)[7])

    return 0


if __name__ == "__main__":
    main()
