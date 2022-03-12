# %%
import numpy as np
import math
import pandas as pd
import re

object_name = "europa"  # europa/ganymde/callisto
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 14  # ..th flyby
information_list = ['year', 'month', 'start_day', 'end_day',
                    'start_hour', 'end_hour', 'start_min', 'end_min']

# %%


def Pick_up_spacecraft_csv():

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
    csv_name = str(complete_selecred_flyby_list['spacecraft_ephemeris_csv'][0])

    # csvから時刻データを抽出
    time_information = []
    for i in information_list:
        time_information.append(str(complete_selecred_flyby_list[i][0]))

    spacecraft_csv_path = '../result_for_yasudaetal2022/spacecraft_ephemeris/' + csv_name
    spacecraft_ephemeris_csv = pd.read_csv(
        spacecraft_csv_path, header=13, skipfooter=7)

    return spacecraft_ephemeris_csv, time_information


def Pick_up_moon_csv():

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
    csv_name = str(complete_selecred_flyby_list['spacecraft_ephemeris_csv'][0])

    # csvから時刻データを抽出
    time_information = []
    for i in information_list:
        time_information.append(str(complete_selecred_flyby_list[i][0]))

    spacecraft_csv_path = '../result_for_yasudaetal2022/spacecraft_ephemeris/' + csv_name
    spacecraft_ephemeris_csv = pd.read_csv(
        spacecraft_csv_path, header=13, skipfooter=7)

    return spacecraft_ephemeris_csv, time_information


def Check_time_validity(time, csv_data):
    first_str = str(csv_data['UTC calendar date'][0])
    last_str = str(csv_data['UTC calendar date'][-1:])

    if int(re.findall(r'\d+', first_str)[0]) != int(time[0]) or int(re.findall(r'\d+', first_str)[1]) != int(time[1]) or int(re.findall(r'\d+', first_str)[2]) != int(time[2]) or int(re.findall(r'\d+', first_str)[3]) != int(time[4]) or int(re.findall(r'\d+', first_str)[4]) != int(time[6]):
        print("wrong time!!!!!")

    if int(re.findall(r'\d+', last_str)[3]) != int(time[3]) or int(re.findall(r'\d+', last_str)[4]) != int(time[5]) or int(re.findall(r'\d+', last_str)[5]) != int(time[7]):
        print(int(re.findall(r'\d+', first_str)[0]))
        print("wrong time!!!!!")


def main():
    spacecraft_epemeris, time = Pick_up_spacecraft_csv()
    Check_time_validity(time, spacecraft_epemeris)

    return 0


if __name__ == "__main__":
    main()


# %%
Rdo = np.loadtxt(
    '/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/All_Gnymede_Radio_data.txt')  # ???

Rdo[:, 12:15] = Rdo[:, 12:15]*71492

GG = np.loadtxt(
    '/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/GLL_GAN_2.txt')

# %%

GG[:, 2] = np.radians(GG[:, 2])  # gallireo position
GG[:, 3] = np.radians(GG[:, 3])
GG[:, 5] = np.radians(GG[:, 5])  # ganymede position
GG[:, 6] = np.radians(GG[:, 6])

G = np.zeros(GG.shape)

G[:, 0] = GG[:, 0]
G[:, 1] = GG[:, 1]
G[:, 2] = GG[:, 4]*np.cos(GG[:, 2])*np.cos(GG[:, 3])  # gallireo position
G[:, 3] = GG[:, 4]*np.sin(GG[:, 2])*np.cos(GG[:, 3])
G[:, 4] = GG[:, 4]*np.sin(GG[:, 3])

G[:, 5] = GG[:, 7]*np.cos(GG[:, 5])*np.cos(GG[:, 6])  # ganymede position
G[:, 6] = GG[:, 7]*np.sin(GG[:, 5])*np.cos(GG[:, 6])
G[:, 7] = GG[:, 7]*np.sin(GG[:, 6])

lr = len(Rdo)
res = np.zeros((len(Rdo), 8))
# %%
Ghour = G[:, 0].copy()
Gmin = G[:, 1].copy()

Xmax = 0
Ymax = 0

for i in range(lr):
    ex = np.array([0.0, 0, 0])
    ey = np.array([0.0, 0, 0])
    r1 = np.array([0.0, 0, 0])
    r2 = np.array([0.0, 0, 0])
    r3 = np.array([0.0, 0, 0])
    rc1 = np.array([0.0, 0, 0])
    rc2 = np.array([0.0, 0, 0])

    res[i][0] = Rdo[i][3].copy()
    res[i][1] = Rdo[i][4].copy()
    res[i][2] = Rdo[i][9].copy()
    res[i][3] = Rdo[i][10].copy()
    res[i][4] = Rdo[i][11].copy()
    res[i][7] = math.degrees(math.atan2(Rdo[i][13], Rdo[i][12]))
    Glow = np.intersect1d(
        np.where(Ghour == Rdo[i][3]), np.where(Gmin == Rdo[i][4]))
    glow = int(Glow)
    r1[0:3] = Rdo[i][12:15]
    r2[0:3] = G[glow, 5:8]  # ganymede position
    r3[0:3] = G[glow, 2:5]  # gallireo position

    ex = (r3-r1) / np.linalg.norm(r3-r1)
    rc1 = np.cross((r2-r1), (r3-r1))
    rc2 = np.cross((rc1 / np.linalg.norm(rc1)), ex)
    ey = rc2 / np.linalg.norm(rc2)

    res[i][6] = np.dot((r3-r2), ey) - 2634.1
    res[i][5] = np.dot((r3-r2), ex)

    if res[i][5] > 0:
        if res[i][5] > Xmax:
            Xmax = res[i][5]

        if res[i][6] > Ymax:
            Ymax = res[i][6]

print(Xmax, Ymax)
"""
np.savetxt(
    '/Users/yasudarikuto/research/icymoon_raytracing/tools/result_for_yasudaetal2022/R_P_fulldata.txt', res)
"""

# %%
