# %%
from re import S
from socket import TIPC_MEDIUM_IMPORTANCE
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
highest_plasma = '4e2'  # 単位は(/cc) 2e2/4e2/16e22
plasma_scaleheight = '6e2'  # 単位は(km) 1.5e2/3e2/6e2

information_list = ['year', 'month', 'start_day', 'end_day',
                    'start_hour', 'end_hour', 'start_min', 'end_min']


Radio_name_cdf = '../result_for_yasudaetal2022/tracing_range_'+spacecraft_name+'_'+object_name + \
    '_'+str(time_of_flybies)+'_flybys/para_' + \
    highest_plasma+'_'+plasma_scaleheight+'.csv'
Radio_Range = pd.read_csv(Radio_name_cdf, header=0)
# [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 座標変換した時のx(tangential point との水平方向の距離),6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度]
Radio_observer_position = np.loadtxt('../result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/calculated_all_' +
                                     spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_Radio_data.txt')  # 電波源の経度を含む

Freq_str = ['3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]

Freq_num = []
for idx in Freq_str:
    Freq_num.append(float(idx)/1000000)

Highest = Radio_Range.highest
Lowest = Radio_Range.lowest
Except = Radio_Range.exc

n = len(Radio_observer_position)
total_radio_number = list(np.arange(n))


boundary_intensity_str = '7e-16'
# boundary_intensity_str = '1e-15'

boundary_intensity = float(boundary_intensity_str)

gal_rad_row = pd.read_csv(
    '../result_sgepss_2021/Survey_Electric_1996-06-27T05-30_1996-06-27T07-00.csv', header=None)

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


res = Radio_observer_position.copy()


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

    complete_selecred_flyby_list.index.tolist()[0]

    # csvから時刻データを抽出
    time_information = []
    for i in information_list:
        time_information.append(str(complete_selecred_flyby_list[i][0]))

    return time_information


def Time_step(time):
    day_range = int(time[3])-int(time[2])
    hour_range = int(time[5])-int(time[4])
    min_range = int(time[7])-int(time[6])

    step_count = day_range*1440*60 + hour_range*60*60 + min_range*60 + \
        1  # フライバイリストからステップ数を計算（今は1step1minを仮定してステップ数を計算）
    # フライバイリストのステップ数と位置データのステップ数が一致する確認（今は1step1minを仮定してステップ数を計算）

    return step_count


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
        spacecraft_csv_path, header=13, skipfooter=7)  # ヘッダーの位置と末端のデータの位置をheaderとskipfooterで設定

    return spacecraft_ephemeris_csv, time_information


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

    spacecraft_csv, t = Pick_up_spacecraft_csv
    spacecraft_longitude_rad = np.array(spacecraft_csv['Longitude (deg)'])

    for k in range(len(judgement)):
        Num = int(judgement[k][0]*60+judgement[k][1] -
                  time_information[4]*60-time_information[6])

        if np.abs(spacecraft_longitude_rad[Num][2]+360-judgement[k][7]) < np.abs(spacecraft_longitude_rad[Num][2]-judgement[k][7]):
            Lon = spacecraft_longitude_rad[Num][2]+360 - judgement[k][7]

        elif np.abs(judgement[k][7]+360-spacecraft_longitude_rad[Num][2]) < np.abs(judgement[k][7]-spacecraft_longitude_rad[Num][2]):
            Lon = spacecraft_longitude_rad[Num][2]-360 - judgement[k][7]

        else:
            Lon = spacecraft_longitude_rad[Num][2] - judgement[k][7]

        Lat = judgement[k][4]

        Fre = np.where(Freq_num == judgement[k][2])
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

# 修正ここから

# ガリレオ探査機の周波数一覧（Hz)とダウンロードした電波強度電波を代入（das2をcsvに変換）


def Prepare_Galileo_data(fleq_tag, rad_row_data):
    gal_fleq_tag = np.array(fleq_tag, dtype='float64')/1000000

    gal_time_tag_prepare = np.array(rad_row_data.iloc[0])
    gal_time_tag_prepare = gal_time_tag_prepare.astype(np.str)
    gal_time_tag = np.zeros(len(gal_time_tag_prepare))
    for i in range(len(gal_time_tag)):
        onesplit = np.char.split(np.char.split(
            gal_time_tag_prepare[:], sep='T')[i][1], sep=[':'])[0]
        onesplit2 = [float(vle) for vle in onesplit]
        gal_time_tag[i] = onesplit2[2] + onesplit2[1] * \
            60 + onesplit2[0]*3600  # 経過時間(sec)に変換

    df = pd.DataFrame(rad_row_data[:][1:])
    DDF = np.array(df).astype(np.float64)
    gal_time_tag = np.array(gal_time_tag-gal_time_tag[0])

    # ガリレオ探査機のデータ取得開始時刻からの経過時間（sec) , ガリレオ探査機のデータ取得周波数（MHz), ガリレオ探査機の取得した電波強度（代入したデータと同じ単位）
    return gal_time_tag, gal_fleq_tag, DDF


def Make_FT_full(DataA, DataB, DataC, DataD, raytrace_time):

    galileo_data_time, galileo_data_freq, galileo_radio_intensity = Prepare_Galileo_data(
        gal_fleq_tag_row, gal_rad_row)

    galileo_radio_intensity_row = galileo_radio_intensity.copy()

    galileo_radio_intensity[boundary_intensity
                            < galileo_radio_intensity] = 1
    galileo_radio_intensity[galileo_radio_intensity <
                            boundary_intensity] = 0
    # print(galileo_data_time.shape)
    # print(galileo_radio_intensity.shape)

    occulted_time = int(np.where(galileo_data_time > 2400)[0][0])
    ingress_time_list = galileo_data_freq.copy()
    egress_time_list = galileo_data_freq.copy()

    for k in range(len(galileo_data_freq)):
        over_judge_time_list = np.array(np.where(
            (galileo_radio_intensity[k][:] > boundary_intensity)))
        # print(over_judge_time_list)
        A = over_judge_time_list[over_judge_time_list < occulted_time]
        B = over_judge_time_list[over_judge_time_list > occulted_time]

        if len(B) > 0:
            b = B[0]
            egress_time_list = np.append(
                egress_time_list, ((galileo_data_time[b] + galileo_data_time[b-1])/2))

        else:
            egress_time_list = np.append(egress_time_list, -100000)

        if len(A) > 0:
            a = A[len(A)-1]
            ingress_time_list = np.append(ingress_time_list, ((
                galileo_data_time[a] + galileo_data_time[a+1])/2))

        else:
            ingress_time_list = np.append(ingress_time_list, 100000)

    ingress_time_list = ingress_time_list.reshape(
        2, int(len(ingress_time_list)/2))
    egress_time_list = egress_time_list.reshape(
        2, int(len(egress_time_list)/2))

    FREQ = np.insert(np.array(Freq_num), 0, 0.36122)

    x, y, DDF = Prepare_Galileo_data(gal_fleq_tag_row, gal_rad_row)
    xx, yy = np.meshgrid(x, y)

    fig, ax = plt.subplots(1, 1)

    plt.contour(xx, yy, galileo_radio_intensity,
                levels=[0.5], colors='red')

    pcm = ax.pcolor(xx, yy, DDF, norm=mpl.colors.LogNorm(
        vmin=1e-16, vmax=1e-12), cmap='Spectral_r')
    fig.colorbar(pcm, extend='max')
    plt.contour(raytrace_time, FREQ, DataA, levels=[0.5], colors='white')
    plt.contour(raytrace_time, FREQ, DataB, levels=[0.5], colors='lightgray')
    plt.contour(raytrace_time, FREQ, DataC, levels=[0.5], colors='darkgray')
    plt.contour(raytrace_time, FREQ, DataD, levels=[0.5], colors='black')
    plt.yscale("log")
    plt.ylim(100000, 5000000)
    plt.xlabel("Time of 27 June 1996")
    plt.ylabel("Frequency (MHz)")
    plt.yscale('log')
    plt.xlim(0, 5400)
    plt.ylim(0.1, 6.0)
    ax.set_xticks([0, 900, 1800, 2700, 3600, 4500, 5400])
    ax.set_xticklabels(
        ["05:30", "05:45", "06:00", "06:15", "06:30", "06:45", "07;00"])
    plt.title("max density "+highest_plasma +
              "(/cc) scale height "+plasma_scaleheight+"(km)")

    fig.savefig(os.path.join('../result_for_yasudaetal2022/' + object_name+'_'+highest_plasma +
                '_'+plasma_scaleheight+'/', object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'_'+boundary_intensity_str+'_f-t.png'))

    fig.savefig(os.path.join('../result_for_yasudaetal2022/f-t_plot_'+object_name+'/',
                object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'_'+boundary_intensity_str+'_f-t.png'))

    plt.show()

    fig, ax = plt.subplots(1, 1)

    plt.contour(xx, yy, galileo_radio_intensity,
                levels=[0.5], colors='red')

    pcm = ax.pcolor(xx, yy, DDF, norm=mpl.colors.LogNorm(
        vmin=1e-16, vmax=1e-12), cmap='Spectral_r')
    fig.colorbar(pcm, extend='max')
    plt.contour(raytrace_time, FREQ, DataA, levels=[0.5], colors='white')
    plt.contour(raytrace_time, FREQ, DataB, levels=[0.5], colors='lightgray')
    plt.contour(raytrace_time, FREQ, DataC, levels=[0.5], colors='darkgray')
    plt.contour(raytrace_time, FREQ, DataD, levels=[0.5], colors='black')
    plt.yscale("log")
    plt.xlabel("Time of 27 June 1996")
    plt.ylabel("Frequency (MHz)")
    plt.yscale('log')
    plt.xlim(2400, 3600)
    plt.ylim(0.1, 6.0)
    ax.set_xticks([2400, 3000, 3600])
    ax.set_xticklabels(["06:10", "06:20", "06:30"])
    plt.title("max density "+highest_plasma +
              "(/cc) scale height "+plasma_scaleheight+"(km)")
    fig.savefig(os.path.join('../result_for_yasudaetal2022/' + object_name+'_'+highest_plasma +
                '_'+plasma_scaleheight+'/', object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'_'+boundary_intensity_str+'_egress_f-t.png'))

    fig.savefig(os.path.join('../result_for_yasudaetal2022/f-t_plot_'+object_name+'_egress/',
                object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'_'+boundary_intensity_str+'_egress_f-t.png'))

    fig, ax = plt.subplots(1, 1)

    plt.contour(xx, yy, galileo_radio_intensity,
                levels=[0.5], colors='red')

    pcm = ax.pcolor(xx, yy, DDF, norm=mpl.colors.LogNorm(
        vmin=1e-16, vmax=1e-12), cmap='Spectral_r')
    fig.colorbar(pcm, extend='max')
    plt.contour(raytrace_time, FREQ, DataA, levels=[0.5], colors='white')
    plt.contour(raytrace_time, FREQ, DataB, levels=[0.5], colors='lightgray')
    plt.contour(raytrace_time, FREQ, DataC, levels=[0.5], colors='darkgray')
    plt.contour(raytrace_time, FREQ, DataD, levels=[0.5], colors='black')
    plt.yscale("log")
    plt.xlabel("Time of 27 June 1996")
    plt.ylabel("Frequency (MHz)")
    plt.yscale('log')
    plt.xlim(600, 2400)
    plt.ylim(0.1, 6.0)
    ax.set_xticks([600, 1200, 1800, 2400])
    ax.set_xticklabels(["05:40", "05:50", "06:00", "06:10"])
    plt.title("max density "+highest_plasma +
              "(/cc) scale height "+plasma_scaleheight+"(km)")

    fig.savefig(os.path.join('../result_for_yasudaetal2022/' + object_name+'_'+highest_plasma +
                '_'+plasma_scaleheight+'/', object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'_'+boundary_intensity_str+'_ingress_f-t.png'))

    fig.savefig(os.path.join('../result_for_yasudaetal2022/f-t_plot_'+object_name+'_ingress/',
                object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'_'+boundary_intensity_str+'_ingress_f-t.png'))

    return 0


def Evaluate_data_coutour():

    galileo_data_time, galileo_data_freq, galileo_radio_intensity = Prepare_Galileo_data(
        gal_fleq_tag_row, gal_rad_row)
    using_galileo_data = galileo_radio_intensity[np.where(
        galileo_data_freq > 1e-1)][:].flatten()

    plt.hist(using_galileo_data, range=(1e-18, 1e-12),
             bins=np.logspace(-18, -12, 30))
    plt.show()


def idx_of_the_nearest(data, value):
    idx = np.argmin(np.abs(np.array(data) - value))
    return idx


def Evaluate_galileo_data():

    galileo_data_time, galileo_data_freq, galileo_radio_intensity = Prepare_Galileo_data(
        gal_fleq_tag_row, gal_rad_row)

    galileo_radio_intensity_row = galileo_radio_intensity.copy()

    galileo_radio_intensity[boundary_intensity
                            < galileo_radio_intensity] = 1
    galileo_radio_intensity[galileo_radio_intensity <
                            boundary_intensity] = 0
    # print(galileo_data_time.shape)
    # print(galileo_radio_intensity.shape)

    occulted_time = int(np.where(galileo_data_time > 2400)[0][0])
    ingress_time_list = galileo_data_freq.copy()
    egress_time_list = galileo_data_freq.copy()

    for k in range(len(galileo_data_freq)):
        over_judge_time_list = np.array(np.where(
            (galileo_radio_intensity[k][:] > boundary_intensity)))
        # print(over_judge_time_list)
        A = over_judge_time_list[over_judge_time_list < occulted_time]
        B = over_judge_time_list[over_judge_time_list > occulted_time]

        if len(B) > 0:
            b = B[0]
            egress_time_list = np.append(
                egress_time_list, ((galileo_data_time[b] + galileo_data_time[b-1])/2))

        else:
            egress_time_list = np.append(egress_time_list, -100000)

        if len(A) > 0:
            a = A[len(A)-1]
            ingress_time_list = np.append(ingress_time_list, ((
                galileo_data_time[a] + galileo_data_time[a+1])/2))

        else:
            ingress_time_list = np.append(ingress_time_list, 100000)

    ingress_time_list = ingress_time_list.reshape(
        2, int(len(ingress_time_list)/2))
    egress_time_list = egress_time_list.reshape(
        2, int(len(egress_time_list)/2))

    np.savetxt('../result_for_yasudaetal2022/'+object_name +
               '_'+boundary_intensity_str+'_ingress_time_data.txt', ingress_time_list)
    np.savetxt('../result_for_yasudaetal2022/'+object_name +
               '_'+boundary_intensity_str+'_ingress_time_data.txt', egress_time_list)

    return ingress_time_list, egress_time_list


def ingress(data, raytrace_time):
    raytrace_freq = np.insert(np.array(Freq_num), 0, 0.36122)
    occulted_time = int(np.where(raytrace_time > 2400)[0][0])
    ingress_time_list = raytrace_freq.copy()
    Data = data

    for k in range(len(raytrace_freq)):
        over_judge_time_list = np.array(np.where(
            (Data[k][:] == 1)))
        # print(over_judge_time_list)
        A = over_judge_time_list[over_judge_time_list < occulted_time]

        if len(A) > 0:
            a = A[len(A)-1]
            ingress_time_list = np.append(ingress_time_list, ((
                raytrace_time[a] + raytrace_time[a+1])/2))

        else:
            ingress_time_list = np.append(ingress_time_list, 100000)

    ingress_time_list = ingress_time_list.reshape(
        2, int(len(ingress_time_list)/2))

    return ingress_time_list


def egress(data, raytrace_time):
    raytrace_freq = np.insert(np.array(Freq_num), 0, 0.36122)
    occulted_time = int(np.where(raytrace_time > 2400)[0][0])
    egress_time_list = raytrace_freq.copy()
    Data = data

    for k in range(len(raytrace_freq)):
        over_judge_time_list = np.array(np.where(
            (Data[k][:] == 1)))
        # print(over_judge_time_list)
        B = over_judge_time_list[over_judge_time_list > occulted_time]

        if len(B) > 0:
            b = B[0]
            egress_time_list = np.append(
                egress_time_list, ((raytrace_time[b] + raytrace_time[b-1])/2))

        else:
            egress_time_list = np.append(egress_time_list, -100000)

    egress_time_list = egress_time_list.reshape(
        2, int(len(egress_time_list)/2))

    return egress_time_list


class Evaluate_raytrace_data:

    def __init__(self, Data, time_data):
        self.data = Data
        self.ingress = ingress(Data, time_data)
        self.egress = egress(Data, time_data)


def Evaluate_ionosphere_density(raytrace_data, galileo_data):

    raytrace_freq = np.insert(np.array(Freq_num), 0, 0.36122)
    time_defference_index = raytrace_data.copy()
    num = raytrace_data.shape[len(raytrace_data)-1]
    for i in range(num):
        nearest_galileo_freq = idx_of_the_nearest(
            galileo_data[0][:], raytrace_data[0][i])
        time_defference = abs(
            galileo_data[1][nearest_galileo_freq] - raytrace_data[1][i])
        time_defference_index[1][i] = time_defference

    return time_defference_index


# %%


def main():
    time_information = Pick_up_cdf()
    time_step = Time_step(time_information)
    print(time_step)

    time_list = np.arange(0, time_step, 60)  # エクスプレスコードで計算している時間幅（sec)を60で割る

    detectable_radio = np.loadtxt('../result_for_yasudaetal2022/raytracing_'+object_name+'_results/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight +
                                  '/' + object_name+'_'+spacecraft_name+'_'+str(time_of_flybies)+'_'+highest_plasma+'_'+plasma_scaleheight+'_dectable_radio_data.txt')

    detectable_A, detectable_B, detectable_C, detectable_D = Prepare_Figure(
        detectable_radio, time_information)

    Make_FT_full(detectable_A, detectable_B,
                 detectable_C, detectable_D, time_list)

    Evaluate_data_coutour()

    ingress_time, egress_time = Evaluate_galileo_data()

    detectable_data = [detectable_A, detectable_B,
                       detectable_C, detectable_D]

    detectable_data_str = ['dataA', 'dataB', 'dataC', 'dataD']

    for i in range(4):
        evaluated_data = Evaluate_raytrace_data(detectable_data[i], time_list)
        time_defference_ingress = Evaluate_ionosphere_density(
            evaluated_data.ingress, ingress_time)

        print(time_defference_ingress)

        time_defference_egress = Evaluate_ionosphere_density(
            evaluated_data.egress, egress_time)

        print(time_defference_egress)

        np.savetxt('../result_for_yasudaetal2022/f-t_'+object_name+'_ingress_difference/'+object_name+'_' +
                   highest_plasma+'_'+plasma_scaleheight+'_ingress_defference_time_'+detectable_data_str[i]+'_'+boundary_intensity_str+'.txt', time_defference_ingress)
        np.savetxt('../result_for_yasudaetal2022/f-t_'+object_name+'_egress_difference/'+object_name+'_' +
                   highest_plasma+'_'+plasma_scaleheight+'_egress_defference_time_'+detectable_data_str[i]+'_'+boundary_intensity_str+'.txt', time_defference_egress)
    return 0


if __name__ == "__main__":
    main()


# %%
