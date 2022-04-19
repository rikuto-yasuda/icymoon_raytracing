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

object_name = 'europa'  # ganydeme/europa/calisto``
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 12  # ..th flyby
highest_plasma = '10e2'  # 単位は(/cc) 2e2/4e2/16e22
plasma_scaleheight = '3e2'  # 単位は(km) 1.5e2/3e2/6e2


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

# %%


def MakeFolder():
    os.makedirs('../result_for_yasudaetal2022/raytracing_'+object_name+'_results/' + object_name +
                '_'+highest_plasma+'_'+plasma_scaleheight)  # レイトレーシングの結果を格納するフォルダを生成


def MoveFile():

    for l in range(len(Freq_num)):
        for j in range(Lowest[l], Highest[l], 2):
            k = str(j)
            os.replace('../../raytrace.tohoku/src/rtc/testing/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                       k+'-FR'+Freq_str[l], '../result_for_yasudaetal2022/raytracing_'+object_name+'_results/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' + k+'-FR'+Freq_str[l])


def Judge_occultation(i):
    # Radio_observation_positionh[0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 座標変換した時のx(tangential point との水平方向の距離),6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度]
    # aa  受かるときは1 受からないとき0を出力

    print(i)

    aa = 0
    detectable_obsever_position_x = Radio_observer_position[i][5]
    detectable_obsever_position_z = Radio_observer_position[i][6]

    if detectable_obsever_position_z > 0 or detectable_obsever_position_x < 0:  # z座標が0以下のものは受かる可能性がレイパスを見る必要もなし
        detectable_frequency = Radio_observer_position[i][2]  # 使うレイの周波数を取得
        # レイの周波数と周波数リスト（Freq＿num）の値が一致する場所を取得　周波数リスト（Freq＿num）とcsvファイルの週数リストが一致しているのでそこからその周波数における電波源の幅を取得
        freq = int(np.where(Freq_num == detectable_frequency)[0])

        for j in range(Lowest[freq], Highest[freq], 2):
            k = str(j)
            # 高度が低い電波源の電波から取得
            Radio_propagation_route = np.genfromtxt("../result_for_yasudaetal2022/raytracing_"+object_name+"_results/"+object_name+"_"+highest_plasma+"_"+plasma_scaleheight +
                                                    "/ray-P"+object_name+"_nonplume_"+highest_plasma+"_"+plasma_scaleheight+"-Mtest_simple-benchmark-LO-Z"+k+"-FR"+Freq_str[freq])
            # たまにレイパスが計算できない初期条件になって入りう時があるのでそのデータを除外
            if Radio_propagation_route.ndim == 2:
                # 電波の出発点が探査機の高度より高くなった場合それ以上の高度では電波は受からないのでfor文の外へ

                detectable_x_index = np.where(
                    Radio_propagation_route[:, 1] > detectable_obsever_position_x)[0]

                detectable_z_index = np.where(
                    Radio_propagation_route[:, 3] < detectable_obsever_position_z)[0]

                detectable_x_index_minus = detectable_x_index - 1

                # レイが探査機のx座標を超えても探査機よりも低高度にレイパスがある→その電波は受かる
                if len(np.intersect1d(detectable_x_index, detectable_z_index)) != 0:
                    aa = 1
                    break

                # レイが探査機のx座標を超える前に探査機よりも上空に行ってしまっている→その電波は受からない
                if len(np.intersect1d(detectable_x_index_minus, detectable_z_index)) == 0:
                    continue

                # 以下、ギリギリだったときに例が探査機の下にあるか上にあるかを判別する部分

                if len(np.intersect1d(detectable_x_index_minus, detectable_z_index)) > 1:
                    print("error")

                else:
                    h = int(np.intersect1d(
                        detectable_x_index_minus, detectable_z_index)[0])+1

                    para = np.abs(
                        Radio_propagation_route[h][1]-Radio_observer_position[i][5])
                    hight = Radio_propagation_route[h][3] - \
                        Radio_observer_position[i][6]
                    x1 = Radio_propagation_route[h][1]
                    z1 = Radio_propagation_route[h][3]
                    x2 = Radio_propagation_route[h-1][1]
                    z2 = Radio_propagation_route[h-1][3]

                    while (para > 10):
                        ddx = (x1+x2)/2
                        ddz = (z1+z2)/2

                        if ddx > Radio_observer_position[i][5]:
                            x1 = ddx
                            z1 = ddz
                        else:
                            x2 = ddx
                            z2 = ddz

                        para = np.abs(
                            x1-Radio_observer_position[i][5])
                        hight = z1-Radio_observer_position[i][6]

                    if hight < 0:
                        # res[i][6]=0
                        aa = 1
                        break

                if Radio_propagation_route[0][3] > detectable_obsever_position_z:
                    break
    return aa


def Replace_Save(judgement, all_radio_data):
    # 想定した電子密度分布で観測可能な電波のリスト
    # [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 座標変換した時のx(tangential point との水平方向の距離),6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度]
    # を想定した電子密度でのレイトレーシング結果が保存されているフォルダに保存
    occultaion_aray = np.array(judgement)
    judge_array = np.where(occultaion_aray[:] == 1)
    all_detectable_radio = all_radio_data[judge_array][:]
    np.savetxt('../result_for_yasudaetal2022/raytracing_'+object_name+'_results/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'/' +
               object_name+'_'+spacecraft_name+'_'+str(time_of_flybies)+'_'+highest_plasma+'_'+plasma_scaleheight+'_dectable_radio_data.txt', all_detectable_radio)

    return all_detectable_radio


def main():

    # MakeFolder()  # フォルダ作成　基本的にはoccultation_range_plot.py で移動しているから基本使わない

    # MoveFile()  # ファイル移動　

    # 受かっているかの検証　processesの引数で並列数を指定

    with Pool(processes=20) as pool:
        result_list = list(pool.map(Judge_occultation, total_radio_number))

    # 受かっている電波のみを保存
    Replace_Save(result_list, Radio_observer_position)

    return 0


if __name__ == "__main__":
    main()
