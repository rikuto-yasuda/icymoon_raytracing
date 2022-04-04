# In[]

from mmap import PROT_READ
import numpy as np
import os
from multiprocessing import Pool
import pandas as pd


# In[]
object_name = 'europa'   # europa/ganymde/callisto
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 14  # ..th flyby
highest_plasma = '0.25e2'  # 単位は(/cc) 2e2/4e2/16e2
plasma_scaleheight = '4e2'  # 単位は(km) 1.5e2/3e2/6e2
raytrace_lowest_altitude = -300  # レイトレーシングの下端の初期高度(km) 100の倍数で
raytrace_highest_altitude = 2600  # レイトレーシング上端の初期高度(km) 500の倍数+100で

information_list = ['year', 'month', 'start_day', 'end_day',
                    'start_hour', 'end_hour', 'start_min', 'end_min']


Freq_str = ['3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]

Freq_num = []
for i in Freq_str:
    Freq_num.append(float(i)/1000000)

kinds_freq = list(np.arange(len(Freq_num)))


def Raytrace_result_makefolder():
    os.makedirs('../result_for_yasudaetal2022/raytracing_'+object_name+'_results/' + object_name +
                '_'+highest_plasma+'_'+plasma_scaleheight)  # レイトレーシングの結果を格納するフォルダを生成


def MoveFile():
    # 周波数ごとにレイトレーシングの結果を格納
    for l in range(len(Freq_num)):
        # zが0以下のレイパスは100kmおきに計算⇨格納
        for j in range(raytrace_lowest_altitude, 0, 100):
            k = str(j)
            os.replace('../../raytrace.tohoku/src/rtc/testing/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                       k+'-FR'+Freq_str[l], '../result_for_yasudaetal2022/raytracing_'+object_name+'_results/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                       k+'-FR'+Freq_str[l])
        # zが0以下のレイパスは0,100,500,600,1000,1100 ..kmで計算⇨格納
        for i in range(0, raytrace_highest_altitude, 500):
            lower_altitude = str(i)
            higher_altitude = str(i + 100)

            os.replace('../../raytrace.tohoku/src/rtc/testing/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                       lower_altitude+'-FR'+Freq_str[l], '../result_for_yasudaetal2022/raytracing_'+object_name+'_results/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                       lower_altitude+'-FR'+Freq_str[l])
            os.replace('../../raytrace.tohoku/src/rtc/testing/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                       higher_altitude+'-FR'+Freq_str[l], '../result_for_yasudaetal2022/raytracing_'+object_name+'_results/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                       higher_altitude+'-FR'+Freq_str[l])


def Pick_up_params(param_name):
    # 計算をしたいフライバイにおける探査機の移動範囲をcsvファイルから取得する用　レイパスが必要な範囲を決定させるため
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
    # 欲しいパラメータを取得
    param = float(complete_selecred_flyby_list[param_name][0])
    return param


def NibunnZ(x1, z1, x2, z2, z_destination):
    # 二点を結んだ直線がz=z_destinationと交わるx座標
    para = np.abs(z2 - z_destination)

    while (para > 0.1):
        ddx = (x1+x2)/2
        ddz = (z1+z2)/2

        if ddz > z_destination:
            x2 = ddx
            z2 = ddz
        else:
            x1 = ddx
            z1 = ddz

        para = np.abs(z2 - z_destination)

    return x2, z2


def NibunnX(x1, z1, x2, z2, x_destination):
    # 二点を結んだ直線がx=x_destinationと交わるz座標
    para1 = np.abs(x1 - x_destination)
    para2 = np.abs(x2 - x_destination)

    while (para1 > 0.1 and para2 > 0.1):
        ddx = (x1+x2)/2
        ddz = (z1+z2)/2

        if ddx > x_destination:
            x2 = ddx
            z2 = ddz
        else:
            x1 = ddx
            z1 = ddz

        para1 = np.abs(x1 - x_destination)
        para2 = np.abs(x2 - x_destination)

    return x2, z2


def Check_too_nearby(raytracing_result):
    # 初期位置ではdz=0でレイパスを飛ばしているという条件下で、x座標1000km分移動した時、z座標の変化が微小であることを確認するもの
    # →z座標の変化がもし大きい場合、衛星（月）周辺での電波の屈折の寄与を過小評価してしまうため、その場合にはより離れたところから電波を飛ばす必要がある
    initial_x = raytracing_result[0][1]
    check_x_position = initial_x + 1000

    check_idx = np.array(np.where(raytracing_result[:][1] > check_x_position))
    P1 = check_idx[0, 0]
    deff_x = raytracing_result[P1+1][1] - raytracing_result[P1][1]
    deff_z = raytracing_result[P1+1][3] - raytracing_result[P1][3]

    check_degree = np.degrees(np.arctan(deff_z/deff_x))
    if check_degree > 0.01:
        print("Start position need to be far from moon")


def Calc_lowest(l):
    # 伝搬経路の計算が必要となる最低高度を調べるもの
    lowest_altitude = 10000
    # 最も低い高度から検証
    for i in range(raytrace_lowest_altitude, 0, 100):
        k = str(i)
        ray_path = np.genfromtxt("../result_for_yasudaetal2022/raytracing_"+object_name+"_results/"+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+"/ray-P" +
                                 object_name+"_nonplume_"+highest_plasma+"_"+plasma_scaleheight+"-Mtest_simple-benchmark-LO-Z"+k+"-FR"+Freq_str[l])
        # 初期ベクトルと背景磁場・プラズマ密度分布の関係から、電波の伝搬経路が解なしになることもあるのでその時のデータは無視するためのif文
        if ray_path.ndim == 2:
            Check_too_nearby(ray_path)  # 電波の計算が十分遠方から行われて入れるかの確認
            n2 = len(ray_path)-1

            if (ray_path[n2][3] > 0 and ray_path[n2][1] > 0):
                # 今回検証で用いたデータのうち最も低高度からのレイが、月に衝突しているかを確認するもの。衝突しない場合より低い高度から電波を飛ばす必要がある。
                break

            if (ray_path[n2][1] < 0):
                lowest_altitude = i  # 電波が電離圏で反射して木星方向に向かう場合、その電波は崩落線になり得ないのでその高度以下での計算が無意味であることが確認できる

            if (ray_path[n2][3] < 0):
                lowest_altitude = i  # 電波が衝突する場合、その高度以下での計算が無意味であることが確認できる

    return lowest_altitude


def Calc_highest(l):
    # 伝搬経路の計算が必要となる最低高度を調べるもの
    highest_altitude = 100
    x_farthest = Pick_up_params("x_farthest")
    z_farthest = Pick_up_params("z_farthest")
    for i in range(0, raytrace_highest_altitude, 500):
        # 500kmごとの高度で検証（ex.500と600 1000と1100など）
        lower_altitude = str(i)
        k = i+100
        higher_altitude = str(k)

        lower_ray_path = np.genfromtxt(("../result_for_yasudaetal2022/raytracing_"+object_name+"_results/"+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+"/ray-P"
                                        + object_name+"_nonplume_"+highest_plasma+"_"+plasma_scaleheight+"-Mtest_simple-benchmark-LO-Z"+lower_altitude+"-FR"+Freq_str[l]))

        higher_ray_path = np.genfromtxt(("../result_for_yasudaetal2022/raytracing_"+object_name+"_results/"+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+"/ray-P"
                                         + object_name+"_nonplume_"+highest_plasma+"_"+plasma_scaleheight+"-Mtest_simple-benchmark-LO-Z"+higher_altitude+"-FR"+Freq_str[l]))

        # 初期ベクトルと背景磁場・プラズマ密度分布の関係から、電波の伝搬経路が解なしになることもあるのでその時のデータは無視するためのif文
        if lower_ray_path.ndim == 2 and higher_ray_path.ndim == 2:
            # 電波の計算が十分遠方から行われて入れるかの確認
            Check_too_nearby(lower_ray_path)
            Check_too_nearby(higher_ray_path)

            n2 = len(lower_ray_path)-1

            # 計算が必要な計算空間内で計算が終わってしまっている場合、もっと長いレイパスで計算する必要がある。その確認
            if (lower_ray_path[n2][1] < x_farthest and lower_ray_path[n2][3] < z_farthest):
                print(Freq_str[l], lower_altitude, "error")

            if (higher_ray_path[n2][1] < x_farthest and higher_ray_path[n2][3] < z_farthest):
                print(Freq_str[l], higher_altitude, "error")

            if (lower_ray_path[n2][3] > z_farthest):
                # もし低高度側のレイパスの終着点のz座標が計算空間の上端より高い場合
                # lower_x, lower_zで計算空間をでた座標を計算
                lower_idx = np.array(
                    np.where(lower_ray_path[:, 3] > z_farthest))
                t2 = lower_idx[0, 0]
                t1 = t2 - 1
                lower_x, lower_z = NibunnZ(
                    lower_ray_path[t1][1], lower_ray_path[t1][3], lower_ray_path[t2][1], lower_ray_path[t2][3], z_farthest)

                if (higher_ray_path[n2][3] > z_farthest):
                    # かつ高高度側のレイパスの終着点のz座標も計算空間の上端より高い場合
                    # higher_x, higher_zで計算空間をでた座標を計算
                    higher_idx = np.array(
                        np.where(higher_ray_path[:, 3] > z_farthest))
                    T2 = higher_idx[0, 0]
                    T1 = T2 - 1
                    higher_x, higher_z = NibunnZ(
                        higher_ray_path[T1][1], higher_ray_path[T1][3], higher_ray_path[T2][1], higher_ray_path[T2][3], z_farthest)

                    if higher_x > lower_x:
                        # 低高度のレイパスの方がより遠くで上端を超えている　→ より高高度のレイパスが包絡線を形成することはないので計算の必要はない
                        highest_altitude = higher_altitude

            else:
                # もし低高度側のレイパスの終着点のz座標が計算空間の上端より低い場合
                # lower_x, lower_z, higher_x, higher_zで計算空間x座標をでた座標を計算
                lower_idx = np.array(
                    np.where(lower_ray_path[:, 1] > x_farthest))
                t2 = lower_idx[0, 0]
                t1 = t2 - 1
                lower_x, lower_z = NibunnX(
                    lower_ray_path[t1][1], lower_ray_path[t1][3], lower_ray_path[t2][1], lower_ray_path[t2][3], x_farthest)

                higher_idx = np.array(
                    np.where(higher_ray_path[:, 1] > x_farthest))
                T2 = higher_idx[0, 0]
                T1 = T2 - 1
                higher_x, higher_z = NibunnX(
                    higher_ray_path[T1][1], higher_ray_path[T1][3], higher_ray_path[T2][1], higher_ray_path[T2][3], x_farthest)

                highest_altitude = higher_altitude

                if higher_z > lower_z:
                    # より高高度側のレイが最後まで高い高度を維持している→それより高い場所を計算する必要はない
                    break

        else:
            higher_altitude = i + 500

    return highest_altitude


def Replace_csv(lowest_list, highest_list):
    """
    radio_range.loc[:, Rowname] = replace_list
    radio_range.to_csv('../result_for_yasudaetal2022/tracing_range_'+spacecraft_name+'_'+object_name +
                       '_'+str(time_of_flybies)+'_flybys/para_' + highest_plasma+'_'+plasma_scaleheight+'.csv', index=False)
    """
    a = np.array(Freq_num)
    df2 = pd.DataFrame(a, columns=['freq'])
    df2['lowest'] = lowest_list
    df2['highest'] = highest_list
    df2['exc'] = -10000
    df2.to_csv('../result_for_yasudaetal2022/tracing_range_'+spacecraft_name+'_'+object_name +
               '_'+str(time_of_flybies)+'_flybys/para_' + highest_plasma+'_'+plasma_scaleheight+'.csv', index=False)

    print(df2)
    # print(radio_range.highest)
    return 0


def main():

    Raytrace_result_makefolder()  # レイトレーシングの結果を格納するフォルダを生成
    MoveFile()  # レイトレーシングの結果を移動

    with Pool(processes=3) as pool:
        lowest_altitude_list = list(pool.map(Calc_lowest, kinds_freq))

    with Pool(processes=3) as pool:
        highest_altitude_list = list(pool.map(Calc_highest, kinds_freq))

    Replace_csv(lowest_altitude_list, highest_altitude_list)

    return 0


if __name__ == "__main__":
    main()

# %%
