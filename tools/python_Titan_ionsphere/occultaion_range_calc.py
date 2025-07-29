import numpy as np
from mmap import PROT_READ
import numpy as np
import os
from multiprocessing import Pool
import pandas as pd
import requests

# In[]
object_name = "titan"  # ganydeme/europa/calisto
spacecraft_name = "cassini"  # galileo/JUICE(?)
time_of_flybies = 15  # ..th flyby
#time_of_flybies = 151  # ..th flyby
#highest_plasma = "0"  # 単位は(/cc) 2e2/4e2/16e22 #12.5 13.5 str
#highest_altitude = "0"  # 単位は(km) 1.5e2/3e2/6e2 str
#standard_deviation = "0"  # 単位は(km)str
highest_plasma = "3900"
peak_altitude = "1400"
standard_deviation = "400"

raytrace_lowest_altitude = -2500  # レイトレーシングの下端の初期高度(km) 100の倍数で
raytrace_highest_altitude = 5000  # レイトレーシング上端の初期高度(km) 500の倍数+100で


output_label = "Nmax_" + str(highest_plasma) + "-hpeak_" + str(peak_altitude) + "-sigma_" + str(standard_deviation)
result_output_path = "/work1/rikutoyasuda/tools/result_titan/"

information_list = [
    "year",
    "month",
    "start_day",
    "end_day",
    "start_hour",
    "end_hour",
    "start_min",
    "end_min",
]



# MHz
# Freq_num = np.array([0.0512426, 0.0537086, 0.0562933, 0.0590024, 0.0618418, 0.0648179, 0.0679373, 0.0712067, 0.0746335, 0.0782252, 0.0819898, 0.0859355, 0.0900711, 0.0944057, 0.098949, 0.1037109, 0.1087019, 0.1139331, 0.1194161, 0.1251629, 0.1311864, 0.1374996, 0.1441167, 0.15105231, 0.15832159, 0.1659408, 0.17392661, 0.18229671, 0.1910697, 0.2002648, 0.2099025, 0.22000391, 0.23059151, 0.2416886, 0.25331979, 0.26551071, 0.2782883, 0.29168079, 0.3057178, 0.3204303, 0.31875, 0.33125, 0.34375, 0.35625, 0.36875, 0.38125, 0.39375, 0.40625, 0.41875, 0.43125, 0.44375, 0.45625, 0.46875, 0.48125, 0.49375, 0.50625, 0.51875, 0.53125, 0.54375, 0.55625, 0.56875, 0.58125, 0.59375, 0.60625, 0.61875, 0.63125, 0.64375, 0.65625, 0.66875, 0.68125, 0.69375, 0.70625, 0.71875, 0.73125, 0.74375, 0.75625, 0.76875, 0.78125, 0.79375, 0.80625, 0.81875, 0.83125, 0.84375, 0.85625, 0.86875, 0.88125, 0.89375, 0.90625, 0.91875, 0.93125, 0.94375, 0.95625, 0.96875, 0.98125, 0.99375])

#Freq_str = ['51242.6', '53708.6', '56293.3', '59002.4', '61841.8', '64817.9', '67937.3', '71206.7', '74633.5', '78225.2', '81989.8', '85935.5', '90071.1', '94405.7', '98949', '103710.9', '108701.9', '113933.1', '119416.1', '125162.9', '131186.4', '137499.6', '144116.7', '151052.31', '158321.59', '165940.8', '173926.61', '182296.71', '191069.7', '200264.8', '209902.5', '220003.91', '230591.51', '241688.6', '253319.79', '265510.71', '278288.3', '291680.79', '305717.8', '320430.3', '318750', '331250', '343750', '356250', '368750', '381250', '393750', '406250', '418750', '431250', '443750', '456250', '468750', '481250', '493750', '506250', '518750', '531250', '543750', '556250', '568750', '581250', '593750', '606250', '618750', '631250', '643750', '656250', '668750', '681250', '693750', '706250', '718750', '731250', '743750', '756250', '768750', '781250', '793750', '806250', '818750', '831250', '843750', '856250', '868750', '881250', '893750', '906250', '918750', '931250', '943750', '956250', '968750', '981250', '993750']
Freq_str = ['90071', '113930', '173930', '241690', '278290', '320430', '368750', '431250', '468750', '531250', '568750', '631250', '668750', '731250', '768750', '831250',  '868750',  '931250', '968750']
#Freq_str = np.array(['51242.6'])
Freq_num = []
for i in Freq_str:
    Freq_num.append(float(i))

kinds_freq = list(np.arange(len(Freq_num)))


def lineNotify(message):
    line_notify_token = "MFCL4nEMoT0m9IyjUXLeVsoePNXCfbAInnBs7tZeGts"
    line_notify_api = "https://notify-api.line.me/api/notify"
    payload = {"message": message}
    headers = {"Authorization": "Bearer " + line_notify_token}
    requests.post(line_notify_api, data=payload, headers=headers)

def Pick_up_params(param_name):
    # 計算をしたいフライバイにおける探査機の移動範囲をcsvファイルから取得する用　レイパスが必要な範囲を決定させるため
    flyby_list_path = result_output_path + "occultation_flyby_list.csv"
    flyby_list = pd.read_csv(flyby_list_path)

    # csvファイルにフライバイごとで使う軌道データを記入しておく　上記のパラメータから必要なデータのファイル名が選ばれて読み込まれる
    # queryが数値非対応なのでまずはフライバイ数で絞り込み
    selected_flyby_list = flyby_list[flyby_list["flyby_time"] == time_of_flybies]
    complete_selecred_flyby_list = selected_flyby_list.query(
        'object == "' + object_name + '" & spacecraft == "' + spacecraft_name + '"'
    )  # queryでフライバイ数以外を絞り込み

    complete_selecred_flyby_list = complete_selecred_flyby_list.reset_index(
        drop=True
    )  # index振り直し
    # 欲しいパラメータを取得
    param = float(complete_selecred_flyby_list[param_name][0])
    return param


def NibunnZ(x1, z1, x2, z2, z_destination):
    # 二点を結んだ直線がz=z_destinationと交わるx座標
    para = np.abs(z2 - z_destination)

    while para > 0.1:
        ddx = (x1 + x2) / 2
        ddz = (z1 + z2) / 2

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

    while para1 > 0.1 and para2 > 0.1:
        ddx = (x1 + x2) / 2
        ddz = (z1 + z2) / 2

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
    deff_x = raytracing_result[P1 + 1][1] - raytracing_result[P1][1]
    deff_z = raytracing_result[P1 + 1][3] - raytracing_result[P1][3]

    check_degree = np.degrees(np.arctan(deff_z / deff_x))
    if check_degree > 0.01:
        lineNotify("Start position need to be far from moon")
        raise Exception("Start position need to be far from moon")


def Calc_lowest(l):
    # 伝搬経路の計算が必要となる最低高度を調べるもの
    lowest_altitude = 10000
    reject = 0
    # 最も低い高度から検証
    for i in range(raytrace_lowest_altitude, 2000, 500):
        k = str(i)
        path = result_output_path + "raytracing_results/" +output_label+"/Pla_titan_gaussian-"+ output_label+"-Mag_test_simple-Mode_RX-Freq_"+ Freq_str[l]+"Hz-X_-6500km-Y_0km-Z_"+k+"km"

        if not os.path.exists(path):
            print(f"File not found:"+path )
            continue

        ray_path = np.genfromtxt(path) # ray_path[n2][1] x座標   ray_path[n2][3] z座標
        
        # 初期ベクトルと背景磁場・プラズマ密度分布の関係から、電波の伝搬経路が解なしになることもあるのでその時のデータは無視するためのif文
        if ray_path.ndim == 2:
            Check_too_nearby(ray_path)  # 電波の計算が十分遠方から行われて入れるかの確認
            n2 = len(ray_path) - 1

            if ray_path[n2][3] > 0 and ray_path[n2][1] > 0:
                # 今回検証で用いたデータのうち最も低高度からのレイが、月に衝突しているかを確認するもの。衝突しない場合より低い高度から電波を飛ばす必要がある。
                break

            if ray_path[n2][1] < 0:
                lowest_altitude = i  # 電波が電離圏で反射して木星方向に向かう場合、その電波は崩落線になり得ないのでその高度以下での計算が無意味であることが確認できる
                reject = 1

            if ray_path[n2][3] < 0:
                lowest_altitude = i  # 電波が衝突する場合、その高度以下での計算が無意味であることが確認できる
                reject = 1
        
    if reject == 0:
        lineNotify("unenough low altitude")
        raise Exception("unenough low altitude") 

    return lowest_altitude

def Calc_highest(l):
    # 伝搬経路の計算が必要となる最低高度を調べるもの
    highest_altitude = 100
    x_farthest = Pick_up_params("x_farthest")
    z_farthest = Pick_up_params("z_farthest")
    for i in range(0, raytrace_highest_altitude, 500):
        # 500kmごとの高度で検証（ex.500と600 1000と1100など）
        lower_altitude = str(i)
        k = i + 100
        higher_altitude = str(k)

        lower_path = result_output_path + "raytracing_results/" +output_label+"/Pla_titan_gaussian-"+ output_label+"-Mag_test_simple-Mode_RX-Freq_"+ Freq_str[l]+"Hz-X_-6500km-Y_0km-Z_"+lower_altitude+"km"

        higher_path = result_output_path + "raytracing_results/" +output_label+"/Pla_titan_gaussian-"+ output_label+"-Mag_test_simple-Mode_RX-Freq_"+ Freq_str[l]+"Hz-X_-6500km-Y_0km-Z_"+higher_altitude+"km"

        
        if not os.path.exists(lower_path):
            print(f"File not found:"+ lower_path )
            continue

        if not os.path.exists(higher_path):
            print(f"File not found:"+ higher_path )
            continue

        lower_ray_path = np.genfromtxt(lower_path) # ray_path[n2][1] x座標   ray_path[n2][3] z座標

        higher_ray_path = np.genfromtxt(higher_path)# ray_path[n2][1] x座標   ray_path[n2][3] z座標


        # 初期ベクトルと背景磁場・プラズマ密度分布の関係から、電波の伝搬経路が解なしになることもあるのでその時のデータは無視するためのif文
        if lower_ray_path.ndim == 2 and higher_ray_path.ndim == 2:
            # 電波の計算が十分遠方から行われて入れるかの確認
            Check_too_nearby(lower_ray_path)
            Check_too_nearby(higher_ray_path)

            n2_lower = len(lower_ray_path) - 1
            n2_higher = len(higher_ray_path) - 1


            # 伝搬経路の終着点が電離圏内で止まっている or 地表面で消失
            if lower_ray_path[n2_lower][1] < 0:
                highest_altitude = higher_altitude
                continue

            if np.sqrt(lower_ray_path[n2_lower][1]**2 + (lower_ray_path[n2_lower][3]+ 2574) **2 )< 3574**2:
                highest_altitude = higher_altitude
                continue


            # 計算が必要な計算空間内で計算が終わってしまっている場合、もっと長いレイパスで計算する必要がある。その確認
            if (
                lower_ray_path[n2_lower][1] < x_farthest
                and lower_ray_path[n2_lower][3] < z_farthest
            ):
                error_message = Freq_str[l] + "lower" +str(lower_altitude) + "error"
                lineNotify(error_message)
                raise Exception(error_message)

            if (
                higher_ray_path[n2_higher][1] < x_farthest
                and higher_ray_path[n2_higher][3] < z_farthest
            ):
                error_message = Freq_str[l] + "lower" +str(lower_altitude) + "error"
                lineNotify(error_message)
                raise Exception(error_message)

            if lower_ray_path[n2_lower][3] > z_farthest:
                # もし低高度側のレイパスの終着点のz座標が計算空間の上端より高い場合
                # lower_x, lower_zで計算空間をでた座標を計算
                lower_idx = np.array(np.where(lower_ray_path[:, 3] > z_farthest))
                t2 = lower_idx[0, 0]
                t1 = t2 - 1
                lower_x, lower_z = NibunnZ(
                    lower_ray_path[t1][1],
                    lower_ray_path[t1][3],
                    lower_ray_path[t2][1],
                    lower_ray_path[t2][3],
                    z_farthest,
                )

                if higher_ray_path[n2_higher][3] > z_farthest:
                    # かつ高高度側のレイパスの終着点のz座標も計算空間の上端より高い場合
                    # higher_x, higher_zで計算空間をでた座標を計算
                    higher_idx = np.array(np.where(higher_ray_path[:, 3] > z_farthest))
                    T2 = higher_idx[0, 0]
                    T1 = T2 - 1
                    higher_x, higher_z = NibunnZ(
                        higher_ray_path[T1][1],
                        higher_ray_path[T1][3],
                        higher_ray_path[T2][1],
                        higher_ray_path[T2][3],
                        z_farthest,
                    )

                    if higher_x > lower_x:
                        # 低高度のレイパスの方がより遠くで上端を超えている　→ より高高度のレイパスが包絡線を形成することはないので計算の必要はない
                        highest_altitude = higher_altitude
                else:
                    highest_altitude = higher_altitude

            else:
                # もし低高度側のレイパスの終着点のz座標が計算空間の上端より低い場合
                # lower_x, lower_z, higher_x, higher_zで計算空間x座標をでた座標を計算
                lower_idx = np.array(np.where(lower_ray_path[:, 1] > x_farthest))
                t2 = lower_idx[0, 0]
                t1 = t2 - 1
                lower_x, lower_z = NibunnX(
                    lower_ray_path[t1][1],
                    lower_ray_path[t1][3],
                    lower_ray_path[t2][1],
                    lower_ray_path[t2][3],
                    x_farthest,
                )

                higher_idx = np.array(np.where(higher_ray_path[:, 1] > x_farthest))
                T2 = higher_idx[0, 0]
                T1 = T2 - 1
                higher_x, higher_z = NibunnX(
                    higher_ray_path[T1][1],
                    higher_ray_path[T1][3],
                    higher_ray_path[T2][1],
                    higher_ray_path[T2][3],
                    x_farthest,
                )

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
                       '_'+str(time_of_flybies)+'_flybys/para_' + highest_plasma+'_'+peak_altitude+'.csv', index=False)
    """
    a = np.array(Freq_num)
    df2 = pd.DataFrame(a, columns=["freq"])
    df2["lowest"] = lowest_list
    df2["highest"] = highest_list
    df2["exc"] = -10000
    df2.to_csv(
        result_output_path 
        + "tracing_range_" 
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_flybys/para_"
        + highest_plasma
        + "_"
        + peak_altitude
        + "_"
        + standard_deviation
        + ".csv",
        index=False,
    )

    print(df2)
    return 0

def lineNotify(message):
    line_notify_token = "MFCL4nEMoT0m9IyjUXLeVsoePNXCfbAInnBs7tZeGts"
    line_notify_api = "https://notify-api.line.me/api/notify"
    payload = {"message": message}
    headers = {"Authorization": "Bearer " + line_notify_token}
    requests.post(line_notify_api, data=payload, headers=headers)


def main():

    with Pool(processes=20) as pool:
        lowest_altitude_list = list(pool.map(Calc_lowest, kinds_freq))

    with Pool(processes=20) as pool:
        highest_altitude_list = list(pool.map(Calc_highest, kinds_freq))

    Replace_csv(lowest_altitude_list, highest_altitude_list)

    return 0


if __name__ == "__main__":
    main()

# %%
