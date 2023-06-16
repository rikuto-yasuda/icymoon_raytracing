# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
import os
import time
import glob
import requests
import sys

# %%
# あらかじめ ../result_sgepss2021/~/~ に必要なレイトレーシング結果とパラメータセットを入れること
args = sys.argv
rng = np.random.default_rng()

object_name = "callisto"  # ganydeme/europa/calisto``
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 30  # ..th flyby
# highest_plasma = "0e2"  # 単位は(/cc) 2e2/4e2/16e22 #12.5 13.5
# plasma_scaleheight = "4e2"  # 単位は(km) 1.5e2/3e2/6e2
highest_plasma = args[1]  # 単位は(/cc) 2e2/4e2/16e22 #12.5 13.5
plasma_scaleheight = args[2]  # 単位は(km) 1.5e2/3e2/6e2


Radio_range_cdf = (
    "../result_for_yasudaetal2022/tracing_range_"
    + spacecraft_name
    + "_"
    + object_name
    + "_"
    + str(time_of_flybies)
    + "_flybys/para_"
    + highest_plasma
    + "_"
    + plasma_scaleheight
    + ".csv"
)
Radio_Range = pd.read_csv(Radio_range_cdf, header=0)
# new [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]
# old [0 hour,1 min,2 frequency(MHz),3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),4 電波源の南北,5 座標変換した時のx(tangential point との水平方向の距離),6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度]
Radio_observer_position = np.loadtxt(
    "../result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/interpolated_calculated_all_"
    + spacecraft_name
    + "_"
    + object_name
    + "_"
    + str(time_of_flybies)
    + "_Radio_data.txt"
)  # 電波源の経度を含む


# europa & ganymede
if object_name == "ganymede":
    Freq_str = [
        "3.984813988208770752e5",
        "4.395893216133117676e5",
        "4.849380254745483398e5",
        "5.349649786949157715e5",
        "5.901528000831604004e5",
        "6.510338783264160156e5",
        "7.181954979896545410e5",
        "7.922856807708740234e5",
        "8.740190267562866211e5",
        "9.641842246055603027e5",
        "1.063650846481323242e6",
        "1.173378825187683105e6",
        "1.294426321983337402e6",
        "1.427961349487304688e6",
        "1.575271964073181152e6",
        "1.737779378890991211e6",
        "1.917051434516906738e6",
        "2.114817380905151367e6",
        "2.332985162734985352e6",
        "2.573659420013427734e6",
        "2.839162111282348633e6",
        "3.132054328918457031e6",
        "3.455161809921264648e6",
        "3.811601638793945312e6",
        "4.204812526702880859e6",
        "4.638587474822998047e6",
        "5.117111206054687500e6",
        "5.644999980926513672e6",
    ]

    if time_of_flybies == 1:
        judge_start_time = np.array([5.0, 30.0])  # 5:30 前は全て受信可能判定
        judge_end_time = np.array([7.0, 0.0])  # 7:00 後は全て受信可能判定

if object_name == "callisto":
    Freq_str = [
        "3.612176179885864258e5",
        "3.984813988208770752e5",
        "4.395893216133117676e5",
        "4.849380254745483398e5",
        "5.349649786949157715e5",
        "5.901528000831604004e5",
        "6.510338783264160156e5",
        "7.181954979896545410e5",
        "7.922856807708740234e5",
        "8.740190267562866211e5",
        "9.641842246055603027e5",
        "1.063650846481323242e6",
        "1.173378825187683105e6",
        "1.294426321983337402e6",
        "1.427961349487304688e6",
        "1.575271964073181152e6",
        "1.737779378890991211e6",
        "1.917051434516906738e6",
        "2.114817380905151367e6",
        "2.332985162734985352e6",
        "2.573659420013427734e6",
        "2.839162111282348633e6",
        "3.132054328918457031e6",
        "3.455161809921264648e6",
        "3.811601638793945312e6",
        "4.204812526702880859e6",
        "4.638587474822998047e6",
        "5.117111206054687500e6",
        "5.644999980926513672e6",
    ]

    if time_of_flybies == 30:
        judge_start_time = np.array([11.0, 15.0])  # 5:30 前は全て受信可能判定
        judge_end_time = np.array([12.0, 20.0])  # 7:00 後は全て受信可能判定

    if time_of_flybies == 9:
        judge_start_time = np.array([13.0, 30.0])  # 5:30 前は全て受信可能判定
        judge_end_time = np.array([14.0, 0.0])  # 7:00 後は全て受信可能判定


Freq_num = []
for idx in Freq_str:
    Freq_num.append(float(idx) / 1000000)

Highest = Radio_Range.highest
Lowest = Radio_Range.lowest
Except = Radio_Range.exc
Calculation_height_interbal = -((Lowest - Highest) // 8) * 2

# %%


def MakeFolder():
    os.makedirs(
        "../result_for_yasudaetal2022/raytracing_"
        + object_name
        + "_results/"
        + object_name
        + "_"
        + highest_plasma
        + "_"
        + plasma_scaleheight
    )  # レイトレーシングの結果を格納するフォルダを生成


def MoveFile():
    for l in range(len(Freq_num)):
        for j in range(Lowest[l], Highest[l], 2):
            k = str(j)
            os.replace(
                "../../raytrace.tohoku/src/rtc/testing/ray-P"
                + object_name
                + "_nonplume_"
                + highest_plasma
                + "_"
                + plasma_scaleheight
                + "-Mtest_simple-benchmark-LO-Z"
                + k
                + "-FR"
                + Freq_str[l],
                "../result_for_yasudaetal2022/raytracing_"
                + object_name
                + "_results/"
                + object_name
                + "_"
                + highest_plasma
                + "_"
                + plasma_scaleheight
                + "/ray-P"
                + object_name
                + "_nonplume_"
                + highest_plasma
                + "_"
                + plasma_scaleheight
                + "-Mtest_simple-benchmark-LO-Z"
                + k
                + "-FR"
                + Freq_str[l],
            )


def Judgemet_array(data):
    before_nonjudge_number = np.intersect1d(
        np.where(data[:, 0] <= judge_start_time[0]),
        np.where(data[:, 1] < judge_start_time[1]),
    )[-1]
    after_nonjudge_number = np.intersect1d(
        np.where(data[:, 0] >= judge_end_time[0]),
        np.where(data[:, 1] > judge_end_time[1]),
    )[0]

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


def Judge_occultation(i):
    # Radio_observation_position
    # new [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]
    # aa  受かるときは1 受からないとき0を出力
    aa = 0
    detectable_obsever_position_x = Radio_observer_position[i][6]
    detectable_obsever_position_z = Radio_observer_position[i][7]
    detectable_frequency = Radio_observer_position[i][3]  # 使うレイの周波数を取得
    # レイの周波数と周波数リスト（Freq＿num）の値が一致する場所を取得　周波数リスト（Freq＿num）とcsvファイルの週数リストが一致しているのでそこからその周波数における電波源の幅を取得

    print(i)
    # print(detectable_frequency)
    # print(np.where(Freq_num == detectable_frequency)[0][0])

    freq = int(np.where(Freq_num == detectable_frequency)[0][0])

    # 電波の出発点が探査機の高度より高くなったものは考えない
    repeat_array1 = np.arange(
        Lowest[freq],
        min(Highest[freq], detectable_obsever_position_z),
        Calculation_height_interbal[freq],
        dtype=int,
    )
    repeat_array2 = np.arange(
        Lowest[freq], min(Highest[freq], detectable_obsever_position_z), 2, dtype=int
    )
    rng.shuffle(repeat_array2)
    repeat_array = np.concatenate([repeat_array1, repeat_array2], 0)

    for j in repeat_array:
        k = str(j)
        # 高度が低い電波源の電波から取得
        Radio_propagation_route = np.genfromtxt(
            "../result_for_yasudaetal2022/raytracing_"
            + object_name
            + "_results/"
            + object_name
            + "_"
            + highest_plasma
            + "_"
            + plasma_scaleheight
            + "/ray-P"
            + object_name
            + "_nonplume_"
            + highest_plasma
            + "_"
            + plasma_scaleheight
            + "-Mtest_simple-benchmark-LO-Z"
            + k
            + "-FR"
            + Freq_str[freq]
        )

        # たまにレイパスが計算できない初期条件になって入りう時があるのでそのデータを除外
        if Radio_propagation_route.ndim == 2:
            detectable_x_index = np.where(
                Radio_propagation_route[:, 1] > detectable_obsever_position_x
            )[0]

            detectable_z_index = np.where(
                Radio_propagation_route[:, 3] < detectable_obsever_position_z
            )[0]

            detectable_x_index_minus = detectable_x_index - 1

            # レイが探査機のx座標を超える前に探査機よりも上空に行ってしまっている→その電波は受からない
            if len(np.intersect1d(detectable_x_index_minus, detectable_z_index)) == 0:
                continue

            # レイが探査機のx座標を超えても探査機よりも低高度にレイパスがある→その電波は受かる
            if len(np.intersect1d(detectable_x_index, detectable_z_index)) != 0:
                aa = 1
                break

            # 以下、ギリギリだったときに例が探査機の下にあるか上にあるかを判別する部分

            if len(np.intersect1d(detectable_x_index_minus, detectable_z_index)) > 1:
                print("error")

            else:
                h = (
                    int(np.intersect1d(detectable_x_index_minus, detectable_z_index)[0])
                    + 1
                )

                para = np.abs(
                    Radio_propagation_route[h][1] - detectable_obsever_position_x
                )
                hight = Radio_propagation_route[h][3] - detectable_obsever_position_z
                x1 = Radio_propagation_route[h][1]
                z1 = Radio_propagation_route[h][3]
                x2 = Radio_propagation_route[h - 1][1]
                z2 = Radio_propagation_route[h - 1][3]

                while para > 10:
                    ddx = (x1 + x2) / 2
                    ddz = (z1 + z2) / 2

                    if ddx > detectable_obsever_position_x:
                        x1 = ddx
                        z1 = ddz
                    else:
                        x2 = ddx
                        z2 = ddz

                    para = np.abs(x1 - detectable_obsever_position_x)
                    hight = z1 - detectable_obsever_position_z

                if hight < 0:
                    # res[i][6]=0
                    aa = 1
                    break

    return aa


def Replace_Save(
    judgement, befrore_judged_array, all_radio_data, expected_detectable_array
):
    # 想定した電子密度分布で観測可能な電波のリスト
    # [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]
    # を想定した電子密度でのレイトレーシング結果が保存されているフォルダに保存
    occultaion_aray = np.array(judgement)
    judge_array = befrore_judged_array[np.where(occultaion_aray[:] == 1)[0]]
    np.savetxt("./judgement.txt", judgement)
    np.savetxt("./judge_array.txt", judge_array)
    # print(judge_array)
    # print(expected_detectable_array)
    detectable_array = np.sort(
        np.concatenate(
            [judge_array, expected_detectable_array],
            0,
        )
    )
    np.savetxt("./detectable_array.txt", detectable_array)
    all_detectable_radio = all_radio_data[detectable_array][:]
    np.savetxt(
        "../result_for_yasudaetal2022/raytracing_"
        + object_name
        + "_results/"
        + object_name
        + "_"
        + highest_plasma
        + "_"
        + plasma_scaleheight
        + "/interpolated_"
        + object_name
        + "_"
        + spacecraft_name
        + "_"
        + str(time_of_flybies)
        + "_"
        + highest_plasma
        + "_"
        + plasma_scaleheight
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
    # MakeFolder()  # フォルダ作成　基本的にはoccultation_range_plot.py で移動しているから基本使わない
    # MoveFile()  # ファイル移動
    (
        judge_array,
        no_judge_detectable_array,
        no_judge_undetectable_array,
    ) = Judgemet_array(Radio_observer_position)

    # n = len(Radio_observer_position)
    # total_radio_number = list(np.arange(n))
    total_radio_number = list(judge_array)
    start = time.time()
    # 受かっているかの検証　processesの引数で並列数を指定

    with Pool(processes=60) as pool:
        result_list = list(pool.map(Judge_occultation, total_radio_number))

    # 受かっている電波のみを保存
    Replace_Save(
        result_list,
        judge_array,
        Radio_observer_position,
        no_judge_detectable_array,
    )
    end = time.time()
    minute = str((end - start) / 60)
    # minute = str(((end - start) * n) / (60 * n1))
    message = (
        "計算完了 / max:"
        + highest_plasma
        + " / scale:"
        + plasma_scaleheight
        + " / total time:"
        + minute
        + "min"
    )
    lineNotify(message)

    return 0


if __name__ == "__main__":
    main()

# %%
