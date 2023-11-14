# In[]
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as mpt
import pandas as pd
from matplotlib.colors import LogNorm

object_name = "ganymede"  # europa/ganymde/callisto
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 1  # ..th flyby
highest_plasma = "1e2"  # 単位は(/cc) 2e2/4e2/16e2
plasma_scaleheight = "10e2"  # 単位は(km) 1.5e2/3e2/6e2
frequency = "6.510338783264160156e5"  # MHz
altitiude_interval = 20
radio_type = "A"  # 複数選択可能にしたい

occultation_begining_hour = 6
occultation_begining_minute = 10

occultation_end_hour = 6
occultation_end_minute = 30


frequency_number = float(frequency) / 1000000
occultation_lowest_frequency = frequency_number - 0.01
occultation_highest_frequecy = frequency_number + 0.01


# 対象のフライバイ時における角周波数での電波の幅をまとめたファイルを取得
Radio_name_cdf = (
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
Radio_Range = pd.read_csv(Radio_name_cdf, header=0)


# 天体ごとにしようしている周波数を出力
def Get_frequency(object):
    if object == "ganymede" or object == "europa":
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

        Freq_underline = 0.36122

    elif object == "callisto":
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

        Freq_underline = 0.32744

    else:
        print("object name is not correct")

    return Freq_str, Freq_underline


def ray_plot(height):
    data = np.loadtxt(
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
        + str(height)
        + "-FR"
        + frequency
        + ""
    )
    x = data[:, [1]]
    z = data[:, [3]]
    plt.plot(x, z, color="red", linewidth=0.5)


def Occultation_timing_select(row_data):
    # 0 hour,1 min, 2 frequency(MHz), 3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000), 4 電波源の南北, 5 座標変換した時のx(tangential point との水平方向の距離), 6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度, 8 探査機の経度]

    data = np.loadtxt(
        "../result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/calculated_all_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_Radio_data.txt"
    )

    def radio_source_select(judgement, radio):
        selected_data = np.zeros_like(judgement)

        for k in range(len(judgement)):
            # 電波源の位置からABCDを判定
            if np.abs(judgement[k][8] + 360 - judgement[k][7]) < np.abs(
                judgement[k][8] - judgement[k][7]
            ):
                Lon = judgement[k][8] + 360 - judgement[k][7]

            elif np.abs(judgement[k][7] + 360 - judgement[k][8]) < np.abs(
                judgement[k][7] - judgement[k][8]
            ):
                Lon = judgement[k][8] - 360 - judgement[k][7]

            else:
                Lon = judgement[k][8] - judgement[k][7]

            Lat = judgement[k][4]

            if "A" in radio:
                if Lon < 0 and Lat > 0:
                    selected_data[k, :] = judgement[k, :].copy()

            if "B" in radio:
                if Lon > 0 and Lat > 0:
                    selected_data[k, :] = judgement[k, :].copy()

            if "C" in radio:
                if Lon < 0 and Lat < 0:
                    selected_data[k, :] = judgement[k, :].copy()

            if "D" in radio:
                if Lon > 0 and Lat < 0:
                    selected_data[k, :] = judgement[k, :].copy()

        selected_data = selected_data[np.all(selected_data != 0, axis=1), :]

        return selected_data

    def time_select_and_xz(
        radio_data,
        begining_hour,
        begining_min,
        end_hour,
        end_min,
        lowest_freq,
        highest_freq,
    ):
        # 0 hour,1 min, 2 frequency(MHz), 3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000), 4 電波源の南北, 5 座標変換した時のx(tangential point との水平方向の距離), 6 座標変換した時のy(tangential pointからの高さ方向の距離),7 電波源の実際の経度, 8 探査機の経度]

        start_hour_select = np.where(radio_data[:, 0] == begining_hour)
        start_minute_select = np.where(radio_data[:, 1] == begining_min)
        start_select = np.intersect1d(start_hour_select, start_minute_select)[0]

        end_hour_select = np.where(radio_data[:, 0] == end_hour)
        end_minute_select = np.where(radio_data[:, 1] == end_min)
        end_select = np.intersect1d(end_hour_select, end_minute_select)[-1]

        time_select = radio_data[start_select:end_select, :]

        freq_select = np.where(
            (time_select[:, 2] > lowest_freq) & (time_select[:, 2] < highest_freq)
        )[0]

        selected_freq_data = time_select[freq_select, :]

        detectable_select = np.where(selected_freq_data[:, 6] > 0)[0]
        selected_detectable_data = selected_freq_data[detectable_select, :]
        z_height = selected_detectable_data[:, 6]
        x_holizon = selected_detectable_data[:, 5]
        select_hour = selected_detectable_data[:, 0]
        select_minute = selected_detectable_data[:, 1]
        return select_hour, select_minute, x_holizon, z_height

    source_type_selected = radio_source_select(data, radio_type)
    h, m, x, z = time_select_and_xz(
        source_type_selected,
        occultation_begining_hour,
        occultation_begining_minute,
        occultation_end_hour,
        occultation_end_minute,
        occultation_lowest_frequency,
        occultation_highest_frequecy,
    )
    print(x, z)
    return h, m, x, z
    # np.savetxt('../result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/calculated_all_' +spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_'+radio_type+'_tangential_point.txt', selected_data, fmt="%s")


def spacecraft_plot():
    # [0 hour, 1 min, 2 frequency(MHz), 3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000), 4 電波源の南北,座標変換した時のx(tangential point との水平方向の距離), 5 座標変換した時のy(tangential pointからの高さ方向の距離),6 電波源の実際の経度]
    data = np.loadtxt(
        "../result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/calculated_all_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_tangential_point_revised.txt"
    )
    H, M, X, Z = Occultation_timing_select(data)
    plt.scatter(X, Z, color="black", linewidth=0.5, label="spacecraft position")

    #  凡例用のダミープロット
    plt.plot(
        [-20000, -20000], [-20000, -20000], color="red", linewidth=0.5, label="ray path"
    )

    # 最大電子密度の動画を作る用
    """
    plt.text(
        3800,
        -380,
        str(float(highest_plasma)) + " (cm⁻³)",
        fontsize=30,
        fontname="Helvetica",
        color="red",
        ha="center",
        va="center",
    )
    """

    # スケールハイト用を作る用
    """
    plt.text(
        3800,
        -380,
        str(float(plasma_scaleheight)) + " (km)",
        fontsize=30,
        fontname="Helvetica",
        color="red",
        ha="center",
        va="center",
    )
    """

    plt.legend(loc="lower right")

    annotations = list(np.zeros(len(H)))
    for i in range(len(H)):
        hour_str = str(int(H[i]))
        min_sr = str(int(M[i]))
        annotations[i] = hour_str + ":" + min_sr

    for i, label in enumerate(annotations):
        plt.annotate(
            label, (X[i], Z[i]), xytext=(X[i] - 150, Z[i] - 70), annotation_clip=None
        )


def Output_moon_radius(moon_name):
    moon_radius = None

    if moon_name == "io":
        moon_radius = 1821.6

    elif moon_name == "europa":
        moon_radius = 1560.8

    elif moon_name == "ganymede":
        moon_radius = 2634.1

    elif moon_name == "callisto":
        moon_radius = 2410.3

    else:
        print(
            "undefined object_name, please check the object_name (moon name) input and def Output_moon_radius function"
        )

    return moon_radius


def main():
    # 対象フライバイにおける電波源の最高高度と最低高度のリスト
    Highest = Radio_Range.highest
    Lowest = Radio_Range.lowest
    Except = Radio_Range.exc
    Freq_str, Freq_underline = Get_frequency(object_name)
    Freq_num = []

    for idx in Freq_str:
        Freq_num.append(float(idx) / 1000000)

    Freq_num = np.array(Freq_num)
    freq = np.where(Freq_num == float(frequency) / 1000000)[0][0]
    raytrace_lowest_altitude = Lowest[freq]
    raytrace_highest_altitude = Highest[freq]

    raytrace_lowest_altitude = Lowest[freq]
    raytrace_highest_altitude = Highest[freq]
    plt.figure(figsize=(16, 4))
    plt.title(
        "Maximum density "
        + highest_plasma
        + "(cm-3), scale height "
        + plasma_scaleheight
        + "(km), frequency "
        + str(round(float(frequency) / 1000000, 2))
        + "(MHz)"
    )
    plt.xlabel("x (km) / tangential direction")
    plt.ylabel("z (km) / normal direction")
    plt.xlim(-2000, 6000)
    plt.ylim(-500, 700)

    print(raytrace_lowest_altitude)

    for i in range(
        raytrace_lowest_altitude, raytrace_highest_altitude, altitiude_interval
    ):
        ray_plot(i)

    radius = Output_moon_radius(object_name)
    t = np.arange(-1 * radius, radius, 2)
    c = np.sqrt(radius * radius - t * t) - radius

    plt.plot(t, c, color="black")
    n = -1600 + t * 0
    plt.plot(t, n, color="black")
    plt.fill_between(t, c, n, facecolor="black")
    plt.annotate("Ganymede", (-450, -300), color="white", fontsize="xx-large")
    spacecraft_plot()
    plt.savefig(
        "../result_for_yasudaetal2022/ray_path_plot_for_paper/maximum density "
        + highest_plasma
        + "(cm-3), scale height "
        + plasma_scaleheight
        + "(km), frequency "
        + str(round(float(frequency) / 1000000, 2))
        + "(MHz).png",
        dpi=1000,
    )
    plt.show()


if __name__ == "__main__":
    main()

# %%

# In[]
