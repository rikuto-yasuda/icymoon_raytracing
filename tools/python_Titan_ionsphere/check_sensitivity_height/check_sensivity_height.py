# In[]
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as mpt
import pandas as pd
from matplotlib.colors import LogNorm

object_name = "titan"  # titan
spacecraft_name = "cassini"  # cassini
time_of_flybies = 15  # ..th flyby
highest_plasma = "1300"  # 単位は(/cc)
peak_height = "1400"  # 単位は(km) 
standard_deviation = "200" #単位は(km)
frequency ='320430' #Freq_str = ['90071', '113930', '173930', '241690', '278290', '320430', '368750', '431250', '468750', '531250', '568750', '631250', '668750', '731250', '768750', '831250',  '868750',  '931250', '968750']
altitiude_interval = 2
radio_type = "D"  # 複数選択可能にしたい

results_path = "/work1/rikutoyasuda/tools/result_titan/"
occultation_begining_hour = 10
occultation_begining_minute = 10 #偶数のみ

occultation_end_hour = 10
occultation_end_minute = 30


frequency_number = float(frequency) / 1000000 # MHz
occultation_lowest_frequency = frequency_number - 0.01
occultation_highest_frequecy = frequency_number + 0.01
using_frequency_array = np.array([0.090071, 0.113930, 0.173930, 0.241690, 0.278290, 0.320430, 0.368750, 0.431250, 0.468750, 0.531250, 0.568750, 0.631250, 0.668750, 0.731250, 0.768750, 0.831250, 0.868750, 0.931250, 0.968750])

sensitivity_start_hour = 10
sensitivity_start_minute = 25
sensitivity_end_hour = 10
sensitivity_end_minute = 30

# 対象のフライバイ時における角周波数での電波の幅をまとめたファイルを取得
Radio_name_cdf = (
    results_path
    + "tracing_range_"
    + spacecraft_name
    + "_"
    + object_name
    + "_"
    + str(time_of_flybies)
    + "_flybys/para_"
    + highest_plasma
    + "_"
    + peak_height
    + "_"
    + standard_deviation
    + ".csv"
)
Radio_Range = pd.read_csv(Radio_name_cdf, header=0)


# 天体ごとにしようしている周波数を出力
def Get_frequency(object):
    if object == "titan":
        Freq_str = ['51242.6', '53708.6', '56293.3', '59002.4', '61841.8', '64817.9', '67937.3', '71206.7', '74633.5', '78225.2', '81989.8', '85935.5', '90071.1', '94405.7', '98949', '103710.9', '108701.9', '113933.1', '119416.1', '125162.9', '131186.4', '137499.6', '144116.7', '151052.31', '158321.59', '165940.8', '173926.61', '182296.71', '191069.7', '200264.8', '209902.5', '220003.91', '230591.51', '241688.6', '253319.79', '265510.71', '278288.3', '291680.79', '305717.8', '320430.3', '318750', '331250', '343750', '356250', '368750', '381250', '393750', '406250', '418750', '431250', '443750', '456250', '468750', '481250', '493750', '506250', '518750', '531250', '543750', '556250', '568750', '581250', '593750', '606250', '618750', '631250', '643750', '656250', '668750', '681250', '693750', '706250', '718750', '731250', '743750', '756250', '768750', '781250', '793750', '806250', '818750', '831250', '843750', '856250', '868750', '881250', '893750', '906250', '918750', '931250', '943750', '956250', '968750', '981250', '993750']

    else:
        print("object name is not correct")

    return Freq_str


def ray_plot(height):
    data = np.loadtxt(
        results_path
        + "raytracing_results/"
        + "Nmax_"
        + highest_plasma
        + "-hpeak_"
        + peak_height
        + "-sigma_"
        + standard_deviation
        + "/Pla_titan_gaussian-Nmax_"
        + highest_plasma
        + "-hpeak_"
        + peak_height
        + "-sigma_"
        + standard_deviation
        + "-Mag_test_simple-Mode_RX-Freq_"
        + frequency
        + "Hz-X_-6500km-Y_0km-Z_"
        + str(height)
        + "km"
    )
    x = data[:, [1]]
    z = data[:, [3]]
    plt.plot(x, z, color="red", linewidth=0.5)

    from_surface_height = np.sqrt(x**2 + (z+2574.7)**2) - 2574.7
    min_height = np.min(from_surface_height)

    return min_height

"""
def check_sensitivity_altitude(height, sensitivity_x, sensitivity_z):
    # height: 2km刻みの高度
    # sensitivity_x: x方向の距離
    # sensitivity_z: z方向の距離

    data = np.loadtxt(
        results_path
        + "raytracing_results/"
        + "Nmax_"
        + highest_plasma
        + "-hpeak_"
        + peak_height
        + "-sigma_"
        + standard_deviation
        + "/Pla_titan_gaussian-Nmax_"
        + highest_plasma
        + "-hpeak_"
        + peak_height
        + "-sigma_"
        + standard_deviation
        + "-Mag_test_simple-Mode_RX-Freq_"
        + frequency
        + "Hz-X_-6500km-Y_0km-Z_"
        + str(height)
        + "km"
    )
    x = data[:, [1]]
    z = data[:, [3]]
    
    x0 = sensitivity_x[0]
    z0 = sensitivity_z[0]
    x1 = sensitivity_x[-1]
    z1 = sensitivity_z[-1]

    if x1 < x0:
        x0, x1 = x1, x0
        z0, z1 = z1, z0

    # xとzの配列によって生成される線が(x0,z1)と(x1,z1)の間を通過するかを確認する
    
    x_start = np.where(x >= x0)[0]
    x_end = np.where(x >= x1)[0]

    if len(x_start) == 0:
        return None
    
    if len(x_end) == 0:
        return None
    
    x_start = x_start[0]
    x_end = x_end[0]

    x_ratio4z0 = x0-x[x_start-1]/x[x_start]- x[x_start-1]
    z0_interpolated = z[x_start-1] + (z[x_start] - z[x_start-1]) * x_ratio4z0


    x_ratio4z1 = x1-x[x_end-1]/x[x_end]- x[x_end-1]
    z1_interpolated = z[x_end-1] + (z[x_end] - z[x_end-1]) * x_ratio4z1

    if (z0_interpolated <= z0 and z1_interpolated >= z1) or (z0_interpolated >= z0 and z1_interpolated <= z1):
        #print("sensitivity height is included in the ray path")
        plt.plot(x, z, color="purple", linewidth=0.5)

        from_surface_height = np.sqrt(x**2 + (z+2574.7)**2) - 2574.7
        min_height = np.min(from_surface_height)
        return min_height
"""

def Occultation_timing_select():
        # [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]

    data = np.loadtxt(
        results_path
        + "calculated_expres_detectable_radio_data_of_each_flyby/interpolated_calculated_all_"
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


            Lat = judgement[k][5]
            Lon = judgement[k][10]



            if judgement[k][2] == 1:
                
                if "A" == radio:
                    if Lon > 0 and Lat > 0:
                        selected_data[k, :] = judgement[k, :].copy()
                        t = judgement[k][1]

                if "B" == radio:
                    if Lon < 0 and Lat > 0:
                        selected_data[k, :] = judgement[k, :].copy()
                        t = judgement[k][1]

                if "C" == radio:
                    if Lon > 0 and Lat < 0:
                        selected_data[k, :] = judgement[k, :].copy()
                        t = judgement[k][1]

                if "D" == radio:
                    if Lon < 0 and Lat < 0:
                        selected_data[k, :] = judgement[k, :].copy()
                        t = judgement[k][1]

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
        # [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]
 
        start_hour_select = np.where(radio_data[:, 0] == begining_hour)
        start_minute_select = np.where(radio_data[:, 1] == begining_min)

        start_select = np.intersect1d(start_hour_select, start_minute_select)[0]

        #print(start_select)


        end_hour_select = np.where(radio_data[:, 0] == end_hour)
        end_minute_select = np.where(radio_data[:, 1] == end_min)
        end_select = np.intersect1d(end_hour_select, end_minute_select)[-1]

        #print(end_select)

        time_select = radio_data[start_select:end_select, :]
        #print(time_select[:, 3])
        freq_select = np.where(
            (time_select[:, 3] > lowest_freq) & (time_select[:, 3] < highest_freq)
        )[0]



        selected_freq_data = time_select[freq_select, :]

        detectable_select = np.where(selected_freq_data[:, 7] > 0)[0]
        selected_detectable_data = selected_freq_data[detectable_select, :]
        z_height = selected_detectable_data[:, 7]
        x_holizon = selected_detectable_data[:, 6]
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
    #print(x, z)
    return h, m, x, z
    # np.savetxt('../result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/calculated_all_' +spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_'+radio_type+'_tangential_point.txt', selected_data, fmt="%s")

"""
def Pick_up_sensitivity_height():
        # [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]

    data = np.loadtxt(
        results_path
        + "calculated_expres_detectable_radio_data_of_each_flyby/interpolated_calculated_all_"
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


            Lat = judgement[k][5]
            Lon = judgement[k][10]



            if judgement[k][2] == 1:
                
                if "A" == radio:
                    if Lon > 0 and Lat > 0:
                        selected_data[k, :] = judgement[k, :].copy()
                        t = judgement[k][1]

                if "B" == radio:
                    if Lon < 0 and Lat > 0:
                        selected_data[k, :] = judgement[k, :].copy()
                        t = judgement[k][1]

                if "C" == radio:
                    if Lon > 0 and Lat < 0:
                        selected_data[k, :] = judgement[k, :].copy()
                        t = judgement[k][1]

                if "D" == radio:
                    if Lon < 0 and Lat < 0:
                        selected_data[k, :] = judgement[k, :].copy()
                        t = judgement[k][1]

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
        # [0 hour,1 min,2 sec, 3 frequency(MHz),4 電波源データの磁力線(根本)の経度  orイオの場合は(-1000),5 電波源の南北,6 座標変換した時のx(tangential point との水平方向の距離),7 座標変換した時のy(tangential pointからの高さ方向の距離),8 電波源の実際の経度,9 探査機の経度,10 電波源タイプ（A,C:1 B,D:-1)]
 
        start_hour_select = np.where(radio_data[:, 0] == begining_hour)
        start_minute_select = np.where(radio_data[:, 1] == begining_min)

        start_select = np.intersect1d(start_hour_select, start_minute_select)[0]



        end_hour_select = np.where(radio_data[:, 0] == end_hour)
        end_minute_select = np.where(radio_data[:, 1] == end_min)
        end_select = np.intersect1d(end_hour_select, end_minute_select)[-1]

        time_select = radio_data[start_select:end_select, :]
        print(time_select[:, 3])
        freq_select = np.where(
            (time_select[:, 3] > lowest_freq) & (time_select[:, 3] < highest_freq)
        )[0]



        selected_freq_data = time_select[freq_select, :]

        detectable_select = np.where(selected_freq_data[:, 7] > 0)[0]
        selected_detectable_data = selected_freq_data[detectable_select, :]
        z_height = selected_detectable_data[:, 7]
        x_holizon = selected_detectable_data[:, 6]
        select_hour = selected_detectable_data[:, 0]
        select_minute = selected_detectable_data[:, 1]
        return select_hour, select_minute, x_holizon, z_height

    source_type_selected = radio_source_select(data, radio_type)


    h, m, x, z = time_select_and_xz(
        source_type_selected,
        sensitivity_start_hour,
        sensitivity_start_minute,
        sensitivity_end_hour,
        sensitivity_end_minute,
        occultation_lowest_frequency,
        occultation_highest_frequecy,
    )
    #print(x, z)
    return h, m, x, z

"""


def spacecraft_plot():
    # [0 hour, 1 min, 2 frequency(MHz), 3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000), 4 電波源の南北,座標変換した時のx(tangential point との水平方向の距離), 5 座標変換した時のy(tangential pointからの高さ方向の距離),6 電波源の実際の経度]

    H, M, X, Z = Occultation_timing_select()

    #print(X, Z)
    plt.scatter(X, Z, color="black", linewidth=0.2, label="spacecraft position")

    #sensitivity_H, sensitivity_M, sensitivity_X, sensitivity_Z = Pick_up_sensitivity_height() 
    """
    print(sensitivity_X, sensitivity_Z)
    plt.scatter(
        sensitivity_X,
        sensitivity_Z,
        color="blue",
        linewidth=1,
        label="sensitivity height",
    )
    """
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
        str(float(peak_height)) + " (km)",
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
            label, (X[i], Z[i]), xytext=(X[i] + 700, Z[i] - 100), annotation_clip=None
        )
    return None
    

def Output_moon_radius(moon_name):
    moon_radius = None

    if moon_name == "titan":
        moon_radius = 2574.7

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
    Freq_str = Get_frequency(object_name)
    Freq_num = []

    for idx in Freq_str:
        Freq_num.append(float(idx) / 1000000)

    #Freq_num = np.array(Freq_num)
    #print(float(frequency) / 1000000)
    #print(using_frequency_array)
    freq = np.where(using_frequency_array == float(frequency) / 1000000)[0][0]
    #print(Lowest)
    raytrace_lowest_altitude = Lowest[freq]
    raytrace_highest_altitude = Highest[freq]

    raytrace_lowest_altitude = Lowest[freq]
    raytrace_highest_altitude = Highest[freq]
    plt.figure(figsize=(9, 4))
    plt.title(
        "Maximum density "
        + highest_plasma
        + "(cm-3), peak height "
        + peak_height
        + "(km), standard deviation "
        + standard_deviation
        + "(km), frequency "
        + str(round(float(frequency) / 1000000, 2))
        + "(MHz)"
    )
    plt.xlabel("x (km) / tangential direction")
    plt.ylabel("z (km) / normal direction")
    plt.xlim(-5000, 25000)
    plt.ylim(-1500, 4000)

    #print(raytrace_lowest_altitude)

    lowest_altitude_list = []

    for i in range(
        raytrace_lowest_altitude, raytrace_highest_altitude, altitiude_interval
    ):
        lowest_altitude_list.append(ray_plot(i))
    
    print(lowest_altitude_list)

    print("lowest altitude: " + str(np.min(lowest_altitude_list)))

    radius = Output_moon_radius(object_name)
    t = np.arange(-1 * radius, radius, 2)
    c = np.sqrt(radius * radius - t * t) - radius

    plt.plot(t, c, color="black")
    n = -1600 + t * 0
    plt.plot(t, n, color="black")
    plt.fill_between(t, c, n, facecolor="black")

    ionosphere_radius = radius + np.min(lowest_altitude_list)

    t2 = np.arange(-1 * ionosphere_radius, ionosphere_radius, 2)
    c2 = np.sqrt(ionosphere_radius * ionosphere_radius - t2 * t2) - radius
    plt.plot(t2, c2, color="pink")
    outline = np.loadtxt(
        results_path
        +"raytracing_results/Nmax_"
        +highest_plasma
        +"-hpeak_"
        +peak_height
        +"-sigma_"
        +standard_deviation
        +"/Outlines_titan_gaussian-Nmax_"
        +highest_plasma
        +"-hpeak_"
        +peak_height
        +"-sigma_"
        +standard_deviation
        +"-Mag_test_simple-Mode_RX-Freq_"
        + frequency
        +"Hz"

    )
    plt.plot(outline[:, 0], outline[:, 1], color="green", linewidth=2)
    plt.plot(outline[:, 0], outline[:, 2], color="blue", linewidth=2)
    plt.plot(outline[:, 0], outline[:, 3], color="black", linewidth=2)
    plt.annotate(object_name, (-1080, -1000), color="white", fontsize="xx-large")
    spacecraft_plot()

    """
    sensitivity_height = []
    for i in range(
        raytrace_lowest_altitude, raytrace_highest_altitude, altitiude_interval
    ):
        sensitivity_height.append(
        check_sensitivity_altitude(i, sensitivity_x, sensitivity_z)
    )
    # 例: sensitivity_height から None を除く
    sensitivity_height = [h for h in sensitivity_height if h is not None]

    print("sensitivity height min:" + str(np.min(sensitivity_height)))
    print("sensitivity height max:" + str(np.max(sensitivity_height)))
    """


    plt.grid()
    plt.savefig(
        "/work1/rikutoyasuda/tools/python_Titan_ionsphere/check_sensitivity_height/test.png",
        dpi=1000,
    )
    plt.show()


if __name__ == "__main__":
    main()

# %%

# In[]
