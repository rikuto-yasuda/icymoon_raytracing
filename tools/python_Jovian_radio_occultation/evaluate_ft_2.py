# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys


# %%
args = sys.argv

####################################################
object_name = "callisto"  # ganydeme/europa/calisto`

spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 9  # ..th flyby
occultaion_type = "egress"  # 'ingress' or 'egress
radio_type_A2D = "D"  # 'A' or 'B' or 'C' or 'D'

# object_name = args[1]  # ganydeme/europa/calisto`
# spacecraft_name = "galileo"  # galileo/JUICE(?)
# time_of_flybies = int(args[2])  # ..th flyby
# occultaion_type = args[3]  # 'ingress' or 'egress
# radio_type_A2D = args[1]  # 'A' or 'B' or 'C' or 'D'
print(radio_type_A2D)


boundary_average_str = "10"  # boundary_intensity_str = '10'⇨ノイズフロアの10倍強度まで

# %%

max = []
scale = []
dif = []
kai2_temp = []
kai_comp = []


def MakeFolder(boundary_intensity_str):
    os.makedirs(
        "../result_for_yasudaetal2022/evaluate_f-t_diagram_plot_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_flyby_radioint_"
        + boundary_intensity_str
    )  # 結果を格納するフォルダを生成


def maxandscale(file):
    filename = file
    sep = "_"
    t = filename.split(sep)
    max_density = t[14]
    scale_height = t[15]

    # print(max_density, scale_height)
    return max_density, scale_height


# ずれ時間の計算関数・保存機能なし


def plot_difference(highest, scaleheight, radio_type, using_frequency_range):
    # [[frequencyの配列] [time_lagの配列]]

    time_diffrence_index = np.loadtxt(
        "../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_flyby_radioint_"
        + boundary_average_str
        + "dB/interpolated_"
        + object_name
        + "_"
        + highest
        + "_"
        + scaleheight
        + "_"
        + occultaion_type
        + "_defference_time_data"
        + radio_type
        + "_"
        + boundary_average_str
        + "dB.txt"
    )

    limited_time_list = np.array(
        np.where(
            (time_diffrence_index[0][:] > using_frequency_range[0])
            & (time_diffrence_index[0][:] < using_frequency_range[1])
        )
    )
    limited_time_minimum = limited_time_list[0][0]  # 最低周波数の位置
    limited_time_maximum = limited_time_list[0][
        len(limited_time_list[0][:]) - 1
    ]  # 最高周波数の位置

    frequency_number = limited_time_maximum + 1 - limited_time_minimum

    average_difference_time = (
        sum(time_diffrence_index[1][limited_time_minimum : limited_time_maximum + 1])
    ) / frequency_number
    """
    print("using frequency")
    print(time_diffrence_index[0][limited_time_minimum : limited_time_maximum + 1])
    """

    # シグマを1と置いたときのkai2じょう
    kai2_temporary = np.dot(
        time_diffrence_index[1][limited_time_minimum : limited_time_maximum + 1],
        time_diffrence_index[1][limited_time_minimum : limited_time_maximum + 1],
    )

    max.append(float(highest))
    scale.append(float(scaleheight))
    dif.append(float(average_difference_time))
    kai2_temp.append(float(kai2_temporary))

    return frequency_number


def mindif_density():
    scale_list = np.unique(np.array(scale))
    # print(scale_list)
    max_np = np.array(max)
    scale_np = np.array(scale)
    dif_np = np.array(dif)

    for sca in scale_list:
        scaleselected_dif_position = np.where(scale_np == sca)
        min_dif_time = np.min(dif_np[scaleselected_dif_position])
        minselected_dif_position = np.where(dif_np == min_dif_time)
        selected_position = np.intersect1d(
            scaleselected_dif_position, minselected_dif_position
        )

        selected_max_density = max_np[selected_position]

        print(
            "scale_height="
            + str(sca)
            + "  dif_time ="
            + str(round(min_dif_time, 1))
            + "  max_den = "
            + str(selected_max_density)
        )

    return 0


def get_frequency_intensity_plotparameter(
    moon_name, flyby_time, ingress_or_egerss, radio_type
):
    # print(moon_name, flyby_time, ingress_or_egerss, radio_type)
    if moon_name == "ganymede":
        plot_scale_list = np.array([50, 100, 300, 600, 1000, 1500])

        if flyby_time == 1:
            if ingress_or_egerss == "ingress":
                maxdensity_max = 420
                maxdensity_min = -20

                scale_max = 2000
                scale_min = 30

                dot_size = 40
                fig_holizontal = 7
                fig_vertical = 5
                time_lag_color_bar = np.linspace(10, 60, 11)

                using_frequency_range = [8.0e-1, 4.5]  # G1 ingress

            elif ingress_or_egerss == "egress":
                maxdensity_max = 420
                maxdensity_min = -20

                scale_max = 2000
                scale_min = 30

                dot_size = 40
                fig_holizontal = 7
                fig_vertical = 5
                time_lag_color_bar = np.linspace(10, 60, 11)

                using_frequency_range = [6.5e-1, 4.5]  # G1 egress

    if moon_name == "callisto":
        plot_scale_list = np.array([400, 600, 900])

        if flyby_time == 30:
            if ingress_or_egerss == "ingress":
                maxdensity_max = 1020
                maxdensity_min = -20

                scale_max = 1000
                scale_min = 350

                dot_size = 40
                fig_holizontal = 7
                fig_vertical = 5
                time_lag_color_bar = np.linspace(10, 20, 11)

                using_frequency_range = [8.0e-1, 4.5]

            elif ingress_or_egerss == "egress":
                maxdensity_max = 1020
                maxdensity_min = -20

                scale_max = 1000
                scale_min = 350

                dot_size = 40
                fig_holizontal = 7
                fig_vertical = 5
                time_lag_color_bar = np.linspace(10, 20, 11)

                using_frequency_range = [6.5e-1, 4.5]

        elif flyby_time == 9:
            if ingress_or_egerss == "egress":
                maxdensity_max = 1020
                maxdensity_min = -20

                scale_max = 1000
                scale_min = 350

                dot_size = 40
                fig_holizontal = 7
                fig_vertical = 5
                time_lag_color_bar = np.linspace(10, 60, 11)
                using_frequency_range = [6.5e-1, 5.0]  # C9 egres

    # print(using_frequency_range)

    return (
        using_frequency_range,
        maxdensity_max,
        maxdensity_min,
        scale_max,
        scale_min,
        dot_size,
        fig_holizontal,
        fig_vertical,
        plot_scale_list,
        time_lag_color_bar,
    )


def fig_and_save_def(
    def_data,
    frequency_range,
    holizontal_size,
    vertical_size,
    dot,
    ymax,
    ymin,
    plot_scale,
    xmax,
    xmin,
    color_bar,
):
    np.savetxt(
        "../result_for_yasudaetal2022/evaluate_f-t_diagram_plot_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_flyby_radioint_"
        + boundary_average_str
        + "dB/interpolated_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "flyby_radiointensity_"
        + boundary_average_str
        + "dB_"
        + occultaion_type
        + "_"
        + radio_type_A2D
        + "_"
        + str(frequency_range)
        + "output_array.csv",
        def_data,
        fmt="%.2f",
        delimiter=",",
    )

    cmap = ListedColormap(
        [
            "#ff1900",
            "#ff7e00",
            "#ffdb00",
            "#aaff4d",
            "#39ffbd",
            "#00beff",
            "#0063fe",
            "#1203ff",
            "#040080",
            "#000000",
        ]
    )
    # bounds = np.linspace(10, 20, 11)
    # bounds = np.linspace(10, 60, 11)
    norm = BoundaryNorm(color_bar, cmap.N)

    plt.figure(figsize=(holizontal_size, vertical_size))
    plt.rcParams["axes.axisbelow"] = True

    # 指定されたスケールハイトの結果のみをプロットするためのfor and if文
    # 四捨五入
    label_swich = 1
    for i in range(len(max)):
        if np.any(plot_scale == scale[i]):
            if round(dif[i], 1) == round(np.min(dif), 1):
                if label_swich == 1:
                    sc = plt.scatter(
                        max[i],
                        scale[i],
                        s=dot * 3,
                        marker="*",
                        c=dif[i],
                        norm=norm,
                        cmap=cmap,
                        edgecolor="black",
                        zorder=2,
                        label="Best-fit parameter set",
                    )
                    label_swich = 0

                else:
                    sc = plt.scatter(
                        max[i],
                        scale[i],
                        s=dot * 3,
                        marker="*",
                        c=dif[i],
                        norm=norm,
                        cmap=cmap,
                        edgecolor="black",
                        zorder=2,
                    )

            else:
                sc = plt.scatter(
                    max[i],
                    scale[i],
                    s=dot,
                    c=dif[i],
                    norm=norm,
                    cmap=cmap,
                    zorder=1,
                )

    plt.legend(loc="lower right")
    plt.yscale("log")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.colorbar(sc, label="Average time lag (sec)")
    plt.xlabel("Maximum density (cm-3)")
    plt.ylabel("Scale height (km)")
    plt.title(
        object_name.capitalize()
        + str(time_of_flybies)
        + "_"
        + occultaion_type
        + "_"
        + radio_type_A2D
        + "_"
        + str(frequency_range)
        + "MHz_time lags"
    )
    plt.savefig(
        os.path.join(
            "../result_for_yasudaetal2022/evaluate_f-t_diagram_plot_"
            + spacecraft_name
            + "_"
            + object_name
            + "_"
            + str(time_of_flybies)
            + "_flyby_radioint_"
            + boundary_average_str
            + "dB",
            "interpolated_"
            + spacecraft_name
            + "_"
            + object_name
            + "_"
            + str(time_of_flybies)
            + "flyby_radiointensity_"
            + boundary_average_str
            + "dB_"
            + occultaion_type
            + "_"
            + radio_type_A2D
            + "_"
            + str(frequency_range)
            + "_f-t_evaluate.jpg",
        ),
        format="jpg",
        dpi=600,
    )
    plt.show()


def main():
    (
        using_frequency_range,
        xmaximum,
        xminimum,
        ymaximum,
        yminimum,
        dot_size,
        fig_holizontal_size,
        fig_vertical_size,
        plot_scale_array,
        color_bar_array,
    ) = get_frequency_intensity_plotparameter(
        object_name, time_of_flybies, occultaion_type, radio_type_A2D
    )

    # MakeFolder(boundary_intensity)  # フォルダ作成　初めだけ使う

    use_files = sorted(
        glob.glob(
            "../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_"
            + spacecraft_name
            + "_"
            + object_name
            + "_"
            + str(time_of_flybies)
            + "_flyby_radioint_"
            + boundary_average_str
            + "dB/interpolated_"
            + object_name
            + "_*_"
            + occultaion_type
            + "_defference_time_data"
            + radio_type_A2D
            + "_"
            + boundary_average_str
            + "dB.txt"
        )
    )

    for file in use_files:
        highest_density_str, plasma_scaleheight_str = maxandscale(file)

        frequency_kinds = plot_difference(
            highest_density_str,
            plasma_scaleheight_str,
            radio_type_A2D,
            using_frequency_range,
        )

    mindif_density()  # スケールハイトごとにずれ最小となる密度を出力

    # ずれ時間を散布図にする部分
    ### ここから#
    output_array = np.array(max + scale + dif)
    output_array = output_array.reshape(3, int(len(output_array) / 3)).T

    # print(output_array)

    fig_and_save_def(
        output_array,
        using_frequency_range,
        fig_holizontal_size,
        fig_vertical_size,
        dot_size,
        ymaximum,
        yminimum,
        plot_scale_array,
        xmaximum,
        xminimum,
        color_bar_array,
    )  # ずれ時間とカラーマップを保存

    return 0


if __name__ == "__main__":
    main()


# %%
