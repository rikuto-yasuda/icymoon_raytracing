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
# object_name = "ganymede"  # ganyindeme/europa/callisto`
object_name = "callisto"  # ganyindeme/europa/callisto`

spacecraft_name = "galileo"  # galileo/JUICE(?)
# time_of_flybies = 9  # ..th flyby
time_of_flybies = 30  # ..th flyby

occultaion_type = "ingress"  # 'ingress' or 'egress

radio_type_A2D = "C"  # 'A' or 'B' or 'C' or 'D'
# radio_type_A2D = "C"  # 'A' or 'B' or 'C' or 'D'

uncertainty_off = "off"

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
        # plot_scale_list = np.array([50, 100, 300, 600, 1000, 1500])
        plot_scale_list = np.array([25, 50, 100, 300, 600, 1000, 1500])

        if flyby_time == 1:
            if ingress_or_egerss == "ingress":
                maxdensity_max = 420
                maxdensity_min = -20

                scale_max = 2000
                scale_min = 20

                dot_size = 40
                fig_holizontal = 7
                fig_vertical = 5
                time_lag_color_bar = np.linspace(10, 60, 11)

                using_frequency_range = [8.0e-1, 4.5]  # G1 ingress
                contor_dif = [30, 40, 50]
                uncertainty = 36.3  # typeB

            elif ingress_or_egerss == "egress":
                maxdensity_max = 420
                maxdensity_min = -20

                scale_max = 2000
                scale_min = 20

                dot_size = 40
                fig_holizontal = 7
                fig_vertical = 5
                time_lag_color_bar = np.linspace(10, 60, 11)

                using_frequency_range = [6.5e-1, 4.5]  # G1 egress
                contor_dif = [20, 30, 40]
                uncertainty = 24.4  # typeA

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
                contor_dif = [10, 11, 12, 13]
                uncertainty = 5.5  # typeC

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
                contor_dif = [35, 40, 45]
                uncertainty = 9.7  # typeD

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
        contor_dif,
        uncertainty,
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
    contor_dif,
    uncertainty,
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
    max_array = np.empty(0)
    scale_array = np.empty(0)
    dif_array = np.empty(0)

    for i in range(len(max)):
        if np.any(plot_scale == scale[i]):
            max_array = np.append(max_array, max[i])
            scale_array = np.append(scale_array, scale[i])
            dif_array = np.append(dif_array, dif[i])
            if round(dif[i], 1) == round(np.min(dif), 1):
                if label_swich == 1:
                    sc = plt.scatter(
                        max[i],
                        scale[i],
                        s=dot * 3,
                        marker="*",
                        c=dif[i],
                        cmap="rainbow_r",
                        edgecolor="black",
                        zorder=2,
                        vmin=color_bar[0],
                        vmax=color_bar[-1],
                        label="Best-fit parameter set",
                    )
                    best_max_value = max[i]
                    best_scale_value = scale[i]

                    label_swich = 0

                else:
                    sc = plt.scatter(
                        max[i],
                        scale[i],
                        s=dot * 3,
                        marker="*",
                        c=dif[i],
                        cmap="rainbow_r",
                        edgecolor="black",
                        vmin=color_bar[0],
                        vmax=color_bar[-1],
                        zorder=2,
                    )

            else:
                sc = plt.scatter(
                    max[i],
                    scale[i],
                    s=dot,
                    c=dif[i],
                    cmap="rainbow_r",
                    zorder=1,
                    vmin=color_bar[0],
                    vmax=color_bar[-1],
                )

    x_int_array = np.arange(max_array.min(), max_array.max(), 0.5)
    y_int_array = np.arange(scale_array.min(), scale_array.max(), 0.5)
    xi, yi = np.meshgrid(x_int_array, y_int_array)

    # データの補間
    zi = griddata((max_array, scale_array), dif_array, (xi, yi), method="cubic")
    contour_plot = plt.contour(xi, yi, zi, levels=contor_dif, colors="gray")
    clabels = plt.clabel(
        contour_plot, inline=True, fontsize=8, colors="black", fmt="%1.1f"
    )

    if uncertainty_off != "off":
        uncertainty_color = "pink"
        contour_plot2 = plt.contour(
            xi,
            yi,
            zi,
            levels=np.array([uncertainty]),
            colors=uncertainty_color,
        )

        clabels = plt.clabel(
            contour_plot2, inline=True, fontsize=8, colors="black", fmt="%1.1f"
        )

    if object_name == "ganymede" and uncertainty_off != "off":
        plt.plot(
            np.arange(1, 3, 1),
            np.arange(1, 3, 1),
            c=uncertainty_color,
            label="Uncertainty range",
        )
        best_max_pos = np.where(x_int_array == best_max_value)[0]
        best_scale_pos = np.where(y_int_array == best_scale_value)[0]

        x_value_range = x_int_array[np.where(zi[best_scale_pos] < uncertainty)[1]]
        y_value_range = np.where(zi[:, best_max_pos].T[0] < uncertainty)[0]

        start_indices = np.where(y_value_range == best_scale_pos)[
            0
        ]  # 10が現れるインデックスを取得
        sequences = []  # 単調数列を格納するリスト

        # 各10を含む単調数列を取得
        for start_index in start_indices:
            sequence = [y_value_range[start_index]]  # 数列の最初の要素を追加
            current_index = start_index

            # 10を含む単調数列を前方に拡張
            while (
                current_index > 0
                and y_value_range[current_index] - y_value_range[current_index - 1] == 1
            ):
                sequence.insert(0, y_value_range[current_index - 1])
                current_index -= 1

            current_index = start_index

            # 10を含む単調数列を後方に拡張
            while (
                current_index < len(y_value_range) - 1
                and y_value_range[current_index + 1] - y_value_range[current_index] == 1
            ):
                sequence.append(y_value_range[current_index + 1])
                current_index += 1

            sequences.append(sequence)

            print("density " + str(x_value_range[0]) + "~" + str(x_value_range[-1]))
            print(
                "scale "
                + str(y_int_array[sequences[0][0]])
                + "~"
                + str(y_int_array[sequences[0][-1]])
            )

            plt.errorbar(
                best_max_value,
                best_scale_value,
                xerr=[
                    [best_max_value - x_value_range[0]],
                    [x_value_range[-1] - best_max_value],
                ],
                yerr=[
                    [best_scale_value - y_int_array[sequences[0][0]]],
                    [y_int_array[sequences[0][-1]] - best_scale_value],
                ],
                fmt=".",
                markersize=0.001,
                ecolor="black",
                elinewidth=0.5,
                markeredgecolor="black",
                color="w",
                capsize=3,
            )
    if object_name == "ganymede":
        plt.legend(bbox_to_anchor=(1, 0.55))

    if object_name == "callisto":
        plt.legend(loc="lower right")

    plt.yscale("log")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.colorbar(sc, label="Average time difference (sec)")
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
    if uncertainty_off != "off":
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
                + "_f-t_evaluate_for_Fig7.jpg",
            ),
            format="jpg",
            dpi=600,
        )
    else:
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
        contor_dif,
        uncertainty,
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
        contor_dif,
        uncertainty,
    )  # ずれ時間とカラーマップを保存

    return 0


if __name__ == "__main__":
    main()


# %%
