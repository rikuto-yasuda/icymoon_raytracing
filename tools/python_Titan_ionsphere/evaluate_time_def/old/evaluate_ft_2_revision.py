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
object_name = "titan"  # ganyindeme/europa/callisto`

spacecraft_name = "cassini"  # galileo/JUICE(?)
# time_of_flybies = 9  # ..th flyby
time_of_flybies = 15  # ..th flyby

occultaion_type = "egress"  # 'ingress' or 'egress

radio_type_A2D = "D"  # 'A' or 'B' or 'C' or 'D'
# radio_type_A2D = "C"  # 'A' or 'B' or 'C' or 'D'

uncertainty_off = "off"

# object_name = args[1]  # ganydeme/europa/calisto`
# spacecraft_name = "galileo"  # galileo/JUICE(S?)
# time_of_flybies = int(args[2])  # ..th flyby
# occultaion_type = args[3]  # 'ingress' or 'egress
# radio_type_A2D = args[1]  # 'A' or 'B' or 'C' or 'D'

boundary_average_str = "10"  # boundary_intensity_str = '10'⇨ノイズフロアの10倍強度まで
        

# %%

max = []
peakh = []
sig = []
time_dif =[]


"""
def MakeFolder(boundary_intensity_str):
    os.makedirs(
        "../../result_titan/evaluate_f-t_diagram_plot_"
        + spacecraft_name
        + "_"
        + object_name
        + "_"
        + str(time_of_flybies)
        + "_flyby_radioint_"
        + boundary_intensity_str
    )  # 結果を格納するフォルダを生成
"""


def get_frequency_intensity_plotparameter(
    moon_name, flyby_time, ingress_or_egerss, radio_type
):
    # print(moon_name, flyby_time, ingress_or_egerss, radio_type)
    if moon_name == "titan":
        # plot_scale_list = np.array([50, 100, 300, 600, 1000, 1500])
        peak_height_array = np.arange(1000, 1401, 100) #4
        max_density_array = np.arange(400, 3501, 100) #61
        sigma_array = np.arange(100, 401, 100) # 3
        using_frequency_array = np.array([0.090071, 0.113930, 0.173930, 0.241690, 0.278290, 0.320430, 0.368750, 0.431250, 0.468750, 0.531250, 0.568750, 0.631250, 0.668750, 0.731250, 0.768750, 0.831250, 0.868750, 0.931250, 0.968750])

        maxdensity_max = 3600
        maxdensity_min = 300

        peak_height_max = 1500
        peak_height_min = 900

        sigma_max = 450
        sigma_min = 50

        if flyby_time == 15:

            if ingress_or_egerss == "ingress":


                dot_size = 40
                fig_holizontal = 7
                fig_vertical = 5
                time_lag_color_bar = np.linspace(60, 180, 11)

                using_frequency_range = [0.1, 0.8]  # T15 ingress
                contor_dif = [30, 40, 50]
                uncertainty = 0  # typeB

            elif ingress_or_egerss == "egress":

                dot_size = 40
                fig_holizontal = 7
                fig_vertical = 5
                time_lag_color_bar = np.linspace(60, 180, 11)

                using_frequency_range = [0.1, 0.65]  # T15 egress
                contor_dif = [20, 30, 40]
                uncertainty = 0  # typeA


    return (
        using_frequency_array,
        using_frequency_range,
        maxdensity_max,
        maxdensity_min,
        peak_height_max,
        peak_height_min,
        sigma_max,
        sigma_min,
        dot_size,
        fig_holizontal,
        fig_vertical,
        peak_height_array,
        max_density_array,
        sigma_array,
        time_lag_color_bar,
        contor_dif,
        uncertainty,
    )

def maxandscale(file):
    filename = file
    sep_str1 = "_"
    sep_str2 = "-"
    t = filename.split(sep_str1)
    max_dens = t[12].split(sep_str2)[0]
    peak_height = t[13].split(sep_str2)[0]
    sigma = t[14].split(sep_str2)[0]

    return max_dens, peak_height, sigma


# ずれ時間の計算関数・保存機能なし


def plot_difference(maxdens, peakheight, sigma, radio_type, using_frequency_array ,using_frequency_range):
    # [[frequencyの配列] [time_lagの配列]]

    time_difference_array = np.loadtxt(
            "../../result_titan/radio_raytracing_occultation_timing_def_"
            + spacecraft_name
            + "_"
            + object_name
            + "_"
            + str(time_of_flybies)
            + "_flyby_radioint_"
            + boundary_average_str
            + "dB/interpolated-"
            + object_name
            + "-Nmax_"
            + maxdens
            + "-hpeak_"
            + peakheight
            + "-sigma_"
            + sigma
            + "-"
            + occultaion_type
            + "_defference_time_data"
            + radio_type_A2D
            + "-"
            + boundary_average_str
            + "dB.txt"
    )
    output_freq_position = np.array(
        np.where(
            (using_frequency_array > using_frequency_range[0])
            & (using_frequency_array < using_frequency_range[1])
        )
    )[0]

    galileo_frequency_array = time_difference_array[0]
    time_difference_array = time_difference_array[1]

    total_time_difference = 0
    total_frequency_number = 0
    max_frequency_difference = 0

    for i in output_freq_position:
        closest_index = np.argmin(abs(galileo_frequency_array - using_frequency_array[i]))
        frequency_difference = abs(
            galileo_frequency_array[closest_index] - using_frequency_array[i]
        )
        total_time_difference += time_difference_array[closest_index]
        total_frequency_number += 1

        if max_frequency_difference < frequency_difference:
            max_frequency_difference = frequency_difference

    
    print("total_frequency_number", total_frequency_number)
    
    average_difference_time = total_time_difference / total_frequency_number

    print("max_frequency_difference", max_frequency_difference)

    max.append(float(maxdens))
    peakh.append(float(peakheight))
    sig.append(float(sigma))
    time_dif.append(float(average_difference_time))

    return 0


def fig_and_save_def(
    def_data,
    frequency_range,
    holizontal_size,
    vertical_size,
    dot,
    maxdensity_max,
    maxdensity_min,
    peak_height_max,
    peak_height_min,
    sigma_max,
    sigma_min,
    color_bar,
    contor_dif,
    uncertainty,
):
    np.savetxt(
        "../../result_titan/evaluate_f-t_diagram_plot_"
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
    best_max_density = def_data[np.argmin(def_data[:,3]),0]
    best_peak_height = def_data[np.argmin(def_data[:,3]),1]
    best_sig = def_data[np.argmin(def_data[:,3]),2]
    print("best_max_density", best_max_density)
    print("best_peak_height", best_peak_height)
    print("best_sig", best_sig)
    print("best_time_difference", def_data[np.argmin(def_data[:,3]),3])

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
    

    min_dens_pos = np.where(def_data[:, 0] == best_max_density)[0]
    min_scale_pos = np.where(def_data[:, 1] == best_peak_height)[0]
    min_sig_pos = np.where(def_data[:, 2] == best_sig)[0]

    ### 最大電子密度固定の場合##############################################
    interpolate_peakheight_array = np.empty(0)
    interpolate_sigma_array = np.empty(0)    
    interpolate_dif_array = np.empty(0)

    for i in min_dens_pos:
        sc = plt.scatter(
            def_data[i,1],
            def_data[i,2],
            s=dot,
            c=def_data[i, 3],
            cmap="rainbow_r",
            zorder=2,
            vmin=color_bar[0],
            vmax=color_bar[-1],
        )
        interpolate_peakheight_array = np.append(
            interpolate_peakheight_array, def_data[i, 1]
        )
        interpolate_sigma_array = np.append(
            interpolate_sigma_array, def_data[i, 2]
        )
        interpolate_dif_array = np.append(
            interpolate_dif_array, def_data[i, 3]
        )
    
    plt.scatter(best_peak_height, best_sig, s=dot * 3, zorder=1,c="black", marker="*", label="Best-fit parameter set")


    x_int_array = np.arange(peak_height_min, peak_height_max, 10)
    y_int_array = np.arange(sigma_min, sigma_max, 10)
    xi, yi = np.meshgrid(x_int_array, y_int_array)

    # データの補間
    zi = griddata((interpolate_peakheight_array, interpolate_sigma_array), interpolate_dif_array, (xi, yi), method="cubic")
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


    plt.xlim(peak_height_min, peak_height_max)
    plt.ylim(sigma_min, sigma_max)
    plt.colorbar(sc, label="Average time difference (sec)")
    plt.xlabel("Peak height (km)")
    plt.ylabel("Sigma (km)")
    plt.title(
        object_name.capitalize()
        + str(time_of_flybies)
        + "_"
        + occultaion_type
        + "_"
        + radio_type_A2D
        + "_peak_height_"
        + str(best_peak_height) 
        + "_"       
        + str(frequency_range)
        + "MHz_time lags"
    )

    plt.savefig(
        os.path.join(
            "../../result_titan/evaluate_f-t_diagram_plot_"
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
            + "_max_density_"
            + str(best_max_density)
            + "_"
            + str(frequency_range)
            + "_f-t_evaluate.jpg",
        ),
        format="jpg",
        dpi=600,
    )

    plt.show()
    plt.close("all")

    ### 最大密度高度固定の場合##############################################

    interpolate_maxdensity_array = np.empty(0)
    interpolate_sigma_array = np.empty(0)    
    interpolate_dif_array = np.empty(0)

    for i in min_scale_pos:
        sc = plt.scatter(
            def_data[i,0],
            def_data[i,2],
            s=dot,
            c=def_data[i, 3],
            cmap="rainbow_r",
            zorder=2,
            vmin=color_bar[0],
            vmax=color_bar[-1],
        )
        interpolate_maxdensity_array = np.append(
            interpolate_maxdensity_array, def_data[i, 0]
        )
        interpolate_sigma_array = np.append(
            interpolate_sigma_array, def_data[i, 2]
        )
        interpolate_dif_array = np.append(
            interpolate_dif_array, def_data[i, 3]
        )
    
    plt.scatter(best_max_density, best_sig, s=dot * 3, zorder=1,c="black", marker="*", label="Best-fit parameter set")


    x_int_array = np.arange(maxdensity_min,maxdensity_max, 10)
    y_int_array = np.arange(sigma_min, sigma_max, 10)
    xi, yi = np.meshgrid(x_int_array, y_int_array)

    # データの補間
    zi = griddata((interpolate_maxdensity_array, interpolate_sigma_array), interpolate_dif_array, (xi, yi), method="cubic")
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


    plt.xlim(maxdensity_min,maxdensity_max)
    plt.ylim(sigma_min, sigma_max)
    plt.colorbar(sc, label="Average time difference (sec)")
    plt.xlabel("Max density (cm-3)")
    plt.ylabel("Sigma (km)")
    plt.title(
        object_name.capitalize()
        + str(time_of_flybies)
        + "_"
        + occultaion_type
        + "_"
        + radio_type_A2D
        + "_peak_height_"
        + str(best_peak_height) 
        + "_"       
        + str(frequency_range)
        + "MHz_time lags"
    )

    plt.savefig(
        os.path.join(
            "../../result_titan/evaluate_f-t_diagram_plot_"
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
            + "_peak_height_"
            + str(best_peak_height)
            + "_"
            + str(frequency_range)
            + "_f-t_evaluate.jpg",
        ),
        format="jpg",
        dpi=600,
    )

    plt.show()
    plt.close("all")

    ### シグマ固定の場合##############################################

    interpolate_maxdensity_array = np.empty(0)
    interpolate_peakheight_array = np.empty(0)   
    interpolate_dif_array = np.empty(0)

    for i in min_sig_pos:
        sc = plt.scatter(
            def_data[i,0],
            def_data[i,1],
            s=dot,
            c=def_data[i, 3],
            cmap="rainbow_r",
            zorder=2,
            vmin=color_bar[0],
            vmax=color_bar[-1],
        )
        interpolate_maxdensity_array = np.append(
            interpolate_maxdensity_array, def_data[i, 0]
        )
        interpolate_peakheight_array = np.append(
            interpolate_peakheight_array, def_data[i, 1])
        interpolate_dif_array = np.append(
            interpolate_dif_array, def_data[i, 3]
        )
    
    plt.scatter(best_max_density, best_peak_height, s=dot * 3, zorder=1,c="black", marker="*", label="Best-fit parameter set")


    x_int_array = np.arange(maxdensity_min,maxdensity_max, 10)
    y_int_array = np.arange(peak_height_min, peak_height_max, 10)
    xi, yi = np.meshgrid(x_int_array, y_int_array)

    # データの補間
    zi = griddata((interpolate_maxdensity_array, interpolate_peakheight_array), interpolate_dif_array, (xi, yi), method="cubic")
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


    plt.xlim(maxdensity_min, maxdensity_max)
    plt.ylim(peak_height_min, peak_height_max)
    plt.colorbar(sc, label="Average time difference (sec)")
    plt.xlabel("Max density (cm-3)")
    plt.ylabel("Peak height (km)")
    plt.title(
        object_name.capitalize()
        + str(time_of_flybies)
        + "_"
        + occultaion_type
        + "_"
        + radio_type_A2D
        + "_sigma_"
        + str(best_sig)
        + "_"       
        + str(frequency_range)
        + "MHz_time lags"
    )

    plt.savefig(
        os.path.join(
            "../../result_titan/evaluate_f-t_diagram_plot_"
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
            + "_sigma_"
            + str(best_sig)
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
        using_frequency_array,
        using_frequency_range,
        maxden_max,
        maxden_min,
        peak_h_max,
        peak_h_min,
        sig_max,
        sig_min,
        dot_size,
        fig_holizontal_size,
        fig_vertical_size,
        plot_height_array,
        plot_max_density_array,
        plot_sigma_array,
        color_bar_array,
        contor_dif,
        uncertainty,
    ) = get_frequency_intensity_plotparameter(
        object_name, time_of_flybies, occultaion_type, radio_type_A2D
    )

    # MakeFolder(boundary_intensity)  # フォルダ作成　初めだけ使う

    use_files = sorted(
        glob.glob(
            "../../result_titan/radio_raytracing_occultation_timing_def_"
            + spacecraft_name
            + "_"
            + object_name
            + "_"
            + str(time_of_flybies)
            + "_flyby_radioint_"
            + boundary_average_str
            + "dB/interpolated-"
            + object_name
            + "*"
            + "ingress_defference_time_data"
            + radio_type_A2D
            + "-"
            + boundary_average_str
            + "dB.txt"
        )
    )

    for file in use_files:

        highest_density_str, plasma_scaleheight_str, sigma_str = maxandscale(file)

        plot_difference(
            highest_density_str,
            plasma_scaleheight_str,
            sigma_str,
            radio_type_A2D,
            using_frequency_array,
            using_frequency_range,
        )

    # ここまでで maxdensity, peakheight, sigma, dif_timeのリストができる => max, peakh, sig, dif

    ### ここから結果の出力
    output_array = np.array(max + peakh + sig+ time_dif)
    output_array = output_array.reshape(4, int(len(output_array) / 4)).T
    print(output_array)

    
    fig_and_save_def(
        output_array,
        using_frequency_range,
        fig_holizontal_size,
        fig_vertical_size,
        dot_size,
        maxden_max,
        maxden_min,
        peak_h_max,
        peak_h_min,
        sig_max,
        sig_min,
        color_bar_array,
        contor_dif,
        uncertainty,
    )  # ずれ時間とカラーマップを保存
    


    return 0


if __name__ == "__main__":
    main()

