# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap, BoundaryNorm


# %%

####################################################
object_name = 'callisto'  # ganydeme/europa/calisto`

spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 9  # ..th flyby
occultaion_type = 'egress'  # 'ingress' or 'egress
radio_type_A2D = 'D'  # 'A' or 'B' or 'C' or 'D'
# callisto 30 flyby egress用　if you want to ignore the exclave structere, choose "True" (Check M-thesis!)
exclave_examine = False
# "time_difference" or "kai_2" please choose what you want to plot
purpose = "time_difference"

# %%

max = []
scale = []
dif = []
kai2_temp = []
kai_comp = []


def MakeFolder(boundary_intensity_str):
    os.makedirs('../result_for_yasudaetal2022/evaluate_f-t_diagram_plot_'+spacecraft_name+'_' +
                object_name+'_'+str(time_of_flybies)+'_flyby_radioint_'+boundary_intensity_str)  # 結果を格納するフォルダを生成


def maxandscale(file):
    filename = file
    sep = '_'
    t = filename.split(sep)

    if exclave_examine:
        max_density = t[14]
        scale_height = t[15]

    else:
        max_density = t[13]
        scale_height = t[14]

    print(max_density, scale_height)
    return max_density, scale_height

# ずれ時間の計算関数・保存機能なし


def plot_difference(highest, scaleheight, boundary_intensity_str, radio_type, using_frequency_range, exclave):

    # [[frequencyの配列] [time_lagの配列]]
    if exclave:
        time_diffrence_index = np.loadtxt('../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_' +
                                          boundary_intensity_str+'_examine/'+object_name+'_' + highest+'_'+scaleheight+'_'+occultaion_type+'_defference_time_data'+radio_type+'_'+boundary_intensity_str+'_examine.txt')
    else:
        time_diffrence_index = np.loadtxt('../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_' +
                                          boundary_intensity_str+'/'+object_name+'_' + highest+'_'+scaleheight+'_'+occultaion_type+'_defference_time_data'+radio_type+'_'+boundary_intensity_str+'.txt')

    limited_time_list = np.array(np.where(
        (time_diffrence_index[0][:] > using_frequency_range[0]) & (time_diffrence_index[0][:] < using_frequency_range[1])))
    limited_time_minimum = limited_time_list[0][0]  # 最低周波数の位置
    limited_time_maximum = limited_time_list[0][len(
        limited_time_list[0][:])-1]  # 最高周波数の位置

    frequency_number = limited_time_maximum+1-limited_time_minimum

    average_difference_time = sum(
        time_diffrence_index[1][limited_time_minimum: limited_time_maximum+1])/frequency_number

    # シグマを1と置いたときのkai2じょう
    kai2_temporary = np.dot(
        time_diffrence_index[1][limited_time_minimum: limited_time_maximum+1], time_diffrence_index[1][limited_time_minimum: limited_time_maximum+1])

    max.append(float(highest))
    scale.append(float(scaleheight))
    dif.append(float(average_difference_time))
    kai2_temp.append(float(kai2_temporary))

    return frequency_number


def mindif_density():
    scale_list = np.unique(np.array(scale))
    print(scale_list)
    max_np = np.array(max)
    scale_np = np.array(scale)
    dif_np = np.array(dif)

    for sca in scale_list:
        scaleselected_dif_position = np.where(scale_np == sca)
        min_dif_time = np.min(dif_np[scaleselected_dif_position])
        minselected_dif_position = np.where(dif_np == min_dif_time)
        selected_position = np.intersect1d(
            scaleselected_dif_position, minselected_dif_position)

        selected_max_density = max_np[selected_position]

        print("scale_height=" + str(sca) +
              "  dif_time =" + str(min_dif_time) +
              "  max_den = " + str(selected_max_density))

    return 0


# カイ二乗計算関数・保存機能なし


def kai2(maximum, scaleheight, kai2, frequency_n):

    minimum_kai_position = np.argmin(kai2)
    minimum_kai_maximum_density = maximum[minimum_kai_position]
    minimum_kai_scale_height = scaleheight[minimum_kai_position]
    minimum_kai_kai2 = kai2[minimum_kai_position]
    minimum_kai_sigma2 = minimum_kai_kai2 / frequency_n
    print(minimum_kai_maximum_density, minimum_kai_scale_height)

    complete_kai2_list = kai2 / minimum_kai_sigma2
    complete_minimum_kai = complete_kai2_list[minimum_kai_position]

    delta_kai_2 = complete_kai2_list - complete_minimum_kai
    print(frequency_n-2)

    return delta_kai_2


def get_frequency_intensity_plotparameter(moon_name, flyby_time, ingress_or_egerss, radio_type):
    print(moon_name, flyby_time, ingress_or_egerss, radio_type)
    if moon_name == 'ganymede':
        if flyby_time == 1:

            if ingress_or_egerss == 'ingress':

                timelag_max = 80
                timelag_min = 20

                scale_max = 2000
                scale_min = 20

                dot_size = 40
                fig_holizontal = 7
                fig_vertical = 5

                if (radio_type == 'A') or (radio_type == 'B') or (radio_type == 'C') or (radio_type == 'D'):
                    using_frequency_range = [8.5e-1, 4]  # G1 ingress
                    boundary_intensity_str = '7e-16'

            elif ingress_or_egerss == 'egress':

                timelag_max = 80
                timelag_min = 20

                scale_max = 2000
                scale_min = 20

                dot_size = 40
                fig_holizontal = 7
                fig_vertical = 5

                if (radio_type == 'A') or (radio_type == 'B') or (radio_type == 'C') or (radio_type == 'D'):
                    using_frequency_range = [5.5e-1, 6]  # G1 egress
                    boundary_intensity_str = '7e-16'

    if moon_name == 'callisto':
        if flyby_time == 30:

            if ingress_or_egerss == 'ingress':

                timelag_max = 150
                timelag_min = 20

                scale_max = 1000
                scale_min = 350

                dot_size = 20
                fig_holizontal = 9
                fig_vertical = 4

                if (radio_type == 'A') or (radio_type == 'B') or (radio_type == 'C') or (radio_type == 'D'):
                    using_frequency_range = [6.0e-1, 6]
                    boundary_intensity_str = '7e-16'

            elif ingress_or_egerss == 'egress':

                timelag_max = 150
                timelag_min = 20

                scale_max = 1000
                scale_min = 350

                dot_size = 20
                fig_holizontal = 9
                fig_vertical = 4

                if radio_type == 'A':
                    using_frequency_range = [4.0e-1, 6]  # C30 egress A
                    boundary_intensity_str = '7e-16'

                elif (radio_type == 'B') or (radio_type == 'C'):
                    using_frequency_range = [7.0e-1, 6]  # C30 egress B&C
                    boundary_intensity_str = '7e-16'

                elif radio_type == 'D':
                    using_frequency_range = [4.5e-1, 6]  # C30 egress D
                    boundary_intensity_str = '7e-16'

        elif flyby_time == 9:
            if ingress_or_egerss == 'egress':

                timelag_max = 150
                timelag_min = 20

                scale_max = 1000
                scale_min = 350

                dot_size = 40
                fig_holizontal = 7
                fig_vertical = 5

                if (radio_type == 'A') or (radio_type == 'B') or (radio_type == 'D'):
                    using_frequency_range = [5.3e-1, 5.5]  # C9 egress A B D
                    boundary_intensity_str = '4e-16'

                elif radio_type == 'C':
                    using_frequency_range = [6.5e-1, 5.5]  # C9 egress C
                    boundary_intensity_str = '4e-16'

    print(using_frequency_range)

    return using_frequency_range, boundary_intensity_str, timelag_max, timelag_min, scale_max, scale_min, dot_size, fig_holizontal, fig_vertical

# 　カイ二乗値補間・プロット関数　保存機能付き


def plot_kai2(max, scale, delta_kai, ymin, ymax):

    # 補間するためのグリッドを作成
    max = np.array(max)
    scale = np.array(scale)
    xi = np.arange(max.min(), max.max(), 25)
    yi = np.arange(scale.min(), scale.max(), 25)
    xi, yi = np.meshgrid(xi, yi)

    # データの補間
    zi = griddata((max, scale), delta_kai, (xi, yi), method='linear')

    # 等高線の描画
    cs = plt.contour(xi, yi, zi, levels=[10, 20, 30, 40])
    plt.clabel(cs)

    # データの散布図を重ねて描画
    plt.scatter(max, scale, c=delta_kai, cmap='jet', vmax=100)

    plt.xlim(0, max.max())
    plt.ylim(ymin, ymax)
    plt.yscale("log")
    plt.colorbar()


def fig_and_save_def(def_data, frequency_range, radio_intensity, holizontal_size, vertical_size, dot, ymin, ymax):

    np.savetxt('../result_for_yasudaetal2022/evaluate_f-t_diagram_plot_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_'+radio_intensity+'/'+spacecraft_name +
               '_'+object_name+'_'+str(time_of_flybies)+'flyby_radiointensity_'+radio_intensity+'_'+occultaion_type+'_'+radio_type_A2D+'_'+str(frequency_range)+'output_array.csv', def_data, fmt='%.2f', delimiter=',')

    cmap = ListedColormap(['#dc143c', '#ffa055', '#a7f89d',
                          '#3be9d6', '#2e7bf7', '#4b0082'])
    bounds = np.linspace(0, 180, 7)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(holizontal_size, vertical_size))

    sc = plt.scatter(max, scale, s=dot, c=dif, norm=norm, cmap=cmap)
    plt.yscale('log')
    plt.ylim(ymin, ymax)
    plt.colorbar(sc, label='average time difference (sec)')
    plt.xlabel("Max density (/cc)")
    plt.ylabel("Scale height (km)")
    plt.title(object_name+'_'+occultaion_type +
              '_'+radio_type_A2D+'_f-t_evaluate')
    plt.savefig(os.path.join('../result_for_yasudaetal2022/evaluate_f-t_diagram_plot_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_'+radio_intensity,
                             spacecraft_name + '_'+object_name+'_'+str(time_of_flybies)+'flyby_radiointensity_'+radio_intensity+'_'+occultaion_type+'_'+radio_type_A2D+'_'+str(frequency_range)+'_f-t_evaluate.png'))
    plt.show()


def main():

    using_frequency_range, boundary_intensity, vmaximum, vminimum, ymaximum, yminimum, dot_size, fig_holizontal_size, fig_vertical_size = get_frequency_intensity_plotparameter(
        object_name, time_of_flybies, occultaion_type, radio_type_A2D)

    # MakeFolder(boundary_intensity)  # フォルダ作成　初めだけ使う
    if exclave_examine:
        use_files = sorted(glob.glob('../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_' +
                                     boundary_intensity+'_examine/'+object_name+'_*_'+occultaion_type+'_defference_time_data'+radio_type_A2D+'_'+boundary_intensity+'_examine.txt'))

    else:
        use_files = sorted(glob.glob('../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_' +
                                     boundary_intensity+'/'+object_name+'_*_'+occultaion_type+'_defference_time_data'+radio_type_A2D+'_'+boundary_intensity+'.txt'))

    for file in use_files:

        highest_density_str,  plasma_scaleheight_str = maxandscale(file)

        frequency_kinds = plot_difference(
            highest_density_str, plasma_scaleheight_str, boundary_intensity, radio_type_A2D, using_frequency_range, exclave_examine)

    mindif_density()  # スケールハイトごとにずれ最小となる密度を出力

    # ずれ時間を散布図にする部分
    ### ここから#
    output_array = np.array(max + scale + dif)
    output_array = output_array.reshape(3, int(len(output_array)/3)).T
    print(output_array)

    if purpose == "time_difference":
        fig_and_save_def(output_array, using_frequency_range, boundary_intensity,
                         fig_holizontal_size, fig_vertical_size, dot_size, yminimum, ymaximum)  # ずれ時間とカラーマップを保存
    ### ここまで###

    elif purpose == "kai_2":
        # カイ二乗を計算する部分
        kai2_completed_list = list(
            kai2(max, scale, kai2_temp, frequency_kinds))
        output_kai2 = np.array(kai2_completed_list)
        # output_kai2 = output_kai2.reshape(3, int(len(output_kai2)/3)).T
        print(output_kai2)

        plot_kai2(max, scale, output_kai2, yminimum, ymaximum)
        plt.savefig(os.path.join('../result_for_yasudaetal2022/evaluate_f-t_diagram_plot_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_'+boundary_intensity,
                                 spacecraft_name + '_'+object_name+'_'+str(time_of_flybies)+'flyby_radiointensity_'+boundary_intensity+'_'+occultaion_type+'_'+radio_type_A2D+'_'+str(using_frequency_range)+'_f-t_kai2.png'))
        plt.show()

    else:
        print("porpose is not correct")

    return 0


if __name__ == "__main__":
    main()


# %%
