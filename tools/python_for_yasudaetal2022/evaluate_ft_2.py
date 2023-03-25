# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


# %%

####################################################
object_name = 'ganymede'  # ganydeme/europa/calisto`

spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 1  # ..th flyby
occultaion_type = 'egress'  # 'ingress' or 'egress
radio_type_A2D = 'A'  # 'A' or 'B' or 'C' or 'D'

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
    max_density = t[13]
    scale_height = t[14]
    print(max_density, scale_height)
    return max_density, scale_height


def plot_difference(highest, scaleheight, boundary_intensity_str, radio_type, using_frequency_range):

    # [[frequencyの配列] [time_lagの配列]]
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

                if radio_type == 'A' or 'B' or 'C' or 'D':
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

                if radio_type == 'A' or 'B' or 'C' or 'D':
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

                if radio_type == 'A' or 'B' or 'C' or 'D':
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

                elif radio_type == 'B' or 'C':
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

                if radio_type == 'A' or 'B' or 'D':
                    using_frequency_range = [5.3e-1, 5.5]  # C9 egress A B D
                    boundary_intensity_str = '4e-16'

                elif radio_type == 'C':
                    using_frequency_range = [6.5e-1, 5.5]  # C9 egress C
                    boundary_intensity_str = '4e-16'

    return using_frequency_range, boundary_intensity_str, timelag_max, timelag_min, scale_max, scale_min, dot_size, fig_holizontal, fig_vertical


def plot_kai_contor(max, scale, delta_kai, ymin, ymax):
    sample_df = pd.DataFrame()
    sample_df['X'] = max
    sample_df['Y'] = scale
    sample_df['value'] = delta_kai

    # データ範囲を取得
    x_min, x_max = sample_df['X'].min(), sample_df['X'].max()

    # 取得したデータ範囲で新しく座標にする配列を作成
    new_x_coord = np.linspace(0, x_max, 32)
    new_y_coord = np.linspace(ymin, ymax, 100)

    # x, yのグリッド配列作成
    xx, yy = np.meshgrid(new_x_coord, new_y_coord)

    # 既知のx, y座標, その値取得
    knew_xy_coord = sample_df[['X', 'Y']].values
    knew_values = sample_df['value'].values

    # 座標間のデータを補間, method='nearest', 'linear' or 'cubic'
    result = griddata(points=knew_xy_coord, values=knew_values,
                      xi=(xx, yy), method='linear')

    # グラフ表示
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(xx, yy, result, levels=[3, 10, 30])
    ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    return 0


def main():

    using_frequency_range, boundary_intensity, vmaximum, vminimum, ymaximum, yminimum, dot_size, fig_holizontal_size, fig_vertical_size = get_frequency_intensity_plotparameter(
        object_name, time_of_flybies, occultaion_type, radio_type_A2D)

    # MakeFolder(boundary_intensity)  # フォルダ作成　初めだけ使う

    use_files = sorted(glob.glob('../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies) +
                                 '_flyby_radioint_'+boundary_intensity+'/'+object_name+'_*_'+occultaion_type+'_defference_time_data'+radio_type_A2D+'_'+boundary_intensity+'.txt'))

    for file in use_files:

        highest_density_str,  plasma_scaleheight_str = maxandscale(file)

        frequency_kinds = plot_difference(
            highest_density_str, plasma_scaleheight_str, boundary_intensity, radio_type_A2D, using_frequency_range)

    # ずれ時間を散布図にする部分
    output_array = np.array(max + scale + dif)
    output_array = output_array.reshape(3, int(len(output_array)/3)).T
    print(output_array)
    """
    np.savetxt('../result_for_yasudaetal2022/evaluate_f-t_diagram_plot_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_'+boundary_intensity+'/'+spacecraft_name +
               '_'+object_name+'_'+str(time_of_flybies)+'flyby_radiointensity_'+boundary_intensity+'_'+occultaion_type+'_'+radio_type_A2D+'_'+str(using_frequency_range)+'output_array.csv', output_array, fmt='%.2f', delimiter=',')
    plt.figure(figsize=(fig_holizontal_size, fig_vertical_size))
    plt.scatter(max, scale, s=dot_size, c=dif,
                cmap='rainbow_r', vmax=vmaximum, vmin=vminimum)
    plt.yscale('log')
    plt.ylim(yminimum, ymaximum)
    plt.colorbar(label='average time difference (sec)')
    plt.xlabel("Max density (/cc)")
    plt.ylabel("Scale height (km)")
    plt.title(object_name+'_'+occultaion_type +
              '_'+radio_type_A2D+'_f-t_evaluate')
    """
    # カイ二乗を計算する部分

    kai2_completed_list = list(kai2(max, scale, kai2_temp, frequency_kinds))
    output_kai2 = np.array(kai2_completed_list)
    #output_kai2 = output_kai2.reshape(3, int(len(output_kai2)/3)).T
    print(output_kai2)

    plot_kai_contor(max, scale, output_kai2, yminimum, ymaximum)

    plt.savefig(os.path.join('../result_for_yasudaetal2022/evaluate_f-t_diagram_plot_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_'+boundary_intensity,
                             spacecraft_name + '_'+object_name+'_'+str(time_of_flybies)+'flyby_radiointensity_'+boundary_intensity+'_'+occultaion_type+'_'+radio_type_A2D+'_'+str(using_frequency_range)+'_f-t_evaluate.png'))

    plt.show()

    return 0


if __name__ == "__main__":
    main()

# %%
