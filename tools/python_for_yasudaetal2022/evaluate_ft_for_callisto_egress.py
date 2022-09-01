# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


# %%

####################################################
object_name = 'callisto'  # ganydeme/europa/calisto`

spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 30  # ..th flyby


"""lowest frequency and highest_freqency(MHz)"""
# using_frequency_range = [8.5e-1, 4]  # G1 ingress
# using_frequency_range = [5.5e-1, 6]  # G1 egress
# using_frequency_range = [2, 6]  # E12

# using_frequency_range = [6.0e-1, 6]  # C30 ingress
using_frequency_range = [4.0e-1, 6]  # C30 egress

boundary_intensity_str = '7e-16'  # '7e-16' '1e-15'
occultaion_type = 'egress'  # 'ingress' or 'egress
radio_type = 'D'  # 'A' or 'B' or 'C' or 'D'
# %%

use_files = sorted(glob.glob('../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies) +
                   '_flyby_radioint_'+boundary_intensity_str+'_examine/'+object_name+'_*_'+occultaion_type+'_defference_time_data'+radio_type+'_'+boundary_intensity_str+'_examine.txt'))


max = []
scale = []
dif = []


def MakeFolder():
    os.makedirs('../result_for_yasudaetal2022/evaluate_f-t_diagram_plot_'+spacecraft_name+'_' +
                object_name+'_'+str(time_of_flybies)+'_flyby_radioint_'+boundary_intensity_str)  # 結果を格納するフォルダを生成


def maxandscale(file):
    filename = file
    sep = '_'
    t = filename.split(sep)
    max_density = t[14]
    scale_height = t[15]
    print(max_density, scale_height)
    return max_density, scale_height


def plot_difference(highest, scaleheight):

    time_diffrence_index = np.loadtxt('../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_' +
                                      boundary_intensity_str+'_examine/'+object_name+'_' + highest+'_'+scaleheight+'_'+occultaion_type+'_defference_time_data'+radio_type+'_'+boundary_intensity_str+'_examine.txt')

    print(time_diffrence_index)
    limited_time_list = np.array(np.where(
        (time_diffrence_index[0][:] > using_frequency_range[0]) & (time_diffrence_index[0][:] < using_frequency_range[1])))
    limited_time_minimum = limited_time_list[0][0]
    limited_time_maximum = limited_time_list[0][len(limited_time_list[0][:])-1]
    print()

    average_difference_time = sum(
        time_diffrence_index[1][limited_time_minimum: limited_time_maximum+1])/(limited_time_maximum+1-limited_time_minimum)

    max.append(float(highest))
    scale.append(float(scaleheight))
    dif.append(float(average_difference_time))

    return 0


def main():
    # MakeFolder()  # フォルダ作成　初めだけ使う

    for file in use_files:

        highest_density_str,  plasma_scaleheight_str = maxandscale(file)

        plot_difference(
            highest_density_str, plasma_scaleheight_str)

    output_array = np.array(max + scale + dif)
    output_array = output_array.reshape(3, int(len(output_array)/3)).T
    print(output_array)
    np.savetxt('../result_for_yasudaetal2022/evaluate_f-t_diagram_plot_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_'+boundary_intensity_str+'/'+spacecraft_name +
               '_'+object_name+'_'+str(time_of_flybies)+'flyby_radiointensity_'+boundary_intensity_str+'_'+occultaion_type+'_'+radio_type+'_'+str(using_frequency_range)+'output_array_examine.csv', output_array, fmt='%.2f', delimiter=',')
    plt.scatter(max, scale, s=25, c=dif, cmap='rainbow_r', vmax=300, vmin=20)
    # plt.xscale('log')
    plt.yscale('log')
    plt.ylim(200, 1000)
    plt.colorbar(label='average time difference (sec)')
    plt.xlabel("Max density (/cc)")
    plt.ylabel("Scale height (km)")
    plt.title(object_name+'_'+occultaion_type+'_'+radio_type+'_f-t_evaluate')
    plt.savefig(os.path.join('../result_for_yasudaetal2022/evaluate_f-t_diagram_plot_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_'+boundary_intensity_str,
                             spacecraft_name + '_'+object_name+'_'+str(time_of_flybies)+'flyby_radiointensity_'+boundary_intensity_str+'_'+occultaion_type+'_'+radio_type+'_'+str(using_frequency_range)+'_f-t_evaluate_examine.png'))
    plt.show()

    return 0


if __name__ == "__main__":
    main()
