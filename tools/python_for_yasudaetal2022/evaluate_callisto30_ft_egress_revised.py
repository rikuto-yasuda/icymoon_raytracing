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
using_frequency_range = [4.5e-1, 6]  # C30 egress D
boundary_intensity_str = '7e-16'  # '7e-16'
occultaion_type = 'egress'  # 'egress
radio_type = 'D'  # 'D'
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


def plot(max_list, scale_list, density_list):
    scale_type = np.array([900, 600, 400])
    fig, ax = plt.subplots(len(scale_type), 1, figsize=(4, 7))

    # 上
    for i in range(len(scale_type)):
        selected_number = np.array(np.where(scale_list == scale_type[i]))[0]
        x = np.array(max_list)[selected_number]
        y = np.array(density_list)[selected_number]
        x_sorted_number = np.argsort(x)
        x_sorted = x[x_sorted_number]
        y_sorted = y[x_sorted_number]
        print(x_sorted_number)
        ax[i].plot(x_sorted, y_sorted)
        ax[i].set_xlim(0, 3000)
        ax[i].set_ylim(50, 200)
        ax[i].set_title('scale_height:'+str(scale_type[i])+'(km)', fontsize=10)

    fig.supxlabel('maximum density (cm-3)')
    fig.supylabel('average time lag (sec) radio source:'+radio_type)
    fig.subplots_adjust(left=0.22)
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(bottom=0.07)

    plt.savefig("../result_for_yasudaetal2022/evaluate_average_time_lag/" +
                object_name+"_"+str(time_of_flybies)+"_"+occultaion_type+radio_type+".png")

    plt.show()

    return 0


def main():
    # MakeFolder()  # フォルダ作成　初めだけ使う

    for file in use_files:

        highest_density_str,  plasma_scaleheight_str = maxandscale(file)

        plot_difference(
            highest_density_str, plasma_scaleheight_str)

    output_array = np.array(max + scale + dif)
    output_array = output_array.reshape(3, int(len(output_array)/3)).T
    np.savetxt('../result_for_yasudaetal2022/evaluate_f-t_diagram_plot_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_'+boundary_intensity_str+'/'+spacecraft_name +
               '_'+object_name+'_'+str(time_of_flybies)+'flyby_radiointensity_'+boundary_intensity_str+'_'+occultaion_type+'_'+radio_type+'_'+str(using_frequency_range)+'output_array.csv', output_array, fmt='%.2f', delimiter=',')

    plot(max, scale, dif)
    # plt.xscale('log')
    return 0


if __name__ == "__main__":
    main()

# %%
