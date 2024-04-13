# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


# %%

####################################################
object_name = 'ganymede'  # ganydeme/europa/calisto`

spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 1  # ..th flyby


"""lowest frequency and highest_freqency(MHz)"""
# using_frequency_range = [8.5e-1, 4]  # G1 ingress
using_frequency_range = [5.5e-1, 6]  # G1 egress

boundary_intensity_str = '7e-16'  # '7e-16' '
occultaion_type = 'egress'  # 'ingress' or 'egress
radio_type = 'A'  # 'A' or 'B' or 'C' or 'D'


# %%

use_files = sorted(glob.glob('../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies) +
                   '_flyby_radioint_'+boundary_intensity_str+'/'+object_name+'_*_'+occultaion_type+'_defference_time_data'+radio_type+'_'+boundary_intensity_str+'.txt'))


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
    max_density = t[13]
    scale_height = t[14]
    print(max_density, scale_height)
    return max_density, scale_height


def plot_difference(highest, scaleheight):

    time_diffrence_index = np.loadtxt('../result_for_yasudaetal2022/radio_raytracing_occultation_timing_def_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies)+'_flyby_radioint_' +
                                      boundary_intensity_str+'/'+object_name+'_' + highest+'_'+scaleheight+'_'+occultaion_type+'_defference_time_data'+radio_type+'_'+boundary_intensity_str+str(using_frequency_range)'.txt')

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
    # scale_type = np.sort(np.array(list(set(scale_list))))
    scale_type = np.array([1500, 1000, 750, 600, 100, 50])
    fig, ax = plt.subplots(len(scale_type)//2, 2, figsize=(9, 8))

    # 上
    for i in range(len(scale_type)):
        selected_number = np.array(np.where(scale_list == scale_type[i]))[0]
        x = np.array(max_list)[selected_number]
        y = np.array(density_list)[selected_number]
        x_sorted_number = np.argsort(x)
        x_sorted = x[x_sorted_number]
        y_sorted = y[x_sorted_number]
        print(x_sorted_number)
        m = i//2
        n = i % 2
        ax[m, n].plot(x_sorted, y_sorted)
        ax[m, n].set_xlim(0, 400)
        ax[m, n].set_ylim(15, 90)
        # ax[m,n].set_yscale('log')
        ax[m, n].set_title(
            'Scale_height:'+str(scale_type[i])+'(km)', fontsize=10)
        ax[m, n].set_yticks(np.array([15, 30, 45, 60, 75, 90]))
        ax[m, n].grid()

    fig.supxlabel('Maximum density (cm-3)')
    fig.supylabel('Average time lag (sec)')
    fig.subplots_adjust(hspace=0.5)
    plt.savefig("../result_for_yasudaetal2022/evaluate_average_time_lag/" +
                object_name+"_"+str(time_of_flybies)+"_"+occultaion_type+radio_type+".jpg", format="jpg", dpi=600)
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
