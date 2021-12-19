import numpy as np
import matplotlib.pyplot as plt
import os

####################################################
object_name = 'ganymede'  # ganydeme

"""lowest frequency and highest_freqency(MHz)"""
# using_frequency_range = [8.5e-1, 6]  # ingress
using_frequency_range = [6e-1, 6]  # egress

highest_density_str = ['1e2', '2e2', '4e2']
plasma_scaleheight_str = ['3e2', '6e2', '9e2', '15e2']
"""
plasma_distribution_list = np.loadtxt() # 工事中
"""

occultaion_type = 'egress'  # 'ingress' or 'egress'
radio_type = 'A'  # 'A' or 'B' or 'C' or 'D'

####################################################

highest_density_num = []
for idx in highest_density_str:
    highest_density_num.append(float(idx))

plasma_scaleheight_num = []
for idx in plasma_scaleheight_str:
    highest_density_num.append(float(idx))

"""
for idx in len(plasma_distribution_list): #工事中
    highest_density_num.append(float(plasma_distribution_list[idx][0]))
    highest_density_num.append(float(plasma_distribution_list[idx][1]))
"""


max = []
scale = []
dif = []


def plot_difference(highest, scaleheight):
    time_diffrence_index = np.loadtxt('../result_for_yasudaetal2022/f-t_ganymede_'+occultaion_type+'_difference/ganymede_' +
                                      highest+'_'+scaleheight+'_'+occultaion_type+'_defference_time_data'+radio_type+'.txt')

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
    for i in range(len(highest_density_str)):
        for j in range(len(plasma_scaleheight_str)):

            plot_difference(
                highest_density_str[i], plasma_scaleheight_str[j])

    """
    for i in range(len(highest_density_num)):
        plot_difference(highest_de)
    
    
    """

    plt.scatter(max, scale, s=100, c=dif,
                cmap='rainbow_r', vmax=180, vmin=0)
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar()
    plt.xlabel("Max density (/cc)")
    plt.ylabel("Scale height (km)")
    plt.title(object_name+'_'+occultaion_type+'_'+radio_type+'_f-t_evaluate')
    plt.savefig(os.path.join('../result_for_yasudaetal2022/evaluate_f-t_diagram_plot',
                             object_name+'_'+occultaion_type+'_'+radio_type+'_f-t_evaluate_1.png'))
    plt.show()

    return 0


if __name__ == "__main__":
    main()
