import numpy as np
import matplotlib.pyplot as plt

object_name = 'ganymede'  # ganydeme

highest_density_str = ['0.125e2', '0.25e2', '0.5e2', '1e2', '2e2', '4e2']
highest_density_num = []
for idx in highest_density_str:
    highest_density_num.append(float(idx))

plasma_scaleheight_str = ['1.5e2', '3e2', '6e2', '9e2']


plasma_scaleheight_num = []
for idx in plasma_scaleheight_str:
    highest_density_num.append(float(idx))

occultaion_type = 'ingress'  # 'ingress' or 'egress'
radio_type = 'A'  # 'A' or 'B' or 'C' or 'D'

# lowest frequency and highest_freqency(MHz)
using_frequency_range = [8e-1, 6]  # ingress
# using_frequency_range = [5.5e-1, 6]  # egress

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

    total_difference_time = sum(
        time_diffrence_index[1][limited_time_minimum: limited_time_maximum+1])

    max.append(float(highest))
    scale.append(float(scaleheight))
    dif.append(float(total_difference_time))

    return 0


def main():
    for i in range(len(highest_density_str)):
        for j in range(len(plasma_scaleheight_str)):

            plot_difference(
                highest_density_str[i], plasma_scaleheight_str[j])

    plt.scatter(max, scale, s=100, c=dif,
                cmap='rainbow_r')
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar()
    plt.show()
    plt.savefig(os.path.join('../result_for_yasudaetal2022/evaluate_f-t_diagram_plot',
                             object_name+'_'+occultaion_type+'_'+radio_type+'_ingress_f-t.png'))

    return 0


if __name__ == "__main__":
    main()
