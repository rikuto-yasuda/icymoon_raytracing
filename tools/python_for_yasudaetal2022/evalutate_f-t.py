import numpy as np
import matplotlib as mpl
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
using_frequency_range = [1e5, 5e6]


def plot_difference(highest, scaleheight):
    time_diffrence_index = np.loadtxt(
        '/Users/yasudarikuto/research/icymoon_raytracing/tools/result_for_yasudaetal2022/R_P_'+object_name+'_fulldata.txt',)

    return 0


def main():
    for i in range(len(highest_density_str)):
        for j in range(len(plasma_scaleheight_str)):

            plot_difference(
                highest_density_str[i], using_frequency_range[j])

    return 0


if __name__ == "__main__":
    main()
