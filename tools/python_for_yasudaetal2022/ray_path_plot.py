# In[]
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as mpt
import pandas as pd
from matplotlib.colors import LogNorm

object_name = 'ganymede'   # europa/ganymde/callisto
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 1  # ..th flyby
highest_plasma = '2e2'  # 単位は(/cc) 2e2/4e2/16e2
plasma_scaleheight = '3e2'  # 単位は(km) 1.5e2/3e2/6e2
frequency = '5.349649786949157715e5'  # MHz
altitiude_interval = 50

begining_egress_hour = 13
begining_egress_minute = 41

end_egress_hour = 13
end_egress_minute = 45

lowest_frequency_egress = 0.53
highest_frequecy_egress = 5.5
radio_type_egress = "D"  # 複数選択可能にしたい


Radio_name_cdf = '../result_for_yasudaetal2022/tracing_range_'+spacecraft_name+'_'+object_name + \
    '_'+str(time_of_flybies)+'_flybys/para_' + \
    highest_plasma+'_'+plasma_scaleheight+'.csv'
Radio_Range = pd.read_csv(Radio_name_cdf, header=0)

Highest = Radio_Range.highest
Lowest = Radio_Range.lowest
Except = Radio_Range.exc

"""# カリスト

Freq_str = ['3.612176179885864258e5', '3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6']
"""
# エウロパ・ガニメデ
Freq_str = ['3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6']

Freq_num = []
for idx in Freq_str:
    Freq_num.append(float(idx)/1000000)

Freq_num = np.array(Freq_num)

print(Freq_num)
print(frequency)
freq = np.where(Freq_num == float(frequency)/1000000)[0][0]
print(freq)
raytrace_lowest_altitude = Lowest[freq]
raytrace_highest_altitude = Highest[freq]


def ray_plot(height):
    data = np.loadtxt('../result_for_yasudaetal2022/raytracing_'+object_name+'_results/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight +
                      '/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z'+str(height)+'-FR'+frequency+'')
    x = data[:, [1]]
    z = data[:, [3]]
    plt.plot(x, z, color='red', linewidth=0.5)


def spacecraft_plot():
    # [0 hour, 1 min, 2 frequency(MHz), 3 電波源データの磁力線(根本)の経度  orイオの場合は(-1000), 4 電波源の南北,座標変換した時のx(tangential point との水平方向の距離), 5 座標変換した時のy(tangential pointからの高さ方向の距離),6 電波源の実際の経度]
    data = np.loadtxt(
        '../result_for_yasudaetal2022/calculated_expres_detectable_radio_data_of_each_flyby/')


def Output_moon_radius(moon_name):
    moon_radius = None

    if moon_name == "io":
        moon_radius = 1821.6

    elif moon_name == "europa":
        moon_radius = 1560.8

    elif moon_name == "ganymede":
        moon_radius = 2634.1

    elif moon_name == "callisto":
        moon_radius = 2410.3

    else:
        print("undefined object_name, please check the object_name (moon name) input and def Output_moon_radius function")

    return moon_radius


def main():
    plt.figure(figsize=(20, 4))
    plt.title("ray paths of Jovian radio waves around Ganymede")
    plt.xlabel("x (km) / tangential direction")
    plt.ylabel("z (km) / normal direction")
    plt.xlim(-7500, 2500)
    plt.ylim(-500, 1500)

    print(raytrace_lowest_altitude)

    for i in range(raytrace_lowest_altitude, raytrace_highest_altitude, altitiude_interval):
        ray_plot(i)

    radius = Output_moon_radius(object_name)
    t = np.arange(-1*radius, radius, 2)
    c = np.sqrt(radius*radius-t*t) - radius

    plt.plot(t, c, color="black")
    n = -1600+t*0
    plt.plot(t, n, color="black")
    plt.fill_between(t, c, n, facecolor='black')
    plt.show()


if __name__ == "__main__":
    main()

# In[]
