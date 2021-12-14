# In[]

import numpy as np
import os
from multiprocessing import Pool
import pandas as pd

# In[]
object_name = 'ganymede'  # ganydeme/

highest_plasma = '0.5e2'  # 単位は(/cc) 2e2/4e2/16e2
plasma_scaleheight = '7.5e2'  # 単位99は(km) 1.5e2/3e2/6e2

x_farthest = 26680.87663426876
z_farthest = 8112.669988476546

raytrace_lowest_altitude = -500  # レイトレーシングの下端の初期高度(km) 100の倍数で
raytrace_highest_altitude = 3100  # レイトレーシングの下端の初期高度(km) 500の倍数で


data_name = '../result_for_yasudaetal2022/tracing_range_'+object_name+'/para_' + \
    highest_plasma+'_'+plasma_scaleheight+'.csv'
radio_range = pd.read_csv(data_name, header=0)


Freq_str = ['3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]

Freq_num = []
for i in Freq_str:
    Freq_num.append(float(i)/1000000)

kinds_freq = list(np.arange(len(Freq_num)))


def MakeFolder():
    os.makedirs('../result_for_yasudaetal2022/' + object_name +
                '_'+highest_plasma+'_'+plasma_scaleheight)


def MoveFile():
    for l in range(len(Freq_num)):
        for j in range(raytrace_lowest_altitude, 0, 100):
            k = str(j)
            os.replace('../../raytrace.tohoku/src/rtc/testing/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                       k+'-FR'+Freq_str[l], '../result_for_yasudaetal2022/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                       k+'-FR'+Freq_str[l])

        for i in range(0, raytrace_highest_altitude, 500):
            lower_altitude = str(i)
            higher_altitude = str(i + 100)

            os.replace('../../raytrace.tohoku/src/rtc/testing/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                       lower_altitude+'-FR'+Freq_str[l], '../result_for_yasudaetal2022/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                       lower_altitude+'-FR'+Freq_str[l])
            os.replace('../../raytrace.tohoku/src/rtc/testing/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                       higher_altitude+'-FR'+Freq_str[l], '../result_for_yasudaetal2022/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                       higher_altitude+'-FR'+Freq_str[l])


def NibunnZ(x1, z1, x2, z2, z_destination):
    para = np.abs(z2 - z_destination)

    while (para > 0.1):
        ddx = (x1+x2)/2
        ddz = (z1+z2)/2

        if ddz > z_destination:
            x2 = ddx
            z2 = ddz
        else:
            x1 = ddx
            z1 = ddz

        para = np.abs(z2 - z_destination)

    return x2, z2


def NibunnX(x1, z1, x2, z2, x_destination):
    para1 = np.abs(x1 - x_destination)
    para2 = np.abs(x2 - x_destination)

    while (para1 > 0.1 and para2 > 0.1):
        ddx = (x1+x2)/2
        ddz = (z1+z2)/2

        if ddx > x_destination:
            x2 = ddx
            z2 = ddz
        else:
            x1 = ddx
            z1 = ddz

        para1 = np.abs(x1 - x_destination)
        para2 = np.abs(x2 - x_destination)

    return x2, z2


def Check_too_nearby(raytracing_result):
    initial_x = raytracing_result[0][1]
    check_x_position = initial_x + 1000

    check_idx = np.array(np.where(raytracing_result[:][1] > check_x_position))
    P1 = check_idx[0, 0]
    deff_x = raytracing_result[P1+1][1] - raytracing_result[P1][1]
    deff_z = raytracing_result[P1+1][3] - raytracing_result[P1][3]

    check_degree = np.degrees(np.arctan(deff_z/deff_x))
    if check_degree > 0.01:
        print("Start position need to be far from moon")


def Calc_lowest(l):
    lowest_altitude = raytrace_lowest_altitude
    for i in range(raytrace_lowest_altitude, 0, 100):
        k = str(i)
        ray_path = np.genfromtxt("../result_for_yasudaetal2022/"+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+"/ray-P" +
                                 object_name+"_nonplume_"+highest_plasma+"_"+plasma_scaleheight+"-Mtest_simple-benchmark-LO-Z"+k+"-FR"+Freq_str[l])

        if ray_path.ndim == 2:
            Check_too_nearby(ray_path)

            n2 = len(ray_path)-1

            if (ray_path[n2][1] > 0 & i == -500):
                lowest_altitude = 10000
                break

            if (ray_path[n2][1] < 0):
                lowest_altitude = i

    return lowest_altitude


def Calc_highest(l):
    highest_altitude = 100

    for i in range(0, raytrace_highest_altitude, 500):

        lower_altitude = str(i)
        k = i+100
        higher_altitude = str(k)

        lower_ray_path = np.genfromtxt(("../result_for_yasudaetal2022/"+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+"/ray-P"
                                       + object_name+"_nonplume_"+highest_plasma+"_"+plasma_scaleheight+"-Mtest_simple-benchmark-LO-Z"+lower_altitude+"-FR"+Freq_str[l]))

        higher_ray_path = np.genfromtxt(("../result_for_yasudaetal2022/"+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+"/ray-P"
                                        + object_name+"_nonplume_"+highest_plasma+"_"+plasma_scaleheight+"-Mtest_simple-benchmark-LO-Z"+higher_altitude+"-FR"+Freq_str[l]))

        if lower_ray_path.ndim == 2 and higher_ray_path.ndim == 2:

            Check_too_nearby(lower_ray_path)
            Check_too_nearby(higher_ray_path)

            n2 = len(lower_ray_path)-1

            if (lower_ray_path[n2][1] < x_farthest and lower_ray_path[n2][3] < z_farthest):
                print(Freq_str[l], lower_altitude, "error")

            if (higher_ray_path[n2][1] < x_farthest and higher_ray_path[n2][3] < z_farthest):
                print(Freq_str[l], higher_altitude, "error")

            if (lower_ray_path[n2][3] > z_farthest):
                lower_idx = np.array(
                    np.where(lower_ray_path[:, 3] > z_farthest))
                t2 = lower_idx[0, 0]
                t1 = t2 - 1
                lower_x, lower_z = NibunnZ(
                    lower_ray_path[t1][1], lower_ray_path[t1][3], lower_ray_path[t2][1], lower_ray_path[t2][3], z_farthest)

                if (higher_ray_path[n2][3] > z_farthest):
                    higher_idx = np.array(
                        np.where(higher_ray_path[:, 3] > z_farthest))
                    T2 = higher_idx[0, 0]
                    T1 = T2 - 1
                    higher_x, higher_z = NibunnZ(
                        higher_ray_path[T1][1], higher_ray_path[T1][3], higher_ray_path[T2][1], higher_ray_path[T2][3], z_farthest)

                    if higher_x > lower_x:
                        highest_altitude = higher_altitude

            else:
                lower_idx = np.array(
                    np.where(lower_ray_path[:, 1] > x_farthest))
                t2 = lower_idx[0, 0]
                t1 = t2 - 1
                lower_x, lower_z = NibunnX(
                    lower_ray_path[t1][1], lower_ray_path[t1][3], lower_ray_path[t2][1], lower_ray_path[t2][3], x_farthest)

                higher_idx = np.array(
                    np.where(higher_ray_path[:, 1] > x_farthest))
                T2 = higher_idx[0, 0]
                T1 = T2 - 1
                higher_x, higher_z = NibunnX(
                    higher_ray_path[T1][1], higher_ray_path[T1][3], higher_ray_path[T2][1], higher_ray_path[T2][3], x_farthest)

                highest_altitude = higher_altitude

                if higher_z > lower_z:
                    break

        else:
            higher_altitude = i + 500

    return highest_altitude


def Replace_csv(Rowname, replace_list):
    radio_range.loc[:, Rowname] = replace_list
    radio_range.to_csv('../result_for_yasudaetal2022/tracing_range_' +
                       object_name+'/para_' + highest_plasma+'_'+plasma_scaleheight+'.csv', index=False)
    print(radio_range.highest)
    return 0


def main():

    MakeFolder()
    MoveFile()

    with Pool(processes=3) as pool:
        lowest_altitude_list = list(pool.map(Calc_lowest, kinds_freq))

    with Pool(processes=3) as pool:
        highest_altitude_list = list(pool.map(Calc_highest, kinds_freq))

    Replace_csv("lowest", lowest_altitude_list)
    Replace_csv("highest", highest_altitude_list)

    return 0


if __name__ == "__main__":
    main()

# %%
