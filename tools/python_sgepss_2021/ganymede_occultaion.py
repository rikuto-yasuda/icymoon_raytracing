# In[]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as mpt
import pandas as pd
from matplotlib.colors import LogNorm

# In[]
object_name = 'ganymede'  # ganydeme/
highest_plasma = '1e2'  # 単位は(/cc) 2e2/4e2/16e2
plasma_scaleheight = '3e2'  # 単位は(km) 1.5e2/3e2/6e2

data_name = '../result_sgepss_2021/'+object_name+'_'+highest_plasma+'_' + \
    plasma_scaleheight+'/para_'+highest_plasma+'_'+plasma_scaleheight+'.csv'
list = pd.read_csv(data_name, header=0)
Highest = list.highest
Lowest = list.lowest
Except = list.exc

Freq_str = ['3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]

Freq_num = []
for i in Freq_str:
    Freq_num.append(float(i)/1000000)

for l in range(len(Freq_num)):

    df = np.genfromtxt('../result_sgepss_2021/ganymede_plasma3')
    l_2d = len(df)
    idx = np.array(np.where(df[:, 0] > -6000))
    print(idx[0, 0])
    r_size = idx[0, 0]
    c_size = int(l_2d/r_size)

    x = df[:, 0].reshape(c_size, r_size)
    print(x.shape)
    y = df[:, 1].reshape(c_size, r_size)
    print(y.shape)
    z = df[:, 2].reshape(c_size, r_size)
    print(z.shape)
    v = df[:, 3].reshape(c_size, r_size).T

    plt.imshow(v, origin='lower', interpolation='nearest')
    plt.colorbar(extend='both')

    idx_list = np.array([])
    i_list = np.zeros([v.shape[1]])

    # 1e5Hz ライン
    F_1e5 = v > 1.4E+8
    F_1e5 = F_1e5.astype(np.uint8)

    F_1e5_up = np.roll(F_1e5, 1, axis=0)
    F_1e5_down = np.roll(F_1e5, -1, axis=0)
    F_1e5_right = np.roll(F_1e5, 1, axis=1)
    F_1e5_left = np.roll(F_1e5, -1, axis=1)

    print_F_1e5 = F_1e5_up + F_1e5_down + F_1e5_right + F_1e5_left - 4*F_1e5
    print_F_1e5 = np.where(print_F_1e5 < 1, np.nan, 1)

    # 5e4Hz ライン
    F_5e4 = v > 3.53E+7
    F_5e4 = F_5e4.astype(np.uint8)

    F_5e4_up = np.roll(F_5e4, 1, axis=0)
    F_5e4_down = np.roll(F_5e4, -1, axis=0)
    F_5e4_right = np.roll(F_5e4, 1, axis=1)
    F_5e4_left = np.roll(F_5e4, -1, axis=1)

    print_F_5e4 = F_5e4_up + F_5e4_down + F_5e4_right + F_5e4_left - 4*F_5e4
    print_F_5e4 = np.where(print_F_5e4 < 1, np.nan, 1)

    # 1e4Hz ライン
    F_1e5 = v > 1.4E+6
    F_1e5 = F_1e5.astype(np.uint8)

    F_1e5_up = np.roll(F_1e5, 1, axis=0)
    F_1e5_down = np.roll(F_1e5, -1, axis=0)
    F_1e5_right = np.roll(F_1e5, 1, axis=1)
    F_1e5_left = np.roll(F_1e5, -1, axis=1)

    print_F_1e5 = F_1e5_up + F_1e5_down + F_1e5_right + F_1e5_left - 4*F_1e5
    print_F_1e5 = np.where(print_F_1e5 < 1, np.nan, 1)

    data = v+0.000001

    #plt.imshow(v, norm=mpl.colors.LogNorm(), origin='lower',interpolation='nearest', extent=[-6000,10450,-500,1450], vmin=200, vmax=4E+8)
    plt.imshow(v, norm=mpl.colors.LogNorm(), origin='lower', interpolation='nearest',
               extent=[-6000, 10450, -500, 1450], vmin=5E+3, vmax=4E+8)
    # plt.colorbar(extend='both')

    #plt.imshow(vvv, cmap='spring', origin='lower', interpolation='nearest', extent=[-1000,997.5,-200,1997.5])

    plt.xlabel("x (km)")
    plt.ylabel("z (km)")

    plt.xlim(-5000, 40000)
    plt.ylim(-500, 3500)

    t = np.arange(-2634.1, 2634.1)
    c = np.sqrt(6938482.8-t*t) - 2634.1
    n = -2634+t*0
    plt.plot(t, c, color="black", linewidth=0.0001)
    ##plt.plot(n, c, color = "black")
    plt.fill_between(t, c, n, facecolor='black')
    # for i in range(Lowest[l],1000,100):
    for i in range(0, 101, 100):
        if Except[l] != i:
            n = str(i)
            N = i

            filename = np.genfromtxt("../result_sgepss_2021/"+object_name+"_"+highest_plasma+"_"+plasma_scaleheight+"/ray-P" +
                                     object_name+"_nonplume_"+highest_plasma+"_"+plasma_scaleheight+"-Mtest_simple-benchmark-LO-Z"+n+"-FR"+Freq_str[l])

            x1 = filename[:, [1]]
            z1 = filename[:, [3]]
            plt.title("ganemede_nonplumeFR_"+Freq_str[l])
            plt.plot(x1, z1, color='red', linewidth=0.5)

    plt.show()

# %%
