
# %%
import pprint
import cdflib
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import datetime
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from matplotlib.colors import LogNorm
# In[]
object_name = 'ganymede'  # ganydeme/
highest_plasma = '2e2'  # 単位は(/cc) 2e2/4e2/16e2
plasma_scaleheight = '3e2'  # 単位は(km) 1.5e2/3e2/6e2

data_name = '../result_sgepss_2021/'+object_name+'_'+highest_plasma+'_' + \
    plasma_scaleheight+'/para_'+highest_plasma+'_'+plasma_scaleheight+'.csv'
list = pd.read_csv(data_name, header=0)
Highest = list.highest
Lowest = list.lowest
Except = list.exc


# %%
data_name = '../result_sgepss_2021/ganymede_test_simple/para_test_simple.csv'

data = np.loadtxt(
    '/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/R_P_fulldata.txt',)

n = len(data)
Freq_str = ['3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]

Freq_num = []
for i in Freq_str:
    Freq_num.append(float(i)/1000000)


# %%
A = np.where(data[:, 0] == 6)
AA = data[A]
B = np.where(AA[:, 1] < 25)
BB = AA[B]
C = np.where(BB[:, 1] > 10)
CC = BB[C]
F = np.where(CC[:, 3] > 96)
FF = CC[F]
D = np.where(FF[:, 3] < 105)
DD = FF[D]
# %%

for l in range(len(Freq_num)):
    E = np.where(DD[:, 2] == float(Freq_num[l]))
    EE = DD[E]
    plt.xlabel("x (km)")
    plt.ylabel("z (km)")

    plt.xlim(-3000, 7000)
    plt.ylim(-250, 750)

    t = np.arange(-2634.1, 2634.1)
    c = np.sqrt(6938482.8-t*t) - 2634.1
    n = -2634+t*0
    plt.plot(t, c, color="black", linewidth=0.0001)
    plt.fill_between(t, c, n, facecolor='black')
    for i in range(Lowest[l], Highest[l], 20):
        if Except[l] != i:
            n = str(i)
            N = i

            filename = np.genfromtxt("../result_sgepss_2021/"+object_name+"_"+highest_plasma+"_"+plasma_scaleheight+"/ray-P" +
                                     object_name+"_nonplume_"+highest_plasma+"_"+plasma_scaleheight+"-Mtest_simple-benchmark-LO-Z"+n+"-FR"+Freq_str[l])

            x1 = filename[:, [1]]
            z1 = filename[:, [3]]
            plt.title("ganemede_nonplumeFR_"+Freq_str[l])
            plt.plot(x1, z1, color='red', linewidth=0.5)
    plt.scatter(EE[:, 5], EE[:, 6], color='black', linewidth=2.0)
    plt.show()


# %%
