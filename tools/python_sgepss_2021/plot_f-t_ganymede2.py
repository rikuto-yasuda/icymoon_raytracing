# %%
import pprint
import cdflib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from multiprocessing import Pool
from tqdm import tqdm
import os
import shutil

# %%
# あらかじめ ../result_sgepss_2021/~/~ に必要なレイトレーシング結果とパラメータセットを入れること

object_name = 'ganymede'  # ganydeme/
highest_plasma = '1e2'  # 単位は(/cc) 2e2/4e2/16e2
plasma_scaleheight = '3e2'  # 単位は(km) 1.5e2/3e2/6e2

Radio_name_cdf = '../result_sgepss_2021/tools/result_for_yasudaetal2022/tracing_range_' + \
    object_name+'/para_'+highest_plasma+'_'+plasma_scaleheight+'.csv'
Radio_Range = pd.read_csv(Radio_name_cdf, header=0)

Radio_observer_position = np.loadtxt(
    '/Users/yasudarikuto/research/raytracing/tools/result_for_yasudaetal2022/R_P_'+object_name+'fulldata.txt',)  # 電波源の経度を含む


galdata = np.loadtxt(
    '/Users/yasudarikuto/research/raytracing/tools/result_sgepss_2021/GLL_GAN_2.txt')


n = len(Radio_observer_position)
Freq_str = ['3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]

date = np.arange(0, 10801, 60)  # エクスプレスコードで計算している時間幅（sec)を60で割る

Freq_num = []
for idx in Freq_str:
    Freq_num.append(float(idx)/1000000)


Highest = Radio_Range.highest
Lowest = Radio_Range.lowest
Except = Radio_Range.exc
res = Radio_observer_position.copy()
total_radio_number = list(np.arange(n))


# %%

def MakeFolder():
    new_dir_path_recursive = '../result_for_yasudaetal2022/' + \
        object_name+'_'+highest_plasma+'_'+plasma_scaleheight
    os.makedirs(new_dir_path_recursive)


def MoveFile():
    for l in range(len(Freq_num)):
        for j in range(Lowest[l], Highest[l], 2):
            k = str(j)
            shutil.move('../../src/rtc/testing/ray-P'+object_name+'_nonplume_'+highest_plasma+'_'+plasma_scaleheight+'-Mtest_simple-benchmark-LO-Z' +
                        k+'-FR'+Freq_str[l], '../result_for_yasudaetal2022/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight)


def Judge_occultation(i):
    aa = 0
    print(i)
    if Radio_observer_position[i][5] < 0:
        res[i][6] = 0
        aa = 1  # ukaru

    else:
        for l in range(len(Freq_num)):
            if Radio_observer_position[i][2] == Freq_num[l]:
                for j in range(Lowest[l], Highest[l], 2):
                    k = str(j)
                    Radio_propagation_route = np.genfromtxt("../result_for_yasudaetal2022/"+object_name+"_"+highest_plasma+"_"+plasma_scaleheight +
                                                            "/ray-P"+object_name+"_nonplume_"+highest_plasma+"_"+plasma_scaleheight+"-Mtest_simple-benchmark-LO-Z"+k+"-FR"+Freq_str[l])
                    n2 = len(Radio_propagation_route)
                    aa = 0

                    if Except[l] == [-10000]:

                        if Radio_propagation_route[n2-1][1] < 0:
                            continue

                        for h in range(n2):

                            if Radio_propagation_route[h][1] > Radio_observer_position[i][5] and Radio_propagation_route[h-1][3] < Radio_observer_position[i][6]:
                                para = np.abs(
                                    Radio_propagation_route[h][1]-Radio_observer_position[i][5])
                                hight = Radio_propagation_route[h][3] - \
                                    Radio_observer_position[i][6]
                                x1 = Radio_propagation_route[h][1]
                                z1 = Radio_propagation_route[h][3]
                                x2 = Radio_propagation_route[h-1][1]
                                z2 = Radio_propagation_route[h-1][3]

                                while (para > 10):
                                    ddx = (x1+x2)/2
                                    ddz = (z1+z2)/2

                                    if ddx > Radio_observer_position[i][5]:
                                        x1 = ddx
                                        z1 = ddz
                                    else:
                                        x2 = ddx
                                        z2 = ddz

                                    para = np.abs(
                                        x1-Radio_observer_position[i][5])
                                    hight = z1-Radio_observer_position[i][6]

                                if hight < 0:
                                    # res[i][6]=0
                                    aa = 1
                                    break

                        if aa == 1:
                            break

                    else:

                        if str(j) not in str(Except[l]):
                            if Radio_propagation_route[n2-1][1] < 0:
                                continue

                            for h in range(n2):
                                if Radio_propagation_route[h][1] > Radio_observer_position[i][5] and Radio_propagation_route[h-1][3] < Radio_observer_position[i][6]:
                                    para = np.abs(
                                        Radio_propagation_route[h][1]-Radio_observer_position[i][5])
                                    hight = Radio_propagation_route[h][3] - \
                                        Radio_observer_position[i][6]
                                    x1 = Radio_propagation_route[h][1]
                                    z1 = Radio_propagation_route[h][3]
                                    x2 = Radio_propagation_route[h-1][1]
                                    z2 = Radio_propagation_route[h-1][3]

                                    while (para > 10):
                                        ddx = (x1+x2)/2
                                        ddz = (z1+z2)/2

                                        if ddx > Radio_observer_position[i][5]:
                                            x1 = ddx
                                            z1 = ddz
                                        else:
                                            x2 = ddx
                                            z2 = ddz

                                        para = np.abs(
                                            x1-Radio_observer_position[i][5])
                                        hight = z1 - \
                                            Radio_observer_position[i][6]

                                    if hight < 0:
                                        # res[i][6]=0 #ukaru
                                        aa = 1
                                        break

                        if aa == 1:
                            break
    return aa


def Replace_Save(judgement, all_radio_data):
    occultaion_aray = np.array(judgement)
    judge_array = np.where(occultaion_aray[:] == 1)
    aLl_detectable_radio = all_radio_data[judge_array][:]
    np.savetxt('../result_for_yasudaetal2022/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'/' +
               object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'_dectable_radio_data.txt', aLl_detectable_radio)

    return aLl_detectable_radio


def Prepare_Figure(judgement):

    DataA = np.zeros(len(date)*(len(Freq_num)+1)
                     ).reshape(len(Freq_num)+1, len(date))
    DataB = np.zeros(len(date)*(len(Freq_num)+1)
                     ).reshape(len(Freq_num)+1, len(date))
    DataC = np.zeros(len(date)*(len(Freq_num)+1)
                     ).reshape(len(Freq_num)+1, len(date))
    DataD = np.zeros(len(date)*(len(Freq_num)+1)
                     ).reshape(len(Freq_num)+1, len(date))

    for k in range(len(judgement)):
        Num = int(judgement[k][0]*60+judgement[k][1]-330)
        if np.abs(galdata[Num][2]+360-Radio_observer_position[k][7]) < np.abs(galdata[Num][2]-Radio_observer_position[k][7]):
            Lon = galdata[Num][2]+360 - judgement[k][7]

        elif np.abs(judgement[k][7]+360-galdata[Num][2]) < np.abs(judgement[k][7]-galdata[Num][2]):
            Lon = galdata[Num][2]-360 - judgement[k][7]

        else:
            Lon = galdata[Num][2] - judgement[k][7]

        Lat = judgement[k][4]

        Fre = np.where(Freq_num == judgement[k][2])
        if Lon < 0 and Lat > 0:
            DataA[int(Fre[0])+1][Num] = 1
        if Lon > 0 and Lat > 0:
            DataB[int(Fre[0])+1][Num] = 1

        if Lon < 0 and Lat < 0:
            DataC[int(Fre[0])+1][Num] = 1

        if Lon > 0 and Lat < 0:
            DataD[int(Fre[0])+1][Num] = 1

    print("complete")

    return DataA, DataB, DataC, DataD


def Male_FT_full(DataA, DataB, DataC, DataD):

    FREQ = np.insert(np.array(Freq_num), 0, 0.36122)

    yTags = [5.620e+00, 1.000e+01, 1.780e+01, 3.110e+01, 4.213e+01, 4.538e+01, 4.888e+01, 5.265e+01, 5.671e+01, 6.109e+01, 6.580e+01, 7.087e+01, 7.634e+01, 8.223e+01, 8.857e+01,
            9.541e+01, 1.028e+02, 1.107e+02, 1.192e+02, 1.284e+02, 1.383e+02, 1.490e+02, 1.605e+02, 1.729e+02, 1.862e+02, 2.006e+02, 2.160e+02, 2.327e+02, 2.507e+02, 2.700e+02, 2.908e+02,
            3.133e+02, 3.374e+02, 3.634e+02, 3.915e+02, 4.217e+02, 4.542e+02, 4.892e+02, 5.270e+02, 5.676e+02, 6.114e+02, 6.586e+02, 7.094e+02, 7.641e+02, 8.230e+02, 8.865e+02, 9.549e+02,
            1.029e+03, 1.108e+03, 1.193e+03, 1.285e+03, 1.385e+03, 1.491e+03, 1.606e+03, 1.730e+03, 1.864e+03, 2.008e+03, 2.162e+03, 2.329e+03, 2.509e+03, 2.702e+03, 2.911e+03, 3.135e+03,
            3.377e+03, 3.638e+03, 3.918e+03, 4.221e+03, 4.546e+03, 4.897e+03, 5.275e+03, 5.681e+03, 6.120e+03, 6.592e+03, 7.100e+03, 7.648e+03, 8.238e+03, 8.873e+03, 9.558e+03, 1.029e+04,
            1.109e+04, 1.194e+04, 1.287e+04, 1.386e+04, 1.493e+04, 1.608e+04, 1.732e+04, 1.865e+04, 2.009e+04, 2.164e+04, 2.331e+04, 2.511e+04, 2.705e+04, 2.913e+04, 3.138e+04, 3.380e+04,
            3.641e+04, 3.922e+04, 4.224e+04, 4.550e+04, 4.901e+04, 5.279e+04, 5.686e+04, 6.125e+04, 6.598e+04, 7.106e+04, 7.655e+04, 8.245e+04, 8.881e+04, 9.566e+04, 1.030e+05, 1.030e+05,
            1.137e+05, 1.254e+05, 1.383e+05, 1.526e+05, 1.683e+05, 1.857e+05, 2.049e+05, 2.260e+05, 2.493e+05, 2.750e+05, 3.034e+05, 3.347e+05, 3.692e+05, 4.073e+05, 4.493e+05, 4.957e+05,
            5.468e+05, 6.033e+05, 6.655e+05, 7.341e+05, 8.099e+05, 8.934e+05, 9.856e+05, 1.087e+06, 1.199e+06, 1.323e+06, 1.460e+06, 1.610e+06,
            1.776e+06, 1.960e+06, 2.162e+06, 2.385e+06, 2.631e+06, 2.902e+06, 3.201e+06, 3.532e+06, 3.896e+06, 4.298e+06, 4.741e+06, 5.231e+06, 5.770e+06]

    Ytag = np.array(yTags, dtype='float64')/1000000
    gal_rad_row = pd.read_csv(
        '../result_sgepss_2021/Survey_Electric_1996-06-27T05-30_1996-06-27T07-00.csv', header=None)
    Xtag = np.array(gal_rad_row.iloc[0])
    Xtag = Xtag.astype(np.str)
    XXtag = np.zeros(len(Xtag))
    for i in range(len(XXtag)):
        onesplit = np.char.split(np.char.split(
            Xtag[:], sep='T')[i][1], sep=[':'])[0]
        onesplit2 = [float(vle) for vle in onesplit]
        XXtag[i] = onesplit2[2] + onesplit2[1] * \
            60 + onesplit2[0]*3600  # 経過時間(sec)に変換

    df = pd.DataFrame(gal_rad_row[:][1:])
    DDF = np.array(df).astype(np.float64)
    xtite = np.array(XXtag-XXtag[0])

    x = xtite
    y = Ytag
    xx, yy = np.meshgrid(x, y)

    fig, ax = plt.subplots(1, 1)

    pcm = ax.pcolor(xx, yy, DDF, norm=mpl.colors.LogNorm(
        vmin=1e-16, vmax=1e-12), cmap='Spectral_r')
    fig.colorbar(pcm, extend='max')
    plt.contour(date, FREQ, DataA, levels=[0.5], colors='white')
    plt.contour(date, FREQ, DataB, levels=[0.5], colors='lightgray')
    plt.contour(date, FREQ, DataC, levels=[0.5], colors='darkgray')
    plt.contour(date, FREQ, DataD, levels=[0.5], colors='black')
    plt.yscale("log")
    plt.ylim(100000, 5000000)
    plt.xlabel("Time of 27 June 1996")
    plt.ylabel("Frequency (MHz)")
    plt.yscale('log')
    plt.xlim(0, 5400)
    plt.ylim(0.1, 5.0)
    ax.set_xticks([0, 900, 1800, 2700, 3600, 4500, 5400])
    ax.set_xticklabels(
        ["05:30", "05:45", "06:00", "06:15", "06:30", "06:45", "07;00"])
    plt.title("max density "+highest_plasma +
            "(/cc) scale height "+plasma_scaleheight+"(km)")

    fig.savefig(os.path.join(save_dir, object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'_f-t.png')


def main():
    """`
    global res
    with tqdm(total=n) as t:
        pool = Pool(processes=8)
        for _ in pool.imap_unordered(calc, range(n)):
            t.update(1)
    """
    MakeFolder()
    MoveFile()

    with Pool(processes=16) as pool:
        result_list=list(pool.map(Judge_occultation, total_radio_number))
        # pool.map(calc,args)
        # pool.map(proc,[0,1,2])
        # args = list(np.arange(0,n,1))

    detectable_radio=Replace_Save(result_list, Radio_observer_position)

    detectabe_A, detectable_B, detectableC, detectableD=Prepare_Figure(
        detectable_radio)

    return 0


# %%
if __name__ == "__main__":
    main()

# date = np.arange('1996-06-27 05:30:00', '1996-06-27 08:31:00',np.timedelta64(1,'m'), dtype='datetime64')


FREQ=np.insert(np.array(Freq_num), 0, 0.36122)

yTags=[5.620e+00, 1.000e+01, 1.780e+01, 3.110e+01, 4.213e+01, 4.538e+01, 4.888e+01, 5.265e+01, 5.671e+01, 6.109e+01, 6.580e+01, 7.087e+01, 7.634e+01, 8.223e+01, 8.857e+01,
         9.541e+01, 1.028e+02, 1.107e+02, 1.192e+02, 1.284e+02, 1.383e+02, 1.490e+02, 1.605e+02, 1.729e+02, 1.862e+02, 2.006e+02, 2.160e+02, 2.327e+02, 2.507e+02, 2.700e+02, 2.908e+02,
         3.133e+02, 3.374e+02, 3.634e+02, 3.915e+02, 4.217e+02, 4.542e+02, 4.892e+02, 5.270e+02, 5.676e+02, 6.114e+02, 6.586e+02, 7.094e+02, 7.641e+02, 8.230e+02, 8.865e+02, 9.549e+02,
         1.029e+03, 1.108e+03, 1.193e+03, 1.285e+03, 1.385e+03, 1.491e+03, 1.606e+03, 1.730e+03, 1.864e+03, 2.008e+03, 2.162e+03, 2.329e+03, 2.509e+03, 2.702e+03, 2.911e+03, 3.135e+03,
         3.377e+03, 3.638e+03, 3.918e+03, 4.221e+03, 4.546e+03, 4.897e+03, 5.275e+03, 5.681e+03, 6.120e+03, 6.592e+03, 7.100e+03, 7.648e+03, 8.238e+03, 8.873e+03, 9.558e+03, 1.029e+04,
         1.109e+04, 1.194e+04, 1.287e+04, 1.386e+04, 1.493e+04, 1.608e+04, 1.732e+04, 1.865e+04, 2.009e+04, 2.164e+04, 2.331e+04, 2.511e+04, 2.705e+04, 2.913e+04, 3.138e+04, 3.380e+04,
         3.641e+04, 3.922e+04, 4.224e+04, 4.550e+04, 4.901e+04, 5.279e+04, 5.686e+04, 6.125e+04, 6.598e+04, 7.106e+04, 7.655e+04, 8.245e+04, 8.881e+04, 9.566e+04, 1.030e+05, 1.030e+05,
         1.137e+05, 1.254e+05, 1.383e+05, 1.526e+05, 1.683e+05, 1.857e+05, 2.049e+05, 2.260e+05, 2.493e+05, 2.750e+05, 3.034e+05, 3.347e+05, 3.692e+05, 4.073e+05, 4.493e+05, 4.957e+05,
         5.468e+05, 6.033e+05, 6.655e+05, 7.341e+05, 8.099e+05, 8.934e+05, 9.856e+05, 1.087e+06, 1.199e+06, 1.323e+06, 1.460e+06, 1.610e+06,
         1.776e+06, 1.960e+06, 2.162e+06, 2.385e+06, 2.631e+06, 2.902e+06, 3.201e+06, 3.532e+06, 3.896e+06, 4.298e+06, 4.741e+06, 5.231e+06, 5.770e+06]

Ytag=np.array(yTags, dtype='float64')/1000000
# np.savetxt('../result_sgepss_2021/Ytag.txt', Ytag)
gal_rad_row=pd.read_csv(
    '../result_sgepss_2021/Survey_Electric_1996-06-27T05-30_1996-06-27T07-00.csv', header=None)
Xtag=np.array(gal_rad_row.iloc[0])
Xtag=Xtag.astype(np.str)
XXtag=np.zeros(len(Xtag))
for i in range(len(XXtag)):
    onesplit=np.char.split(np.char.split(
        Xtag[:], sep='T')[i][1], sep=[':'])[0]
    onesplit2=[float(vle) for vle in onesplit]
    XXtag[i]=onesplit2[2] + onesplit2[1] * \
        60 + onesplit2[0]*3600  # 経過時間(sec)に変換


df=pd.DataFrame(gal_rad_row[:][1:])
DDF=np.array(df).astype(np.float64)
xtite=np.array(XXtag-XXtag[0])
"""
cb_min, cb_max = 1E-16, 1E-12
cb_div = 100
interval_of_cf = np.linspace(cb_min, cb_max, cb_div+1)
"""

x=xtite
y=Ytag
xx, yy=np.meshgrid(x, y)

fig, ax=plt.subplots(1, 1)

pcm=ax.pcolor(xx, yy, DDF, norm=mpl.colors.LogNorm(
    vmin=1e-16, vmax=1e-12), cmap='Spectral_r')
fig.colorbar(pcm, extend='max')
plt.contour(date, FREQ, DataA, levels=[0.5], colors='white')
plt.contour(date, FREQ, DataB, levels=[0.5], colors='lightgray')
plt.contour(date, FREQ, DataC, levels=[0.5], colors='darkgray')
plt.contour(date, FREQ, DataD, levels=[0.5], colors='black')
plt.yscale("log")
plt.ylim(100000, 5000000)
plt.xlabel("Time of 27 June 1996")
plt.ylabel("Frequency (MHz)")
plt.yscale('log')
plt.xlim(0, 5400)
plt.ylim(0.1, 5.0)
ax.set_xticks([0, 900, 1800, 2700, 3600, 4500, 5400])
ax.set_xticklabels(
    ["05:30", "05:45", "06:00", "06:15", "06:30", "06:45", "07;00"])
plt.title("max density "+highest_plasma +
          "(/cc) scale height "+plasma_scaleheight+"(km)")

fig.savefig(object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'_f-t.png')
# tick label
# %%
fig, ax=plt.subplots(1, 1)

pcm=ax.pcolor(xx, yy, DDF, norm=mpl.colors.LogNorm(
    vmin=1e-16, vmax=1e-12), cmap='Spectral_r')
fig.colorbar(pcm, extend='max')
plt.contour(date, FREQ, DataA, levels=[0.5], colors='white')
plt.contour(date, FREQ, DataB, levels=[0.5], colors='lightgray')
plt.contour(date, FREQ, DataC, levels=[0.5], colors='darkgray')
plt.contour(date, FREQ, DataD, levels=[0.5], colors='black')
plt.yscale("log")
plt.xlabel("Time of 27 June 1996")
plt.ylabel("Frequency (MHz)")
plt.yscale('log')
plt.xlim(2400, 3600)
plt.ylim(0.1, 5.0)
ax.set_xticks([2400, 3000, 3600, ])
ax.set_xticklabels(["06:10", "06:20", "06:30"])
plt.title("max density "+highest_plasma +
          "(/cc) scale height "+plasma_scaleheight+"(km)")

fig.savefig(object_name+'_'+highest_plasma+'_' +
            plasma_scaleheight+'_detail_f-t.png')

# %%
"""
data = DDF

fig = plt.figure(figsize=(4, 4))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.5,
                 label_mode='L', cbar_location='right', cbar_mode='each', cbar_pad=0.2)
# im = grid[0].imshow(data, origin=‘lower’)
# grid[0].set_title(‘linear scale’)
# cbar = grid.cbar_axes[0].colorbar(im)
im = grid[0].imshow(data, origin='lower', norm=LogNorm())
grid[0].set_title('log scale')
cbar = grid.cbar_axes[0].colorbar(im)  # , locator=LogLocator())
cbar.norm.vmax = 1.0e-12
cbar.norm.vmin = 1.0e-16
# cbar.cbar_axis.set_major_formatter(LogFormatterSciNotation())
# cbar.cbar_axis.set_minor_locator(LogLocator(subs=‘auto’))
cbar.ax.xaxis.set_visible(False)
# %%

plt.imshow(DDF,norm=mpl.colors.LogNorm(),origin='lower',
           interpolation='nearest',aspect=4,vmin=1E-16, vmax=1E-12,cmap="Spectral_r")
plt.ylim(109,151)


# %%
# 型をstrに変換
Xtag = Xtag.astype(np.str)
# 'T'と':'をどちらも適用できていない
np.char.split(Xtag[0:2], sep=['T',':'])
plt.imshow(Gll_Rad_data, origin='lower', interpolation='nearest')

# %%

fig, ax = plt.subplots(1, 1)

pcm = ax.pcolor(xx, yy, DDF, norm=mpl.colors.LogNorm(
    vmin=1e-16, vmax=1e-12),cmap='Spectral_r')
fig.colorbar(pcm, extend='max')
plt.yscale("log")
plt.ylim(100000,5000000)
"""
