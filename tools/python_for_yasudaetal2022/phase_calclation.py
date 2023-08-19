# ガニメデ電離圏を通過してとらえた電波がどれくらい回転して見えるか検証
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pathlib

a = 0
# tangential to jupiter (m) // ガニメデ公転半径　1070400 km = 1070400000 m とりあえず1000km
b = 10000000
n = int(b/1000)  # 分割数は一キロ分解能になるように
magnetic_field = 750*(10**-9)  # 　磁場強度 T //ガニメデ赤道表面 750 nT 木星磁場 100 nT
max_density1 = 100*(10**6)  # 最大電子密度 m-3 //n(/cc)=n*10**6(/m3)  250 or 100 /cc
scale_height1 = 1000*(10**3)  # スケールハイト m //l(km)=l*10**3(m) 1500 or 300 km

max_density2 = 25*(10**6)  # 最大電子密度 m-3 //n(/cc)=n*10**6(/m3)  250 or 100 /cc
scale_height2 = 100*(10**3)  # スケールハイト m //l(km)=l*10**3(m) 1500 or 300 km

radius = 2634100  # 半径 m ガニメデ半球 2634.1 km = 2634100 m
diameter_raio = 0.2  # 楕円の長辺と短辺の比率(0-1)円偏波度の考慮はここで起こる

# 短冊の幅Δx
dx = (b-a)/n
# frequency = np.arange(200000, 10000000, 100)  # 0.1MHzから10MHzを0.01MHz間隔で分解
# 0.1MHzから10MHzを0.01MHz間隔で分解 位相計算用
frequency = np.arange(1000000, 10000000, 100)

def calc_psi_deg(max_density,scale_height):

    # 周波数 Hz // 1MHz 1000000Hz
    # 積分する関数の定義
    K = 2.42*(10**4)
    coefficient = (K*magnetic_field) / \
        (1.41421*frequency*frequency)  # 0.1MHzから10MHzの係数


    def f(x):
        integrand = np.exp(-1*(np.sqrt((x*x)+radius*radius)-radius)/scale_height)
        return integrand


    # 面積の総和
    s = 0
    for i in range(n):
        x1 = a + dx*i
        x2 = a + dx*(i+1)
        f1 = f(x1)    # 上底
        f2 = f(x2)    # 下底
        # 面積
        s += dx * (f1+f2)/2

    TEC = s*max_density
    psi_rad = TEC*coefficient  # 0.1MHzから10MHzのψ角
    e_magnitude = np.reciprocal(np.sqrt(np.square(np.cos(psi_rad)) +
                                        np.square(np.sin(psi_rad)/diameter_raio)))
    radio_intensity = np.square(e_magnitude)
    psi_deg = np.rad2deg(psi_rad)
    return psi_deg

psi_deg1 = calc_psi_deg(max_density1,scale_height1)
psi_deg2 = calc_psi_deg(max_density2, scale_height2)



deg_1MHz = psi_deg[int(np.where(frequency == 1000000)[0])]

deg_1MHz_plus_pi = deg_1MHz+180
deg_1MHz_plus_2pi = deg_1MHz+360
deg_1MHz_plus_3pi = deg_1MHz+540

fre_1MHz_plus_pi = frequency[np.where(psi_deg > deg_1MHz_plus_pi)[0][-1]]
fre_1MHz_plus_2pi = frequency[np.where(psi_deg > deg_1MHz_plus_2pi)[0][-1]]
fre_1MHz_plus_3pi = frequency[np.where(psi_deg > deg_1MHz_plus_3pi)[0][-1]]

width_1 = 1000000-fre_1MHz_plus_pi
width_2 = fre_1MHz_plus_pi-fre_1MHz_plus_2pi
width_3 = fre_1MHz_plus_2pi-fre_1MHz_plus_3pi
width_total = 1000000-fre_1MHz_plus_3pi


print(fre_1MHz_plus_pi)
# 結果の表示(小数点以下10桁)
print("raypath_rength(km):"+str(int(b/1000)) +
      "/ magnetic_field(nT):"+str(magnetic_field*(10**9)))
print("max_density(/cc):"+str(max_density/(10**6))
      + " / scale_height(km):"+str(scale_height/1000))
print("psi_radian:", psi_rad)
print("psi_degree:", psi_deg)

# 以下電波強度用

reshape_intensity = np.reshape(radio_intensity, (1, len(radio_intensity)))
c = np.concatenate([reshape_intensity, reshape_intensity]).T
# print(b)
#X, Y = np.mgrid[:2, :9700]

xx, yy = np.meshgrid([0, 1], frequency)
#print(xx, yy)
fig, ax = plt.subplots(1, 2)

# ガリレオ探査機の電波強度をカラーマップへ
pcm = ax.pcolormesh(xx, yy, c, norm=mpl.colors.LogNorm(
    vmin=1e-3, vmax=10), cmap='Spectral_r')
fig.colorbar(pcm, extend='max', label='normalized radio intensity')

ax[0].set_yscale("log")
ax[0].set_ylim(100000, 6000000)
ax[0].axhline(y=1000000, xmin=0, xmax=1, color="green",
           label='1MHz', linestyle="dashed")
ax[0].axhline(y=fre_1MHz_plus_pi, xmin=0, xmax=1, color="blue",
           label='1MHz-pi', linestyle="dashed")
ax[0].axhline(y=fre_1MHz_plus_2pi, xmin=0, xmax=1, color="grey",
           label='1MHz-2pi', linestyle="dashed")
ax[0].axhline(y=fre_1MHz_plus_3pi, xmin=0, xmax=1, color="purple",
           label='1MHz-3pi', linestyle="dashed")
ax[0].set_ylabel("radio frequency (Hz)")
ax[0].set_title("max:"+str(max_density/1000000) + "(/cc) h_s " +
             str(scale_height/1000)+"(km) TEC:"+str('{:.2e}'.format(TEC))+"(/m2)"+" \n width(MHz):"+str('{:.2e}'.format(width_1/1000000))+","+str('{:.2e}'.format(width_2/1000000))+","+str('{:.2e}'.format(width_3/1000000)), fontsize=10)
ax[0].axes.xaxis.set_visible(False)
ax[0].legend()



print("total_width:"+str(width_total/1000000))

# 以下電波位相用


psi_deg1_mod = np.mod(psi_deg1, 360)
plt.xlim(1000000, 10000000)
plt.ylim(0, 200)
# print(b)
#plt.plot(frequency, psi_deg_mod)
plt.plot(frequency, psi_deg1,label='1.max:100 /cc scale: 1000 km')
plt.plot(frequency,psi_deg2,label='2.max:25 /cc scale: 100 km')
plt.legend()
plt.title("faraday rotation effect", fontsize=10)
#plt.title("max:"+str(max_density/1000000) + "(/cc) h_s " +str(scale_height/1000)+"(km) TEC:"+str('{:.2e}'.format(TEC))+"(/m2)", fontsize=10)
plt.xlabel("radio frequency (MHz)")
plt.ylabel("rotation angle (deg)")
plt.xticks([1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000,8000000,9000000,10000000], ['1', '2', '3', '4', '5', '6', '7','8','9','10'])
plt.savefig("others/max_" + str(int(max_density/1000000)) +
            "_cc_scaleheight_"+str(int(scale_height/1000))+"_km.png")

plt.show()

# %%
