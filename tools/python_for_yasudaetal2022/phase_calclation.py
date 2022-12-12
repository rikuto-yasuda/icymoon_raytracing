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
max_density = 200*(10**6)  # 最大電子密度 m-3 //n(/cc)=n*10**6(/m3)  250 or 100 /cc
scale_height = 50*(10**3)  # スケールハイト m //l(km)=l*10**3(m) 1500 or 300 km
radius = 2634100  # 半径 m ガニメデ半球 2634.1 km = 2634100 m
diameter_raio = 0.2  # 楕円の長辺と短辺の比率(0-1)円偏波度の考慮はここで起こる


# 短冊の幅Δx
dx = (b-a)/n
frequency = np.arange(200000, 10000000, 100)  # 0.1MHzから10MHzを0.01MHz間隔で分解
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
psi_rad = TEC*coefficient + 1.570796  # 0.1MHzから10MHzのψ角
e_magnitude = np.reciprocal(np.sqrt(np.square(np.cos(psi_rad)) +
                                    np.square(np.sin(psi_rad)/diameter_raio)))
radio_intensity = np.square(e_magnitude)
psi_deg = np.rad2deg(psi_rad)

deg_1MHz = psi_deg[int(np.where(frequency == 1000000)[0])]
deg_1MHz_plus_pi = deg_1MHz+180
fre_1MHz_plus_pi = frequency[np.where(psi_deg > deg_1MHz_plus_pi)[0][-1]]
print(fre_1MHz_plus_pi)
# 結果の表示(小数点以下10桁)
print("raypath_rength(km):"+str(int(b/1000)) +
      "/ magnetic_field(nT):"+str(magnetic_field*(10**9)))
print("max_density(/cc):"+str(max_density/(10**6))
      + " / scale_height(km):"+str(scale_height/1000))
print("psi_radian:", psi_rad)
print("psi_degree:", psi_deg)
"""
plt.plot(frequency, psi_deg)
plt.plot(frequency, e_magnitude)
plt.plot(frequency, radio_intensity)
plt.show()
"""
b = np.reshape(radio_intensity, (len(radio_intensity), 1))
c = np.concatenate([b, b])
# print(b)
X, Y = np.mgrid[:2, :9700]
"""
cout = plt.contourf(X, Y, c, cmap='Spectral_r', locator=ticker.LogLocator(),)
plt.colorbar(cout)
cout.set_clim(0.001, 1)
"""

xx, yy = np.meshgrid([0, 1], frequency)
#print(xx, yy)
fig, ax = plt.subplots(1, 1)

# ガリレオ探査機の電波強度をカラーマップへ
pcm = ax.pcolormesh(xx, yy, b, norm=mpl.colors.LogNorm(
    vmin=1e-3, vmax=10), cmap='Spectral_r')
fig.colorbar(pcm, extend='max', label='normalized radio intensity')

ax.set_yscale("log")
ax.set_ylim(100000, 6000000)
ax.axhline(y=1000000, xmin=0, xmax=1, color="green",
           label='1MHz', linestyle="dashed")
ax.axhline(y=fre_1MHz_plus_pi, xmin=0, xmax=1, color="blue",
           label='1MHz-pi/2', linestyle="dashed")
ax.set_ylabel("radio frequency (Hz)")
ax.set_title("max:"+str(max_density/1000000) + "(/cc) h_s " +
             str(scale_height/1000)+"(km) TEC:"+str('{:.2e}'.format(TEC))+"(/m2)"+" wipth:"+str('{:.2e}'.format((1000000-fre_1MHz_plus_pi)/1000000))+"MHz", fontsize=10)
ax.axes.xaxis.set_visible(False)
ax.legend()
plt.savefig("max_" + str(int(max_density/1000000)) +
            "_cc_scaleheight_"+str(int(scale_height/1000))+"_km.png")
plt.show()
# %%
