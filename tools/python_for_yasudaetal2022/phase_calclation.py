# ガニメデ電離圏を通過してとらえた電波がどれくらい回転して見えるか検証
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pathlib

max_electron_density = 100 * (
    10**6
)  # 最大電子密度 m-3 //n(/cc)=n*10**6(/m3)  250 or 100 /cc
electron_density_scale_height = 100 * (
    10**3
)  # スケールハイト m //l(km)=l*10**3(m) 1500 or 300 km
frequency_array = np.arange(10000, 10000000, 1)
magnetic_field_intensity = 750 * (10**-9)  # 　磁場強度 T //ガニメデ赤道表面 750 nT 木星磁場 100 nT
moon_radius = 2634100  # 半径 m ガニメデ半球 2634.1 km = 2634100 m
diameter_raio = 0.2  # 楕円の長辺と短辺の比率(0-1)円偏波度の考慮はここで起こる

# ガリレオ周波数バンド幅 (Hz)
gal_fleq_band = 1340

# ガリレオ周波数チャンネル (Hz)
gal_fleq_tag_row = np.array(
    [
        1.030e05,
        1.137e05,
        1.254e05,
        1.383e05,
        1.526e05,
        1.683e05,
        1.857e05,
        2.049e05,
        2.260e05,
        2.493e05,
        2.750e05,
        3.034e05,
        3.347e05,
        3.692e05,
        4.073e05,
        4.493e05,
        4.957e05,
        5.468e05,
        6.033e05,
        6.655e05,
        7.341e05,
        8.099e05,
        8.934e05,
        9.856e05,
        1.087e06,
        1.199e06,
        1.323e06,
        1.460e06,
        1.610e06,
        1.776e06,
        1.960e06,
        2.162e06,
        2.385e06,
        2.631e06,
        2.902e06,
        3.201e06,
        3.532e06,
        3.896e06,
        4.298e06,
        4.741e06,
        5.231e06,
        5.770e06,
    ],
    dtype="float64",
)

# JUICE バンド幅
juice_freq_band = 37000
# JUICE周波数チャンネル
juice_fleq_tag_row = np.linspace(80000, 2096000, 72)


def calc_psi_deg(max_density, scale_height, frequency, magnetic_field, radius):
    a = 0
    # tangential to jupiter (m) // ガニメデ公転半径　1070400 km = 1070400000 m とりあえず1000km
    b = 10000000
    n = int(b / 1000)  # 分割数は一キロ分解能になるよう
    # 短冊の幅Δx
    dx = (b - a) / n

    # 積分する関数の定義
    K = 2.42 * (10**4)
    coefficient = (K * magnetic_field) / (
        1.41421 * frequency * frequency
    )  # 0.1MHzから10MHzの係数

    def f(x):
        integrand = np.exp(
            -1 * (np.sqrt((x * x) + radius * radius) - radius) / scale_height
        )
        return integrand

    # 面積の総和
    s = 0
    for i in range(n):
        x1 = a + dx * i
        x2 = a + dx * (i + 1)
        f1 = f(x1)  # 上底
        f2 = f(x2)  # 下底
        # 面積
        s += dx * (f1 + f2) / 2

    TEC = s * max_density
    psi_rad = TEC * coefficient  # 0.1MHzから10MHzのψ角
    e_magnitude = np.reciprocal(
        np.sqrt(np.square(np.cos(psi_rad)) + np.square(np.sin(psi_rad) / diameter_raio))
    )
    radio_int = np.square(e_magnitude)
    psi_deg = np.rad2deg(psi_rad)
    return psi_deg, radio_int, TEC


####### 理想的な周波数分解能を有するときのファラデー回転縞を描写するコード ################


def plot_original_faraday_stripe(frequency, radio_intensity, total_electron_content):
    reshape_intensity = np.reshape(radio_intensity, (1, len(radio_intensity)))

    c = np.concatenate([reshape_intensity, reshape_intensity]).T

    xx, yy = np.meshgrid([0, 1], frequency)
    # print(xx, yy)
    fig, ax = plt.subplots(1, 1)

    # ガリレオ探査機の電波強度をカラーマップへ
    pcm = ax.pcolormesh(
        xx, yy, c, norm=mpl.colors.LogNorm(vmin=1e-3, vmax=10), cmap="Spectral_r"
    )
    fig.colorbar(pcm, extend="max", label="Radio intensity (nomarized)")

    ax.set_yscale("log")
    ax.set_ylim(100000, 2000000)
    ax.set_ylabel("Frequency (Hz)")
    ax.axes.xaxis.set_visible(False)
    ax.set_title(
        "original radio intensity\nmax:"
        + str(max_electron_density / 1000000)
        + "(/cc) h_s "
        + str(electron_density_scale_height / 1000)
        + "(km) TEC:"
        + str("{:.2e}".format(total_electron_content))
        + "(/m2)",
        fontsize=10,
    )
    plt.savefig(
        "../result_for_yasudaetal2022/faraday_stripe/row_data/max_"
        + str(int(max_electron_density / 1000000))
        + "_cc_scaleheight_"
        + str(int(electron_density_scale_height / 1000))
        + "_km.png"
    )
    return


def plot_binning_raraday_stripe(
    radio_frequency,
    instrment_frequency,
    instrment_freq_band,
    radio_intensity,
    total_electron_content,
    instrument,
):
    instrment_intensity = np.zeros(len(instrment_frequency))

    for i in range(len(instrment_frequency)):
        minimum_freqency_in_band = instrment_frequency[i] - (instrment_freq_band / 2)
        maximum_freqency_in_band = instrment_frequency[i] + (instrment_freq_band / 2)

        minimum_freqency_position = np.where(
            radio_frequency > minimum_freqency_in_band
        )[0][0]
        maximum_freqency_position = np.where(
            radio_frequency < maximum_freqency_in_band
        )[0][-1]

        average_intensity_in_band = np.mean(
            radio_intensity[minimum_freqency_position:maximum_freqency_position]
        )

        instrment_intensity[i] = average_intensity_in_band

    reshape_intensity = np.reshape(instrment_intensity, (1, len(instrment_intensity)))

    c = np.concatenate([reshape_intensity, reshape_intensity]).T

    xx, yy = np.meshgrid([0, 1], instrment_frequency)
    # print(xx, yy)
    fig, ax = plt.subplots(1, 1)

    # ガリレオ探査機の電波強度をカラーマップへ
    pcm = ax.pcolormesh(
        xx, yy, c, norm=mpl.colors.LogNorm(vmin=1e-3, vmax=10), cmap="Spectral_r"
    )
    fig.colorbar(pcm, extend="max", label="Radio intensity (nomarized)")

    ax.set_yscale("log")
    ax.set_ylim(100000, 2000000)
    ax.set_ylabel("Frequency (Hz)")
    ax.axes.xaxis.set_visible(False)
    ax.set_title(
        instrument
        + " radio intensity\nmax:"
        + str(max_electron_density / 1000000)
        + "(/cc) h_s "
        + str(electron_density_scale_height / 1000)
        + "(km) TEC:"
        + str("{:.2e}".format(total_electron_content))
        + "(/m2)",
        fontsize=10,
    )

    plt.savefig(
        "../result_for_yasudaetal2022/faraday_stripe/"
        + instrument
        + "/max_"
        + str(int(max_electron_density / 1000000))
        + "_cc_scaleheight_"
        + str(int(electron_density_scale_height / 1000))
        + "_km.png"
    )

    return


def main():
    angle_array, radio_intensity, total_electron_content = calc_psi_deg(
        max_electron_density,
        electron_density_scale_height,
        frequency_array,
        magnetic_field_intensity,
        moon_radius,
    )

    plot_original_faraday_stripe(
        frequency_array, radio_intensity, total_electron_content
    )

    plot_binning_raraday_stripe(
        frequency_array,
        gal_fleq_tag_row,
        gal_fleq_band,
        radio_intensity,
        total_electron_content,
        "galileo_pws",
    )

    plot_binning_raraday_stripe(
        frequency_array,
        juice_fleq_tag_row,
        juice_freq_band,
        radio_intensity,
        total_electron_content,
        "juice_rpwi",
    )


if __name__ == "__main__":
    main()


"""
ax.axhline(
    y=fre_1MHz_plus_pi,
    xmin=0,
    xmax=1,
    color="blue",
    label="1MHz-pi",
    linestyle="dashed",
)
ax.axhline(
    y=fre_1MHz_plus_2pi,
    xmin=0,
    xmax=1,
    color="grey",
    label="1MHz-2pi",
    linestyle="dashed",
)
ax.axhline(
    y=fre_1MHz_plus_3pi,
    xmin=0,
    xmax=1,
    color="purple",
    label="1MHz-3pi",
    linestyle="dashed",
)
ax.legend()
"""


# 以下電波位相用

"""
psi_deg1_mod = np.mod(psi_deg1, 360)
plt.xlim(1000000, 10000000)
plt.ylim(0, 200)
# print(b)
# plt.plot(frequency, psi_deg_mod)
plt.plot(frequency, psi_deg1, label="1.max:100 /cc scale: 1000 km")
plt.plot(frequency, psi_deg2, label="2.max:25 /cc scale: 100 km")
plt.legend()
plt.title("faraday rotation effect", fontsize=10)
# plt.title("max:"+str(max_density/1000000) + "(/cc) h_s " +str(scale_height/1000)+"(km) TEC:"+str('{:.2e}'.format(TEC))+"(/m2)", fontsize=10)
plt.xlabel("radio frequency (MHz)")
plt.ylabel("rotation angle (deg)")
plt.xticks(
    [
        1000000,
        2000000,
        3000000,
        4000000,
        5000000,
        6000000,
        7000000,
        8000000,
        9000000,
        10000000,
    ],
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
)

"""
"""
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
"""


# %%
