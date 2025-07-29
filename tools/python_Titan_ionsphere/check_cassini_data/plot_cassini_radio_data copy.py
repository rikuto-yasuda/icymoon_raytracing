# %%
from maser.data import Data
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import os

# %%
path = "/Users/yasudarikuto/research/icymoon_raytracing/tools/python_Titan_ionsphere/Cassini_radio_data/"
plot_result_path = "/Users/yasudarikuto/research/icymoon_raytracing/tools/result_titan/radio_plot/"

Cassini_flyby = "T15" # "T15", "manual"

start_year = 2006
start_date = 106
start_hour = 18
duration = 4


if Cassini_flyby == "T15":
    start_year = 2006
    start_date = 183
    start_hour = 8
    duration = 3
    save_path = plot_result_path + Cassini_flyby + "/"
    os.makedirs(plot_result_path + Cassini_flyby, exist_ok=True)  # フォルダが存在しない場合は作成
    
elif Cassini_flyby == "manual":
    save_path = plot_result_path + Cassini_flyby + "/"
    os.makedirs(plot_result_path + Cassini_flyby, exist_ok=True)  # フォルダが存在しない場合は作成

# %%
def download_data(url, path, file_name):

    # 保存先のフォルダ
    os.makedirs(path, exist_ok=True)  # フォルダが存在しない場合は作成

    # 保存するファイル名
    save_path = os.path.join(path, file_name)

    # データをダウンロード
    response = requests.get(url)
    response.raise_for_status()  # エラーチェック

    # ダウンロードしたデータをファイルに保存
    with open(save_path, "wb") as file:
        file.write(response.content)

# %%
def load_intensity_data(path, year, date, hour, duration):

    n2_path = path + "n2/"
    n3_path = path + "n3e/"

    for i in range(duration):
        n3_data_name = "N3e_dsq" + str(year) + "{:03}".format(date+((hour + i)//24)) + "." + "{:02}".format((hour + i)%24)
        n2_data_name = "P" + str(year) + "{:03}".format(date+((hour + i)//24)) + "." + "{:02}".format((hour + i)%24)

        if (0 < date) and (date < 91):
            download_folder_str = "001_090"
        
        elif (90 < date) and (date < 181):
            download_folder_str = "091_180"
        
        elif (180 < date) and (date < 271):
            download_folder_str = "181_270"
        
        else:
            download_folder_str = "271_366"
        
        n2_data_download_url = "https://lesia.obspm.fr/kronos/data/" + str(year) + "_" + download_folder_str + "/n2/" + n2_data_name
        n3_data_download_url = "https://lesia.obspm.fr/kronos/data/" + str(year) + "_" + download_folder_str + "/n3e/" + n3_data_name


        if os.path.exists(n3_path + n3_data_name)==False:
            download_data(n3_data_download_url, n3_path, n3_data_name)
            download_data(n2_data_download_url, n2_path, n2_data_name)

        # データの読み込み and 強度や周波数、時間配列をインプット
        data_rpwi = Data(n3_path + n3_data_name)
        xr_st = data_rpwi.as_xarray()

        if i == 0:
            Radio_intensity = xr_st["s"].values # ?
            Epoch = xr_st["time"].values
        
        else:
            Radio_intensity = np.concatenate((Radio_intensity, xr_st["s"].values), axis=1)
            Epoch = np.concatenate([Epoch, xr_st["time"].values])


    # 周波数と時間を単純化した一次元配列を生成（周波数 Hz, 時間 0:00からの秒）
    # Epoch = [parse(iso_str) for iso_str in Epoch_row]
    Frequency_1d = xr_st["frequency"].values[0]  # kHz

    day_array = (
        np.array([np.datetime64(dt, "D").astype(int) for dt in Epoch])
        - np.array([np.datetime64(dt, "D").astype(int) for dt in Epoch])[0]
    )
    hour_array = np.array([np.datetime64(dt, "h").astype(int) % 24 for dt in Epoch])
    minute_array = np.array([np.datetime64(dt, "m").astype(int) % 60 for dt in Epoch])
    second_array = np.array([np.datetime64(dt, "s").astype(int) % 60 for dt in Epoch])
    mili_second_array = np.array(
        [np.datetime64(dt, "ms").astype(int) % 1000 for dt in Epoch]

    )

    Epoch_from_0min_1d = (
        day_array * 60 * 60 * 24
        + hour_array * 60 * 60
        + minute_array * 60
        + second_array
        + mili_second_array / 1000
        - day_array[10] * 60 * 60 * 24
    )

    return hour_array, minute_array, second_array, Epoch_from_0min_1d, Frequency_1d, Radio_intensity

# %%
def load_polarizaiton_data(path, year, date, hour, duration):

    n3_path = path + "n3e/"

    for i in range(duration):
        n3_data_name = "N3e_dsq" + str(year) + "{:03}".format(date+((hour + i)//24)) + "." + "{:02}".format((hour + i)%24)

        # データの読み込み and 強度や周波数、時間配列をインプット
        data_rpwi = Data(n3_path + n3_data_name)
        xr_st = data_rpwi.as_xarray()

        if i == 0:
            Radio_polarization = xr_st["v"].values # ?
        
        else:
            Radio_polarization = np.concatenate((Radio_polarization, xr_st["v"].values), axis=1)

    return Radio_polarization



def plot_ft_nomal(
    Hour_array,
    Start_hour,
    Duration,
    Epoch,
    Frequency,
    Intensity,
    instrument_nameandunit,
):


    xx, yy = np.meshgrid(Epoch, Frequency)
    fig, ax = plt.subplots(1, 1)


    pcm = ax.pcolormesh(
        xx,
        yy,
        Intensity,
        cmap="viridis",
        norm=LogNorm()
        )
    
    np.where(Hour_array == 0, 1, Intensity)

    for i in range(Duration):
        if i == 0:
            x_array = [Epoch[0]]
            x_label = [str(Start_hour)]
        
        else:
            x_pos = np.where(Hour_array == (Start_hour + i)%24)[0][0]
            x_array = np.append(x_array, Epoch[x_pos])
            x_label = np.append(x_label, str((Start_hour + i)%24))



    ax.set_xticks(x_array)
    ax.set_xticklabels(x_label)

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Frequency (kHz)")
    ax.set_yscale("log")
    ax.set_ylim(3, 1500)  # 3kHz~2MHz
    #ax.set_xlim(time_range[0], time_range[1])  # 80kHz~1MHz
    ax.set_yscale("log")

    fig.colorbar(
        pcm,
        extend="max",
        label=instrument_nameandunit,
    )

    plt.savefig(save_path+"intensity.png")
    plt.show()


    return 0


def plot_ft_polarization(
    Hour_array,
    Start_hour,
    Duration,
    Epoch,
    Frequency,
    Intensity,
    instrument_nameandunit,
):

    print(Frequency/1000)
    xx, yy = np.meshgrid(Epoch, Frequency)
    fig, ax = plt.subplots(1, 1)


    pcm = ax.pcolormesh(
        xx,
        yy,
        Intensity,
        cmap="viridis",
        vmin=-1, vmax=1
        )
    
    np.where(Hour_array == 0, 1, Intensity)

    for i in range(Duration):
        if i == 0:
            x_array = [Epoch[0]]
            x_label = [str(Start_hour)]
        
        else:
            x_pos = np.where(Hour_array == (Start_hour + i)%24)[0][0]
            x_array = np.append(x_array, Epoch[x_pos])
            x_label = np.append(x_label, str((Start_hour + i)%24))



    ax.set_xticks(x_array)
    ax.set_xticklabels(x_label)

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Frequency (kHz)")
    ax.set_yscale("log")
    ax.set_ylim(3, 1500)  # 3kHz~2MHz
    #ax.set_xlim(time_range[0], time_range[1])  # 80kHz~1MHz
    ax.set_yscale("log")

    fig.colorbar(
        pcm,
        extend="max",
        label=instrument_nameandunit,
    )

    plt.savefig(save_path+"polarization.png")
    plt.show()


    return 0

# %%
Hour, Minute, Second, Epoch_from_0min_1d, Frequency_1d, Radio_intensity = load_intensity_data(path, start_year, start_date, start_hour, duration)

Radio_polarization = load_polarizaiton_data(path, start_year, start_date, start_hour, duration)

plot_ft_nomal(
    Hour,
    start_hour,
    duration,
    Epoch_from_0min_1d,
    Frequency_1d,
    Radio_intensity,
    "Intensity")

plot_ft_polarization(
    Hour,
    start_hour,
    duration,
    Epoch_from_0min_1d,
    Frequency_1d,
    Radio_polarization,
    "Polarization")

# %%
