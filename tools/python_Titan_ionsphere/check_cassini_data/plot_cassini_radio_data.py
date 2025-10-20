# %%
from maser.data import Data
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import os
from scipy.interpolate import interp1d

# %%
path = "/Users/yasudarikuto/research/icymoon_raytracing/tools/python_Titan_ionsphere/Cassini_radio_data/"
plot_result_path = "/Users/yasudarikuto/research/icymoon_raytracing/tools/result_titan/radio_plot/"

Cassini_flyby = "T15" # "T15", "manual"

start_year = 2006
start_date = 106
start_hour = 18
duration = 4
mask_frequency = [98,103,191,200,291,305,318,331,393,406,493,506,593,606,643,656,693,706,756,793,806,856,893,906,993]


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

def moving_average_intensity(Epoch, Intensity, window_sec=60):
    """
    各時間ステップで前後window_sec秒の移動平均を計算
    Intensity: shape = (周波数, 時間)
    Epoch: shape = (時間,)
    """
    freq_num, time_num = Intensity.shape
    averaged_Intensity = np.zeros_like(Intensity)
    for t in range(time_num):
        # 前後window_sec秒の範囲を取得
        t_min = Epoch[t] - window_sec
        t_max = Epoch[t] + window_sec
        idx = np.where((Epoch >= t_min) & (Epoch <= t_max))[0]
        # 各周波数ごとに平均
        averaged_Intensity[:, t] = np.mean(Intensity[:, idx], axis=1)
    return averaged_Intensity

def moving_average_frequency(Intensity, window_ch=5):
    """
    周波数方向に移動平均を計算
    Intensity: shape = (周波数, 時間)
    window_ch: 平均を取る周波数チャンネル数（奇数推奨）
    NaNが含まれる場合はその周波数帯は平均計算に使わない
    """
    freq_num, time_num = Intensity.shape
    averaged_Intensity = np.full_like(Intensity, np.nan)
    half_window = window_ch // 2

    for t in range(time_num):
        for f in range(freq_num):
            # 周波数方向のウィンドウ範囲
            f_min = max(0, f - half_window)
            f_max = min(freq_num, f + half_window + 1)
            window = Intensity[f_min:f_max, t]
            # NaNが含まれている場合は平均計算しない
            averaged_Intensity[f, t] = np.nanmean(window)
    return averaged_Intensity


def mask_data(Frequency, Intensity, mask_frequency):
    """
    データにマスクを適用する
    """
    Intensity = Intensity.copy()  # ここでコピーを作る
    for freq in mask_frequency:
        freq_mask = np.argmin(np.abs(Frequency-freq))
        Intensity[freq_mask, :] = np.nan
    return Intensity

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
    Radio_intensity = mask_data(Frequency, Intensity, mask_frequency)
    #averaged_Intensity = moving_average_intensity(Epoch_from_0min_1d, Radio_intensity, window_sec=30)
    #averaged_Intensity = moving_average_frequency(averaged_Intensity, window_ch=5)
    averaged_Intensity = Radio_intensity

    pcm = ax.pcolormesh(
        xx,
        yy,
        averaged_Intensity,
        cmap="jet",
        norm=LogNorm(vmin=1e-16, vmax=1e-12)
        )

    # 時間範囲を指定
    time_min = 3600 * 10.25
    time_max = 3600 * 10.42
    time_mask = (Epoch >= time_min) & (Epoch <= time_max)
    Epoch_in_range = Epoch[time_mask]
    idx_in_range = np.where(time_mask)[0]

    # maskした周波数のインデックスを取得
    mask_indices = [np.argmin(np.abs(Frequency - freq)) for freq in mask_frequency]

    # 各周波数ごとに最大変化比時刻にscatter（mask周波数は除外）
    for freq_idx in range(averaged_Intensity.shape[0]):
        if freq_idx in mask_indices:
            continue  # mask周波数はスキップ
        intensity_in_range = averaged_Intensity[freq_idx, idx_in_range]
        # 変化比（次の値/前の値）を計算
        # 0除算を避けるため、前の値が0の場合はnp.nanにする
        prev = intensity_in_range[:-1]
        next_ = intensity_in_range[1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(prev != 0, next_ / prev, np.nan)
        # 最大変化比のインデックス
        if np.all(np.isnan(ratio)):
            continue
        max_ratio_idx = np.nanargmax(ratio)
        scatter_time = Epoch_in_range[max_ratio_idx + 1]  # np.diffと同じく+1
        scatter_freq = Frequency[freq_idx]
        ax.scatter(scatter_time, scatter_freq, color="red", s=30, label="max ratio" if freq_idx==0 else "")



    np.where(Hour_array == 0, 1, Intensity)

    for i in range(Duration):
        if i == 0:
            x_array = [Epoch[0]]
            x_label = [str(Start_hour)]
        
        else:
            x_pos = np.where(Hour_array == (Start_hour + i)%24)[0][0]
            x_array = np.append(x_array, Epoch[x_pos])
            x_label = np.append(x_label, str((Start_hour + i)%24))


    ax.axvline(x=time_min, color="white", linestyle="--")
    ax.axvline(x=time_max, color="white", linestyle="--")

    ax.set_xticks(x_array)
    ax.set_xticklabels(x_label)

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Frequency (kHz)")
    ax.set_yscale("log")
    ax.set_ylim(100, 900)  # 3kHz~2MHz
    ax.set_xlim(3600*10.25, 3600*10.46)

    ax.set_yscale("log")

    fig.colorbar(
        pcm,
        extend="max",
        label=instrument_nameandunit,
    )

    plt.savefig(save_path+"intensity.png")
    plt.show()


    return 0

def plot_ft_diff(
        Hour_array,
        Start_hour,
        Duration,
        Epoch,
        Frequency,
        Intensity,
        instrument_nameandunit,
    ):
        """
        強度の時間差分（隣接時刻間の差分）をft図で可視化
        """
        xx, yy = np.meshgrid(Epoch, Frequency)
        Radio_intensity = mask_data(Frequency, Intensity, mask_frequency)
        averaged_Intensity = Radio_intensity
        # 時間方向の差分を計算
        diff_Intensity = np.diff(averaged_Intensity, axis=1)
        # 時刻軸（Epoch）は1つ短くなる
        xx_diff, yy_diff = np.meshgrid(Epoch[1:], Frequency)
        fig, ax = plt.subplots(1, 1)
        vmax = np.nanpercentile(np.abs(diff_Intensity), 90)
        pcm = ax.pcolormesh(
            xx_diff,
            yy_diff,
            diff_Intensity,
            cmap="bwr",
            vmin=-vmax, vmax=vmax
        )
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_yscale("log")
        ax.set_ylim(100, 900)
        ax.set_xlim(3600*10.25, 3600*10.46)
        fig.colorbar(pcm, label="Intensity Difference")
        plt.savefig(save_path+"intensity_diff.png")
        plt.show()
        return 0

def plot_ft_window_diff(
    Hour_array,
    Start_hour,
    Duration,
    Epoch,
    Frequency,
    Intensity,
    instrument_nameandunit,
    window_sec=30
):
    """
    各時刻で前後window_sec秒の平均値の差分（未来平均−過去平均）をft図で可視化
    """
    Radio_intensity = mask_data(Frequency, Intensity, mask_frequency)
    freq_num, time_num = len(Frequency), len(Epoch)

    diff_window = np.full_like(Radio_intensity, np.nan)
    for t in range(time_num):
        t_val = Epoch[t]
        # 過去window_sec秒の範囲
        idx_past = np.where((Epoch >= t_val - window_sec) & (Epoch < t_val))[0]
        # 未来window_sec秒の範囲
        idx_future = np.where((Epoch > t_val) & (Epoch <= t_val + window_sec))[0]
        if len(idx_past) == 0 or len(idx_future) == 0:
            continue
        past_mean = np.nanmean(Radio_intensity[:, idx_past], axis=1)
        future_mean = np.nanmean(Radio_intensity[:, idx_future], axis=1)
        diff_window[:, t] = future_mean / past_mean
    xx, yy = np.meshgrid(Epoch, Frequency)
    fig, ax = plt.subplots(1, 1)
    vmax = np.nanpercentile(np.abs(diff_window), 99)
    """
    pcm = ax.pcolormesh(
        xx,
        yy,
        diff_window,
        cmap="bwr",
        vmin=-vmax, vmax=vmax
    )
    """
    pcm = ax.pcolormesh(
        xx,
        yy,
        Intensity,
        cmap="jet",
        norm=LogNorm(vmin=1e-16, vmax=1e-12)
    )

    # 各周波数ごとに最大差分時刻にscatter
    # 時間区間を指定（例: 10.25h～10.42h）
    time_min = 37000
    time_max = 37500
    time_mask = (Epoch >= time_min) & (Epoch <= time_max)

    for freq_idx in range(diff_window.shape[0]):
        freq_diff = diff_window[freq_idx, :]
        # 区間内のみで探索
        freq_diff_in_range = np.where(time_mask, freq_diff, np.nan)
        if np.all(np.isnan(freq_diff_in_range)):
            continue
        max_idx = np.nanargmax(np.abs(freq_diff_in_range))
        scatter_time = Epoch[max_idx]
        scatter_freq = Frequency[freq_idx]
        ax.scatter(scatter_time, scatter_freq, color="red", s=30, label="max window diff" if freq_idx==0 else "")
        print(f"freq={scatter_freq:.1f}kHz, time={scatter_time:.2f}s, diff={freq_diff[max_idx]:.2e}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_yscale("log")
    ax.set_ylim(100, 900)
    ax.set_xlim(3600*10.25, 3600*10.46)
    fig.colorbar(pcm, label=f"Future-Past Mean Diff ({window_sec}s)")
    plt.savefig(save_path+f"intensity_windowdiff_{window_sec}s.png")
    plt.show()
    return 0


def interpolate_nan_freq(Intensity, Frequency):
    """
    周波数方向でNaNを線形補間する
    Intensity: shape = (周波数, 時間)
    Frequency: shape = (周波数,)
    """
    Intensity_interp = Intensity.copy()
    freq_num, time_num = Intensity.shape
    for t in range(time_num):
        y = Intensity[:, t]
        nan_mask = np.isnan(y)
        if np.all(nan_mask):
            continue  # 全部NaNなら補間しない
        # 補間関数を作成（NaN以外の値のみ使用）
        f_interp = interp1d(Frequency[~nan_mask], y[~nan_mask], kind='linear', bounds_error=False, fill_value="extrapolate")
        # NaN部分を補間値で埋める
        Intensity_interp[nan_mask, t] = f_interp(Frequency[nan_mask])
    return Intensity_interp

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

    Intensity = mask_data(Frequency, Intensity, mask_frequency)
    #Intensity = moving_average_intensity(Epoch, Intensity, window_sec=10)

        # 周波数方向でNaNを補間
    #Intensity_interp = interpolate_nan_freq(Intensity, Frequency)

    pcm = ax.pcolormesh(
        xx,
        yy,
        Intensity,
        cmap="bwr",
        vmin=-1, vmax=1
        )

    contour_levels = [-0.3, 0.3]
    cs = ax.contour(
        xx, yy, Intensity,
        levels=contour_levels,
        colors=["black", "black"],
        linewidths=1.5
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
    ax.set_ylim(100, 900)  # 3kHz~2MHz
    ax.set_xlim(3600*9.25, 3600*9.5)
    #ax.set_xlim(3600*9, 3600*10.46)
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

print(Epoch_from_0min_1d)
Radio_polarization = load_polarizaiton_data(path, start_year, start_date, start_hour, duration)


plot_ft_nomal(
    Hour,
    start_hour,
    duration,
    Epoch_from_0min_1d,
    Frequency_1d,
    Radio_intensity,
    "Intensity")
"""
plot_ft_diff(
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
"""
plot_ft_window_diff(
    Hour,
    start_hour,
    duration,
    Epoch_from_0min_1d,
    Frequency_1d,
    Radio_intensity,
    "Intensity",
    window_sec= 60
)

# %%
