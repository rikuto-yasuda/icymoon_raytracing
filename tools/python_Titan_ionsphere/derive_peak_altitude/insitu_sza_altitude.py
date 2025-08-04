# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

sza_values = [50, 70, 90, 110, 130]

csx_data_path = "/work1/rikutoyasuda/tools/python_Titan_ionsphere/derive_peak_altitude/wpd_datasets.csv"
data = pd.read_csv(csx_data_path, header=1)

data_amount = int(data.shape[1]/2)

sza_array = np.zeros(data_amount)
altitude_array = np.zeros(data_amount)
upper_altitude_array = np.zeros(data_amount)
lower_altitude_array = np.zeros(data_amount)

print(data)
# %%

for i in range(data_amount):
    sza_array[i] = (data.iloc[0, i * 2] + data.iloc[1, i * 2] + data.iloc[2, i * 2]) / 3
    altitude_array[i] = data.iloc[0, i * 2 + 1]
    upper_altitude_array[i] = data.iloc[1, i * 2 + 1]
    lower_altitude_array[i] = data.iloc[2, i * 2 + 1]



# %%
# グラフの描写　散布グラフ　横軸はSZA 0~180, 縦軸は高度 900~1400
# altitudeは上下にエラーバーをつける　エラーバーの範囲はupper_altitude_arrayとlower_altitude_arrayの間

plt.figure(figsize=(10, 6))
plt.errorbar(sza_array, altitude_array, yerr=[altitude_array - lower_altitude_array, upper_altitude_array - altitude_array], fmt='o', ecolor='lightgray',
             elinewidth=2, capsize=4, label='Altitude with Error Bars',
             color='blue', alpha=0.7)
plt.xlim(0, 180)
plt.ylim(900, 1400)
plt.xlabel('Solar Zenith Angle (SZA) [degrees]')
plt.ylabel('Altitude [m]')
plt.title('Solar Zenith Angle vs Altitude')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('sza_vs_altitude.png', dpi=300, bbox_inches='tight')


# %%
# WLS（加重最小二乗法）を行って回帰直線を求める
# 本来であれば、エラーσが観測値の上下に同じく存在するときにつかうものであるが、
# ここでは上下のエラーを平均してσとする
sigma = (upper_altitude_array - lower_altitude_array) / 2 # シグマ
w = 1 / (sigma ** 2)

# 説明変数に定数項（切片）を追加
X = sm.add_constant(sza_array)

# WLSモデルの構築とフィッティング
model = sm.WLS(altitude_array, X, weights=w)
results = model.fit()

# 結果表示
print(results.params)       # 回帰係数（切片, 傾き）
print(results.conf_int())   # 95%信頼区間

# %%
plt.figure(figsize=(10, 6))
plt.errorbar(sza_array, altitude_array, yerr=[altitude_array - lower_altitude_array, upper_altitude_array - altitude_array], fmt='o', ecolor='lightgray',
             elinewidth=2, capsize=4, label='Altitude with Error Bars',
             color='blue', alpha=0.7)

x_array = np.linspace(0, 180, 100)
y_array = results.params[0] + results.params[1] * x_array  # 回帰直線の計算
plt.plot(x_array, y_array, color='red', label='Regression Line', linewidth=2)
plt.xlim(0, 180)
plt.ylim(900, 1400)
plt.xlabel('Solar Zenith Angle (SZA) [degrees]')
plt.ylabel('Altitude [m]')
plt.title('Solar Zenith Angle vs Altitude')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('sza_vs_altitude_regression.png', dpi=300, bbox_inches='tight')

# %%
# 複数のSZA値で予測

new_X = np.column_stack([np.ones(len(sza_values)), sza_values])  # 定数項とSZA値

print(f"new_X shape: {new_X.shape}")  # (5, 2) になるはず

# get prediction + interval
pred = results.get_prediction(new_X)
summary = pred.summary_frame(alpha=0.05)  # 95%区間

print(summary[['mean', 'obs_ci_lower', 'obs_ci_upper']])
# %%
