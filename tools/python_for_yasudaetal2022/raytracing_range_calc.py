# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
import os
import time
import glob

# %%
# あらかじめ ../result_sgepss2021/~/~ に必要なレイトレーシング結果とパラメータセットを入れること
# highest_plasma = '10.5e2'  # 単位は(/cc) 2e2/4e2/16e22

object_name = 'ganymede'  # ganydeme/europa/calisto``
spacecraft_name = "galileo"  # galileo/JUICE(?)
time_of_flybies = 1  # ..th flyby

# europa & ganymede
"""
Freq_str = ['3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]

Freq_underline = 0.36122

####

# callisto

Freq_str = ['3.612176179885864258e5', '3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
            '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
            '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
            '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
            '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
            '5.117111206054687500e6', '5.644999980926513672e6', ]

Freq_underline = 0.32744
"""


# %%
def Get_usefile(object, spacecraft, flyby):
    use_files = sorted(glob.glob('../result_for_yasudaetal2022/tracing_range_'+spacecraft_name+'_'+object_name+'_'+str(time_of_flybies) +
                                 '_flybys/para_*.csv'))
    return use_files


# 作ったけど使ってない（他のコードで使えるかも）
def Get_frequency(object):
    if object == 'ganymede' or object == 'europa':
        Freq_str = ['3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
                    '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
                    '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
                    '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
                    '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
                    '5.117111206054687500e6', '5.644999980926513672e6', ]

        Freq_underline = 0.36122

    if object == 'callisto':
        Freq_str = ['3.612176179885864258e5', '3.984813988208770752e5', '4.395893216133117676e5', '4.849380254745483398e5', '5.349649786949157715e5', '5.901528000831604004e5', '6.510338783264160156e5',
                    '7.181954979896545410e5', '7.922856807708740234e5', '8.740190267562866211e5', '9.641842246055603027e5', '1.063650846481323242e6',
                    '1.173378825187683105e6', '1.294426321983337402e6', '1.427961349487304688e6', '1.575271964073181152e6', '1.737779378890991211e6',
                    '1.917051434516906738e6', '2.114817380905151367e6', '2.332985162734985352e6', '2.573659420013427734e6', '2.839162111282348633e6',
                    '3.132054328918457031e6', '3.455161809921264648e6', '3.811601638793945312e6', '4.204812526702880859e6', '4.638587474822998047e6',
                    '5.117111206054687500e6', '5.644999980926513672e6', ]

        Freq_underline = 0.32744

    else:
        print('object name is not correct')

    return Freq_str, Freq_underline


def Highest_and_lowest_souce(range_file):

    lowest_altitude_in_all_ionosphere_case = 0
    highest_altitude_in_all_ionosphere_case = 0
    for file in range_file:
        range_list = pd.read_csv(file)
        lowest_altitude_list = range_list['lowest']
        highest_altitude_list = range_list['highest']
        lowest_altitude = np.min(lowest_altitude_list)
        highest_altitude = np.max(highest_altitude_list)

        if lowest_altitude < lowest_altitude_in_all_ionosphere_case:
            lowest_altitude_in_all_ionosphere_case = lowest_altitude

        if highest_altitude > highest_altitude_in_all_ionosphere_case:
            highest_altitude_in_all_ionosphere_case = highest_altitude

    print(highest_altitude_in_all_ionosphere_case,
          lowest_altitude_in_all_ionosphere_case)

    return 0


def main():
    range_list = Get_usefile(object_name, spacecraft_name, time_of_flybies)
    frequency_list = Get_frequency(object_name)
    Highest_and_lowest_souce(range_list)
    return 0


if __name__ == "__main__":
    main()

# %%
