import numpy as np
import math
import os
import spacepy.pycdf
import netCDF4

# import CDF_LIB
os.environ["CDF_LIB"] = "/Applications/cdf/cdf39_0-dist/lib"


def read_ionosphere_data(ionospheric_data_path):

    # データの読み込み and 強度や周波数、時間配列をインプット
    f = netCDF4.Dataset(ionospheric_data_path)
    print(f.variables.keys())
    # EuEu = cdf["EuEu"][...]

    return 0

    # data_wind.quicklook(keys=["STOKES_I"], cmap="viridis")
