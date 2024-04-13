# %%
import netCDF4
import sys

# %%
# import calibration lib
sys.path.append(".")
import read_model_lib as mod_lib

# write juice SID data path
juice_data_name = "Elew_22_03_23_t00600.nc"
moon = "Europa"
juice_data_path = "../" + moon + "/" + juice_data_name


mod_lib.read_ionosphere_data(juice_data_path)

# %%
