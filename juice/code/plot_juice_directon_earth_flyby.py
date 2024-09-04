#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import juice_lib
from astropy.constants import au
import sys
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from matplotlib import patches
from datetime import datetime

# In[2]:


# SPICE test
spice.tkvrsn("TOOLKIT")


# Import JUICE Lib
# In[3]:
# import JUICE lib
sys.path.append("/Users/yasudarikuto/research/icymoon_raytracing/juice/lib")
# Load SPICE kernels
import juice_lib

# In[4]:


# load SPICE ketnels
spice_dir = "/Users/yasudarikuto/spice/JUICE/kernels/"
juice_lib.spice_ini(spice_dir)


save_folder_path = "/Users/yasudarikuto/research/icymoon_raytracing/juice/results/"
# JUICE orbit near Moon

# In[5]:
# set date/time
utctim = "2024-08-19T20:00:00"  # start date/time
et_ex = spice.str2et(utctim)  # seconds
nd = 60 * 60 * 2  # number of data
dt = 1  # time step [second]
et = et_ex + dt * np.arange(0, nd)

x, y, z, r, lat, lon = juice_lib.get_direction_from_juice(et,"EARTH")


# %%
