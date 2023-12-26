# %%
import pprint
import cdflib
import numpy as np
import re

# %%
cdf_file = cdflib.CDF(
    "/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/expres_gll_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_19960627_v11.cdf"
)

x = cdf_file.varget(
    "SrcPosition", startrec=0, endrec=180
)  # epoch frequency longtitude source position

time = cdf_file.varget(
    "Epoch"
)  # time (need to check galireo spacecraft position as time)
TIME2 = cdflib.cdfepoch.breakdown(time[:])

fre = cdf_file.varget("Frequency")  # frequency (important for altitude)
"""
print(fre)
"""
long = cdf_file.varget(
    "Src_ID_Label"
)  # longtitude from which magnetic field line (north 360 and south 360)
"""
print(long)
"""
y = cdf_file.varget(
    "Src_Pos_Coord"
)  # galireo spacecraft can catch the radio or not (if can, where the radio is emitted)
"""
print(y)
"""

idx = np.where(x > -1.0e31)
"""
print(idx)
"""

times = time[idx[0]]

fres = fre[idx[1]]

longs = np.array(long[idx[2]], dtype=object)
"""
print(times)
print(times.shape)

print(fres)
print(fres.shape)

print(longs)
print(longs.shape)
"""

# %%
n = int(times.shape[0] / 3)

position = x[idx].reshape(n, 3)

TIME = np.array(cdflib.cdfepoch.breakdown(times.reshape(n, 3)[:, 0]))

FRE = fres.reshape(n, 3)[:, 0]
FRES = np.reshape(FRE, [FRE.shape[0], 1])

LONG = longs.reshape(n, 3)[:, 0]

LONG2 = np.reshape(LONG, [LONG.shape[0], 1])
"""
print('TIME')
print(TIME)
print('FRE')
print(FRE)
print (position)

print(FRE.shape)
print(TIME4.shape)
print(position.shape)

print(LONG)
print(LONG.shape)
"""

# %%


LON = np.zeros(len(LONG2))

for i in range(len(LONG2)):
    LON[i] = int(re.search(r"\d+", str(LONG2[i].copy())).group())

LONGS = np.reshape(LON, [LON.shape[0], 1])

POL = np.zeros(len(LONG2))

for i in range(len(LONG2)):
    POL[i] = str(LONG2[i].copy()).find("NORTH")

POLSS = np.where(POL < 0, POL, 1)
POLS = np.reshape(POLSS, [POLSS.shape[0], 1])

# %%

"""
pprint.pprint(cdf_file.cdf_info())

print(cdf_file.varget("Freq_Label"))

A= np.where(TIME4==10)
B= np.where(TIME4==25)

print(A)
print(B)

TIME5 = TIME4[8067:11369,:]
FRE5 = FRE[8067:11369]
position5 = position[8067:11369,:]

print(FRE5.shape)
print(TIME5.shape)
print(position5.shape)
"""
# DATA = np.hstack((TIME4, np.reshape(FRE, [FRE.shape[0],1]),LONG,position))
DATA = np.hstack((TIME, FRES, LONGS, POLS, position))

print(DATA.shape)
# np.savetxt('/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/All_Gnymede_Radio_data.txt', DATA, fmt="%s")

# %%
