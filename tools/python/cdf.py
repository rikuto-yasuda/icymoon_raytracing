# %%
import pprint 
import cdflib
import numpy as np

# %% 
cdf_file=cdflib.CDF('~/Downloads/expres_gll_jupiter_0d-30r_jrm09_lossc-wid1deg_5kev_19960627_v11.cdf')

x= cdf_file.varget("SrcPosition", startrec =0, endrec = 180) #epoch frequency longtitude source position
# %%
'''
print(x)

'''
time = cdf_file.varget("Epoch") #time (need to check galireo spacecraft position as time)
TIME2 = cdflib.cdfepoch.breakdown(time[:])

fre = cdf_file.varget("Frequency") #frequency (important for altitude)
'''
print(fre)

long = cdf_file.varget("Src_ID_Label")  # longtitude from which magnetic field line (north 360 and south 360)
print(long)


y = cdf_file.varget("Src_Pos_Coord")  # galireo spacecraft can catch the radio or not (if can, where the radio is emitted)
print(y)
'''

idx= np.where(x>-1.0e+31)
"""
print(idx)
"""
timeindex = idx[0]
times=time[timeindex]

freindex = idx[1]
fres=fre[freindex]

"""
print(times)
print(times.shape)

print(fres)
print(fres.shape)
"""

n = int(times.shape[0]/3)

position= x[idx].reshape(n,3)
timess= times.reshape(n,3)
TIME= timess[:,0]
TIME3 =  cdflib.cdfepoch.breakdown(TIME)
TIME4 =  np.array(TIME3)
fress= fres.reshape(n,3)
FRE= fress[:,0]

print('TIME')
print(TIME4)
print('FRE')
print(FRE)
print (position)

print(FRE.shape)
print(TIME4.shape)
print(position.shape)


"""
pprint.pprint(cdf_file.cdf_info())

print(cdf_file.varget("Freq_Label"))
"""
A= np.where(TIME4==10)
B= np.where(TIME4==25)

"""
print(A)
print(B)
"""

TIME5 = TIME4[8067:11369,:]
FRE5 = FRE[8067:11369]
position5 = position[8067:11369,:]

print(FRE5.shape)
print(TIME5.shape)
print(position5.shape)

DATA = np.hstack((TIME5, np.reshape(FRE5, [FRE5.shape[0],1]),position5))
print(DATA.shape)
np.savetxt('/Users/yasudarikuto/research/raytracing/tools/results/Radio_data.txt', DATA)

