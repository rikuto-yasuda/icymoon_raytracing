 # %%
import pprint 
import cdflib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime


# %%
data_name = '../result_sgepss_2021/ganymede_test_simple/para_test_simple.csv'

data = np.loadtxt('/Users/yasudarikuto/research/raytracing/tools/result_sgepss_2021/R_P_fulldata.txt',)
Radio_Range = pd.read_csv(data_name, header=0)

n = len(data)
Freq_str = ['3.984813988208770752e5','4.395893216133117676e5','4.849380254745483398e5','5.349649786949157715e5','5.901528000831604004e5','6.510338783264160156e5',\
    '7.181954979896545410e5','7.922856807708740234e5','8.740190267562866211e5','9.641842246055603027e5','1.063650846481323242e6',\
    '1.173378825187683105e6','1.294426321983337402e6','1.427961349487304688e6','1.575271964073181152e6','1.737779378890991211e6',\
    '1.917051434516906738e6','2.114817380905151367e6','2.332985162734985352e6','2.573659420013427734e6','2.839162111282348633e6',\
    '3.132054328918457031e6','3.455161809921264648e6','3.811601638793945312e6','4.204812526702880859e6','4.638587474822998047e6',\
    '5.117111206054687500e6','5.644999980926513672e6',]

Freq_num = []
for i in Freq_str:
    Freq_num.append(float(i)/1000000)


Highest = Radio_Range.highest
Lowest = Radio_Range.lowest
Except = Radio_Range.exc
res = data.copy()

# %%

for i in range (n):
    if data[i][5]<0:
        res[i][6] = 0
        aa = 1
        continue

    for l in range (len(Freq_num)):
        if data[i][2] == Freq_num[l]:
            for j in range(Lowest[l],Highest[l],2):
                k = str(j)
                data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR"+Freq_str[l])
                n2 = len(data2)
                aa=0

                if Except[l]==[-10000]:
                    for h in range(n2):

                        if data2[h][1] > data[i][5] and data2[h-1][3]<data[i][6]:
                            para = np.abs(data2[h][1]-data[i][5])
                            hight = data2[h][3]-data[i][6] 
                            x1 = data2[h][1]
                            z1 = data2[h][3]
                            x2 = data2[h-1][1]
                            z2 = data2[h-1][3]
                            
                            while (para>10):
                                ddx = (x1+x2)/2
                                ddz = (z1+z2)/2

                                if ddx > data[i][5]:
                                    x1 = ddx
                                    z1 = ddz
                                else:
                                    x2 = ddx
                                    z2 = ddz

                                para = np.abs(x1-data[i][5])
                                hight = z1-data[i][6]
                            
                            if hight < 0:
                                res[i][6]=0
                                aa=1
                                break
                    
                    if aa==1:
                        break 
                
                else :
                    if str(j) not in str(Except[l]):
                        for h in range(n2):
                            if data2[h][1] > data[i][5] and data2[h-1][3] < data[i][6]:
                                para = np.abs(data2[h][1]-data[i][5])
                                hight = data2[h][3]-data[i][6] 
                                x1 = data2[h][1]
                                z1 = data2[h][3]
                                x2 = data2[h-1][1]
                                z2 = data2[h-1][3]
                                
                                while (para>10):
                                    ddx = (x1+x2)/2
                                    ddz = (z1+z2)/2

                                    if ddx > data[i][5]:
                                        x1 = ddx
                                        z1 = ddz
                                    else:
                                        x2 = ddx
                                        z2 = ddz

                                    para = np.abs(x1-data[i][5])
                                    hight = z1-data[i][6]
                                
                                if hight < 0:
                                    res[i][6]=0 #ukaru
                                    aa=1
                                    break
                        
                    if aa==1:
                        break 


A = np.where(res[:,6]==0)
res2 = data[A][:]
np.savetxt('../result_sgepss_2021/ganymede_test_simple/ganymede_noreflection_occultaion_data_longver1.txt', res)
np.savetxt('../result_sgepss_2021/ganymede_test_simple/ganymede_noreflection_occultaion_data_longver2.txt', res2)
"""
plt.yscale('log').
plt.scatter(res2[:,1], res2[:,2])
plt.plot()
"""
print("complete")

# %%
data4 = np.loadtxt('../result_sgepss_2021/ganymede_test_simple/ganymede_noreflection_occultaion_data_longver1.txt') #電波源の経度を含
detail_data4=data4.copy()
galdata=np.loadtxt('/Users/yasudarikuto/research/raytracing/tools/result_sgepss_2021/GLL_GAN_2.txt')
date = np.arange('1996-06-27 05:30:00', '1996-06-27 08:31:00',np.timedelta64(1,'m'), dtype='datetime64')
DataA= np.zeros(len(date)*(len(Freq_num)+1)).reshape(len(Freq_num)+1,len(date))
DataB= np.zeros(len(date)*(len(Freq_num)+1)).reshape(len(Freq_num)+1,len(date))
DataC= np.zeros(len(date)*(len(Freq_num)+1)).reshape(len(Freq_num)+1,len(date))
DataD= np.zeros(len(date)*(len(Freq_num)+1)).reshape(len(Freq_num)+1,len(date))
"""
Hdate=date[0].tolist().time().hour
Mdate=date[0].tolist().time().minute
print(Hdate)
"""
for k in range (len(data4)):
    Num = int(data4[k][0]*60+data4[k][1]-330)
    if np.abs(galdata[Num][2]+360-data[k][7])<np.abs(galdata[Num][2]-data[k][7]):
        Lon = galdata[Num][2]+360- data4[k][7]

    elif np.abs(data4[k][7]+360-galdata[Num][2])<np.abs(data4[k][7]-galdata[Num][2]):
        Lon = galdata[Num][2]-360 - data4[k][7]

    else:
        Lon = galdata[Num][2] - data4[k][7]

    Lat = data4[k][4]

    if data4[k][6]==0:
        Fre=np.where(Freq_num==data4[k][2])
        if Lon < 0 and Lat > 0:
            DataA[int(Fre[0])+1][Num]=1
            detail_data4[k][5]=0 
        
        if Lon > 0 and Lat > 0:
            DataB[int(Fre[0])+1][Num]=1

        if Lon < 0 and Lat < 0:
            DataC[int(Fre[0])+1][Num]=1

        if Lon > 0 and Lat < 0:
            DataD[int(Fre[0])+1][Num]=1

print("complete")
np.savetxt('../result_sgepss_2021/ganymede_test_simple/ft-radioA.txt', DataA) 
B = np.where(detail_data4[:,5]==0)
res3 = detail_data4[B][:]
np.savetxt('../result_sgepss_2021/ganymede_test_simple/ft-radio-detail.txt', res3) 
# %%

plt.xlabel("1996/7/26 6 o'clock (min))")
plt.ylabel("Frequency (MHz)")
plt.yscale('log')

plt.xlim(date[0],date[90])
plt.ylim(0.1,5.0)
plt.title("ganymede no reflection")
FREQ=np.insert(np.array(Freq_num), 0, 0.36122)
plt.contour(date, FREQ, DataA,levels=[0.5],cmap='gray',alpha =0.1)
plt.contour(date, FREQ, DataB,levels=[0.5],cmap='gray',alpha =0.4)
plt.contour(date, FREQ, DataC,levels=[0.5],cmap='gray',alpha =0.7)
plt.contour(date, FREQ, DataD,levels=[0.5],cmap='gray',alpha =1.0)

# %%
data3 = np.loadtxt('../result_sgepss_2021/ganymede_test_simple/ganymede_noreflection_occultaion_data_longver2.txt')
plt.xlabel("1996/7/26 6 o'clock (min))")
plt.ylabel("Frequency (MHz)")
plt.yscale('log')

plt.title("ganymede no reflection")
plt.ylim(0.3,10.0)
plt.scatter(data3[:,1], data3[:,2])
plt.plot()

print("complete")

# %%

yTags=[5.620e+00,1.000e+01,1.780e+01,3.110e+01,4.213e+01,4.538e+01,4.888e+01,5.265e+01,5.671e+01,6.109e+01,6.580e+01,7.087e+01,7.634e+01,8.223e+01,8.857e+01,\
9.541e+01,1.028e+02,1.107e+02,1.192e+02,1.284e+02,1.383e+02,1.490e+02,1.605e+02,1.729e+02,1.862e+02,2.006e+02,2.160e+02,2.327e+02,2.507e+02,2.700e+02,2.908e+02,\
3.133e+02,3.374e+02,3.634e+02,3.915e+02,4.217e+02,4.542e+02,4.892e+02,5.270e+02,5.676e+02,6.114e+02,6.586e+02,7.094e+02,7.641e+02,8.230e+02,8.865e+02,9.549e+02,\
1.029e+03,1.108e+03,1.193e+03,1.285e+03,1.385e+03,1.491e+03,1.606e+03,1.730e+03,1.864e+03,2.008e+03,2.162e+03,2.329e+03,2.509e+03,2.702e+03,2.911e+03,3.135e+03,\
3.377e+03,3.638e+03,3.918e+03,4.221e+03,4.546e+03,4.897e+03,5.275e+03,5.681e+03,6.120e+03,6.592e+03,7.100e+03,7.648e+03,8.238e+03,8.873e+03,9.558e+03,1.029e+04,\
1.109e+04,1.194e+04,1.287e+04,1.386e+04,1.493e+04,1.608e+04,1.732e+04,1.865e+04,2.009e+04,2.164e+04,2.331e+04,2.511e+04,2.705e+04,2.913e+04,3.138e+04,3.380e+04,\
3.641e+04,3.922e+04,4.224e+04,4.550e+04,4.901e+04,5.279e+04,5.686e+04,6.125e+04,6.598e+04,7.106e+04,7.655e+04,8.245e+04,8.881e+04,9.566e+04,1.030e+05,1.030e+05,\
1.137e+05,1.254e+05,1.383e+05,1.526e+05,1.683e+05,1.857e+05,2.049e+05,2.260e+05,2.493e+05,2.750e+05,3.034e+05,3.347e+05,3.692e+05,4.073e+05,4.493e+05,4.957e+05,\
5.468e+05,6.033e+05,6.655e+05,7.341e+05,8.099e+05,8.934e+05,9.856e+05,1.087e+06,1.199e+06,1.323e+06,1.460e+06,1.610e+06,\
1.776e+06,1.960e+06,2.162e+06,2.385e+06,2.631e+06,2.902e+06,3.201e+06,3.532e+06,3.896e+06,4.298e+06,4.741e+06,5.231e+06,5.770e+06]

Ytag=np.array(yTags)

data = np.loadtxt('../result_sgepss_2021/Survey_Electric_1996-06-27T05-30_1996-06-27T07-00.d2s')

# %%
