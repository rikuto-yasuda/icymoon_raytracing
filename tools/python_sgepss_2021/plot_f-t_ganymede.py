 # %%
import pprint 
import cdflib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% 
# あらかじめ ../result_sgepss_2021/~/~ に必要なレイトレーシング結果とパラメータセットを入れること

object_name='ganymede' # ganydeme/
highest_plasma='1e2' #単位は(/cc) 2e2/4e2/16e2
plasma_scaleheight='6e2' #単位は(km) 1.5e2/3e2/6e2

# %%
data_name = '../result_sgepss_2021/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'/para_'+highest_plasma+'_'+plasma_scaleheight+'.csv'

data = np.loadtxt('/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/R_P_data2.txt',)
list = pd.read_csv(data_name, header=0)

n = len(data)
Freq_str = ['3.984813988208770752e5','4.395893216133117676e5','5.349649786949157715e5','5.901528000831604004e5','6.510338783264160156e5',\
    '7.181954979896545410e5','7.922856807708740234e5','8.740190267562866211e5','9.641842246055603027e5','1.063650846481323242e6',\
    '1.173378825187683105e6','1.294426321983337402e6','1.427961349487304688e6','1.575271964073181152e6','1.737779378890991211e6',\
    '1.917051434516906738e6','2.114817380905151367e6','2.332985162734985352e6','2.573659420013427734e6','2.839162111282348633e6',\
    '3.132054328918457031e6','3.455161809921264648e6','3.811601638793945312e6','4.204812526702880859e6','4.638587474822998047e6',\
    '5.117111206054687500e6','5.644999980926513672e6',]

Freq_num = []
for i in Freq_str:
    Freq_num.append(float(i)/1000000)


Highest = list.highest
Lowest = list.lowest
Except = list.exc
res = data.copy()

# %%

for i in range (n):
    for l in range (len(Freq_num)):
        if data[i][1] == Freq_num[l]:
            for j in range(Lowest[l],Highest[l],2):
                k = str(j)
                data2 = np.genfromtxt("../result_sgepss_2021/"+object_name+"_"+highest_plasma+"_"+plasma_scaleheight+"/ray-P"+object_name+"_nonplume_"+highest_plasma+"_"+plasma_scaleheight+"-Mtest_simple-benchmark-LO-Z"+k+"-FR"+Freq_str[l])
                n2 = len(data2)
                aa=0

                if Except[l]==[-10000]:
                    for h in range(n2):
                        if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                            para = np.abs(data2[h][1]-data[i][2])
                            hight = data2[h][3]-data[i][3] 
                            x1 = data2[h][1]
                            z1 = data2[h][3]
                            x2 = data2[h-1][1]
                            z2 = data2[h-1][3]
                            
                            while (para>10):
                                ddx = (x1+x2)/2
                                ddz = (z1+z2)/2

                                if ddx > data[i][2]:
                                    x1 = ddx
                                    z1 = ddz
                                else:
                                    x2 = ddx
                                    z2 = ddz

                                para = np.abs(x1-data[i][2])
                                hight = z1-data[i][3]
                            
                            if hight < 0:
                                res[i][3]=0
                                aa=1
                                break
                    
                    if aa==1:
                        break 
                
                else :
                    if str(j) not in str(Except[l]):
                        for h in range(n2):
                            if data2[h][1] > data[i][2] and data2[h-1][3] < data[i][3]:
                                para = np.abs(data2[h][1]-data[i][2])
                                hight = data2[h][3]-data[i][3] 
                                x1 = data2[h][1]
                                z1 = data2[h][3]
                                x2 = data2[h-1][1]
                                z2 = data2[h-1][3]
                                
                                while (para>10):
                                    ddx = (x1+x2)/2
                                    ddz = (z1+z2)/2

                                    if ddx > data[i][2]:
                                        x1 = ddx
                                        z1 = ddz
                                    else:
                                        x2 = ddx
                                        z2 = ddz

                                    para = np.abs(x1-data[i][2])
                                    hight = z1-data[i][3]
                                
                                if hight < 0:
                                    res[i][3]=0
                                    aa=1
                                    break
                        
                    if aa==1:
                        break 


A = np.where(res[:,3]==0)
res2 = data[A][:]
np.savetxt('../result_sgepss_2021/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'_occultaion_data.txt', res2)
"""
plt.yscale('log')
plt.scatter(res2[:,0], res2[:,1])
plt.plot()


# %%
data3 = np.loadtxt('../result_sgepss_2021/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'/'+object_name+'_'+highest_plasma+'_'+plasma_scaleheight+'_occultaion_data.txt')
plt.xlabel("1996/7/26 6 o'clock (min))")
plt.ylabel("Frequency (MHz)")
plt.yscale('log')
highest=str(int(float(highest_plasma)))
scale=str(int(float(plasma_scaleheight)))
print(scale)
plt.title("max density "+highest+"(/cc) scale height "+scale+"(km)")
plt.xlim(14.5,25.5)
plt.ylim(0.3,7.0)
plt.scatter(data3[:,0], data3[:,1])
plt.plot()

"""

# %%
