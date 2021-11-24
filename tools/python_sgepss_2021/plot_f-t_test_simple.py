# %%
import pprint 
import cdflib
import numpy as np
import matplotlib.pyplot as plt

# %% 
data = np.loadtxt('/Users/yasudarikuto/research/raytracing/tools/result_sgepss_2021/R_P_data2.txt',)
n = len(data)
list1 = ['3.984813988208770752e5','4.395893216133117676e5','5.349649786949157715e5','5.901528000831604004e5','6.510338783264160156e5',\
    '7.181954979896545410e5','7.922856807708740234e5','8.740190267562866211e5','9.641842246055603027e5','1.063650846481323242e6',\
    '1.173378825187683105e6','1.294426321983337402e6','1.427961349487304688e6','1.575271964073181152e6','1.737779378890991211e6',\
    '1.917051434516906738e6','2.114817380905151367e6','2.332985162734985352e6','2.573659420013427734e6','2.839162111282348633e6',\
    '3.132054328918457031e6','3.455161809921264648e6','3.811601638793945312e6','4.204812526702880859e6','4.638587474822998047e6',\
    '5.117111206054687500e6','5.644999980926513672e6',]

res = data


# %%
aaa=0
for i in range (n):
    if data[i][1] == 3.984813988208770752e-01:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR3.984813988208770752e5")
            n2 = len(data2)
            aa=0

            for h in range(n2):

                if data2[h][1] > data[i][2] and data2[h-1][3] < data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>100000):
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
    
    if data[i][1] == 4.395893216133117676e-01:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR4.395893216133117676e5")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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

    if data[i][1] == 5.349649786949157715e-01:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR5.349649786949157715e5")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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

    if data[i][1] == 5.901528000831604004e-01:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR5.901528000831604004e5")
            n2 = len(data2)
            aa=0
            for h in range(n2):

                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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

    if data[i][1] == 6.510338783264160156e-01:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR6.510338783264160156e5")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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


    if data[i][1] == 7.181954979896545410e-01:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR7.181954979896545410e5")
            n2 = len(data2)
            aa=0

            for h in range(n2):

                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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

    if data[i][1] == 7.922856807708740234e-01:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR7.922856807708740234e5")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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

    if data[i][1] == 8.740190267562866211e-01:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR8.740190267562866211e5")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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


    if data[i][1] == 9.641842246055603027e-01:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR9.641842246055603027e5")
            n2 = len(data2)
            aa=0

            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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

    if data[i][1] == 1.063650846481323242e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR1.063650846481323242e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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


    if data[i][1] == 1.173378825187683105e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR1.173378825187683105e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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
            


    if data[i][1] == 1.294426321983337402e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR1.294426321983337402e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):

                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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


    if data[i][1] == 1.427961349487304688e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR1.427961349487304688e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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
            

    if data[i][1] == 1.575271964073181152e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR1.575271964073181152e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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

    if data[i][1] == 1.737779378890991211e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR1.737779378890991211e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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


    if data[i][1] == 1.917051434516906738e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR1.917051434516906738e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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

    if data[i][1] == 2.114817380905151367e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR2.114817380905151367e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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
            


    if data[i][1] == 2.332985162734985352e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR2.332985162734985352e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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
            


    if data[i][1] == 2.573659420013427734e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR2.573659420013427734e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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
            


    if data[i][1] == 2.839162111282348633e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR2.839162111282348633e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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


    if data[i][1] == 3.132054328918457031e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR3.132054328918457031e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                

                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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
            

    if data[i][1] == 3.455161809921264648e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR3.455161809921264648e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                

                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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
            


    if data[i][1] == 3.811601638793945312e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR3.811601638793945312e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                

                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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

    if data[i][1] == 4.204812526702880859e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR4.204812526702880859e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                

                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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
            


    if data[i][1] == 5.117111206054687500e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR5.117111206054687500e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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
            



    if data[i][1] == 5.644999980926513672e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR5.644999980926513672e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                

                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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
            


    if data[i][1] == 4.638587474822998047e+00:
        for j in range(1,2,1):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_test_simple/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR4.638587474822998047e6")
            n2 = len(data2)
            aa=0
            for h in range(n2):
                
                if data2[h][1] > data[i][2] and data2[h-1][3]<data[i][3]:
                    para = np.abs(data2[h][1]-data[i][2])
                    hight = data2[h][3]-data[i][3] 
                    x1 = data2[h][1]
                    z1 = data2[h][3]
                    x2 = data2[h-1][1]
                    z2 = data2[h-1][3]
                    
                    while (para>1000):
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
res2 = res[A][:]
plt.xlabel("1996/7/26 6 o'clock (min))")
plt.ylabel("Frequency (MHz)")
plt.yscale('log')
plt.title("no reflection")
plt.xlim(14.5,25.5)
plt.ylim(0.3,7.0)
plt.scatter(res2[:,0], res2[:,1])
plt.plot()




# %%
"""
for i in range (n):
    if data[i][1] == 3.984813988208770752e-01:

        for j in range(-500,1201,100):
            k = str(j)
            data2 = np.genfromtxt("../result_sgepss_2021/ganymede_FR3.984813988208770752e5/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z"+k+"-FR3.984813988208770752e5")
            index = data2[:,0]

plt.yscale('log')
plt.scatter(data[:,0], data[:,1])
plt.plot()
"""

# %%
