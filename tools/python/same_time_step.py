import numpy as np
import matplotlib.pyplot as plt




for j in range (3,9):
    Ord = j
    ORDER = str(j)
    for k in range (1,10):
        COE = str(k)
        Coe = k
        for l in range (300,331,5):
            Deg = l
            DEG = str(l)

            filename ="/Users/yasudarikuto/research/raytracing/tools/results/time_step_error_data2/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z1-FR1e6-TIME-RANGE"+COE+"e-"+ORDER+":1e-13-PITCH"+DEG 
            data = np.genfromtxt(filename)
            l_2d = len(data) 
            change = 0

            for i in range(l_2d - 1):

                if data[i][7] != data[i+1][7] :
                    change += 1
                    
            print (COE,'e-',ORDER,'(sec)',DEG,'(degree)')
            print(change)
            