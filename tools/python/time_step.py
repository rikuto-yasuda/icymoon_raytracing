import numpy as np
import matplotlib.pyplot as plt

step = np.array([0.02,-100])
err = 0


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

            sx = 0.0
            sz = 1.0
            stheta = 360.0 - l #the angle of incidence
            freq =1000000.0
            a = 50000.0 # coefficient of simple plasma model

            hight = 0.0
            mz = 0.0
            mx = 0.0 

            for i in range(l_2d):

                if data[i][3] >= hight :
                    hight = data[i][3]
                    mx = data[i][1]
                    if i == l_2d-1:
                        print("error(too short ray)")
                        error = 1
                        break
                else:
                    print("non error")
                    error = 0
                    break

            seemhight = (mx-sx)/np.tan(np.radians(stheta))

            wt = freq*freq*np.cos(np.radians(stheta))*np.cos(np.radians(stheta))*(4*np.pi*np.pi)
            k = 3182.607*a*1000.0
            h = (np.sqrt(1-(k*sz/wt)))*2*wt/k 
            seemhight2 = h + sz

            distance = seemhight - seemhight2

            atan = np.degrees(np.arctan(mx/h))
            eangle = stheta - atan
            """
            print (mx,'(km)')
            print (hight,'(km)')
            print (seemhight,'(km)' )
            print (seemhight2,'(km)')
            print (stheta,'(degree)')
            print (atan,'(degree)')
            print ("---error distance----")
            print (distance,'(km)')
            print ("---error angle----")
            """
            num = Coe * (10**(-Ord))
            ti = data[1][7] 
            add_step = np.array([ti,abs(eangle)])

            if error == 0 :
                step = np.append(step,add_step)

            print (COE,'e-',ORDER,'(sec)',eangle,'(degree)(',mx,',',hight,')')
            print(ti)

print(step)
leng = len(step)
leng2 = int(leng/2)

step2 = step.reshape(leng2,2)
x = step2[:,0]
y = step2[:,1]
print (x)
print (y)

plt.scatter(x,y, color = "black")
plt.ylim(0.05,30)
plt.xscale("log")
plt.yscale("log")

plt.show()

np.savetxt('step2.txt', step2)
 
