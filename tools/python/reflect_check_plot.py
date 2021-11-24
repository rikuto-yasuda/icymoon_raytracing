import numpy as np
import matplotlib.pyplot as plt
 

step2 = np.genfromtxt("/Users/yasudarikuto/research/icymoon_raytracing/tools/results/reflect_check_result.txt")
x = step2[:,0]
y = step2[:,2]
print (x)
print (y)

plt.scatter(x,y, color = "black")
plt.ylim(-0.001,0.001)
plt.xlabel("Angle of incidence (degree)")
plt.ylabel("Angle of incidence  -  Angle of reflection (degree) ")

plt.show()

np.savetxt('plot_step.txt', step2)
