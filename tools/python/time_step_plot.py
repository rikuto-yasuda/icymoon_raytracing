import numpy as np
import matplotlib.pyplot as plt

step2 = np.genfromtxt("/Users/yasudarikuto/research/raytracing/tools/step2.txt")
x = step2[:,0]
y = step2[:,1]
print (x)
print (y)

plt.scatter(x,y, color = "black")
plt.ylim(0.0003,60)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("time_step(s)")
plt.ylabel("error_degree(°)")

plt.show()

np.savetxt('plot_step.txt', step2)
