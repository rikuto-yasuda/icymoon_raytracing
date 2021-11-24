# In[]
import numpy as np
import matplotlib.pyplot as plt

# In[]
class test:

    def __init__ (self,filename):
        data = np.genfromtxt(filename)
        x = data[:,[1]]
        z = data[:,[3]]
        self.x = x
        self.z = z
        self.a=0
        self.b=0

    def plot (self):
        plt.scatter(self.x,self.z, color = 'black',s=0.1)
    
# In[]
a=0
b=0
for j in range (0,10,1):
    J = str(j)
    for h in range (0,10,1):
        H = str(h)
        filename = "/Users/yasudarikuto/research/raytracing/tools/result_21_06_02/dice_check_20/ray-Ptest_simple-Mtest_simple-benchmark-LO-X"+J+"."+H
        data_ = test(filename)
        data_.plot()

        f = np.genfromtxt(filename)
        if f[[60],[3]] > 0:
            a += 1
            b += 1
        else:
            b += 1

plt.xlim(0,15)
plt.ylim(-10,10)
plt.show()
print(a/b)
# %%
