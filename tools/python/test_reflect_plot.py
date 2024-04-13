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
    
    def plot (self):
        plt.scatter(self.x,self.z, color = 'black',s=0.1)


    
# In[]
for j in range (5,90,5):
    J = str(j)
    filename ="/Users/yasudarikuto/research/icymoon_raytracing/tools/results/ray_reflection_check/ray-Ptest_simple-Mtest_simple-benchmark-LO-Deg"+J
    data_ = test(filename)
    data_.plot()

plt.xlim(0,60)
plt.ylim(0,11)
plt.show()
# %%
