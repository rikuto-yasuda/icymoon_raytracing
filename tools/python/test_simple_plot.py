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
    
    def plot (self,n):

        n  = str(n)

        plt.imshow(v, origin='lower',interpolation='nearest', extent=[-1000,995,0,995], vmin=0, vmax=0.6E+10)
        plt.colorbar(extend='both')
        plt.scatter(self.x,self.z, color = 'white',s=0.1)

        plt.title("TIME-RANGE1e-"+n+":1e-13")
        plt.xlabel("x (km)")
        plt.ylabel("z (km)")

        plt.xlim(0,20)
        plt.ylim(0,20)

        plt.show()
    
# In[]

df = np.genfromtxt ('/Users/yasudarikuto/research/icymoon_raytracing/raytrace.tohoku/src/rtc/tools/map_model/test_simple_plasma_map')
l_2d = len(df)

idx = np.array(np.where(df[:,0]>-1000)) # np.whereは条件を満たす要素番号を返す　idxはnp.arrayは
print(idx[0,0])
r_size = idx[0,0]
c_size = int(l_2d/r_size)

x = df[:,0].reshape(c_size, r_size)
print(x.shape)
y = df[:,1].reshape(c_size, r_size)
print(y.shape)
z = df[:,2].reshape(c_size, r_size)
print(z.shape)
v = df[:,3].reshape(c_size, r_size).T
print(v.shape)

# In[]

data = []
for i in range(3,9):
    n = str(i)
    N = i
    filename = "/Users/yasudarikuto/research/icymoon_raytracing/tools/results/time_step_error_data/ray-Ptest_simple-Mtest_simple-benchmark-LO-Z1-FR1e6-TIME-RANGE1e-"+n+":1e-13" 

    data_ = test(filename)
    data_.plot(n)
    data.append(data_)

# %%
data[3].plot(n)