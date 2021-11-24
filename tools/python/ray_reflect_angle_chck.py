import numpy as np
import matplotlib.pyplot as plt

class test:

    def __init__ (self,filename):
        data = np.genfromtxt(filename)
        x = data[:,[1]]
        z = data[:,[3]]
        self.x = x
        self.z = z
    
    def first_angle (self):
        x1 = self.x[0]
        z1 = self.z[0]
        x2 =0
        z2= 10
        i=0
        while (z2 > 1 ):
            x2 = self.x[i]
            z2 = self.z[i]
            i += 1

        self.first_angle = np.degrees(-np.arctan((z2-z1)/(x2-x1)))


    def second_angle (self):
        x1 = 0
        z1 = 10
        i = 0
        while (z1 < 1 or self.z[i] > self.z[i+1]):
            x1 = self.x[i]
            z1 = self.z[i]
            i +=1
        
        z2 = 0
        x2 = 0
        while (z2 < 10 or self.z[i] > self.z[i+1]):
            x2 = self.x[i]
            z2 = self.z[i]
            i += 1
        
        self.second_angle =  np.degrees(np.arctan((z2-z1)/(x2-x1)))

result = []
for j in range (5,90,5):
    J = str(j)
    filename ="/Users/yasudarikuto/research/raytracing/tools/results/ray_reflection_check/ray-Ptest_simple-Mtest_simple-benchmark-LO-Deg"+J

    data_ = test(filename)
    data_.first_angle()
    data_.second_angle()

    error_angle = data_.first_angle - data_.second_angle
    result = np.append(result,[data_.first_angle, data_.second_angle,error_angle])

print(result)

size = int(len(result)/3)

re_result = result.reshape(size,3)
print(re_result)

np.savetxt('reflect_check_result.txt', re_result)