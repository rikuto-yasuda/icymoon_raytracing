# %%
import pprint 
import cdflib
import numpy as np
import matplotlib.pyplot as plt

# %% 
Rdo = np.loadtxt('/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/Radio_data.txt')

####print(Rdo.shape)

Rdo[:,10:13] = Rdo[:,10:13]*71492
print(Rdo[0][9])

# %%
GG = np.loadtxt('/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/GLL_GAN.txt', delimiter=',')
##print(GG)

GG[:,1] = np.radians(GG[:,1]) # gallireo position
GG[:,2] = np.radians(GG[:,2])
GG[:,4] = np.radians(GG[:,4]) # ganymede position
GG[:,5] = np.radians(GG[:,5])

##print(GG)
 
 # %%
G=np.zeros(GG.shape)

G[:,1] = GG[:,3]*np.cos(GG[:,1])*np.cos(GG[:,2]) # gallireo position
G[:,2] = GG[:,3]*np.sin(GG[:,1])*np.cos(GG[:,2])
G[:,3] = GG[:,3]*np.sin(GG[:,2])

G[:,4] = GG[:,6]*np.cos(GG[:,4])*np.cos(GG[:,5]) # ganymede position
G[:,5] = GG[:,6]*np.sin(GG[:,4])*np.cos(GG[:,5])
G[:,6] = GG[:,6]*np.sin(GG[:,5])

##print(G)
# %%
lr = len(Rdo)
res =np.zeros((len(Rdo),4))
##print(lr)
# %%
for i in range(lr):
    ex = np.array([0.0,0,0])
    ey = np.array([0.0,0,0])
    r1 = np.array([0.0,0,0])
    r2 = np.array([0.0,0,0])
    r3 = np.array([0.0,0,0])
    rc1 = np.array([0.0,0,0]) 
    rc2 = np.array([0.0,0,0])

    res[i][1] = Rdo[i][9]
    
    for j in range(10,26):

        if Rdo[i][4]==j:

            res[i][0] = j
            r1[0:3] = Rdo[i][10:13] 
            r2[0:3] = G[j-10][4:7] # ganymede position
            r3[0:3] = G[j-10][1:4] # gallireo position

            ex = (r3-r1) / np.linalg.norm(r3-r1)
            rc1 = np.cross((r2-r1),(r3-r1))
            rc2 = np.cross((rc1 / np.linalg.norm(rc1)), ex)
            ey = rc2 / np.linalg.norm(rc2)
            res[i][3] = np.dot((r3-r2),ey) - 2634.1
            res[i][2] = np.dot((r3-r2),ex)
# %%

np.savetxt('/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/R_P_data.txt', res)

A = np.where(res[:,3]>0)

res2 = res[A][:]
print(res2)

np.savetxt('/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/R_P_data2.txt', res2)

# %%
res2 =np.loadtxt('/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/R_P_data2.txt')
plt.yscale('log')
plt.scatter(res2[:,0], res2[:,1])
plt.plot()

# %%
