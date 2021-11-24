# %%
import pprint
import cdflib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# %%
Rdo = np.loadtxt(
    '/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/All_Gnymede_Radio_data2.txt')  # ???

# print(Rdo.shape)

Rdo[:, 12:15] = Rdo[:, 12:15]*71492


# %%
GG = np.loadtxt(
    '/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/GLL_GAN_2.txt')


# %%

GG[:, 2] = np.radians(GG[:, 2])  # gallireo position
GG[:, 3] = np.radians(GG[:, 3])
GG[:, 5] = np.radians(GG[:, 5])  # ganymede position
GG[:, 6] = np.radians(GG[:, 6])
# print('{:,30g}'.format(GG[1,6])) 確認済み
# print(GG)

# %%
G = np.zeros(GG.shape)

G[:, 0] = GG[:, 0]
G[:, 1] = GG[:, 1]
G[:, 2] = GG[:, 4]*np.cos(GG[:, 2])*np.cos(GG[:, 3])  # gallireo position
G[:, 3] = GG[:, 4]*np.sin(GG[:, 2])*np.cos(GG[:, 3])
G[:, 4] = GG[:, 4]*np.sin(GG[:, 3])

G[:, 5] = GG[:, 7]*np.cos(GG[:, 5])*np.cos(GG[:, 6])  # ganymede position
G[:, 6] = GG[:, 7]*np.sin(GG[:, 5])*np.cos(GG[:, 6])
G[:, 7] = GG[:, 7]*np.sin(GG[:, 6])

# %%
lr = len(Rdo)
#res =np.zeros((len(Rdo),7))
res = np.zeros((len(Rdo), 8))
# print(lr)
# %%
Ghour = G[:, 0].copy()
Gmin = G[:, 1].copy()

# for i in range(4436,4437):
for i in range(lr):
    ex = np.array([0.0, 0, 0])
    ey = np.array([0.0, 0, 0])
    r1 = np.array([0.0, 0, 0])
    r2 = np.array([0.0, 0, 0])
    r3 = np.array([0.0, 0, 0])
    rc1 = np.array([0.0, 0, 0])
    rc2 = np.array([0.0, 0, 0])

    res[i][0] = Rdo[i][3].copy()
    res[i][1] = Rdo[i][4].copy()
    res[i][2] = Rdo[i][9].copy()
    res[i][3] = Rdo[i][10].copy()
    res[i][4] = Rdo[i][11].copy()
    res[i][7] = math.degrees(math.atan2(Rdo[i][13], Rdo[i][12]))
    Glow = np.intersect1d(
        np.where(Ghour == Rdo[i][3]), np.where(Gmin == Rdo[i][4]))
    glow = int(Glow)
    r1[0:3] = Rdo[i][12:15]
    r2[0:3] = G[glow, 5:8]  # ganymede position
    r3[0:3] = G[glow, 2:5]  # gallireo position

    ex = (r3-r1) / np.linalg.norm(r3-r1)
    rc1 = np.cross((r2-r1), (r3-r1))

    """
    print('{:.18g}'.format(rc1[2]))
    print('{:.18g}'.format(rc1_2[2]))
    """
    rc2 = np.cross((rc1 / np.linalg.norm(rc1)), ex)
    ey = rc2 / np.linalg.norm(rc2)

    res[i][6] = np.dot((r3-r2), ey) - 2634.1
    res[i][5] = np.dot((r3-r2), ex)

np.savetxt(
    '/Users/yasudarikuto/research/icymoon_raytracing/tools/result_sgepss_2021/R_P_fulldata2.txt', res)

# %%
"""
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
"""
