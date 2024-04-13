# %%
import pprint
import cdflib
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import math

radg = np.arange(0, 2000, 10)
eldensity1 = 100*np.exp(-radg/1000)
eldensity2 = 50*np.exp(-radg/600)
eldensity3 = 25*np.exp(-radg/100)
eldensity4 = 200*np.exp(-radg/1000)
eldensity5 = 150*np.exp(-radg/600)
eldensity6 = 100*np.exp(-radg/100)


plt.plot(radg, eldensity1, label="Trail/scale 1000km", linestyle="solid", c="r")
plt.plot(radg, eldensity2, label="Trail/scale 600km",
         linestyle="dashed", c="r")
plt.plot(radg, eldensity3, label="Trail/scale 100km",
         linestyle="dotted", c="r")
plt.plot(radg, eldensity4, label="Lead/scale 1000km", linestyle="solid", c="b")
plt.plot(radg, eldensity5, label="Lead/scale 600km",
         linestyle="dashed", c="b")
plt.plot(radg, eldensity6, label="Lead/scale 100km",
         linestyle="dotted",  c="b")

radgG1 = np.arange(835, 2000, 5)
eldensity7 = 100*np.exp(-radgG1/1000)
plt.plot(radgG1, eldensity7, label="G1 flyby radio",
         linestyle="solid", c="0.3")

radgG2 = np.arange(288.80, 1605.9, 5)
eldensity8 = 400*np.exp(-radgG2/600)
plt.plot(radgG2, eldensity8, label="G2 flyby radio",
         linestyle="solid", c="0.5")

plt.vlines(0, 0, 4000, color="0", linestyles='dotted',
           label="all flyby range")
plt.scatter(0, 4000, label="G8 flyby radio", ec="0.3", c="w", s=20)


plt.yscale('log')
plt.xlabel('height(km)')
plt.ylabel('density(/cc)')
plt.ylim(1, 5000)

plt.legend(loc="best", fontsize="small")
#plt.savefig('plasma_density.png', format="png", dpi=300)
plt.show()
