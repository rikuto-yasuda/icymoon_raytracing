# %%
import pprint
import cdflib
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import math

radg = np.arange(0, 2000, 10)
eldensity1 = 300*np.exp(-radg/900)
eldensity2 = 550*np.exp(-radg/600)
eldensity3 = 1000*np.exp(-radg/400)
eldensity4 = 600*np.exp(-radg/900)
eldensity5 = 1100*np.exp(-radg/600)
eldensity6 = 2100*np.exp(-radg/400)


plt.plot(radg, eldensity1, label="A scale 900", linestyle="solid", c="r")
plt.plot(radg, eldensity2, label="A scale 600",
         linestyle="dashed", c="r")
plt.plot(radg, eldensity3, label="A scale 400",
         linestyle="dotted", c="r")
plt.plot(radg, eldensity4, label="D scale 900", linestyle="solid", c="b")
plt.plot(radg, eldensity5, label="D scale 600",
         linestyle="dashed", c="b")
plt.plot(radg, eldensity6, label="D scale 400",
         linestyle="dotted",  c="b")

plt.scatter(1136, 100, label="C3 flyby in situ", c="0.5", marker="x")
plt.scatter(535, 400, label="C10 flyby in situ", c="0.3", marker="x")
plt.vlines(0, 0, 17400, color="0", linestyles='dotted',
           label="all flyby range")

plt.scatter(0, 15300, label="C22 flyby radio", ec="0.5", c="w", s=20)
plt.scatter(0, 17400, label="C23 flyby radio", ec="0.3", c="w", s=20)


plt.yscale('log')
plt.xlabel('height(km)')
plt.ylabel('density(/cc)')

plt.legend(loc="best", fontsize="small")
#plt.savefig('plasma_density.png', format="png", dpi=300)
plt.show()
