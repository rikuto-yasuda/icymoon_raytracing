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
eldensity6 = 1200*np.exp(-radg/400)


plt.plot(radg, eldensity1, label="max 300 scale 900")
plt.plot(radg, eldensity2, label="max 550 scale 600")
plt.plot(radg, eldensity3, label="max 1000 scale 400")
plt.plot(radg, eldensity4, label="max 600 scale 900")
plt.plot(radg, eldensity5, label="max 1100 scale 600")
plt.plot(radg, eldensity6, label="max 1200 scale 400 (now calculating..)")

plt.yscale('log')
plt.legend()
#plt.savefig('plasma_density.png', format="png", dpi=300)
plt.show()
