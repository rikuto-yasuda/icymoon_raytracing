# %%
import pprint
import cdflib
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import math

radg = np.arange(1, 2.7, 0.01)
radkm = radg*2634.1
eldensity1 = 100*np.exp(-(radkm-2634.1)/150)
eldensity2 = 100*np.exp(-(radkm-2634.1)/300)
eldensity3 = 200*np.exp(-(radkm-2634.1)/600)
eldensity4 = 200*np.exp(-(radkm-2634.1)/900)

plt.yscale('log')
plt.xlim(1.0, 2.7)
plt.ylim(1, 1000)
plt.plot(radg, eldensity1)
plt.plot(radg, eldensity2)
plt.plot(radg, eldensity3)
plt.plot(radg, eldensity4)
plt.savefig('plasma_density.png', format="png", dpi=300)
plt.show()


# %%
