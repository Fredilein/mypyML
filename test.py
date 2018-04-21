import datagen

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 100, num=100)
f = lambda x: x

x, y = datagen.fromfunc(f, x)

plt.scatter(x, y)
plt.savefig('plot.png')
