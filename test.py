import datagen

import numpy as np
import matplotlib.pyplot as plt



x, y = datagen.generate_polynomial_data(np.array([3, 1]))

plt.scatter(x[:, 0], y)
plt.savefig('plot.png')
