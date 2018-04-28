import datagen
import regression
import utils
import validation

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval



ground_truth = np.array([-.5, .5, 1, -1])
x, y = datagen.generate_polynomial_data(ground_truth, num_points=100, noise=0.6)

steps = 100
lin_reg = regression.linear_regression(x, y, ada_lr=True, steps=steps)

validation.cross_validation(regression.linear_regression, x, y)

print('Weights:', lin_reg['weights'])

plt.plot(np.arange(steps), lin_reg['loss'])
plt.savefig('loss.png')
plt.clf()


plt = utils.saveplot(x, y, lin_reg['weights'])
plt.savefig('plot.png')


