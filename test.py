import datagen
from regression import LinearRegression
import utils
from validation import cross_validation

import numpy as np
import matplotlib.pyplot as plt



ground_truth = np.array([1, 0, 0])
x, y = datagen.generate_polynomial_data(ground_truth, num_points=50, noise=0.6)


steps = 100
lin_reg = LinearRegression()
lin_reg.fit(x, y)
print("Weights:", lin_reg.weights)

print("Crossval scores:")
print(cross_validation(LinearRegression(), x, y))


# plt.plot(np.arange(steps), lin_reg['loss'])
# plt.savefig('loss.png')
# plt.clf()


plt = utils.saveplot(x, y, lin_reg.weights)
plt.savefig('plot.png')


