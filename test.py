"""
test.py

Only used for testing purposes
"""

import datagen
from regression import LinearRegression
from classification import BinaryClassification
import utils
from validation import cross_validation

import numpy as np
import matplotlib.pyplot as plt
import plot_helpers



def linreg_test():
    ground_truth = np.array([1, 0])
    x, y = datagen.generate_polynomial_data(ground_truth, num_points=50, noise=0.6)


    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    print("Weights:", lin_reg.weights)

    print("Crossval scores:")
    print(cross_validation(LinearRegression(), x, y))


    # plt.plot(np.arange(steps), lin_reg['loss'])
    # plt.savefig('loss.png')
    # plt.clf()


    plot = utils.saveplot(x, y, lin_reg.weights)
    plot.savefig('plot.png')


num_points = 100  # Number of points per class
noise = 0.5  # Noise Level (needed for data generation).
X, Y = datagen.generate_linear_separable_data(num_points, noise=noise, dim=2)

indexes = np.arange(0, 2*num_points, 1)
np.random.shuffle(indexes)
num_train = int(np.ceil(2*.05*num_points))

X_train = X[indexes[:num_train]]
Y_train = Y[indexes[:num_train]]

X_test = X[indexes[num_train:]]
Y_test = Y[indexes[num_train:]]

fig = plt.subplot(111)

opt = {'marker': 'ro', 'fillstyle': 'full', 'label': '+ Train', 'size': 8}
plot_helpers.plot_data(X_train[np.where(Y_train == 1)[0], 0], X_train[np.where(Y_train == 1)[0], 1], fig=fig, options=opt)
opt = {'marker': 'bs', 'fillstyle': 'full', 'label': '- Train', 'size': 8}
plot_helpers.plot_data(X_train[np.where(Y_train == -1)[0], 0], X_train[np.where(Y_train == -1)[0], 1], fig=fig, options=opt)

opt = {'marker': 'ro', 'fillstyle': 'none', 'label': '+ Test', 'size': 8}
plot_helpers.plot_data(X_test[np.where(Y_test == 1)[0], 0], X_test[np.where(Y_test == 1)[0], 1], fig=fig, options=opt)
opt = {'marker': 'bs', 'fillstyle': 'none', 'label': '- Test', 'size': 8, 
       'x_label': '$x$', 'y_label': '$y$', 'legend': True}
plot_helpers.plot_data(X_test[np.where(Y_test == -1)[0], 0], X_test[np.where(Y_test == -1)[0], 1], fig=fig, options=opt)

plt.savefig('class_plot.png')


bin_class = BinaryClassification()
bin_class.fit(X_train, Y_train)
print("Weights:", bin_class.weights)

bin_pred = bin_class.predict(X_test)


print("Score:", np.where(bin_pred == Y_test)[0].shape[0], "out of", len(Y_test))

steps = 50
plt.clf()
plt.plot(np.arange(steps), bin_class.loss)
plt.savefig('class_loss.png')
plt.clf()