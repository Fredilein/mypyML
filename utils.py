import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval


def wat_saveplot(x, y, w):
    y_hat = polyval(x[:, -2], w)
    y_hat = w[0] * x[:, 0]**2 + w[1] * x[:, 1] + w[2]

    # Sort x and y_hat by x-axis
    xy = np.c_[y_hat, x]
    xy = xy[np.argsort(xy[:, -2])]

    plt.plot(xy[:, -2], xy[:, 0], 'r')
    plt.scatter(x[:, -2], y)
    plt.savefig('plot.png')


def saveplot(x, y, w):
    x = x[:, -2]
    x_axis = np.linspace(min(x), max(x), len(y))
    y_hat = np.polyval(w, x_axis)
    
    plt.plot(x_axis, y_hat, 'r')
    plt.scatter(x, y)
    return plt

