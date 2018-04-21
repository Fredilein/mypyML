import numpy

def linear_regression(X, y, steps=50, learning_rate=1):
    n = len(y)

    # Set arbitrary starting point
    w_hat = np.array([1, 0])

    for t in range(steps):
        grad = 


