import numpy as np
import matplotlib.pyplot as plt

# @PARAMS
# X:        Feature Matrix
# y:        Labels
# learning_rate:        
def linear_regression(X, y, steps=500, learning_rate=0.0001, ada_lr=False):
    n = len(y)
    dim = X.shape[1]
    loss = np.array([])
    # For adaptive learning_rate
    grad_tm1 = np.zeros(dim)
    c_inc = 1.1
    c_dec = 0.5

    # Set arbitrary starting point
    w = np.zeros(dim)

    for t in range(steps):
        # Compute Gradient
        sum = np.zeros(dim)
        for i in range(n):
            sum += (y[i] - np.dot(w.T, X[i])) * X[i]
        grad = -2 * sum

        # Update weights
        w -= learning_rate * grad

        # Update learning rate
        if ada_lr:
            if (np.linalg.norm(grad) < np.linalg.norm(grad_tm1)):   learning_rate *= c_inc
            else:                                                   learning_rate *= c_dec
            grad_tm1 = grad

        # Compute Loss
        l1 = 0
        for i in range(n):
            l1 += (y[i] - np.dot(w.T, X[i])) ** 2
        loss = np.append(loss, l1)

    return {"weights":  w,
            "loss":     loss}


