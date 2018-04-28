import numpy as np
import matplotlib.pyplot as plt



class LinearRegression:


    def __init__(self, ada_lr=True, learning_rate=0.0001):
        # Constructor
        self.weights = []
        self.scores = []
        self.loss = []
        self.ada_lr = ada_lr
        self.learning_rate = learning_rate


    def fit(self, X, y, steps=500):
        n = len(y)
        dim = X.shape[1]
        lr_loss = np.array([])
        # For adaptive learning_rate
        grad_prev = np.zeros(dim)
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
            w -= self.learning_rate * grad

            # Update learning rate
            if self.ada_lr:
                if (np.linalg.norm(grad) < np.linalg.norm(grad_prev)):  self.learning_rate *= c_inc
                else:                                                   self.learning_rate *= c_dec
                grad_prev = grad

            # Compute Loss
            l1 = 0
            for i in range(n):
                l1 += (y[i] - np.dot(w.T, X[i])) ** 2
            lr_loss = np.append(lr_loss, l1)

        self.weights = w
        self.loss = lr_loss


    def predict(self, X):
        if self.weights == []:
            print("Model has no weights yet. Call fit() first.")
            return []
        return np.polyval(self.weights, X)
        
