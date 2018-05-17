"""
classification.py

= Description =
Used for Classification Tasks.
After constructing an object of this class you can fit a model on some input.

Binary Classification currently uses the Surrogate Loss function.


= Functions =


__init__
    Constructs a Object and initializes parameters

    @(IN)       ada_lr              -- updates the learning rate in every iteration depending on the gradient. Results in faster convergence
    @(IN)       learning_rate       -- initial learning rate


fit()
    Takes Data X and labels y. Updates the objects weights and loss

    @IN         X                   -- Data vector. Each row corresponding to a sample of dimension [cols]          
    @IN         y                   -- Label vector. One dimensional, same amount of rows as X
    @(IN)       steps               -- Number of iterations. Defaults to 500.


predict()
    Takes Data X and returns a label vector calculated from the weights. fit() needs to be called first.

    @IN         X                   -- Data vector.
"""

import numpy as np



class BinaryClassification:

    C_INC = 1.1
    C_DEC = 0.5


    def __init__(self, ada_lr=True, learning_rate=1.e-5):
        self.weights       = np.array([])
        self.loss          = np.array([])
        self.ada_lr        = ada_lr
        self.learning_rate = learning_rate

    
    def fit(self, X, y, steps=50):

        n         = len(y)
        dim       = X.shape[1]
        lr_loss   = np.array([])

        grad_prev = np.zeros(dim)

        w         = np.zeros(dim)
        
        for _ in range(steps):

            # Compute Gradient
            grad_sum = np.zeros(dim)
            for i in range(n):
                if y[i] != np.sign(np.dot(w.T, X[i])):
                    grad_sum += y[i] * X[i]
            grad = grad_sum * -1.

            w -= self.learning_rate * grad

            # Update learning rate
            if self.ada_lr:
                if (np.linalg.norm(grad) < np.linalg.norm(grad_prev)):  self.learning_rate *= self.C_INC
                else:                                                   self.learning_rate *= self.C_DEC
                grad_prev = grad

            # Compute Loss
            loss_sum = 0
            for i in range(n):
                loss_sum += max(0., -1. * y[i] * np.dot(w.T, X[i]))
            lr_loss = np.append(lr_loss, loss_sum)

        self.weights = np.array(w)
        self.loss    = lr_loss

    
    def predict(self, X):
        return np.array([int(np.sign(np.dot(self.weights.T, X_i))) for X_i in X])
