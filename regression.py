"""
regression.py

= Description =
Used for Linear Regression Tasks.
After constructing an object of this class you can fit a model on some input.


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



import numpy             as np



class LinearRegression:

    C_INC = 1.1
    C_DEC = 0.5


    def __init__(self, ada_lr=True, learning_rate=0.0001):
        self.weights       = []
        self.scores        = []
        self.loss          = []
        self.ada_lr        = ada_lr
        self.learning_rate = learning_rate


    def fit(self, X, y, steps=500):

        n         = len(y)
        dim       = X.shape[1]
        lr_loss   = np.array([])

        # For adaptive learning_rate
        grad_prev = np.zeros(dim)

        # Set arbitrary starting point
        w         = np.zeros(dim)


        # Repeat updating weights [steps] times
        for _ in range(steps):

            # Compute Gradient
            grad_sum = np.zeros(dim)
            for i in range(n):
                grad_sum += (y[i] - np.dot(w.T, X[i])) * X[i]
            grad = -2 * grad_sum

            # Update weights
            w -= self.learning_rate * grad

            # Update learning rate
            if self.ada_lr:
                if (np.linalg.norm(grad) < np.linalg.norm(grad_prev)):  self.learning_rate *= self.C_INC
                else:                                                   self.learning_rate *= self.C_DEC
                grad_prev = grad

            # Compute Loss
            l1 = 0
            for i in range(n):
                l1 += (y[i] - np.dot(w.T, X[i])) ** 2
            lr_loss = np.append(lr_loss, l1)


        # Write weights and loss to the object
        self.weights = w
        self.loss    = lr_loss


    def predict(self, X):
        if self.weights == []:
            print("Model has no weights yet. Call fit() first.")
            return []

        return np.polyval(self.weights, X)
        
