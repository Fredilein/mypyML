import numpy as np


def from_func(f, x, noise=10):
    noise_vec = (np.random.rand(len(x)) - 0.5) * noise
    x += noise_vec
    y = f(x) + noise_vec * 10
    return x, y


# From IntroML course, located in util.py
def generate_polynomial_data(w, num_points=100, noise=0.6):
    dim = w.size - 1
    # Generate feature vector 
    x = np.random.normal(size=(num_points, 1))
    x1 = np.power(x, 0)
    for d in range(dim):
        x1 = np.concatenate((np.power(x, 1 + d), x1), axis=1)  # X = [x, 1].
    y = np.dot(x1, w) + np.random.normal(size=(num_points,)) * noise  # y = Xw + eps

    return x1, y

