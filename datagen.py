import numpy as np

def fromfunc(f, x, noise=10):
    noise_vec = (np.random.rand(len(x)) - 0.5) * noise
    x += noise_vec
    y = f(x) + noise_vec * 10
    return x, y
