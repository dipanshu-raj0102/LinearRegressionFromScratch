import numpy as np


def regularization(X):
    mean = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mean) / sigma

    return X_norm


