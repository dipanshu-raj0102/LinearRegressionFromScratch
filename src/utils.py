import numpy as np
import pandas as pd


def load_data(path):
    data = pd.read_csv(path)

    data = pd.get_dummies(data, columns=["ocean_proximity"], dtype=float)
    data["total_bedrooms"] = data["total_bedrooms"].fillna(data["total_bedrooms"].median())

    X = data.drop("median_house_value", axis=1).to_numpy(dtype=np.float64)
    y = data["median_house_value"].to_numpy(dtype=np.float64).reshape(-1, 1)

    return X, y

def split_data(X, y):
    np.random.seed(42)
    indices = np.random.permutation(len(X))

    X = X[indices]
    y = y[indices]

    split = int(0.8 * len(X))

    X_train = X[:split]
    X_test  = X[split:]

    y_train = y[:split]
    y_test  = y[split:]

    return X_train, X_test, y_train, y_test


def normalize(X):
    mean = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mean) / sigma

    return X_norm, mean, sigma


