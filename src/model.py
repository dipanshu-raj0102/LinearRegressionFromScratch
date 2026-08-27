import numpy as np


def predict(X, w, b):
    y_hat = np.matmul(X, w) + b 
    return y_hat 

def compute_cost(X, y, w, b):
    m = X.shape[0]
    y_hat = predict(X, w, b)
    cost = 0
    for i in range(m):
        loss = (y_hat[i] - y[i]) ** 2
        cost += loss
    cost = cost / (2 * m)
    return cost

def gradient_descent(X, y, w, b, alpha, iterations):
    m = X.shape[0]
    X_T = X.T

    J_history = []
    for i in range(iterations):
        y_hat = predict(X, w, b)
        w = w - alpha * ((1 / m) * X_T @ (y_hat - y))
        b = b - alpha * ((1 / m) * np.sum(y_hat - y))

        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        if ((i % 10) == 0):
            print(f"Iteration {i:4d} | Cost: {cost:.6f}")

    return w, b, J_history




