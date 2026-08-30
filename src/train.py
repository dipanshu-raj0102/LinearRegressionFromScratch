import numpy as np 
import matplotlib.pyplot as plt 
from model import *;
from utils import *;

X, y = load_data("../data/housing.csv")

X_train, X_test, y_train, y_test = split_data(X, y)

X_train, mean, sigma = normalize(X_train)
X_test = (X_test - mean) / sigma

y_train, mu, sig = normalize(y_train)
y_test = (y_test - mu) / sig

w = np.zeros((X_train.shape[1], 1))
b = 0 

alpha = 0.004
iterations = 1000

w, b , J_history = gradient_descent(X_train, y_train, w, b, alpha, iterations)

train_cost = compute_cost(X_train, y_train, w, b)
test_cost = compute_cost(X_test, y_test, w, b)

train_rmse = rmse(X_train, y_train, w, b)
test_rmse = rmse(X_test, y_test, w, b)

train_r2 = r2_score(X_train, y_train, w, b)
test_r2 = r2_score(X_test, y_test, w, b)

print("\n----------------------------")
print("Model Performance")
print("----------------------------")
print(f"Train Cost : {train_cost:.2f}")
print(f"Test Cost  : {test_cost:.2f}")
print(f"Train RMSE : {train_rmse:,.2f}")
print(f"Test RMSE  : {test_rmse:,.2f}")
print(f"Train R²   : {train_r2:.4f}")
print(f"Test R²    : {test_r2:.4f}")

#-----------------------------
# Cost Curve
# ----------------------------
plt.figure(figsize=(7,4))
plt.plot(J_history, linewidth=2)
plt.title("Gradient Descent Convergence")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)

plt.savefig(
    "../images/cost_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ----------------------------
# Actual vs Predicted
# ----------------------------
y_pred = predict(X_test, w, b)

plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.35
)

mn = min(y_test.min(), y_pred.min())
mx = max(y_test.max(), y_pred.max())

plt.plot(
    [mn, mx],
    [mn, mx],
    "r--",
    linewidth=2
)

plt.title(
    f"Actual vs Predicted House Prices\n"
    f"R² = {test_r2:.3f} | RMSE = ${test_rmse:,.0f}"
)

plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)

plt.savefig(
    "../images/predicted_vs_actual.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ----------------------------
# Residual Plot
# ----------------------------
residuals = y_test - y_pred

plt.figure(figsize=(7,4))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.35
)

plt.axhline(
    0,
    color="red",
    linestyle="--"
)

plt.title("Residual Plot")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residual ($)")
plt.grid(True)

plt.savefig(
    "../images/residual_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
