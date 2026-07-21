"""
21CSC305P — Machine Learning Lab
Experiment 3: Linear Regression using Ordinary Least Squares (OLS)

AIM: To implement linear regression to perform prediction.

ALGORITHM:
1. Initialize Parameters: Start by initializing the parameters like coefficients
   (slope and intercept).
2. Input Data: Gather the dataset containing the independent variable (X) and
   dependent variable (Y).
3. Feature Scaling (Optional): Normalize or standardize the input data if necessary
   to ensure better convergence.
4. Split Data: Divide the dataset into training and testing sets to evaluate the model.
5. Model Training: Implement a method to optimize the parameters (coefficients) based
   on the training data using Normal Equations.
6. Prediction: Use the learned parameters to predict outcomes for new data points.
7. Evaluation: Measure the performance using MSE, R-squared.
"""

import numpy as np
import matplotlib.pyplot as plt

# ─── Generate synthetic dataset ───────────────────────────────────────
# Simulating house prices: size (sq ft) vs price ($)
np.random.seed(42)
X = np.array([600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
              2600, 2800, 3000, 3200, 3400], dtype=float)
# Price = 50 * size + 30000 + some noise
Y = 50 * X + 30000 + np.random.randn(len(X)) * 8000

# ─── Train-test split ────────────────────────────────────────────────
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = X[:split] * 50 + 30000 + np.random.randn(split) * 8000, Y[split:]


# ─── Linear Regression using OLS (Normal Equation) ───────────────────
def linear_regression_ols(X, Y):
    """
    theta = (X^T * X)^(-1) * X^T * Y
    This gives us the slope and intercept that minimize squared error.
    """
    # Add column of ones for intercept term
    X_b = np.c_[np.ones((len(X), 1)), X]
    # Normal Equation
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
    return theta


def predict(X, theta):
    """Y_pred = X_b * theta"""
    X_b = np.c_[np.ones((len(X), 1)), X]
    return X_b.dot(theta)


def mse(Y_actual, Y_predicted):
    """Mean Squared Error"""
    return np.mean((Y_actual - Y_predicted) ** 2)


def r_squared(Y_actual, Y_predicted):
    """R-squared (coefficient of determination)"""
    ss_res = np.sum((Y_actual - Y_predicted) ** 2)
    ss_tot = np.sum((Y_actual - np.mean(Y_actual)) ** 2)
    return 1 - (ss_res / ss_tot)


# ─── Train the model ─────────────────────────────────────────────────
theta = linear_regression_ols(X_train, Y_train)
print(f"Intercept (theta_0): {theta[0]:.2f}")
print(f"Slope (theta_1): {theta[1]:.2f}")

# ─── Make predictions ────────────────────────────────────────────────
Y_pred_train = predict(X_train, theta)
Y_pred_test = predict(X_test, theta)

print(f"\nTraining MSE: {mse(Y_train, Y_pred_train):.2f}")
print(f"Testing MSE: {mse(Y_test, Y_pred_test):.2f}")
print(f"Training R²: {r_squared(Y_train, Y_pred_train):.4f}")
print(f"Testing R²: {r_squared(Y_test, Y_pred_test):.4f}")

# ─── Predict for new values ──────────────────────────────────────────
X_new = np.array([1500, 2500, 3500])
predictions = predict(X_new, theta)
print(f"\nPredictions for new sizes:")
for size, price in zip(X_new, predictions):
    print(f"  {size:.0f} sq ft → ${price:,.2f}")

# ─── Visualization ───────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.scatter(X_train, Y_train, color='blue', label='Training data', alpha=0.7)
plt.scatter(X_test, Y_test, color='green', label='Testing data', alpha=0.7)
# Plot the regression line
X_line = np.linspace(X.min() - 100, X.max() + 100, 100)
Y_line = predict(X_line, theta)
plt.plot(X_line, Y_line, color='red', linewidth=2, label='Regression line')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Linear Regression — House Price Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_regression_plot.png', dpi=150)
plt.show()

print("\nResult: Linear regression using OLS executed successfully.")
