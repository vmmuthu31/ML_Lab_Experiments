"""
21CSC305P — Machine Learning Lab
Experiment 4.2: SVM Classification using RBF Kernel

AIM: To implement SVM for classification.

ALGORITHM:
1. Initialize Parameters: Set kernel type (RBF), regularization (C), and gamma.
2. Input Data: Gather features (X) and binary labels (Y).
3. Model Training: Fit the SVM model to maximize the margin between classes.
4. Prediction: Predict classes for test data.
5. Evaluation: Report accuracy, precision, recall, F1-score.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ─── Generate dataset (two interleaving half-moons) ──────────────────
# This dataset is NOT linearly separable — perfect for RBF kernel
X, Y = make_moons(n_samples=500, noise=0.25, random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ─── Train SVM with RBF kernel ───────────────────────────────────────
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, Y_train)

# ─── Predictions ─────────────────────────────────────────────────────
Y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

# ─── Decision Boundary Visualization ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (kernel, title) in enumerate([
    ('rbf', 'RBF Kernel'),
    ('linear', 'Linear Kernel'),
    ('poly', 'Polynomial Kernel')
]):
    model = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    model.fit(X_train, Y_train)
    acc = accuracy_score(Y_test, model.predict(X_test))

    ax = axes[idx]
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap='RdYlBu',
               edgecolors='black', s=50)
    ax.set_title(f'{title} (Accuracy: {acc:.2%})')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.suptitle('SVM Decision Boundaries — Different Kernels', fontsize=13)
plt.tight_layout()
plt.savefig('svm_kernels.png', dpi=150)
plt.show()

print("\nResult: SVM classification with RBF kernel executed successfully.")
