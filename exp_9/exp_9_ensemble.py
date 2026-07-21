"""
21CSC305P — Machine Learning Lab
Experiment 9: Ensemble Learning (Random Forest + AdaBoost)

AIM: To implement ensemble learning models to perform classification.

ALGORITHM:
Bagging (Random Forest):
1. Create multiple subsets by bootstrap sampling.
2. Train a base model (decision tree) on each subset.
3. Aggregate predictions by majority vote.

Boosting (AdaBoost):
1. Initialize weights for all training examples.
2. Train a base model weighted by current weights.
3. Increase weights of misclassified examples.
4. Combine models with weight proportional to accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# ─── Load Iris dataset ───────────────────────────────────────────────
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

# ─── Train models ───────────────────────────────────────────────────
# Single Decision Tree (baseline)
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train_2d, y_train)
dt_acc = accuracy_score(y_test, dt_clf.predict(X_test_2d))

# Random Forest (Bagging)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_2d, y_train)
rf_acc = accuracy_score(y_test, rf_clf.predict(X_test_2d))

# AdaBoost (Boosting)
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_clf.fit(X_train_2d, y_train)
ada_acc = accuracy_score(y_test, ada_clf.predict(X_test_2d))

print("Accuracy Comparison:")
print(f"  Single Decision Tree: {dt_acc:.4f}")
print(f"  Random Forest:        {rf_acc:.4f}")
print(f"  AdaBoost:             {ada_acc:.4f}")

# ─── Effect of n_estimators ──────────────────────────────────────────
print("\nRandom Forest — Effect of n_estimators:")
for n in [1, 5, 10, 50, 100, 200]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train_2d, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test_2d))
    print(f"  n_estimators={n:>3d} → Accuracy: {acc:.4f}")

# ─── Visualization ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = [
    (dt_clf, 'Decision Tree', dt_acc),
    (rf_clf, 'Random Forest (n=100)', rf_acc),
    (ada_clf, 'AdaBoost (n=100)', ada_acc),
]

h = 0.02
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))

for idx, (model, title, acc) in enumerate(models):
    ax = axes[idx]
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap='RdYlBu',
               edgecolors='black', s=50)
    ax.set_title(f'{title}\nAccuracy: {acc:.2%}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

plt.suptitle('Ensemble Learning — Decision Boundary Comparison', fontsize=13)
plt.tight_layout()
plt.savefig('ensemble_result.png', dpi=150)
plt.show()

print("\nResult: Ensemble learning models executed successfully.")
