"""
21CSC305P — Machine Learning Lab
Experiment 8: CART Decision Tree

AIM: To implement CART learning algorithm to perform categorization.

ALGORITHM:
1. Load dataset (Iris).
2. Split into training and testing sets.
3. Train a Decision Tree classifier using Gini impurity.
4. Evaluate accuracy on test set.
5. Visualize the decision tree.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─── Load Iris dataset ───────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {list(target_names)}")

# ─── Split into train and test ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")

# ─── Train Decision Tree (Gini impurity) ────────────────────────────
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# ─── Predictions ─────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# ─── Feature importance ──────────────────────────────────────────────
print("Feature Importance:")
for name, importance in zip(feature_names, clf.feature_importances_):
    print(f"  {name}: {importance:.4f}")

# ─── Visualization ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Decision Tree
ax1 = axes[0]
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=list(target_names),
    filled=True,
    rounded=True,
    ax=ax1,
    fontsize=10
)
ax1.set_title('CART Decision Tree (Gini, max_depth=3)')

# Right: Feature importance
ax2 = axes[1]
y_pos = np.arange(len(feature_names))
ax2.barh(y_pos, clf.feature_importances_, color='steelblue', alpha=0.7)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(feature_names)
ax2.set_xlabel('Importance')
ax2.set_title('Feature Importance (Gini)')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('cart_result.png', dpi=150)
plt.show()

# ─── Effect of max_depth ─────────────────────────────────────────────
print("\nEffect of max_depth on accuracy:")
for depth in [1, 2, 3, 5, None]:
    clf_temp = DecisionTreeClassifier(criterion='gini', max_depth=depth,
                                       random_state=42)
    clf_temp.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf_temp.predict(X_test))
    print(f"  max_depth={str(depth):>4s} → Accuracy: {acc:.2f}")

print("\nResult: CART decision tree executed successfully.")
