"""
21CSC305P — Machine Learning Lab
Experiment 6: Principal Component Analysis (PCA)

AIM: To implement PCA for dimensionality reduction.

ALGORITHM:
1. Standardize the Data: Center by subtracting mean of each feature.
2. Compute Covariance Matrix: Calculate covariance of centered data.
3. Eigenvalues & Eigenvectors: Find eigenvalues and eigenvectors of covariance matrix.
4. Sort: Sort eigenvectors by decreasing eigenvalues.
5. Transform: Project original data onto new feature space.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


def pca(X, n_components):
    """
    PCA from scratch using eigenvalue decomposition.
    """
    # Step 1: Center the data (subtract mean)
    X_centered = X - np.mean(X, axis=0)

    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Step 3: Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort by decreasing eigenvalue
    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_idx]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]

    # Step 5: Select top n_components
    eigenvector_subset = sorted_eigenvectors[:, :n_components]

    # Step 6: Transform data
    X_reduced = X_centered.dot(eigenvector_subset)

    return X_reduced, sorted_eigenvalues, sorted_eigenvectors


# ─── Load Iris dataset ───────────────────────────────────────────────
iris = load_iris()
X = iris.data  # 4 features
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Original shape: {X.shape} (150 samples, 4 features)")

# ─── Apply PCA (reduce 4 features to 2) ─────────────────────────────
X_reduced, eigenvalues, eigenvectors = pca(X, n_components=2)
print(f"Reduced shape: {X_reduced.shape} (150 samples, 2 features)")

# ─── Explained variance ──────────────────────────────────────────────
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance
print(f"\nEigenvalues: {eigenvalues}")
print(f"Explained variance ratio: {explained_variance_ratio}")
print(f"PC1 explains {explained_variance_ratio[0]:.2%} of variance")
print(f"PC2 explains {explained_variance_ratio[1]:.2%} of variance")
print(f"Total explained: {sum(explained_variance_ratio[:2]):.2%}")

# ─── Visualization ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: PCA scatter plot
ax1 = axes[0]
for i, target_name in enumerate(target_names):
    mask = y == i
    ax1.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                label=target_name, alpha=0.7, s=50, edgecolors='white')
ax1.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
ax1.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
ax1.set_title('PCA — Iris Dataset (4D → 2D)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Middle: Variance explained
ax2 = axes[1]
components = range(1, len(eigenvalues) + 1)
ax2.bar(components, explained_variance_ratio, color='steelblue', alpha=0.7,
        edgecolor='white', label='Individual')
ax2.plot(components, np.cumsum(explained_variance_ratio), 'ro-',
         linewidth=2, label='Cumulative')
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Variance Explained')
ax2.set_title('Scree Plot')
ax2.set_xticks(list(components))
ax2.legend()
ax2.grid(True, alpha=0.3)

# Right: Eigenvectors (loadings)
ax3 = axes[2]
loading_matrix = eigenvectors[:, :2]
im = ax3.imshow(loading_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['PC1', 'PC2'])
ax3.set_yticks(range(len(feature_names)))
ax3.set_yticklabels(feature_names)
ax3.set_title('Feature Loadings')
plt.colorbar(im, ax=ax3)

plt.tight_layout()
plt.savefig('pca_result.png', dpi=150)
plt.show()

print("\nResult: PCA executed successfully.")
