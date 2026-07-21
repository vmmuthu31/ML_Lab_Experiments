"""
21CSC305P — Machine Learning Lab
Experiment 5.3: Hierarchical Clustering

AIM: To implement hierarchical clustering to categorize the data.

ALGORITHM:
1. Start: Treat each data point as a singleton cluster.
2. Merge: Find the pair of clusters that are closest and merge them.
3. Repeat: Step 2 until only a single cluster remains.
4. Cut: Cut the dendrogram at the desired level to extract clusters.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


def generate_data(n_samples=200, n_centers=4, random_seed=42):
    """Generate synthetic data with n_centers clusters."""
    np.random.seed(random_seed)
    points_per_center = n_samples // n_centers
    centers = np.array([[1, 1], [5, 5], [1, 5], [5, 1]], dtype=float)
    X = np.vstack([center + np.random.randn(points_per_center, 2) * 0.6
                   for center in centers])
    return X


# ─── Generate data ───────────────────────────────────────────────────
X = generate_data()
print(f"Dataset shape: {X.shape}")

# ─── Hierarchical Clustering using scipy ─────────────────────────────
# linkage computes the merge history
# 'ward' minimizes variance within clusters (most common)
Z = linkage(X, method='ward')

# ─── Cut the dendrogram to get 4 clusters ────────────────────────────
n_clusters = 4
cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
print(f"Number of clusters: {n_clusters}")
print(f"Cluster sizes: {[np.sum(cluster_labels == i) for i in range(1, n_clusters+1)]}")

# ─── Visualization ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Scatter plot of clusters
ax1 = axes[0]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for j in range(1, n_clusters + 1):
    mask = cluster_labels == j
    ax1.scatter(X[mask, 0], X[mask, 1], c=colors[j-1], label=f'Cluster {j}',
                alpha=0.6, s=50, edgecolors='white')
ax1.set_title(f'Hierarchical Clustering ({n_clusters} clusters)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Dendrogram
ax2 = axes[1]
dendrogram(
    Z,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True,
    ax=ax2
)
ax2.axhline(y=7, color='r', linestyle='--', label='Cut line')
ax2.set_title('Dendrogram (truncated)')
ax2.set_xlabel('Sample Index or Cluster Size')
ax2.set_ylabel('Distance')
ax2.legend()

plt.tight_layout()
plt.savefig('hierarchical_result.png', dpi=150)
plt.show()

print("\nResult: Hierarchical clustering executed successfully.")
