"""
21CSC305P — Machine Learning Lab
Experiment 5.1: K-Means Clustering

AIM: To implement K-Means clustering to categorize the data.

ALGORITHM:
1. Initialize: Randomly select K initial centroids from the data.
2. Assignment: Assign each data point to the nearest centroid.
3. Update: Recalculate centroids by taking the mean of assigned points.
4. Repeat: Steps 2-3 until convergence.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_samples=300, n_centers=4, random_seed=42):
    """Generate synthetic data with n_centers clusters."""
    np.random.seed(random_seed)
    points_per_center = n_samples // n_centers
    centers = np.array([[2, 2], [-2, -2], [2, -2], [-2, 2]], dtype=float)
    X = np.vstack([center + np.random.randn(points_per_center, 2) * 0.8
                   for center in centers])
    return X


def k_means(X, k, max_iters=100):
    """
    K-Means algorithm:
    1. Pick K random points as initial centroids
    2. Assign each point to nearest centroid
    3. Move centroids to mean of assigned points
    4. Repeat until convergence
    """
    # Initialize centroids randomly from data points
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices].copy()

    for iteration in range(max_iters):
        # Step 1: Assign each point to nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Step 2: Calculate new centroids
        new_centroids = np.array([
            X[clusters == j].mean(axis=0) if np.any(clusters == j) else centroids[j]
            for j in range(k)
        ])

        # Step 3: Check for convergence
        if np.allclose(centroids, new_centroids):
            print(f"  Converged after {iteration + 1} iterations")
            break
        centroids = new_centroids

    # Calculate within-cluster sum of squares (WCSS)
    wcss = sum(
        np.sum((X[clusters == j] - centroids[j]) ** 2)
        for j in range(k)
    )
    print(f"  WCSS: {wcss:.2f}")

    return clusters, centroids


# ─── Generate data and run K-Means ──────────────────────────────────
X = generate_data()
print("Running K-Means with K=4:")
clusters, centroids = k_means(X, k=4)

# ─── Elbow Method (find optimal K) ──────────────────────────────────
print("\nElbow method (finding optimal K):")
wcss_values = []
K_range = range(2, 10)
for k in K_range:
    _, c = k_means(X, k=k)
    distances = np.linalg.norm(X[:, np.newaxis] - c, axis=2)
    assignments = np.argmin(distances, axis=1)
    wcss = sum(np.sum((X[assignments == j] - c[j]) ** 2) for j in range(k))
    wcss_values.append(wcss)
    print(f"  K={k}: WCSS={wcss:.2f}")

# ─── Visualization ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Clustering result
ax1 = axes[0]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
          '#1abc9c', '#e67e22', '#34495e']
for j in range(4):
    mask = clusters == j
    ax1.scatter(X[mask, 0], X[mask, 1], c=colors[j], label=f'Cluster {j+1}',
                alpha=0.6, s=50, edgecolors='white')
ax1.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200,
            label='Centroids', zorder=5)
ax1.set_title('K-Means Clustering (K=4)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Elbow plot
ax2 = axes[1]
ax2.plot(list(K_range), wcss_values, 'bo-', linewidth=2, markersize=8)
ax2.set_title('Elbow Method')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Within-Cluster Sum of Squares')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_result.png', dpi=150)
plt.show()

print("\nResult: K-Means clustering executed successfully.")
