"""
21CSC305P — Machine Learning Lab
Experiment 5.2: Gaussian Mixture Model (GMM)

AIM: To implement GMM to categorize the data using the EM algorithm.

ALGORITHM:
1. Initialize: Choose initial parameters for K Gaussian components
   (means, covariances, mixing coefficients).
2. E-step: Calculate probability of each data point belonging to each component.
3. M-step: Update parameters using the probabilities from E-step.
4. Repeat: Steps 2-3 until convergence.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_samples=300, n_centers=4, random_seed=42):
    """Generate synthetic data with n_centers clusters."""
    np.random.seed(random_seed)
    points_per_center = n_samples // n_centers
    centers = np.array([[3, 3], [-3, -3], [3, -3], [-3, 3]], dtype=float)
    X = np.vstack([center + np.random.randn(points_per_center, 2) * 1.0
                   for center in centers])
    return X


def gaussian_pdf(x, mean, cov):
    """
    Multivariate Gaussian probability density function.
    p(x) = (1 / sqrt((2*pi)^d * det(cov))) * exp(-0.5 * (x-mu)^T * cov^-1 * (x-mu))
    """
    d = x.shape[0]
    diff = x - mean
    cov_inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    exponent = -0.5 * diff.dot(cov_inv).dot(diff)
    return np.exp(exponent) / np.sqrt((2 * np.pi) ** d * det)


def gmm(X, k, max_iters=100):
    """
    Gaussian Mixture Model using EM algorithm.
    """
    n_samples, n_features = X.shape

    # ── Initialize parameters ────────────────────────────────────────
    # Pick K random points as initial means
    indices = np.random.choice(n_samples, k, replace=False)
    means = X[indices].copy()
    # Identity covariance for each component
    covariances = np.array([np.eye(n_features) for _ in range(k)])
    # Equal mixing coefficients
    weights = np.ones(k) / k

    log_likelihoods = []

    for iteration in range(max_iters):
        # ── E-step: Calculate responsibilities ──────────────────────
        responsibilities = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j in range(k):
                responsibilities[i, j] = weights[j] * gaussian_pdf(
                    X[i], means[j], covariances[j]
                )
            # Normalize so responsibilities sum to 1
            total = np.sum(responsibilities[i])
            if total > 0:
                responsibilities[i] /= total

        # ── M-step: Update parameters ──────────────────────────────
        for j in range(k):
            N_j = np.sum(responsibilities[:, j])
            if N_j < 1e-10:
                continue

            # Update mean
            means[j] = np.sum(responsibilities[:, j, np.newaxis] * X, axis=0) / N_j

            # Update covariance
            diff = X - means[j]
            covariances[j] = (
                (responsibilities[:, j, np.newaxis] * diff).T.dot(diff) / N_j
            )

            # Update mixing coefficient
            weights[j] = N_j / n_samples

        # ── Calculate log-likelihood ────────────────────────────────
        ll = 0
        for i in range(n_samples):
            ll_i = 0
            for j in range(k):
                ll_i += weights[j] * gaussian_pdf(X[i], means[j], covariances[j])
            ll += np.log(ll_i + 1e-10)
        log_likelihoods.append(ll)

        # Check convergence
        if len(log_likelihoods) > 1:
            if abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-4:
                print(f"  Converged after {iteration + 1} iterations")
                break

    # Assign clusters
    clusters = np.argmax(responsibilities, axis=1)
    return clusters, means, log_likelihoods


# ─── Generate data and run GMM ──────────────────────────────────────
X = generate_data()
print("Running GMM with K=4:")
clusters, means, log_likelihoods = gmm(X, k=4)

# ─── Visualization ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Clustering result
ax1 = axes[0]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for j in range(4):
    mask = clusters == j
    ax1.scatter(X[mask, 0], X[mask, 1], c=colors[j], label=f'Cluster {j+1}',
                alpha=0.6, s=50, edgecolors='white')
ax1.scatter(means[:, 0], means[:, 1], c='black', marker='X', s=200,
            label='Means', zorder=5)
ax1.set_title('GMM Clustering (K=4)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Log-likelihood convergence
ax2 = axes[1]
ax2.plot(log_likelihoods, 'b-', linewidth=2)
ax2.set_title('Log-Likelihood Convergence')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Log-Likelihood')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gmm_result.png', dpi=150)
plt.show()

print("\nResult: Gaussian Mixture Model executed successfully.")
