"""
21CSC305P — Machine Learning Lab
Experiment 4.1: Bayesian Logistic Regression using Metropolis-Hastings MCMC

AIM: To implement Bayesian logistic regression for classification.

ALGORITHM:
1. Initialize Parameters: Start with prior distributions for the model parameters
   (coefficients, intercept) and likelihood distributions based on the data.
2. Input Data: Gather the dataset containing features (X) and binary labels (Y).
3. Posterior Calculation: Use Metropolis-Hastings MCMC to compute the posterior
   distribution over the parameters given the data.
4. Prediction: Use the posterior distribution to predict class probabilities.
5. Evaluation: Measure accuracy, precision, recall, and F1-score.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─── Generate synthetic dataset ───────────────────────────────────────
X, Y = make_classification(n_samples=500, n_features=10, random_state=42)

# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Add intercept (bias) column
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]


# ─── Sigmoid function ────────────────────────────────────────────────
def sigmoid(z):
    """Maps any real number to (0, 1)."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


# ─── Log-likelihood ──────────────────────────────────────────────────
def log_likelihood(X, Y, theta):
    """Log-likelihood of the data given parameters theta."""
    probs = sigmoid(X.dot(theta))
    return np.sum(Y * np.log(probs + 1e-10) + (1 - Y) * np.log(1 - probs + 1e-10))


# ─── Log-prior (uninformative / flat) ───────────────────────────────
def log_prior(theta):
    """Flat prior — all parameters equally likely (ignores constant)."""
    return 0


# ─── Log-posterior ───────────────────────────────────────────────────
def log_posterior(X, Y, theta):
    return log_likelihood(X, Y, theta) + log_prior(theta)


# ─── Metropolis-Hastings sampling ────────────────────────────────────
def metropolis_hastings(X, Y, n_samples=2000, burn_in=500, step_size=0.1):
    """
    Metropolis-Hastings MCMC:
    1. Start at random theta
    2. Propose new theta = current + Gaussian noise
    3. Accept if posterior improves; otherwise accept with probability ratio
    4. Keep samples after burn-in
    """
    n_params = X.shape[1]
    theta_current = np.random.randn(n_params) * 0.5
    samples = np.zeros((n_samples, n_params))
    accepted = 0

    current_log_post = log_posterior(X, Y, theta_current)

    for i in range(n_samples):
        # Propose new parameters
        proposal = theta_current + np.random.randn(n_params) * step_size

        # Calculate acceptance probability
        proposal_log_post = log_posterior(X, Y, proposal)
        log_acceptance = proposal_log_post - current_log_post

        # Accept or reject
        if np.log(np.random.rand()) < log_acceptance:
            theta_current = proposal
            current_log_post = proposal_log_post
            accepted += 1

        samples[i] = theta_current

    acceptance_rate = accepted / n_samples
    print(f"Acceptance rate: {acceptance_rate:.4f}")

    # Return samples after burn-in
    return samples[burn_in:]


# ─── Run MCMC ────────────────────────────────────────────────────────
print("Running Metropolis-Hastings MCMC (this may take a moment)...")
trace = metropolis_hastings(X_train_b, Y_train, n_samples=2000, burn_in=500)

# ─── Predict using posterior mean ────────────────────────────────────
theta_mean = trace.mean(axis=0)
logits = X_test_b.dot(theta_mean)
Y_pred = (sigmoid(logits) >= 0.5).astype(int)

accuracy = accuracy_score(Y_test, Y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

# ─── Visualize coefficient distributions ─────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.hist(trace[:, i], bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(theta_mean[i], color='red', linestyle='--', linewidth=1.5)
    ax.set_title(f'Coefficient {i}', fontsize=10)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
plt.suptitle('Posterior Distribution of Coefficients (red line = mean)', fontsize=13)
plt.tight_layout()
plt.savefig('bayesian_coefficients.png', dpi=150)
plt.show()

print("\nResult: Bayesian logistic regression using Metropolis-Hastings executed successfully.")
