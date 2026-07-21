# 21CSC305P — Machine Learning Lab

**Department of Computer Science and Engineering**
**SRM Institute of Science and Technology, Ramapuram**
**III / V / F — Academic Year 2026-27**

---

## How to Use This Repository

Each experiment is inside its own folder. To run any experiment:

```bash
cd experiments/exp_1&2   # or whichever experiment
python3 exp_1&2_run.py
```

### Requirements

Install these before running anything:

```bash
pip install numpy pandas matplotlib scikit-learn
```

That's it. No extra libraries, no special setup.

---

## Experiment Index

| Exp | Topic | Algorithm | Key Library |
|-----|-------|-----------|-------------|
| 1 & 2 | Load dataset, view statistics | Pandas | `pandas` |
| 3 | Linear Regression | OLS (Normal Equation) | `numpy` |
| 4.1 | Bayesian Logistic Regression | Metropolis-Hastings MCMC | `numpy`, `sklearn` |
| 4.2 | SVM Classification | RBF Kernel SVM | `sklearn.svm` |
| 5.1 | K-Means Clustering | Centroid-based clustering | `numpy` |
| 5.2 | Gaussian Mixture Model | EM Algorithm | `numpy` |
| 5.3 | Hierarchical Clustering | Agglomerative merge | `numpy` |
| 6 | PCA | Eigenvalue Decomposition | `numpy` |
| 7 | Hidden Markov Model | Viterbi Algorithm | `numpy` |
| 8 | CART Decision Tree | Gini Impurity | `sklearn.tree` |
| 9 | Ensemble Learning | Random Forest + AdaBoost | `sklearn.ensemble` |

---

## What Each Experiment Does

### Experiments 1 & 2 — Loading and Summarizing Data

**What's happening:** You read a CSV file into Python, peek at the first few rows, and compute basic stats like mean, median, and standard deviation.

**Why it matters:** Before you build any model, you need to understand what your data looks like. How many rows? Any missing values? What's the range of each column?

**Key commands to remember:**
- `pd.read_csv("file.csv")` — loads the data
- `df.head(10)` — shows first 10 rows
- `df.describe()` — gives you count, mean, std, min, max, and quartiles
- `df["column"].median()` — median of a specific column

**Common mistakes:**
- Forgetting `encoding='unicode_escape'` — you'll get weird characters or errors
- Not checking `df.shape` first — you won't know if the load actually worked

---

### Experiment 3 — Linear Regression

**What's happening:** You're fitting a straight line through data points using the Normal Equation (Ordinary Least Squares). The formula is:

```
theta = (X^T * X)^(-1) * X^T * Y
```

This finds the slope and intercept that minimize the sum of squared errors between your predictions and actual values.

**Why it matters:** Linear regression is the simplest supervised learning algorithm. If you can't do this, you can't do anything else.

**Key concepts:**
- `X_b = np.c_[np.ones((len(X), 1)), X]` — adds a column of 1s for the intercept term
- `np.linalg.inv()` — matrix inverse
- `.dot()` — matrix multiplication

**What the output means:**
- The scatter plot shows your original data points (blue) and the fitted line (red)
- `predictions` shows what the model thinks Y would be for new X values

**Try this:** Change the `X` and `Y` arrays to something non-linear (like `Y = X**2`). Watch the fit get worse. Linear regression can only model straight lines.

---

### Experiment 4.1 — Bayesian Logistic Regression

**What's happening:** Instead of finding one "best" set of coefficients (like regular logistic regression), you're sampling from a distribution of possible coefficients using Metropolis-Hastings MCMC.

**Why it matters:** This gives you uncertainty estimates. You don't just know "the answer is 0.73" — you know "the answer is probably between 0.68 and 0.78."

**The algorithm (simplified):**
1. Start with random coefficients
2. Propose a small random change
3. Calculate if the new coefficients are better (higher posterior probability)
4. If better, accept. If worse, accept with some probability.
5. Repeat 1000 times. The last 800 samples (after burn-in) represent your posterior.

**Key terms:**
- **Prior** — what you believed before seeing data (uninformative here)
- **Likelihood** — how well the parameters explain the data
- **Posterior** — prior × likelihood (what you believe after seeing data)
- **Acceptance rate** — fraction of proposals accepted. Too low (< 0.1) means your step size is too big. Too high (> 0.5) means you're not exploring enough.

**What the output means:**
- `Acceptance rate: 0.005` — very low, the model is struggling to find good proposals. In practice you'd tune the proposal distribution.
- `Accuracy: 0.69` — not great, but remember this is a probabilistic method with a simple implementation.
- The histogram shows the distribution of each coefficient. Wide peaks = high uncertainty.

---

### Experiment 4.2 — SVM Classification

**What's happening:** You're training a Support Vector Machine with an RBF (Radial Basis Function) kernel to separate two classes of data.

**Why it matters:** SVMs work well when the decision boundary isn't a straight line. The RBF kernel maps data into higher dimensions where it becomes linearly separable.

**Key parameters:**
- `kernel='rbf'` — uses the radial basis function kernel
- `C=1.0` — regularization. Higher C = less tolerance for misclassification
- `gamma='scale'` — controls how far the influence of a single training point reaches

**What the output means:**
- The accuracy report shows precision, recall, and F1-score for each class
- If you had 2D data, you'd see the decision boundary (the line where the model switches from predicting class 0 to class 1)

**Try this:** Change `kernel='linear'` and compare. For some datasets, linear is faster and just as accurate.

---

### Experiment 5.1 — K-Means Clustering

**What's happening:** You're grouping data into K clusters by:
1. Picking K random centroids
2. Assigning each point to the nearest centroid
3. Moving centroids to the average of their assigned points
4. Repeating until nothing changes

**Why it matters:** This is the most common unsupervised learning algorithm. You use it when you don't have labels and want to find natural groupings.

**Key decisions:**
- How do you pick K? Try different values and look at the "elbow" in the within-cluster sum of squares
- What if clusters have different sizes? K-Means assumes roughly equal-sized spherical clusters

**What the output means:**
- Colored dots = data points assigned to different clusters
- Red X marks = final centroid positions

---

### Experiment 5.2 — Gaussian Mixture Model

**What's happening:** Instead of hard-assigning points to clusters (like K-Means), GMM assigns probabilities. Each point has a probability of belonging to each cluster.

**Why it matters:** Real data rarely falls into neat groups. GMM handles overlapping clusters better than K-Means.

**The algorithm:**
1. Initialize K Gaussian distributions (each with its own mean and covariance)
2. **E-step:** Calculate probability that each point belongs to each Gaussian
3. **M-step:** Update the Gaussian parameters using these probabilities
4. Repeat until convergence

**Key difference from K-Means:**
- K-Means: each point belongs to exactly one cluster (hard assignment)
- GMM: each point has a probability for each cluster (soft assignment)

---

### Experiment 5.3 — Hierarchical Clustering

**What's happening:** You're building a tree of clusters by repeatedly merging the two closest clusters until you have one big cluster.

**Why it matters:** You don't need to specify the number of clusters upfront. You can "cut" the tree at different levels to get different numbers of clusters.

**Key decisions:**
- **Linkage:** How do you measure distance between clusters? (single = closest pair, complete = farthest pair, average = mean distance)
- **Where to cut:** Look at the dendrogram (tree diagram) and find the biggest vertical gap

**What the output means:**
- Points with the same color belong to the same cluster
- The algorithm starts with every point as its own cluster and merges downward

---

### Experiment 6 — PCA (Principal Component Analysis)

**What's happening:** You're reducing the number of features while keeping as much information as possible. It works by finding the directions (principal components) where the data varies the most.

**Why it matters:** High-dimensional data is hard to visualize and slow to process. PCA lets you keep, say, 2 components out of 20 features and still capture 95% of the variance.

**The algorithm:**
1. Center the data (subtract mean)
2. Compute covariance matrix
3. Find eigenvalues and eigenvectors
4. Sort by eigenvalue (largest first)
5. Pick top K eigenvectors
6. Project data onto these new axes

**Key output:**
- `eigenvalues` — how much variance each component captures. Bigger = more important.
- `eigenvectors` — the directions of the new feature space
- `X_reduced` — your data in the new, lower-dimensional space

---

### Experiment 7 — Hidden Markov Model (Viterbi Algorithm)

**What's happening:** You have a system that jumps between hidden states (like weather: Rainy/Sunny) and emits observations (like activities: Walk/Shop/Clean). Given a sequence of observations, you want to find the most likely sequence of hidden states.

**Why it matters:** HMMs are used in speech recognition, bioinformatics, and anywhere you have sequential data with hidden patterns.

**The Viterbi algorithm:**
1. For each time step and each state, calculate the probability of being in that state
2. Keep track of which previous state led to the highest probability
3. At the end, trace back through the best previous states to get the full path

**Key parameters:**
- `start_probability` — probability of starting in each state
- `transition_probability` — probability of moving from one state to another
- `emission_probability` — probability of observing each output from each state

**What the output means:**
- `['Sunny', 'Rainy', 'Rainy', 'Sunny']` — the most likely weather sequence given the activities
- `Probability: 0.0024192` — the joint probability of this path. Small numbers are normal because you're multiplying many probabilities.

---

### Experiment 8 — CART Decision Tree

**What's happening:** You're building a tree that makes decisions by asking yes/no questions about features. At each node, the algorithm picks the feature and threshold that best splits the data into pure groups.

**Why it matters:** Decision trees are easy to understand and explain. You can literally draw the tree and show it to someone who doesn't know ML.

**Key parameters:**
- `criterion='gini'` — measures impurity. Gini = 0 means all points in a node belong to one class
- `max_depth=3` — limits how deep the tree grows. Deeper = more complex = risk of overfitting

**What the output means:**
- The tree visualization shows each node with the splitting rule, Gini impurity, number of samples, and class distribution
- An accuracy of 1.0 on Iris means the tree perfectly classified all test points (Iris is an easy dataset)

**Common mistake:** Don't set `max_depth=None` on real datasets — the tree will grow until every leaf is pure, which means it memorizes the training data and fails on new data.

---

### Experiment 9 — Ensemble Learning (Random Forest + AdaBoost)

**What's happening:** You're combining multiple weak models to create one strong model. Two approaches:

**Random Forest (Bagging):**
- Create 100 random subsets of the training data (with replacement)
- Train a decision tree on each subset
- Each tree votes on the final prediction. Majority wins.

**AdaBoost (Boosting):**
- Train a weak model
- Find the points it got wrong
- Give those points higher weight
- Train a new model that focuses on the hard points
- Repeat. Each model gets a vote based on its accuracy.

**Why it matters:** Ensembles almost always outperform single models. Random Forest handles overfitting better than a single tree. AdaBoost is especially good at fixing mistakes.

**Key difference:**
- Random Forest: trees are independent, trained in parallel
- AdaBoost: each model depends on the previous one's mistakes, trained sequentially

**What the output means:**
- The decision boundary plots show how each ensemble classifies the feature space
- Smoother boundaries = better generalization

---

## Lab Evaluation

| Component | Marks |
|-----------|-------|
| Problem Understanding | 10 |
| Dataset Selection | 10 |
| Data Preprocessing | 15 |
| Model Implementation | 25 |
| Result Analysis | 15 |
| Visualization | 10 |
| Viva | 10 |
| Code Quality | 5 |
| **Total** | **100** |

---

## Tips for Presentations

1. **Start with the dataset.** Show `.head()`, `.describe()`, `.shape`. Don't skip this.
2. **Explain the algorithm in plain English** before showing code. "K-Means finds K centers by repeatedly assigning points to the nearest center and moving the centers."
3. **Show the visualization.** A good plot is worth 100 lines of code output.
4. **Discuss what went wrong.** If accuracy is low, say why. If the model overfits, explain what that means.
5. **Know your parameters.** Be ready to answer "what does C do in SVM?" or "why Gini instead of entropy?"

---

## Repository Structure

```
experiments/
├── exp_1&2/
│   ├── exp_1&2_run.py        # Program 1 & 2: Load dataset, statistics
│   └── spotify_songs_demo.csv
├── exp_3/                     # Linear Regression
├── exp_4/                     # Bayesian Logistic Regression & SVM
├── exp_5/                     # K-Means, GMM, Hierarchical Clustering
├── exp_6/                     # PCA
├── exp_7/                     # HMM (Viterbi)
├── exp_8/                     # CART Decision Tree
├── exp_9/                     # Ensemble Learning
└── README.md                  # This file
```

---

## Credits

**Faculty Advisor:** Dr. E. Saraswathi
**HOD, CSE:** Dr. K. Raja
**SRM IST Ramapuram** — 2026-27
