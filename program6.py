import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs 

# --- Configuration ---
K_CLUSTERS = 4  # The target number of clusters (k)
N_SAMPLES = 500 # Total number of data points

# --- 1. Load the Image Dataset (SIMULATED) ---
# Generating synthetic data instead of loading images.
# X: The features (data points, shape 500x2)
# y_true: The true labels (we don't use these for unsupervised clustering, but they exist)
X, y_true = make_blobs(
    n_samples=N_SAMPLES, 
    centers=K_CLUSTERS, 
    cluster_std=1.0, 
    random_state=42
)
print(f"Generated {N_SAMPLES} data points, shape: {X.shape}")

# Initial Visualization (before clustering)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=20)
plt.title("Initial Data Points (Unlabeled)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")


# --- 2. Apply K-Means Clustering ---

# Instantiate the K-Means object. 
# n_init=10 is the number of times the K-Means algorithm will be run with different 
# random centroid seeds (Step 1: Initialize).
kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)

# Fit the model: This executes the Assignment (A) and Update (B) steps 
# iteratively until convergence (C).
kmeans.fit(X)

# Get the final cluster labels and centroids (Step 3: Output)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# --- 3. Visualization of Final Clusters ---

plt.subplot(1, 2, 2)

# Scatter plot the data points, colored by their assigned cluster label
# (Visualizing the Assignment Step outcome)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='viridis')

# Plot the final centroids on top of the data
plt.scatter(centroids[:, 0], centroids[:, 1], 
            marker='X', s=200, linewidths=3, 
            color='red', edgecolors='black', 
            label='Final Centroids')

plt.title(f"K-Means Clustering (K={K_CLUSTERS})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

print("\nFinal Centroids (Cluster Centers):\n", centroids)