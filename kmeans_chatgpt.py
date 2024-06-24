import numpy as np
import matplotlib.pyplot as plt
import random


class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        """
        Initialize the KMeans class.

        Parameters:
        n_clusters (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance to declare convergence.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, data):
        """
        Compute k-means clustering.

        Parameters:
        data (ndarray): The data to cluster.
        """
        # Step 1: Initialize centroids
        self.centroids = self._initialize_centroids(data)

        for i in range(self.max_iter):
            # Step 2: Assign clusters
            labels = self._assign_clusters(data)

            # Step 3: Update centroids
            new_centroids = self._update_centroids(data, labels)

            # Check for convergence
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tol):
                break

            self.centroids = new_centroids

    def _initialize_centroids(self, data):
        """
        Initialize centroids by randomly selecting points from the data.

        Parameters:
        data (ndarray): The data to cluster.

        Returns:
        ndarray: Initial centroids.
        """
        random_indices = random.sample(range(data.shape[0]), self.n_clusters)
        return data[random_indices]

    def _assign_clusters(self, data):
        """
        Assign each data point to the nearest centroid.

        Parameters:
        data (ndarray): The data to cluster.

        Returns:
        ndarray: Array of cluster labels for each data point.
        """
        distances = np.zeros((data.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(data - self.centroids[k], axis=1)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, data, labels):
        """
        Update centroids by computing the mean of all points assigned to each centroid.

        Parameters:
        data (ndarray): The data to cluster.
        labels (ndarray): Cluster labels for each data point.

        Returns:
        ndarray: Updated centroids.
        """
        new_centroids = np.zeros((self.n_clusters, data.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = data[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
        return new_centroids

    def predict(self, data):
        """
        Predict the closest cluster each sample in data belongs to.

        Parameters:
        data (ndarray): New data to predict.

        Returns:
        ndarray: Array of cluster labels for each data point.
        """
        return self._assign_clusters(data)

    def plot_clusters(self, data, labels):
        """
        Plot the clustered data along with the centroids.

        Parameters:
        data (ndarray): The data to cluster.
        labels (ndarray): Cluster labels for each data point.
        """
        for k in range(self.n_clusters):
            cluster_points = data[labels == k]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {k}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=300, c='red', label='Centroids')
        plt.legend()
        # plt.show()


# Example usage:

# Generating some sample data
np.random.seed(42)
data = np.random.rand(300, 2)

# Initializing and fitting the KMeans model
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

# Predicting the cluster labels
labels = kmeans.predict(data)

# Plotting the clustered data
kmeans.plot_clusters(data, labels)
