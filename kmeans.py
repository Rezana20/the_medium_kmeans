import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator


class LearnKMeans:
    def __init__(self):
        """Initialize the LearnKMeans class by loading the Iris dataset and creating a DataFrame."""
        self.kmeans = None
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.df = pd.DataFrame(data=self.X, columns=self.iris.feature_names)
        self.df['species'] = self.y

    def prepare_data(self):
        """Prepare the data by checking for missing values, scaling numerical columns, and dropping unnecessary
        columns."""
        print('Preparing data...')
        print('Shape of data:', self.df.shape)
        print('Missing values:\n', self.df.isnull().sum())
        print(self.df.info())
        self.df.dropna(axis=0, how='any', inplace=True)
        print('Describe:\n', self.df.describe())

        numerical_cols = self.df.columns[:-1]
        scaler = StandardScaler()
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])

    def visualise_data(self, y_value: str, image_name: str):
        """Visualize the data by plotting sepal and petal dimensions."""
        colors = {0: 'red', 1: 'purple', 2: 'orange'}
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for species in self.df[y_value].unique():
            subset = self.df[self.df[y_value] == species]
            axes[0].scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], color=colors[species],
                            label=species)
        axes[0].set_xlabel('Sepal Length (cm)')
        axes[0].set_ylabel('Sepal Width (cm)')
        axes[0].set_title('Sepal Dimensions')
        axes[0].legend()

        for species in self.df[y_value].unique():
            subset = self.df[self.df[y_value] == species]
            axes[1].scatter(subset['petal length (cm)'], subset['petal width (cm)'], color=colors[species],
                            label=species)
        axes[1].set_xlabel('Petal Length (cm)')
        axes[1].set_ylabel('Petal Width (cm)')
        axes[1].set_title('Petal Dimensions')
        axes[1].legend()

        plt.savefig(image_name)
        plt.clf()
        self.df.drop(y_value, axis=1, inplace=True)

    def select_k_with_elbow(self) -> int:
        """Select the optimal number of clusters using the elbow method."""
        inertia = []
        max_clusters = 10

        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=24)
            kmeans.fit(self.df)
            inertia.append(kmeans.inertia_)

        plt.plot(range(1, max_clusters + 1), inertia)
        plt.title('Elbow method for selecting k clusters')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.savefig('kmeans_inertia.png')
        plt.clf()

        kneedle = KneeLocator(range(1, max_clusters + 1), inertia, curve='convex', direction='decreasing')
        optimal_k = kneedle.elbow
        return optimal_k + 1

    def train(self, n_clusters: int):
        """Train the KMeans model with the specified number of clusters and visualize the clustered data using PCA."""
        print('Training data...')
        print('Number of clusters:', n_clusters)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=24)
        self.kmeans.fit(self.df)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.df)
        centroids_pca = pca.transform(self.kmeans.cluster_centers_)
        labels = self.kmeans.labels_

        colors = {0: 'red', 1: 'purple', 2: 'orange'}
        plt.figure(figsize=(15, 6))
        for i in range(n_clusters):
            plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], color=colors[i], label=f'Cluster {i}')
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=70, c='black', marker='X', label='Centroids')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('K-Means Clustering with PCA')
        plt.legend()
        plt.savefig('PCA_kmeans.png')
        plt.clf()

        print("Cluster centroids:")
        print(self.kmeans.cluster_centers_)
        print("\nCluster labels:")
        print(labels)
        self.df['labels'] = labels


if __name__ == '__main__':
    kmeans = LearnKMeans()
    kmeans.visualise_data('species', 'kmeans_petals_and_sepals.png')
    kmeans.prepare_data()

    optimal_k = kmeans.select_k_with_elbow()
    print('Optimal number of clusters:', optimal_k)
    kmeans.train(n_clusters=optimal_k)

    kmeans.visualise_data('labels', 'trained_kmeans_petals_and_sepals.png')
    print(kmeans.kmeans.predict([[0.1, 0.4, 0, 0.3]]))
