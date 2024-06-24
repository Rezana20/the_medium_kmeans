# load a dataset for learning
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class LearnKMeans:
    def __init__(self):
        self.kmeans = None
        # Load the dataset
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.df = pd.DataFrame(data=self.X, columns=self.iris.feature_names)
        self.df['species'] = self.y

    def prepare_data(self):
        print('Preparing data...')
        print('Shape of data:', self.df.shape)
        print('Missing values:\n', self.df.isnull().sum())
        print(self.df.info)
        # Drop rows where any columns have missing values
        self.df.dropna(axis=0, how='any')
        print('Describe:\n', self.df.describe())

        # Extract the numerical columns to scale
        numerical_cols = self.df.columns[:-1]

        scaler = StandardScaler()
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])

    def visualise_data(self, y_value: str, image_name: str):

        # Define colors for each species
        colors = {0: 'red', 1: 'purple', 2: 'orange'}

        # Create subplots for sepal and petal
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Sepal plot
        for species in self.df[y_value].unique():
            subset = self.df[self.df[y_value] == species]
            axes[0].scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
                            color=colors[species], label=species)
        axes[0].set_xlabel('Sepal Length (cm)')
        axes[0].set_ylabel('Sepal Width (cm)')
        axes[0].set_title('Sepal Dimensions')
        axes[0].legend()

        # Petal plot
        for species in self.df[y_value].unique():
            subset = self.df[self.df[y_value] == species]
            axes[1].scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                            color=colors[species], label=species)
        axes[1].set_xlabel('Petal Length (cm)')
        axes[1].set_ylabel('Petal Width (cm)')
        axes[1].set_title('Petal Dimensions')
        axes[1].legend()

        # Save the plot
        plt.savefig(image_name)
        plt.clf()
        self.df.drop(y_value, axis=1, inplace=True)

    def select_k_with_elbow(self) -> int:
        inertia = []
        for k in range(1, 5):
            self.kmeans = KMeans(n_clusters=k, random_state=24)
            self.kmeans.fit(self.df)
            inertia.append(self.kmeans.inertia_)

        plt.plot(range(1, 5), inertia)
        plt.title('Elbow method for selecting k clusters')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.savefig('kmeans_inertia.png')
        plt.clf()

        kneedle = KneeLocator(range(1, 5), inertia, curve='convex', direction='decreasing')
        optimal_k = kneedle.elbow

        # Selecting the next optimal value because we know there are three clusters.
        return optimal_k + 1

    def train(self, n_clusters: int):
        print('Training data...')
        print(n_clusters)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=24)
        self.kmeans.fit(self.df)

        # Use PCA to reduce the data to 2D
        pca = PCA(n_components=2)

        centroids = self.kmeans.cluster_centers_
        labels = self.kmeans.labels_

        X_pca = pca.fit_transform(self.df)
        centroids_pca = pca.transform(centroids)

        # Define colors for each species
        colors = {0: 'red', 1: 'purple', 2: 'orange'}
        # Plot the clustered data
        plt.figure(figsize=(15, 6))
        for i in range(n_clusters):
            # Plot the points in each cluster
            plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], color=colors[i], label=f'Cluster {i}')
        # Plot the centroids
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=70, c='black', marker='X', label='Centroids')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('K-Means Clustering with PCA')
        plt.legend()
        plt.savefig('PCA_kmeans.png')
        plt.clf()

        # Print cluster centroids and labels
        print("Cluster centroids:")
        print(centroids)
        print("\nCluster labels:")
        print(labels)
        self.df['labels'] = labels


if __name__ == '__main__':
    kmeans = LearnKMeans()

    kmeans.visualise_data('species', 'kmeans_petals_and_sepals.png')

    kmeans.prepare_data()

    optimal_k = kmeans.select_k_with_elbow()
    print('Optimal k:', optimal_k)
    kmeans.train(n_clusters=optimal_k)

    kmeans.visualise_data('labels', 'trained_kmeans_petals_and_sepals.png')

    print(kmeans.kmeans.predict([[0.1, 0.4, 0, 0.3]]))
