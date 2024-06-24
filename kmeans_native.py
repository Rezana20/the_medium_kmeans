import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator


class NativeKMeans:
    """
    A class to perform K-means clustering using native Python with sklearn and kneed libraries.

    Attributes:
        data (pd.DataFrame): Pre-processed data for clustering.
        centroids (pd.DataFrame): DataFrame to store centroid coordinates.

    Methods:
        pre_process_data():
            Reads and pre-processes data from 'driver-data.csv', scaling numeric features and dropping IDs.
        find_optimal_centroids() -> int:
            Finds the optimal number of centroids (k) using the elbow method with KneeLocator.
        initialise_centroids(k: int):
            Initializes centroids by randomly selecting k samples from pre-processed data.
        squared_euclidian_distance(x1, x2, y1, y2) -> float:
            Computes the squared Euclidean distance between two points (x1, y1) and (x2, y2).
        train_kmeans_cluster(tolerance=1e-4, training_iterations=10):
            Trains the K-means clustering algorithm with a specified tolerance and maximum iterations.
        update_centroids(tolerance) -> bool:
            Updates centroids based on assigned cluster data and checks for convergence.
        plot_trained_data():
            Plots the clustered data points after training.

    Example usage:
        kmeans = NativeKMeans()
        optimal_k = kmeans.find_optimal_centroids()
        kmeans.initialise_centroids(optimal_k)
        kmeans.train_kmeans_cluster()
    """

    def __init__(self, max_iter=100, tol=1e-4):
        """
        Initializes NativeKMeans with pre-processed data and empty centroids DataFrame.
        """
        self.max_iter = max_iter
        self.tolerance = tol
        self.data = pd.DataFrame()
        self.centroids = pd.DataFrame()
        self.pre_process_data()
        self.n_clusters = self.find_optimal_centroids()
        self.initialise_centroids()

    def pre_process_data(self):
        """
        Reads and pre-processes data from 'driver-data.csv'.

        Returns:
            pd.DataFrame: Pre-processed DataFrame with scaled numeric features and dropped IDs.
        """
        self.data = pd.read_csv('data/driver-data.csv')
        self.data.info()
        self.data.describe()
        self.data.dropna(inplace=True)
        self.data.isnull().sum()

        scaler = StandardScaler()
        self.data[['mean_dist_day', 'mean_over_speed_perc']] = scaler.fit_transform(
            self.data[['mean_dist_day', 'mean_over_speed_perc']])
        self.data = self.data.drop(['id'], axis=1)

    def find_optimal_centroids(self):
        """
        Finds the optimal number of centroids (k) using the elbow method with KneeLocator.

        Returns:
            int: Optimal number of centroids (k) for K-means clustering.
        """
        inertia = []
        max_clusters = 5  # Adjust based on your dataset and problem

        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(self.data[['mean_dist_day', 'mean_over_speed_perc']])
            inertia.append(kmeans.inertia_)

        plt.plot(range(1, max_clusters + 1), inertia)
        plt.show()

        kneedle = KneeLocator(range(1, max_clusters + 1), inertia, curve='convex', direction='decreasing')
        optimal_k = kneedle.elbow
        if optimal_k is None:
            self.n_clusters = 4
        else:
            self.n_clusters = optimal_k



    def initialise_centroids(self):
        """
        Initializes centroids by randomly selecting k samples from pre-processed data.

        """
        self.centroids = self.data[['mean_dist_day', 'mean_over_speed_perc']].sample(n=self.n_clusters)
        print(self.centroids)

    def squared_euclidian_distance(self, x1, x2, y1, y2) -> float:
        """
        Computes the squared Euclidean distance between two points (x1, y1) and (x2, y2).

        Args:
            x1 (float): x-coordinate of point 1.
            x2 (float): x-coordinate of point 2.
            y1 (float): y-coordinate of point 1.
            y2 (float): y-coordinate of point 2.

        Returns:
            float: Squared Euclidean distance between (x1, y1) and (x2, y2).
        """
        return np.sqrt((((x2 - x1) ** 2) + ((y2 - y1) ** 2)))

    def train_kmeans_cluster(self):
        """
        Trains the K-means clustering algorithm with a specified tolerance and maximum iterations.

        Args:
            tolerance (float): Convergence threshold for centroid updates.
            training_iterations (int): Maximum number of training iterations.

        """
        print('Training...')
        iteration = 1
        self.data['labels'] = np.zeros(self.data.shape[0])

        while iteration <= self.max_iter:
            print('Iteration {}'.format(iteration))
            # Calculate distances to all centroids
            # distances = np.linalg.norm(
            #     self.data[['mean_dist_day', 'mean_over_speed_perc']].values[:, np.newaxis] - self.centroids.values,
            #     axis=2)
            #
            # # Assign labels based on the nearest centroid
            # self.data['labels'] = np.argmin(distances, axis=1)

            for index, data_point in self.data.iterrows():
                y2 = data_point['mean_dist_day']
                x2 = data_point['mean_over_speed_perc']
                temp_min = []

                for centroid in self.centroids.values:
                    y1 = centroid[0]
                    x1 = centroid[1]
                    temp_min.append(self.squared_euclidian_distance(x1, x2, y1, y2))

                self.data.at[index, 'labels'] = temp_min.index(min(temp_min))

            if self.update_centroids():
                break

            iteration += 1

        self.plot_trained_data()

    def update_centroids(self) -> bool:
        """
        Updates centroids based on assigned cluster data and checks for convergence.

        Args:
            tolerance (float): Convergence threshold for centroid updates.

        Returns:
            bool: True if centroids have converged (within tolerance), False otherwise.
        """
        new_centroids = self.centroids.copy()

        for i in range(len(self.centroids)):
            cluster_data = self.data[self.data['labels'] == i]
            if not cluster_data.empty:
                new_centroids.loc[i] = cluster_data[['mean_dist_day', 'mean_over_speed_perc']].mean()

        centroid_shift = np.linalg.norm(new_centroids - self.centroids)

        if centroid_shift < self.tolerance:
            return True

        self.centroids = new_centroids
        return False

    def plot_trained_data(self):
        """
        Plots the clustered data points after training.
        """
        for label in np.unique(self.data['labels']):
            subset = self.data[self.data['labels'] == label]
            plt.scatter(subset['mean_over_speed_perc'], subset['mean_dist_day'], label=f'Cluster {label}')

        plt.scatter(self.centroids['mean_over_speed_perc'], self.centroids['mean_dist_day'], color='red', marker='x',
                    label='Centroids')
        plt.legend()
        plt.xlabel('Mean Over Speed Percentile')
        plt.ylabel('Mean Distance Day')
        plt.title('K-means Clustering')
        plt.savefig('images/k_means_native_with_sklearn_trained.png')
        # plt.show()


if __name__ == '__main__':
    kmeans = NativeKMeans()
    kmeans.train_kmeans_cluster()
