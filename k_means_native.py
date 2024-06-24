import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator


class NativeKMeans:
    def __init__(self):
        self.data = self.pre_process_data()
        self.centroids = pd.DataFrame()

        # mean_dist_day is y
        # mean_over_speed_perc is x

    def pre_process_data(self):

        data = pd.read_csv('driver-data.csv')
        data.info()
        data.describe()
        data.dropna(inplace=True)
        data.isnull().sum()

        scaler = StandardScaler()
        data[['mean_dist_day', 'mean_over_speed_perc']] = scaler.fit_transform(
            data[['mean_dist_day', 'mean_over_speed_perc']])
        data = data.drop(['id'], axis=1)

        return data

    def find_optimal_centroids(self):

        inertia = []

        for i in range(1, 5):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(self.data[['mean_dist_day', 'mean_over_speed_perc']])
            inertia.append(kmeans.inertia_)

        kneedle = KneeLocator(range(1, 5), inertia, curve='convex', direction='decreasing')
        optimal_k = kneedle.elbow
        return optimal_k

    def initialise_centroids(self, k: int):
        # Select k unique random samples for initial centroids
        self.centroids = self.data[['mean_dist_day', 'mean_over_speed_perc']].sample(n=k)
        print(self.centroids)

    def squared_euclidian_distance(self, x1, x2, y1, y2):
        return np.sqrt((((x2 - x1) ** 2) + ((y2 - y1) ** 2)))

    def train_kmeans_cluster(self, tolerance=1e-4, training_iterations=10):
        print('training...')
        iteration = 1
        self.data['labels'] = np.zeros(self.data.shape[0])

        while iteration <= training_iterations:

            for index, data in self.data.iterrows():
                y2 = data['mean_dist_day']
                x2 = data['mean_over_speed_perc']
                temp_min = []
                for centroid in self.centroids.values:
                    y1 = centroid[0]
                    x1 = centroid[1]
                    temp_min.append(self.squared_euclidian_distance(x1, x2, y1, y2))

                self.data.at[index, 'labels'] = temp_min.index(min(temp_min))

            # Group by 'labels' and calculate mean for each feature
            means = self.data.groupby('labels').mean()
            # updates to the centroid
            new_centroids = self.centroids.copy()

            for i in range(len(self.centroids)):
                cluster_data = self.data[self.data['labels'] == i]
                if not cluster_data.empty:
                    new_centroids.loc[i] = cluster_data[['mean_dist_day', 'mean_over_speed_perc']].mean()


            # check for convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            if centroid_shift < tolerance:
                print(f'Converged after {iteration+1} iterations.')
                break

            iteration += 1

        for label in np.unique(self.data['labels']):
            subset = self.data[self.data['labels'] == label]
            plt.scatter(subset['mean_over_speed_perc'], subset['mean_dist_day'], label=label)

        plt.legend()
        plt.xlabel('Mean Over Speed Percentile')
        plt.ylabel('Mean Distance Day')
        plt.savefig('k_means_native_with_sklearn_trained_full.png')


if __name__ == '__main__':
    kmeans = NativeKMeans()
    optimal_k = kmeans.find_optimal_centroids()
    optimal_k = 4
    kmeans.initialise_centroids(optimal_k)
    kmeans.train_kmeans_cluster()
    # print(kmeans.squared_euclidian_distance(3, 4, 2, 1))
    #  x1, y1 and x2 and y2
    # P(3,2) and Q(4, 1)
    # √ (3-4)^2 + (2-1)^2
    # √2 = 1.41
