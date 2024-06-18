# load a dataset for learning
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from kneed import KneeLocator
from sklearn.preprocessing import MinMaxScaler


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
        self.df.dropna(axis=0, how='any')
        print('Describe:\n', self.df.describe())

        # Extract the numerical columns to scale
        numerical_cols = self.df.columns[:-1]

        scaler = MinMaxScaler()
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])

    def visualise_data(self):

        # Define colors for each species
        colors = {0: 'red', 1: 'purple', 2: 'orange'}

        # Create subplots for sepal and petal
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Sepal plot
        for species in self.df['species'].unique():
            subset = self.df[self.df['species'] == species]
            axes[0].scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
                            color=colors[species], label=self.iris.target_names[species])
        axes[0].set_xlabel('Sepal Length (cm)')
        axes[0].set_ylabel('Sepal Width (cm)')
        axes[0].set_title('Sepal Dimensions')
        axes[0].legend()

        # Petal plot
        for species in self.df['species'].unique():
            subset = self.df[self.df['species'] == species]
            axes[1].scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                            color=colors[species], label=self.iris.target_names[species])
        axes[1].set_xlabel('Petal Length (cm)')
        axes[1].set_ylabel('Petal Width (cm)')
        axes[1].set_title('Petal Dimensions')
        axes[1].legend()

        # save the plot
        plt.savefig('kmeans_petals_and_sepals.png')
        plt.clf()
        self.df.drop('species', axis=1, inplace=True)

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


if __name__ == '__main__':
    kmeans = LearnKMeans()
    kmeans.visualise_data()
    kmeans.prepare_data()
    optimal_k = kmeans.select_k_with_elbow()
    print('Optimal k:', optimal_k)

