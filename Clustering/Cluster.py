import os

os.system("cls")
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        self.X = X
        n_sampels, n_feratures = X.shape
        rand_indx = np.random.choice(n_sampels, self.n_clusters, replace=False)
        self.centroids = [self.X[indx] for indx in rand_indx]

        for _ in range(self.max_iters):
            self.labels = self._make_clusters(self.centroids)

            old_centroids = self.centroids
            self.centroids = self._update_centroids(self.labels)
            if self._is_convenged(old_centroids, self.centroids):
                break

    def _make_clusters(self, centroids):
        clusters = [[] for _ in range(self.n_clusters)]
        for idx,val in enumerate(self.X):
            centroids_idx = self._nearest_centroid(val)
            clusters[centroids_idx].append(idx)
            
        return clusters
            
    def _nearest_centroid(self, sample):
        distances = [euclidean_distance(sample, point) for point in self.centroids]
        nearest_index = np.argmin(distances)
        return nearest_index

    def _update_centroids(self, labels):
        n_samples, n_features = self.X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        for idx, cluster in enumerate(labels):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[idx] = cluster_mean
        return centroids

    def _is_convenged(self, old_centroids, new_centroids, tol=1e-4):
        distances = [euclidean_distance(old_centroids[i], new_centroids[i]) for i in range(self.n_clusters)]
        return sum(distances) < tol
    
    def predict(self, X):
        predicted_list = []
        for sample in X:
            predicted_list.append(self._nearest_centroid(sample))                        
        return predicted_list
    
class HierarchicalClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.labels = None

    def fit(self, X):
        self.X = X
        n_samples = X.shape[0]
        self.labels = np.arange(n_samples)

        while len(np.unique(self.labels)) > self.n_clusters:
            min_dist = np.inf
            merge_idx = None

            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = self._distance(X[i], X[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_idx = (i, j)

            self._merge_clusters(merge_idx)

    def _distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _merge_clusters(self, merge_idx):
        cluster1, cluster2 = merge_idx
        mask = np.logical_or(self.labels == cluster1, self.labels == cluster2)
        self.labels[mask] = np.max(self.labels) + 1

    def predict(self, X):
        predicted_list = []
        for sample in X:
            distances = [self._distance(sample, self.X[i]) for i in range(len(self.X))]
            nearest_index = np.argmin(distances)
            predicted_list.append(self.labels[nearest_index])
        return predicted_list
