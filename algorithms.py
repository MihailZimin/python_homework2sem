from typing import Callable
import numpy as np
import kdTree as kdtree


def euclidean_dist(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.sum((x-y)**2))


class WeightedKNearestNeighbors:
    def __init__(
        self,
        n_neighbors: int = 20,
        calc_distances: Callable = euclidean_dist,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.calc_distances = calc_distances

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.tree = kdtree.KDTree(self.X_train, self.calc_distances)
        self.dim = X_train.shape[1]

    def nearest_indexes_tree(self, el: np.ndarray) -> np.ndarray:
        return self.tree.find_k_nearest(el, self.n_neighbors)

    def nearest_indexes_vectorized(self, el: np.ndarray) -> np.ndarray:
        distances = np.array([self.calc_distances(el, i) for i in self.X_train])
        nearest_indices = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
        return nearest_indices

    def predict(self, X_test: np.ndarray):
        predictions = []
        for el in X_test:
            if self.dim >= 10:
                indexes = self.nearest_indexes_vectorized(el)
            else:
                indexes = self.nearest_indexes_tree(el)
            features = self.X_train[indexes]
            distances = np.array([self.calc_distances(el, i) for i in features])
            h = np.max(distances)
            weights = 3 / 4 * (1 - (distances / h) ** 2)
            nearest_labels = self.y_train[indexes]

            unique_parts = np.unique(self.y_train)
            part_weights = {part: 0 for part in unique_parts}

            for part, weight in zip(nearest_labels, weights):
                part_weights[part] += weight

            predicted_label = max(part_weights.items(), key=lambda x: x[1])[0]
            predictions.append(predicted_label)

        return np.array(predictions)

    def get_info(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        correctness = self.predict(X_test) == y_test
        radiuses = []
        for el in X_test:
            if self.dim >= 10:
                indexes = self.nearest_indexes_vectorized(el)
            else:
                indexes = self.nearest_indexes_tree(el)
            distances = np.array([self.calc_distances(el, i) for i in self.X_train[indexes]])
            h = np.max(distances)
            radiuses.append(h)
        radiuses = np.array(radiuses)
        return correctness, radiuses


class KNearestNeighbors:
    def __init__(
        self,
        n_neighbors: int = 20,
        calc_distances: Callable = euclidean_dist
    ) -> None:
        self.n_neighbors = n_neighbors
        self.calc_distances = calc_distances

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.tree = kdtree.KDTree(X_train, self.calc_distances)
        self.dim = X_train.shape[1]

    def nearest_indexes_tree(self, el: np.ndarray) -> np.ndarray:
        return self.tree.find_k_nearest(el, self.n_neighbors)

    def nearest_indexes_vectorized(self, el: np.ndarray) -> np.ndarray:
        distances = np.array([self.calc_distances(el, i) for i in self.X_train])
        nearest_indices = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
        return nearest_indices

    def predict(self, X_test: np.ndarray):
        predictions = []
        for el in X_test:
            if self.dim >= 10:
                indexes = self.nearest_indexes_vectorized(el)
            else:
                indexes = self.nearest_indexes_tree(el)
            nearest_labels = self.y_train[indexes]
            unique_vals, unique_inds = np.unique(nearest_labels, return_counts=True)
            indmax = np.argmax(unique_inds)
            predictions.append(unique_vals[indmax])

        return np.array(predictions)

    def get_info(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        correctness = self.predict(X_test) == y_test
        radiuses = []
        for el in X_test:
            if self.dim >= 10:
                indexes = self.nearest_indexes_vectorized(el)
            else:
                indexes = self.nearest_indexes_tree(el)
            distances = np.array([self.calc_distances(el, i) for i in self.X_train[indexes]])
            h = np.max(distances)
            radiuses.append(h)
        radiuses = np.array(radiuses)
        return correctness, radiuses


def accuracy(
    true_targets: np.ndarray,
    prediction: np.ndarray,
) -> float:
    return np.sum(true_targets == prediction) / prediction.shape[0]
