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
        self.n_neighbours = n_neighbors
        self.calc_distances = calc_distances

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.tree = kdtree.KDTree(self.X_train, self.calc_distances)
        print("fitted!")

    def nearest_indexes_vectorized(self, el: np.ndarray) -> np.ndarray:
        distances = np.array([self.calc_distances(el, i) for i in self.X_train])
        nearest_indices = np.argpartition(distances, self.n_neighbours)[:self.n_neighbours]
        return nearest_indices

    def nearest_indexes_tree(self, el: np.ndarray) -> np.ndarray:
        return self.tree.find_k_nearest(el, self.n_neighbours)

    def predict(self, X_test: np.ndarray):
        predictions = []
        for el in X_test:
            if X_test.shape[0] >= 0:
                indexes = self.nearest_indexes_tree(el)
            else:
                indexes = self.nearest_indexes_vectorized(el)
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
            distances = np.array([self.calc_distances(el, i) for i in self.X_train])
            indexes = np.argpartition(distances, self.n_neighbours)[:self.n_neighbours]
            if X_test.shape[0] > 1e4:
                indexes = self.nearest_indexes_tree(el)
            else:
                indexes = self.nearest_indexes_vectorized(el)
            h = np.max(distances[indexes])
            radiuses.append(h)
        radiuses = np.array(radiuses)
        return correctness, radiuses


def accuracy(
    true_targets: np.ndarray,
    prediction: np.ndarray,
) -> float:
    return np.sum(true_targets == prediction) / prediction.shape[0]
