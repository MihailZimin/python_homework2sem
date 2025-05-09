from typing import Callable
import numpy as np
import heapq


class Node:
    def __init__(self, features: np.ndarray, ind: int) -> None:
        self.features = features
        self.ind = ind
        self.left = None
        self.right = None


class KDTree:
    def __init__(self, data: np.ndarray, dist_func: Callable) -> None:
        indexes = np.arange(data.shape[0])
        np.random.shuffle(indexes)
        self.data = data
        self.root = Node(data[indexes[0]], indexes[0])
        self.dist_func = dist_func
        self.dim = data.shape[1]
        for ind in indexes[1:]:
            node = Node(data[ind], ind)
            self._add_vertex(self.root, node)

    def _add_vertex(self, cur: Node, node: Node) -> None:
        depth = 0
        while True:
            if (cur.features[depth] >= node.features[depth]):
                if not cur.left:
                    cur.left = node
                    return
                cur = cur.left
            else:
                if not cur.right:
                    cur.right = node
                    return
                cur = cur.right
            depth = (depth + 1) % self.dim

    def find_k_nearest(self, cur: Node, target: np.ndarray, k) -> np.ndarray:
        heap = []

        def search(node: Node, depth: int = 0):
            if not node:
                return

            axis = depth % self.dim
            dist = self.dist_func(node.features, target)

            if len(heap) < k:
                heapq.heappush(heap, (-dist, node.ind))
            else:
                if dist < -heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (-dist, node.ind))

            if target[axis] < node.features[axis]:
                search(node.left, depth+1)
                if len(heap) < k or abs(node.features[axis] - target[axis]) < -heap[0][0]:
                    search(node.right, depth + 1)
            else:
                search(node.right, depth+1)
                if len(heap) < k or abs(target[axis] - node.features[axis]) < -heap[0][0]:
                    search(node.left, depth + 1)

        search(cur)
        return np.array([ind for (_, ind) in sorted(heap, reverse=True)])


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
        self.tree = KDTree(self.X_train, self.calc_distances)

    def nearest_indexes_vectorized(self, el: np.ndarray) -> np.ndarray:
        distances = np.array([self.calc_distances(el, i) for i in self.X_train])
        nearest_indices = np.argpartition(distances, self.n_neighbours)[:self.n_neighbours]
        return nearest_indices

    def nearest_indexes_tree(self, el: np.ndarray) -> np.ndarray:
        return self.tree.find_k_nearest(self.tree.root, el, self.n_neighbours)

    def predict(self, X_test: np.ndarray):
        predictions = []
        for el in X_test:
            if X_test.shape[0] > 1e4:
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
