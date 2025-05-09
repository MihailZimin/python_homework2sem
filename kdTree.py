import numpy as np
from typing import Callable
import heapq


class KDTree:
    def __init__(self, data: np.ndarray, dist_func: Callable) -> None:
        self.points = data
        self.dim = self.points.shape[1]
        self.size = data.shape[0]
        self.dist_func = dist_func
        self.root = self.build_tree(np.arange(self.size), depth=0)

    def build_tree(self, indexes: np.ndarray, depth: int):
        if indexes.shape[0] == 0:
            return None

        axis = depth % self.dim
        sorted_indexes = indexes[np.argsort(self.points[indexes, axis])]
        mid = sorted_indexes.shape[0] // 2

        return {
            "index": sorted_indexes[mid],
            "point": self.points[sorted_indexes[mid]],
            "left": self.build_tree(sorted_indexes[:mid], depth + 1),
            "right": self.build_tree(sorted_indexes[mid+1:], depth + 1)
        }

    def find_k_nearest(self, target: np.ndarray, k) -> np.ndarray:
        heap = []

        def search(cur, depth: int = 0):
            if not cur:
                return

            dist = self.dist_func(cur["point"], target)

            if len(heap) < k:
                heapq.heappush(heap, (-dist, cur["index"]))
            elif dist < -heap[0][0]:
                heapq.heappop(heap)
                heapq.heappush(heap, (-dist, cur["index"]))

            axis = depth % self.dim
            if target[axis] < cur["point"][axis]:
                search(cur["left"], depth + 1)
                if len(heap) < k or abs(cur["point"][axis] - target[axis]) < -heap[0][0]:
                    search(cur["right"], depth + 1)
            else:
                search(cur["right"], depth+1)
                if len(heap) < k or abs(target[axis] - cur["point"][axis]) < -heap[0][0]:
                    search(cur["left"], depth + 1)

        search(self.root)
        return np.array([ind for (_, ind) in sorted(heap, reverse=True)])
