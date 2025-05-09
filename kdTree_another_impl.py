import numpy as np
from typing import Callable
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
