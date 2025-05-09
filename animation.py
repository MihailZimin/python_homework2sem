import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from algorithms import WeightedKNearestNeighbors
from functools import partial


class AnimationKNN:
    def __init__(self, points, labels):
        self.figure, self.axis = plt.subplots(figsize=(16, 9))
        self.scatter = self.axis.scatter(points[:, 0], points[:, 1], c=labels, cmap="viridis")
        self.circles = []

    def update_frame(
            self,
            frame_id: int,
            X_test: np.ndarray,
            correctness: np.ndarray,
            radiuses: np.ndarray
            ) -> tuple[plt.Axes, plt.Circle]:
        for circle in self.circles:
            circle.remove()
        self.circles.clear()

        color = "green" if correctness[frame_id] else "red"

        circle = plt.Circle(
            X_test[frame_id],
            radius=radiuses[frame_id],
            fill=True,
            color=color,
            linewidth=2,
            alpha=0.5
        )
        self.circles.append(circle)
        self.axis.add_patch(circle)

        return self.scatter, self.circles[-1]

    def init_func(self):
        return self.scatter,

    def create_animation(
            self,
            knn: WeightedKNearestNeighbors,
            true_targets: np.ndarray,
            true_labels: np.ndarray,
            cnt_of_frames: int,
            path_to_save="animation.gif"
            ) -> FuncAnimation:
        correctness, radiuses = knn.get_info(true_targets, true_labels)
        cnt_of_frames = min(cnt_of_frames, true_targets.shape[0])

        animation = FuncAnimation(
            self.figure,
            partial(
                self.update_frame, X_test=true_targets, correctness=correctness, radiuses=radiuses
            ),
            frames=cnt_of_frames,
            interval=500,
            blit=True,
            init_func=self.init_func,
            repeat=False
        )

        try:
            animation.save(path_to_save, writer='pillow')
        except Exception as e:
            print(f"Ошибка при сохранении анимации: {e}")

        return animation
