import sklearn.datasets
from visualise_funtions import VisualiseData
import numpy as np
from typing import Union, Callable, Any
import sklearn
import matplotlib.pyplot as plt
from enum import StrEnum

plt.style.use("ggplot")
np.random.seed(42)


def get_data():
    return sklearn.datasets.make_moons(n_samples=10000, noise=0.3)
    # return sklearn.datasets.make_classification(n_samples=5000)


points, labels = sklearn.datasets.make_moons(n_samples=1000, noise=0.3)


class AxisNames(StrEnum):
    X = "X"
    Y = "Y"


class DiagramTypes(StrEnum):
    Violin = "Violin"
    Hist = "Hist"
    Boxplot = "Boxplot"


class ShapeMismatchError(Exception):
    pass


def visualize_distribution(
    points: np.ndarray,
    diagram_types: Union[DiagramTypes, list[DiagramTypes]],
    diagram_axis: Union[AxisNames, list[AxisNames]],
    path_to_save: str = "",
) -> None:
    vd = VisualiseData()
    rows = len(diagram_types)
    cols = len(diagram_axis)
    fig, axis = plt.subplots(rows, cols, figsize=(9, 6))

    if rows == 1 and cols == 1:
        axis = np.array([axis])
    elif rows == 1 or cols == 1:
        axis = axis.reshape(rows, cols)

    for col_ind, axis_name in enumerate(diagram_axis):
        for row_ind, diag_name in enumerate(diagram_types):
            cur_axis = axis[row_ind, col_ind]
            cur_data_axis = int(axis_name == 'Y')
            vd.plot_by_name(cur_axis, points[:, cur_data_axis], diag_name)
            if row_ind == 0:
                cur_axis.set_title(axis_name)

    if path_to_save:
        fig.savefig(path_to_save)


def get_boxplot_outliers(
    data: np.ndarray,
    key: Callable[[Any], Any],
) -> np.ndarray:
    vkey = np.vectorize(key)

    indexes = np.argsort(vkey(data))
    sorted_data = data[indexes]
    sz = data.shape[0]
    q1 = sorted_data[int(sz*0.25)]
    q3 = sorted_data[int(sz*0.75)]

    eps = (q3 - q1) * 1.5
    mask = (vkey(sorted_data) < vkey(q1 - eps)) | (vkey(sorted_data) > vkey(q3 + eps))
    return indexes[mask]


def train_test_split(
    features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if features.shape[0] != targets.shape[0]:
        raise ShapeMismatchError(
            (
                f"length of features: "
                f"{features.shape[0]} not equal length of targets: {targets.shape[0]}"
            )
        )
    unique_parts = np.unique(targets)
    indexes_of_parts = {part: np.where(targets == part)[0] for part in unique_parts}

    train_indexes = np.ndarray(shape=(1), dtype=int)
    test_indexes = np.ndarray(shape=(1), dtype=int)

    for part in unique_parts:
        indexes = indexes_of_parts[part]
        if shuffle:
            np.random.shuffle(indexes)
        cnt_of_train = int(indexes.shape[0] * train_ratio)
        train_indexes = np.hstack((train_indexes, indexes[:cnt_of_train]))
        test_indexes = np.hstack((test_indexes, indexes[cnt_of_train:]))
    if shuffle:
        np.random.shuffle(train_indexes)
        np.random.shuffle(test_indexes)
    return (features[train_indexes],
            targets[train_indexes],
            features[test_indexes],
            targets[test_indexes])


# visualize_distribution(points,
#     [DiagramTypes.Hist, DiagramTypes.Boxplot],
#     [AxisNames.Y],
#     path_to_save="visualisation"
# )
# plt.show()
