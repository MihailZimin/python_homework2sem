import algorithms
import data_preprocessing
import time
import animation
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")
np.random.seed(42)


# 1) считывание и обработка данных

points, labels = data_preprocessing.get_data()

# визуализация графиков
data_preprocessing.visualize_distribution(
    points,
    diagram_types=["Hist", "Violin"],
    diagram_axis=["X", "Y"],
    path_to_save="visuals/graphics"
    )

# находим индексы выбросов
outliers = data_preprocessing.get_boxplot_outliers(
    points,
    key=lambda point: np.sqrt(np.sum(point**2))
)

# убираем выбросы
mask = np.ones(points.shape[0], dtype=bool)
mask[outliers] = False
points = points[mask]
labels = labels[mask]


# разбиваем выборку
X_train, y_train, X_test, y_test = data_preprocessing.train_test_split(
    points, labels, train_ratio=0.8, shuffle=True
)

# используем взвешенный KNN
print("взвешенный KNN:")
KNN = algorithms.WeightedKNearestNeighbors()
KNN.fit(X_train, y_train)
print("fitted!\n")
start = time.time()
predictions = KNN.predict(X_test)
end = time.time()
print("accuracy: ", algorithms.accuracy(y_test, predictions))
print("time: ", end - start, end="\n\n\n")

# используем обычный KNN
print("обычный KNN:")
KNN_default = algorithms.KNearestNeighbors()
KNN_default.fit(X_train, y_train)
start = time.time()
predictions = KNN_default.predict(X_test)
end = time.time()
print("accuracy: ", algorithms.accuracy(y_test, predictions))
print("time: ", end - start, end="\n\n\n")

# визуализируем алгоритм
anim = animation.AnimationKNN(points, labels)
animate = anim.create_animation(
    KNN,
    X_test,
    y_test,
    cnt_of_frames=10,
    path_to_save="visuals/animation.gif"
)
print("ok!")
