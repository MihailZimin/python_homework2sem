import algorithms
import data_preprocessing
import time
# import animation


if __name__ == "__main__":
    points, labels = data_preprocessing.get_data()
    X_train, y_train, X_test, y_test = data_preprocessing.train_test_split(
        points, labels, train_ratio=0.8, shuffle=True
    )
    KNN = algorithms.WeightedKNearestNeighbors()
    KNN.fit(X_train, y_train)
    print("fitted!")
    start = time.time()
    predictions = KNN.predict(X_test)
    end = time.time()
    print("accuracy: ", algorithms.accuracy(y_test, predictions))
    print("time: ", end - start)
    KNN_bad = algorithms.KNearestNeighbors()
    KNN_bad.fit(X_train, y_train)
    start = time.time()
    predictions = KNN_bad.predict(X_test)
    end = time.time()
    print("accuracy: ", algorithms.accuracy(y_test, predictions))
    print("time: ", end - start)
    # anim = animation.AnimationKNN(points, labels)
    # animate = anim.create_animation(KNN, X_test, y_test, cnt_of_frames=10)
    print("ok")
