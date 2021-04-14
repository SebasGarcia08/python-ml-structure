from base_model import SupervisedModel
import numpy as np


class KNN(SupervisedModel):
    """

    """

    def __init__(self, k):
        super(KNN, self).__init__()
        self.k = k
        self.data = None
        self.classes = None

    def fit(self, X, y):
        self.data = X
        self.classes = y

    def predict(self, X):
        preds = []
        for dtp in X:
            preds.append(self.step_predict(dtp))
        return np.array(preds)

    def step_fit(self, x, y):
        pass

    def step_predict(self, x):
        idxs = self.get_knn(x)
        candidates_classes = [self.classes[idx] for idx in idxs]

        counts = np.bincount(candidates_classes)
        return np.argmax(counts)

    def get_knn(self, dtp):
        distances = np.linalg.norm(self.data-dtp, axis=1)
        nearest_n = np.argsort(distances)[:self.k]
        return nearest_n


if __name__ == "__main__":
    from src.data import get_iris

    X, y = get_iris()

    knn = KNN(k=3)
    (train_x, train_y), (test_x, test_y) = (X[100:, :], y[100:]), (X[100:, :], y[:100])
    knn.fit(train_x, train_y)
    print(knn.predict(test_x))