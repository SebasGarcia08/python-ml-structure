from sklearn.datasets import load_iris


def get_iris():
    X, y = load_iris(return_X_y=True)
    return X, y