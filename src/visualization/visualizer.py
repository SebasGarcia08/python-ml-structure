import plotly
import plotly.graph_objs as go
import json
from sklearn.decomposition import PCA
from src.data import get_iris


class KNNVisualizer:

    def __init__(self):
        self.json_figure = None

    def plot_feature_space_iris(self):
        data = self._compute_pca_iris()

        data = [
            go.Bar(
                x=data[0],
                y=data[1]
            )
        ]

        self.json_figure = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    def _compute_pca_iris(self):
        X, _ = get_iris()
        pca = PCA(n_components=2)
        iris_2d = pca.fit_transform(X)
        return iris_2d
