import numpy as np
import utils
import matplotlib.pyplot as plt
from linear_model import LinearModel


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LinearModel, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR

        Args:
            x: training set input data
            y: training set label data
        """
        self.x = x
        self.y = y

    def predict(self, x):
        """Predict input

        Args:
            x: input data to predict shape(m,n)
        Returns:
            predictions
        """

        m, n = x.shape
        y_pred = np.zeros(m)
        x_xi_norm = np.linalg.norm(self.x[None] - x[:, None], axis=2) ** 2
        weights = np.exp(-x_xi_norm / (2 * self.tau ** 2))
        for i, weight in enumerate(weights):
            W = np.diag(weight)
            inv = np.einsum("ij,ik->jk", self.x, W)
            inv = np.einsum("ij,jk->ik", inv, self.x)
            inv = np.linalg.inv(inv)
            theta = np.einsum("ij,j->i", W, self.y)
            theta = np.einsum("ji,j->i", self.x, theta)
            theta = np.einsum("ij,j->i", inv, theta)
            y_pred[i] = x[i].dot(theta)
        return y_pred
