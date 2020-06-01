import numpy as np
import utils
from linear_model import LinearModel


class GaussianDiscriminantAnalysis(LinearModel):
    """Gaussian Discriminant Analysis."""

    def fit(self, x, y):
        """Find parameters given training set x,y

        Args:
            x: Training data inputs, shape(m,n)
            y: Training data labels, shape(m,n)
        Returns:
            theta: GDA model parameters.
        """
        m, n = x.shape
        phi = np.count_nonzero(y == 1) / m
        x_0 = np.array([x[i] for i in range(len(x)) if y[i] == 0])
        x_1 = np.array([x[i] for i in range(len(x)) if y[i] == 1])
        mu_0 = np.sum(x_0, axis=0) / len(x_0)
        mu_1 = np.sum(x_1, axis=0) / len(x_1)
        sigma = (
            np.einsum("ij,ik->jk", x_0 - mu_0, x_0 - mu_0)
            + np.einsum("ij,ik->jk", x_1 - mu_1, x_1 - mu_1)
        ) / m
        sigma_inv = np.linalg.inv(sigma)
        theta_0 = 1 / 2 * (
            np.einsum("i,ij,j", mu_0, sigma_inv, mu_0)
            - np.einsum("i,ij,j", mu_1, sigma_inv, mu_1)
        ) - np.log((1 - phi) / phi)
        theta = np.einsum("ij,j->i", sigma_inv, mu_1 - mu_0)
        theta = np.hstack([theta_0, theta])
        self.theta = theta
        return theta

    def predict(self, x):
        """Make predictions on data x

        Args:
            x: Inputs of shape (m,n)
        Returns:
            Outputs of shape (m,)
        """
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        x = utils.add_intercept(x)
        probs = sigmoid(np.einsum("ij,j->i", x, self.theta))
        preds = (probs >= 0.5).astype(np.int)
        return preds
