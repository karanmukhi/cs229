import numpy as np
import utils
from linear_model import LinearModel


class PoissonRegression(LinearModel):
    """Poisson regression with gradient ascent for fitting."""

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood of Poisson regression.
        Args:
            x: Training example inputs, shape(m,n)
            y: Training example labels, shape(m,).
        """
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
        g = lambda z: np.exp(z)
        for i in range(self.max_iter):
            theta = self.theta
            z = np.einsum("ij,j->i", x, self.theta)
            self.theta = self.theta + self.step_size * (1 / m) * np.einsum(
                "i,ij->j", y - g(z), x
            )
            if np.linalg.norm(self.theta - theta, ord=1) < self.eps:
                break

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape(m,n)
        Returns:
            Predictions for inputs shape(m,)
        """
        return np.exp(np.einsum("j,ij->i", self.theta, x))
