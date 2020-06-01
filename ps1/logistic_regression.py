import numpy as np


from linear_model import LinearModel


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs, shape (m,n)
            y: Training example labels, shape (m,)
        """
        m, n = x.shape

        self.theta = np.zeros(n)
        g = lambda z: 1 / (1 + np.exp(-z))
        while True:
            theta = self.theta
            h = g(np.einsum("ij,j->i", x, theta))
            j_dash = -1 / m * np.einsum("i,ij->j", (y - h), x)
            H = (
                1
                / m
                * np.einsum(
                    "i,ijk->jk",
                    np.einsum("i,i->i", h, 1 - h),
                    np.einsum("ij,ik->ijk", x, x),
                )
            )
            H_inverse = np.linalg.inv(H)
            self.theta = theta - np.einsum("ij,j->i", H_inverse, j_dash)
            convergence = np.linalg.norm(self.theta - theta, ord=1)
            if convergence < self.eps:
                break

    def predict(self, x):
        """
        Make a prediction of x given the current model

        Args:
            x: Inputs of shape (m,n)
        Returns:
            y_hat: predictions for each input x_i, shape(m,)

        """

        g = lambda x: 1 / (1 + np.exp(-x))
        z = np.einsum("ij, j->i", x, self.theta)
        y_hat = g(z)

        return y_hat
