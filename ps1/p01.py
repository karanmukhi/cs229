import utils
import numpy as np
from logistic_regression import LogisticRegression
from gaussian_discriminant_analysis import GaussianDiscriminantAnalysis


def p01b(train_path, eval_path, pred_path):
    """Logistic regression with Newton's Method

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Train classifier
    x_train, y_train = utils.load_dataset(train_path, add_intercept=True)
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    # Validate classifier
    x_val, y_val = utils.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)
    utils.plot(x_val, y_val, model.theta, "{}.png".format(pred_path))
    np.savetxt(pred_path, y_pred)


def p01e(train_path, eval_path, pred_path):
    """Gaussian discriminant analysis

    Args:
        train_path: path to csv file for training data
        eval_path: path to csv file for validation data
        pred_path: path to save predictions
    """

    x_train, y_train = utils.load_dataset(train_path, add_intercept=False)
    model = GaussianDiscriminantAnalysis()
    model.fit(x_train, y_train)
    x_val, y_val = utils.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_val)
    utils.plot(x_val, y_val, model.theta, "{}.png".format(pred_path))

    # Use np.savetxt to save outputs from validation set to pred_path
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***
