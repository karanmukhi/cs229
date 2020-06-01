import utils
import numpy as np
from poisson_regression import PoissonRegression


def p03(train_path, eval_path, pred_path):
    """Poisson regression with gradient ascent

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """

    x_train, y_train = utils.load_dataset(train_path)
    model = PoissonRegression(max_iter=10000, step_size=1e-7, eps=1e-7)
    model.fit(x_train, y_train)

    x_val, y_val = utils.load_dataset(eval_path)
    y_pred = model.predict(x_val)
    np.savetxt(pred_path, y_pred)
