import utils
import numpy as np
from logistic_regression import LogisticRegression

WILDCARD = "X"


def p02cde(train_path, valid_path, test_path, pred_path):
    """Logistic regression with Newton's Method

    Args:
        train_path: Path to CSV file containing dataset for training.
        validation_path: Path to CSV file containing dataset for evaluation.
        test_path: Path to CSV file containing dataset for testing.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, "c")
    pred_path_d = pred_path.replace(WILDCARD, "d")
    pred_path_e = pred_path.replace(WILDCARD, "e")

    # Part (c)
    # Train classifier
    x_train, y_train = utils.load_dataset(train_path, label_col="t", add_intercept=True)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # Validate classifier
    x_test, y_test = utils.load_dataset(valid_path, label_col="t", add_intercept=True)
    t_pred = model.predict(x_test)
    utils.plot(x_test, y_test, model.theta, "{}.png".format(pred_path_c))
    np.savetxt(pred_path_c, t_pred)

    # Part (d)
    x_train, y_train = utils.load_dataset(test_path, label_col="y", add_intercept=True)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # Validate classifier
    x_test, y_test = utils.load_dataset(test_path, label_col="t", add_intercept=True)
    y_pred = model.predict(x_test)
    utils.plot(x_test, y_test, model.theta, "{}.png".format(pred_path_d))
    np.savetxt(pred_path_d, y_pred)

    # Part (e) find corrections
    x_val, y_val = utils.load_dataset(valid_path, label_col="y", add_intercept=True)
    x_in_V = [x_train[i] for i in len(x_train) if y_train == 1]
    h = model.predict(x_in_V)
    alpha = np.mean(h)
