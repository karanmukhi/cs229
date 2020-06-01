import utils
import matplotlib.pyplot as plt
import numpy as np
from locally_weighted_regression import LocallyWeightedLinearRegression


def p05b(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = utils.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)
    # Get MSE value on the validation set
    x_val, y_val = utils.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)
    mse = ((y_pred - y_val) ** 2).mean()

    plt.figure()
    plt.plot(x_train, y_train, "bx")
    plt.plot(x_val, y_pred, "ro")
    plt.savefig("output/p05b.png")


def p05c(tau_values, train_path, valid_path, test_path, pred_path):
    """Tune bandwidth parameter tau for LWR

    Args:
        tau_values: values of tau to try
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        test_path: Path to CSV file containing dataset for testing.
        pred_path: Path to save predictions of test data
    """
    # Load training set
    x_train, y_train = utils.load_dataset(train_path, add_intercept=True)
    best_tau = None
    best_mse = float("inf")
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        x_val, y_val = utils.load_dataset(valid_path, add_intercept=True)
        y_pred = model.predict(x_val)
        mse = ((y_pred - y_val) ** 2).mean()
        if mse < best_mse:
            best_mse = mse
            best_tau = tau
        plt.figure()
        plt.title("$tau = {}$".format(tau))
        plt.plot(x_train, y_train, "bx")
        plt.plot(x_val, y_pred, "ro")
        plt.savefig("output/tau_{}.png".format(tau))
    print("tau: {}, best_mse: {}".format(best_tau, best_mse))
    model = LocallyWeightedLinearRegression(best_tau)
    model.fit(x_train, y_train)
    x_test, y_test = utils.load_dataset(test_path, add_intercept=True)
    y_pred = model.predict(x_test)
    mse = ((y_pred - y_test) ** 2).mean()
    print("test mse: {}".format(mse))
    plt.figure()
    plt.title("$tau = {}$".format(tau))
    plt.plot(x_train, y_train, "bx")
    plt.plot(x_test, y_pred, "rx")
    plt.savefig("output/final.png")
