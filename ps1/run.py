from p01 import p01b, p01e
from p02 import p02cde
from p03 import p03
from p05 import p05b, p05c


def main():
    p01b(
        train_path="./data/ds1_train.csv",
        eval_path="./data/ds1_valid.csv",
        pred_path="./output/p01b_pred_1.txt",
    )

    p01b(
        train_path="./data/ds2_train.csv",
        eval_path="./data/ds2_valid.csv",
        pred_path="./output/p01b_pred_2.txt",
    )

    p01e(
        train_path="./data/ds1_train.csv",
        eval_path="./data/ds1_valid.csv",
        pred_path="output/p01e_pred_1.txt",
    )

    p01e(
        train_path="./data/ds2_train.csv",
        eval_path="./data/ds2_valid.csv",
        pred_path="output/p01e_pred_2.txt",
    )

    # p02cde(
    #     train_path="./data/ds3_train.csv",
    #     valid_path="./data/ds3_valid.csv",
    #     test_path="./data/ds3_test.csv",
    #     pred_path="output/p02X_pred.txt",
    # )

    p03(
        train_path="./data/ds4_train.csv",
        eval_path="./data/ds4_valid.csv",
        pred_path="output/p03d_pred.txt",
    )

    p05b(0.5, train_path="./data/ds5_train.csv", eval_path="./data/ds5_valid.csv")

    p05c(
        tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
        train_path="./data/ds5_train.csv",
        valid_path="./data/ds5_valid.csv",
        test_path="./data/ds5_test.csv",
        pred_path="output/p05c_pred.txt",
    )


if __name__ == "__main__":
    main()
    print("Completed")
