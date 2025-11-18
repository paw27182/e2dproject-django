import os
import pathlib
import numpy as np
from datetime import datetime
import codecs
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.svm import LinearSVR
# from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.neural_network import MLPRegressor
# import lightgbm as lgb

import warnings

matplotlib.use('Agg')  # バックエンドを指定
scikit_version = sklearn.__version__
pd.options.display.float_format = "{:.2f}".format
warnings.filterwarnings("ignore")


def reg_compare_algorithms(command, df, output_dir):
    print(f"{command}")

    launch_time = datetime.now()

    # tc_time = "0:00:00"
    # return f"NG.  {command} failed.", tc_time  # debug

    # check "target" column
    if "target" not in df.columns:
        alert = 'NG', 'error: data must have "target" column.'
        tc_time = "0:00:00"
        return alert, tc_time

    # data reading
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # standardization
    scaler = StandardScaler()
    scaler.fit(X_train.astype(np.float64))

    X_train = scaler.transform(X_train.astype(np.float64))
    X_test = scaler.transform(X_test.astype(np.float64))

    # define algorithms
    allAlgorithms = {
        "LinearSVR": LinearSVR(),
        # "SVR": SVR(),
        "SGDRegressor": SGDRegressor(),

        "LinearRegression": LinearRegression(),
        # "Ridge": Ridge(),
        # "Lasso": Lasso(),

        "KNeighborsRegressor": KNeighborsRegressor(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        # "MLPRegressor": MLPRegressor(),
        # "LGBMRegressor": lgb.LGBMRegressor(objective=None),
    }

    porec = list()

    scores = {}  # score dictionary

    kfold_cv = KFold(n_splits=3, shuffle=True)  # default 5

    for name, algorithm in allAlgorithms.items():

        try:
            start_time = datetime.now()  # 2021.3.5
            reg = algorithm

            reg.fit(X_train, y_train)

            if hasattr(reg, "score"):  # score()メソッド持つアルゴリズムは交差検証使用
                score_train = np.mean(cross_val_score(reg, X_train, y_train, cv=kfold_cv))  # cv = kfold_cv or 整数値指定
            else:
                score_train = reg.score(X_train, y_train)

            score_test = reg.score(X_test, y_test)

            elapsed_time = (datetime.now() - start_time).total_seconds()
            print("{}\t{}".format(datetime.now() - start_time, name))

            if (score_train >= 0) and (score_test >= 0):
                score = list()
                score.append(score_train,)
                score.append(score_test)
                score.append(elapsed_time)

                scores[name] = score  # add score to dictionary

        except Exception as err:
            score = [0, 0, 0]  # error occurred
            scores[name] = score  # add score to dictionary

    #     break

    scores_sorted = sorted(scores.items(), key=lambda x: x[1][1], reverse=True)  # sort by dict value

    msg = "algorithm, R^2(train), R^2(test), elapsed_time"
    print(msg)
    porec.append(msg)

    for s in ["{},{:.4f},{:.4f},{:.4f}".format(
            item[0],     # algorithm
            item[1][0],  # R^2(train)
            item[1][1],  # R^2(test)
            item[1][2]) for item in scores_sorted]:
        porec.append(s)
        print(s)

    # algorithm score - train score vs test score
    scores2 = sorted(scores.items(), key=lambda x: x[1][1], reverse=False)  # sort by dict value

    x_tr = [item[1][0] for item in scores2]
    x_te = [item[1][1] for item in scores2]

    plt.style.use("ggplot")

    plt.figure(figsize=(5, 5))
    plt.xlim(0., 1.0)
    plt.ylim(0., 1.0)
    plt.xlabel("X_train score")
    plt.ylabel("X_test score")
    plt.title("R^2 score")

    plt.scatter(x_tr, x_te)

    xx = np.linspace(0., 1.0, 10)
    plt.plot(xx, xx, color="gray", linestyle="dashed")  # Upper Left:underfitting ,Lower Right: overfitting
    plt.savefig(os.path.join(output_dir, "algorithmscore.png"))
    plt.close()

    # algorithm score - test score

    scores2 = sorted(scores.items(), key=lambda x: x[1][1], reverse=False)  # sort by dict value

    names = [item[0] for item in scores2]
    sc = [item[1][1] for item in scores2]

    plt.style.use("ggplot")
    plt.figure(figsize=(5, 10))
    plt.title("algorithm TEST score R^2")
    plt.barh(range(len(names)), sc, align="center", tick_label=names)
    plt.savefig(os.path.join(output_dir, "testscore_barchart.png"), bbox_inches="tight")
    plt.close()

    # algorithm score - elapsed_time
    scores2 = sorted(scores.items(), key=lambda x: x[1][1], reverse=False)  # sort by dict value

    names = [item[0] for item in scores2]
    sc = [item[1][2] for item in scores2]  # for elapsed_time

    plt.style.use("ggplot")
    plt.figure(figsize=(5, 10))
    plt.title("elapsed time")
    plt.barh(range(len(names)), sc, align="center", tick_label=names, color="b")
    plt.savefig(os.path.join(output_dir, "elapsedtime_barchart.png"), bbox_inches="tight")
    plt.close()

    # r2 vs. elapsed time
    scores2 = sorted(scores.items(), key=lambda x: x[1][1], reverse=False)  # sort by dict value

    elapsed_time = [item[1][2] for item in scores2]  # for elapsed_time
    x_te = [item[1][1] for item in scores2]

    plt.figure(figsize=(5, 5))
    plt.xlabel("elapsed time")
    plt.ylabel("R^2(test score)")
    plt.title("R^2 vs. elapsed time")

    plt.scatter(elapsed_time, x_te, color="blue")

    # annotation
    for i, label in enumerate(names):
        plt.annotate(label, (elapsed_time[i], x_te[i]))

    plt.savefig(os.path.join(output_dir, "r2_elapsedtime.png"))
    plt.close()

    # create result file
    result_file = os.path.join(output_dir, "result.txt")

    if pathlib.Path(result_file).exists():
        pathlib.Path(result_file).unlink()

    result_str = ""
    for line in porec:
        result_str += line + "\n"

    with codecs.open(result_file, "w", "utf-8") as rf:
        print(result_str, file=rf)

    tc_time = "{}".format(datetime.now() - launch_time)
    match = re.search('[0-9]{1,2}:[0-9]{2}:[0-9]{2}', tc_time)  # 0:00:00
    tc_time = match.group(0)

    alert = "OK", "Done."

    return alert, tc_time


if __name__ == "__main__":
    command = "reg_compare_algorithms"
    data_file = r"W:\dataset\reg\Boston\housing.csv"
    df = pd.read_csv(data_file, encoding="utf-8", header=0)
    output_dir = "../../../tests/unittest/output"

    alert, tc_time = reg_compare_algorithms(command, df, output_dir)
    print(f"{alert= } {tc_time= }")
