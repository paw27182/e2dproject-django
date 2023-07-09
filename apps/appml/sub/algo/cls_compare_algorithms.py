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
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics

from sklearn.svm import LinearSVC
# from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier
# import lightgbm as lgb

from sklearn.metrics import log_loss

matplotlib.use('Agg')  # バックエンドを指定
scikit_version = sklearn.__version__
pd.options.display.float_format = "{:.2f}".format


def cls_compare_algorithms(command, df, output_dir):
    print(f"{command}")

    launch_time = datetime.now()
    tc_time = "0:00:00"

    # return f"NG.  {command} failed.", tc_time  # debug

    # check "class" column
    if "class" not in df.columns:
        alert = 'NG', 'error: data must have "class" column.'
        return alert, tc_time

    # data reading
    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # standardization
    scaler = StandardScaler()
    scaler.fit(X_train.astype(np.float64))

    X_train = scaler.transform(X_train.astype(np.float64))
    X_test = scaler.transform(X_test.astype(np.float64))

    # define algorithms
    allAlgorithms = {
        "LinearSVC": LinearSVC(),
        # "SVC": SVC(gamma='auto'),  # default 'scale'
        "SGDClassifier": SGDClassifier(),

        "LogisticRegression": LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000),
        # "RidgeClassifier": RidgeClassifier(),

        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        # "MLPClassifier": MLPClassifier(),
        # "LGBMClassifier": lgb.LGBMClassifier(),
    }

    porec = list()

    scores = {}  # score dictionary

    kfold_cv = KFold(n_splits=3, shuffle=True)  # default 5

    for name, algorithm in allAlgorithms.items():
        try:
            start_time = datetime.now()  # 2021.3.5
            clf = algorithm

            clf.fit(X_train, y_train)

            if hasattr(clf, "score"):  # score()メソッド持つアルゴリズムは交差検証使用
                score_train = np.mean(cross_val_score(clf, X_train, y_train, cv=kfold_cv))  # cv = kfold_cv or 整数値指定
            else:
                score_train = clf.score(X_train, y_train)

            pred = clf.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, pred)
            precision = metrics.precision_score(y_test, pred, average="weighted")
            recall = metrics.recall_score(y_test, pred, average="weighted")
            f1 = metrics.f1_score(y_test, pred, average="weighted")
            kappa = cohen_kappa_score(y_test, pred, weights='quadratic')

            # calculate logloss
            try:
                pred_proba = clf.predict_proba(X_test)
                logloss = log_loss(y_test, pred_proba)
            except Exception as err:  # unable to calculate the probability
                logloss = 0.0

            score = [score_train, accuracy, precision, recall, f1, kappa, logloss]

            elapsed_time = (datetime.now() - start_time).total_seconds()
            print("{}\t{}".format(datetime.now() - start_time, name))

            score.append(elapsed_time)

            scores[name] = score  # add score to dictionary

        except Exception as err:
            score = [0, 0, 0, 0, 0, 0, 0, 0]  # error occurred
            scores[name] = score  # add score to dictionary'

    #     break

    scores_sorted = sorted(scores.items(), key=lambda x: x[1][1], reverse=True)  # sort by dict value

    msg = "algorithm, acc(train), acc(test), precision, recall, f1, kappa, logloss, elapsed_time"
    print(msg)
    porec.append(msg)

    for s in ["{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
            item[0],     # algorithm
            item[1][0],  # acc(train)
            item[1][1],  # acc(test)
            item[1][2],  # precision
            item[1][3],  # recall
            item[1][4],  # f1
            item[1][5],  # kappa
            item[1][6],  # logloss
            item[1][7]) for item in scores_sorted]:
        porec.append(s)
        print(s)

    # algorithm score - train score vs test score
    scores2 = sorted(scores.items(), key=lambda x: x[1][1], reverse=False)  # sort by dict value

    x_tr = [item[1][0] for item in scores2]
    x_te = [item[1][1] for item in scores2]

    plt.style.use("ggplot")

    plt.figure(figsize=(5, 5))
    plt.xlim(0.5, 1.0)
    plt.ylim(0.5, 1.0)
    plt.xlabel("X_train score")
    plt.ylabel("X_test score")
    plt.title("accuracy")

    plt.scatter(x_tr, x_te)

    xx = np.linspace(0.5, 1.0, 10)
    plt.plot(xx, xx, color="gray", linestyle="dashed")  # Upper Left:underfitting ,Lower Right: overfitting

    plt.savefig(os.path.join(output_dir, "algorithmscore.png"))
    plt.close()

    # algorithm score - test score
    scores2 = sorted(scores.items(), key=lambda x: x[1][1], reverse=False)  # sort by dict value

    names = [item[0] for item in scores2]
    sc = [item[1][1] for item in scores2]  # for test data

    plt.style.use("ggplot")
    plt.figure(figsize=(5, 10))
    plt.title("algorithm TEST score")
    plt.barh(range(len(names)), sc, align="center", tick_label=names)

    plt.savefig(os.path.join(output_dir, "testscore_barchart.png"), bbox_inches="tight")
    # plt.show()
    plt.close()

    # algorithm score - elapsed_time
    scores2 = sorted(scores.items(), key=lambda x: x[1][1], reverse=False)  # sort by dict value

    names = [item[0] for item in scores2]
    sc = [item[1][7] for item in scores2]  # for elapsed_time  2021.3.5

    plt.style.use("ggplot")
    plt.figure(figsize=(5, 10))
    plt.title("elapsed time")
    plt.barh(range(len(names)), sc, align="center", tick_label=names, color="b")

    plt.savefig(os.path.join(output_dir, "elapsedtime_barchart.png"), bbox_inches="tight")
    plt.close()

    # accuracy vs. elapsed time
    scores2 = sorted(scores.items(), key=lambda x: x[1][1], reverse=False)  # sort by dict value

    elapsed_time = [item[1][7] for item in scores2]  # for elapsed_time  2021.3.5
    x_te = [item[1][1] for item in scores2]

    plt.figure(figsize=(5, 5))
    plt.xlabel("elapsed time")
    plt.ylabel("accuracy(est score)")
    plt.title("accuracy vs. elapsed time")

    plt.scatter(elapsed_time, x_te, color="blue")

    # annotation
    for i, label in enumerate(names):
        plt.annotate(label, (elapsed_time[i], x_te[i]))

    plt.savefig(os.path.join(output_dir, "accuracy_elapsedtime.png"))
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
    command = "cls_compare_algorithms"
    data_file = r"W:\dataset\clf\Wine\wine.csv"
    df = pd.read_csv(data_file, encoding="utf-8", header=0)
    output_dir = "../../../tests/unittest/output"

    alert, tc_time = cls_compare_algorithms(command, df, output_dir)
    print(f"{alert= } {tc_time= }")
