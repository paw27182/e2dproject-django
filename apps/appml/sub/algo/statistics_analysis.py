import os
import pathlib
import numpy as np
import pandas as pd
import re
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

# from scipy.stats import norm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import f_classif
# from mlxtend.plotting import heatmap

import plotly.offline as po
import plotly.io as pio
import plotly.express as px

matplotlib.use('Agg')  # バックエンドを指定
pd.options.display.float_format = "{:.2f}".format
pd.options.display.max_columns = None  # 常に全ての列（カラム）を表示

# plt.rcParams['font.family'] = 'MS Gothic'  # 'IPAexGothic'
# plt.rc('font', family='MS Gothic')  # 'serif'
rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'YuGothic', 'Meiryo']  # WSL Ubuntu error 2023.5.29

mid = pathlib.Path(__file__).name


# select features and "class" or "target" through KBest algorithm
def select_columns(df, max_len=4):
    if "class" in df.columns:
        key = "class"
    elif "target" in df.columns:
        key = "target"
    else:
        return []

    if len(df.columns) > max_len:
        X = df.drop(key, axis=1)
        y = df[key]

        X_std = StandardScaler().fit_transform(X.astype(np.float64))

        k = 3  # three features

        if key == "class":
            skb = SelectKBest(score_func=f_classif, k=k)
        else:
            skb = SelectKBest(score_func=f_regression, k=k)

        skb.fit_transform(X_std, y)

        mask = skb.get_support()
        selected_features = X.columns.values[mask]

        cols = [*selected_features, key]

    else:
        cols = list(df.columns)  # 2021.9.7
        for word in ["class", "target"]:
            if word in cols:
                cols.remove(word)
                cols = [*cols, word]  # move "class" or "target" to the last position to draw a 3D graph

    return cols


def statistics_analysis(command, df, output_dir):
    print(f"[{mid}] {command}")

    launch_time = datetime.now()

    # tc_time = "0:00:00"
    # return f"NG.  {command} failed.", tc_time  # debug

    # check "class" column
    if ("class" not in df.columns) and ("target" not in df.columns):
        alert = ('NG', f'error: data must have "class" or "target" column.')
        tc_time = "0:00:00"
        return alert, tc_time

    # fundamental statistics
    try:
        # stats = df.describe().T
        cols = df.columns.to_list()
        stats = df.describe().T.to_numpy().tolist()  # pandas -> ndarray -> list
        # print(f"@@@@@ {df.describe()= }")

    except Exception as err:
        alert = ("NG", f"error: data is inadequate to analyse. {err= }")
        tc_time = "0:00:00"
        return alert, tc_time

    # v = [line.split() for line in str(stats).split("\n")[1:]]

    with open(os.path.join(output_dir, 'result.txt'), "w", encoding="utf-8") as f:
        # f.write(",".join(h) + "\n")
        header = "name, count, mean, std, min, 25%, 50%, 75%, max"
        f.write(header + "\n")

        # for line in v:
        #     f.write(",".join(line) + "\n")
        for i, item in enumerate(stats):
            text = cols[i] + ", " + ", ".join([str(np.round(e, 2)) for e in item]) + "\n"  # numeric -> str -> join
            # print(text)
            f.write(text)

    # select 3 features and "class" or "target" to cut processing time
    cols = select_columns(df)
    print(f"{cols= }")
    if not cols:
        alert = "NG", "error: data must include 'class' or 'target' column"
        tc_time = "0:00:00"
        return alert, tc_time

    # histogram
    plt.style.use("ggplot")

    for item in cols:  # in df.columns:
        mean = np.mean(df[item])  # average
        std = np.std(df[item])  # standard deviation
        qt = np.percentile(df[item], q=[0, 25, 50, 75, 100])  # quartile

        # Cut the window in 2 parts
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

        # Add a graph in each part
        sns.boxplot(df[item], ax=ax_box)

        if "class" in df.columns:
            key = "class"
        else:
            key = None

        sns.histplot(data=df,
                     x=item,
                     hue=key,
                     element="step",
                     kde=True,
                     ax=ax_hist,
                     )

        ax_box.set(xlabel="")  # Remove x axis name for the boxplot
        ax_box.set_title("quartile [{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}]".format(
            qt[0],
            qt[1],
            qt[2],
            qt[3],
            qt[4]),
            fontsize=10)
        ax_hist.set_title(r"norm: $\mu= {:.2f}$ $\sigma= {:.2f}$".format(mean, std), fontsize=14)
        ax_hist.set_ylabel("kernel density")

        # plt.legend()
        plt.savefig(os.path.join(output_dir, "histogram_" + item + ".png"))
        # plt.show()
        plt.close()

        # break

    data = df[cols]

    # Pearson's correlation coefficient
    corr = data.corr()
    sns.heatmap(corr, vmax=1, vmin=-1, center=0, annot=True, cmap="viridis")

    plt.title("Pearson's r")
    plt.savefig(os.path.join(output_dir, "piersons_r.png"))
    plt.show()
    plt.close()

    # 3D-scatter graph for the top three features
    x = cols[0]
    y = cols[1]
    z = cols[2]

    if len(cols) < 4:  # "4" means three features and "class"/"target"
        alert = "NG", "error: At least four columns is needed, that is, three features and 'class' or 'target' column."
        return alert, "0:00:00"  # tc_time

    c = cols[3]  # "class" or "target"

    try:
        fig = px.scatter_3d(data, x=x, y=y, z=z, color=c, title="Tips: the axes through SelectKBest algorithm.")
        po.plot(fig, filename=os.path.join(output_dir, "3DscatterDiagram.html"), auto_open=False)  # save as HTML
        pio.write_image(fig, os.path.join(output_dir, "3DscatterDiagram.png"))  # save as PNG (or JPEG/WebP/PDF/SVG/EPS)
    except Exception as err:
        print(f"[Statistics] 3D scatter diagram {err= }")

    # 3D boundary surface graph for the top two features
    import plotly.graph_objects as go
    from sklearn.svm import SVC, SVR
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if "class" in df.columns:
        key = "class"
        model = SVC(C=1., gamma="auto")
        # model = RandomForestClassifier(n_estimators=100)
    elif "target" in df.columns:
        key = "target"
        # model = SVR(C=1., gamma="auto")
        model = RandomForestRegressor(n_estimators=100)
    else:
        key = ""
        model = ""

    x1 = cols[0]
    x2 = cols[1]
    z = key

    X = df[[x1, x2]]
    y = df[z]

    model.fit(X, y)

    # create mesh grid
    margin = 0

    x_min, x_max = X[x1].min() - margin, X[x1].max() + margin
    y_min, y_max = X[x2].min() - margin, X[x2].max() + margin

    x_mesh_size = np.round((x_max - x_min) / 100, 5)  # .02
    y_mesh_size = np.round((y_max - y_min) / 100, 5)
    # print(f"{x_min= }, {x_max= }, {x_mesh_size= }")

    x_range = np.arange(x_min, x_max, x_mesh_size)
    y_range = np.arange(y_min, y_max, y_mesh_size)
    xx, yy = np.meshgrid(x_range, y_range)

    # predict for each mesh grid
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)

    try:
        fig = px.scatter_3d(df, x=x1, y=x2, z=z, color=c, title="Tips: the axes through SelectKBest algorithm.")
        fig.update_traces(marker=dict(size=5))
        fig.add_traces(go.Surface(x=x_range, y=y_range, z=pred, name='pred_surface'))
        po.plot(fig, filename=os.path.join(output_dir, "3DboundarySurface.html"), auto_open=False)  # save as HTML
        pio.write_image(fig,
                        os.path.join(output_dir, "3DboundarySurface.png"))  # save as PNG (or JPEG/WebP/PDF/SVG/EPS)
    except Exception as err:
        print(f"[Statistics] 3D boundary surface {err= }")

    tc_time = "{}".format(datetime.now() - launch_time)
    match = re.search('[0-9]{1,2}:[0-9]{2}:[0-9]{2}', tc_time)  # 0:00:00
    tc_time = match.group(0)

    alert = "OK", "Done."

    return alert, tc_time


if __name__ == "__main__":
    command = "statistics_analysis"
    data_file = r"W:\dataset\clf\Wine\wine.csv"
    df = pd.read_csv(data_file, encoding="utf-8", header=0)
    output_dir = "../../../tests/unittest/output"

    alert, tc_time = statistics_analysis(command, df, output_dir)
    print(f"{alert= } {tc_time= }")
