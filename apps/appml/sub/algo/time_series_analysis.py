import ast
import re
"""
Apache環境でplot_model機能せず．強制的にリタンする
"""
# from tensorflow.keras.utils import plot_model  # 2021.9.16
import logging

import os
import sys
import joblib
import pathlib
import numpy as np
from datetime import datetime
import codecs
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import statsmodels.api as sm

import warnings

pd.options.display.float_format = "{:.2f}".format
tf_version = tf.__version__

mid = pathlib.Path(__file__).name
ftime = "%Y/%m/%d %H:%M:%S"

np.random.seed(42)
tf.random.set_seed(42)


def pre_processing_debug():
    data_file = r"W:\dataset\tsa\Climate\jena_climate_small.csv"
    df = pd.read_csv(data_file, encoding="utf-8", header=0)
    output_dir = "./output"
    params = {"model_name": "GRU_Model", "units":32, "lookback": 1440, "step": 6, "delay": 144, "batch_size": 128, "epochs": 5}
    user_log_file = "./output/user.log"

    # set logger
    log_file = user_log_file
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    handler = logging.FileHandler(log_file, "a", encoding="utf-8")  # or whatever
    handler.setFormatter(logging.Formatter(' %(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logging.getLogger('werkzeug').disabled = True

    return output_dir, {}, params, df, logger


def pre_processing(args):
    working_dir = args[1]
    output_dir = os.path.join(working_dir, "output")

    with open(os.path.join(working_dir, "subprocess_call.txt"), "r", encoding="utf-8") as f:
        text = f.read()

    print('******************* text= ', text)

    call_dict = ast.literal_eval(text)
    params = call_dict["executeParameters"]
    params = ast.literal_eval(params)
    user_log_file = call_dict["user_log_file"]

    # set logger
    log_file = user_log_file
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    handler = logging.FileHandler(log_file, "a", encoding="utf-8")  # or whatever
    handler.setFormatter(logging.Formatter(' %(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logging.getLogger('werkzeug').disabled = True

    msg = f"[{mid}] {args= }"
    print(datetime.now().strftime(ftime) + " " + msg)
    logger.info(msg)

    file_name = os.path.join(working_dir, "subprocess_call.csv")
    df = pd.read_csv(file_name, encoding="utf-8", header=0)

    return output_dir, call_dict, params, df, logger


def post_processing(output_dir, call_dict):
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
    call_dict["tc_time"] = tc_time

    working_dir = pathlib.Path(output_dir).parent
    with open(os.path.join(working_dir, "subprocess_call.txt"), "w", encoding="utf-8") as f:
        f.write(str(call_dict))

    print(f"[{mid}] {alert= } {tc_time= }")

    msg = f"[{mid}] {alert= } {tc_time= }"
    logger.info(msg)


"""
Pre Processing
"""
output_dir, call_dict, params, df, logger = pre_processing(sys.argv)
# output_dir, call_dict, params, df, logger = pre_processing_debug()
launch_time = datetime.now()


"""
Main Processing
"""
# In[5]:


result_file = os.path.join(output_dir, "result.txt")

# In[6]:


# contents of result file

porec = []  # print out recording initialization

msg = "*** TimeSeriesAnalysis - AnomalyDetectionByRNN, {}".format(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
print(msg)
porec.append(msg)

# ## Step1: DATA PREPARATION

# #### data reading

# In[7]:


# data reading

# if "npz" in pathlib.Path(data_file).suffix:
#     data = np.load(data_file, allow_pickle=True)
#     df = pd.DataFrame(data["data"])
#     df.columns = data["features"]
# elif "csv" in pathlib.Path(data_file).suffix:
#     df = pd.read_csv(data_file, encoding="utf-8", header=0)
# else:
#     print("unsupported file extension")
#
# df.head()

# #### data wrangling

# In[8]:


# basic data information

df.info(True)  # False

# In[9]:


# statistical information

df.describe()

# In[10]:


# sequence graph

plt.style.use("ggplot")

target = df["target"]
plt.plot(range(len(target)), target)

plt.title("target value graph")
plt.savefig(os.path.join(output_dir, "target_graph.png"))

warnings.simplefilter("ignore", UserWarning)

plt.close()


# In[11]:


# スムージング　フランソワ　５章

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points


N = 50
plt.plot(range(len(target[:N])), smooth_curve(target[:N]), 'bo-', label='Smoothed target')
plt.plot(range(len(target[:N])), target[:N], 'ro', label='Original target')
plt.legend()


# In[12]:


# statsmodels
# Observed(元データ), Trend(トレンド), Seasonal(周期変動/季節変動), Residual(残差)

seasonal = sm.tsa.seasonal_decompose(df["target"].values, period=365, extrapolate_trend='freq')
seasonal.plot()

plt.savefig(os.path.join(output_dir, "trend_graph.png"))

warnings.simplefilter("ignore", UserWarning)

plt.close()

# In[13]:


# target - histogram

plt.style.use("ggplot")

num_bins = 30
df["target"].hist(bins=num_bins, figsize=(5, 3), color="blue")  # histogram

plt.title("target's histogram")
plt.xlabel("target")
plt.ylabel("frequency")
plt.savefig(os.path.join(output_dir, "histogram.png"))
plt.close()

# #### data preparation

# In[14]:


# features and target setting

float_data = df.values  # dataframe to numpy.ndarray format

msg = "number of data= {}\nnumber of columns= {}\nnumber of features= {}".format(float_data.shape[0],
                                                                                 float_data.shape[1],
                                                                                 float_data.shape[1] - 1)
print(msg)
porec.append(msg)

target_index = df.columns.values.tolist().index("target")

msg = "target_index= {}".format(target_index)
print(msg)
porec.append(msg)

# In[15]:


# training and test data splitting

TRAIN_GEN_MAX_INDEX = int(float_data.shape[0] * (2 / 4))
VALID_GEN_MAX_INDEX = int(float_data.shape[0] * (3 / 4))

msg = "Number of data= {}\nTRAIN_GEN_MAX_INDEX= {}\nVALID_GEN_MAX_INDEX= {}".format(float_data.shape[0],
                                                                                    TRAIN_GEN_MAX_INDEX,
                                                                                    VALID_GEN_MAX_INDEX)

print(msg)
porec.append(msg)

# #### data refining

# In[16]:


# standardization

# standardize all the data using the mean an std of train data
# (x-μ)÷σ

mean = float_data[:TRAIN_GEN_MAX_INDEX].mean(axis=0)
float_data -= mean
std = float_data[:TRAIN_GEN_MAX_INDEX].std(axis=0)
float_data /= std

msg = "column mean=\n{}\ncolumn std=\n{}".format(mean, std)
print(msg)
porec.append(msg)

# In[17]:


target_mean = mean[target_index]
target_std = std[target_index]

msg = "target_mean= {:.8f}\ntarget_std= {:.8f}".format(target_mean, target_std)
print(msg)
porec.append(msg)

# In[18]:


# scaler

scaler = {}

scaler["mean"] = mean
scaler["std"] = std
scaler["target_mean"] = target_mean
scaler["target_std"] = target_std

# ## Step2 : TRAINING

# #### set time series analysis parameters

# In[19]:


# set time series analysis parameters

# lookback = params["lookback"]
# step = params["step"]
# delay = params["delay"]
lookback = params["lookback"] if "lookback" in params else 1440
step = params["step"] if "step" in params else 6
delay = params["delay"] if "delay" in params else 144

batch_size = params["batch_size"] if "batch_size" in params else 128

msg = "time series analysis parameters:"
print(msg)
porec.append(msg)

msg = "lookback= {}\nstep= {}\ndelay= {}\nbatch_size= {}".format(lookback, step, delay, batch_size)
print(msg)
porec.append(msg)

# In[20]:


# save to scaler

scaler["lookback"] = lookback
scaler["step"] = step


# #### define data generator

# In[21]:


def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1

    i = min_index + lookback

    while 1:
        if shuffle:  # train data
            rows = np.random.randint(
                min_index + lookback,
                max_index,
                size=batch_size)  # この区間からランダムに128個の整数値を生成
        #             print("rows= ", rows)  # debug code

        else:  # validation data or test data
            if i + batch_size >= max_index:
                i = min_index + lookback

            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),  # 128 (バッチサイズ)
                            lookback // step,  # 240 (=lookback/step = 1440/6)
                            data.shape[-1]))  # 14 (特徴量数)

        targets = np.zeros((len(rows),))  # 14

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)  # 1時間毎(step=6)に，計240(=1440÷6)個サンプリング
            #             print("indices= ", indices)  # debug code

            samples[j] = data[indices]
            #             print("samples[{}] = {}".format(j, samples[j]))  # debug code

            #             targets[j] = data[rows[j] + delay][1]  # delay分未来の2列目のデータ（＝温度）
            targets[j] = data[rows[j] + delay][target_index]

        yield samples, targets


# In[22]:


train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,  # 0
                      max_index=TRAIN_GEN_MAX_INDEX,  # 200,000
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=TRAIN_GEN_MAX_INDEX + 1,  # 200,001
                    max_index=VALID_GEN_MAX_INDEX,  # 300,000
                    step=step,
                    batch_size=batch_size)

test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=VALID_GEN_MAX_INDEX + 1,  # 300,001
                     max_index=None,  # last
                     step=step,
                     batch_size=batch_size)

# This is how many steps to draw from val_gen in order to see the whole validation set:
val_steps = (VALID_GEN_MAX_INDEX - (TRAIN_GEN_MAX_INDEX + 1) - lookback) // batch_size  # 769個

# This is how many steps to draw from test_gen in order to see the whole test set:
test_steps = (len(float_data) - (VALID_GEN_MAX_INDEX + 1) - lookback) // batch_size  # 930個

print("val_steps= {}\ttest_steps= {}".format(val_steps, test_steps))

# In[23]:


# generator test

for samples, targets in train_gen:
    print("samples.shape= {}\ntargets.shape= {}".format(samples.shape, targets.shape))  # (128, 240, 14) (128,)

    break  # generates only one data and leave


# #### modeling

# 6-37 全結合モデルの訓練と評価

# In[24]:


# Fully Connected Model

def create_model_Dense(units=32, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), n_out=1):
    model = Sequential()

    model.add(Flatten(input_shape=(lookback // step, float_data.shape[-1])))  # (30, 6)

    model.add(Dense(units=units, activation="relu"))

    model.add(Dense(n_out))

    model.summary()

    model.compile(optimizer=optimizer, loss="mae")
    #     model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    return model


# 6-39 GRUベースのモデルの訓練と評価

# In[25]:


# # GRU Model

# def create_model_GRU(units=32, optimizer=RMSprop(lr=0.001), n_out=1):

#     model = Sequential()

# #     model.add(GRU(units=units,
# #                   input_shape=(None, float_data.shape[-1])))

#     model.add(GRU(units=units,
#                   activation='tanh',
#                   recurrent_activation='sigmoid',
#                   kernel_initializer='glorot_normal',
#                   recurrent_initializer='orthogonal',
#                   input_shape=(None, float_data.shape[-1]),
#                  ))


#     model.add(Dense(n_out, activation='linear'))

#     model.summary()

#     model.compile(optimizer=optimizer, loss="mae")

#     return model


# 6-40 ドロップアウトで正則化したGRUベースのモデルの訓練と評価

# In[26]:


# GRU with Dropout Model

def create_model_GRUwithDropout(units=32, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), n_out=1):
    model = Sequential()

    model.add(GRU(units=units,
                  dropout=0.2,
                  recurrent_dropout=0.2,

                  activation='tanh',
                  recurrent_activation='sigmoid',
                  kernel_initializer='glorot_normal',
                  recurrent_initializer='orthogonal',

                  input_shape=(None, float_data.shape[-1]),
                  ))

    model.add(Dense(n_out, activation='linear'))

    model.summary()

    model.compile(optimizer=optimizer, loss="mae")

    return model


# 6-41 ドロップアウトで正則化されたスッタキングGRUモデルでの訓練と評価

# In[27]:


# GRU with Dropout and Stacking Model

def create_model_GRUwithDropoutAndStacking(units=32, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), n_out=1):
    model = Sequential()

    model.add(GRU(units=units,
                  dropout=0.1,
                  recurrent_dropout=0.5,

                  return_sequences=True,  # リカレント層をスタッキング（層を増やす）

                  activation='tanh',
                  recurrent_activation='sigmoid',
                  kernel_initializer='glorot_normal',
                  recurrent_initializer='orthogonal',

                  input_shape=(None, float_data.shape[-1]),
                  ))

    model.add(GRU(units=units * 2,
                  activation="relu",
                  dropout=0.1,
                  recurrent_dropout=0.5))

    model.add(Dense(n_out, activation='linear'))

    model.summary()

    model.compile(optimizer=optimizer, loss="mae")

    return model


# 6-44 GRUベースの双方向RNNの訓練と評価

# In[28]:


# GRU based Bidirectional Model

def create_model_GRUbasedBidirectional(units=32, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), n_out=1):
    model = Sequential()

    model.add(Bidirectional(GRU(units=units,
                                activation='tanh',
                                recurrent_activation='sigmoid',
                                kernel_initializer='glorot_normal',
                                recurrent_initializer='orthogonal',
                                ),
                            input_shape=(None, float_data.shape[-1])))

    model.add(Dense(n_out, activation='linear'))

    model.summary()

    model.compile(optimizer=optimizer, loss="mae")

    return model


# 6-47 単純な１次元CNNの訓練と評価

# In[29]:


# # One Dimensional CNN Model

# def create_model_1dCNN(optimizer=RMSprop(lr=0.001)):

#     model = Sequential()

#     model.add(Conv1D(32, 5, activation="relu", input_shape=(None, float_data.shape[-1])))
#     model.add(MaxPooling1D(2))

#     model.add(Conv1D(32, 5, activation="relu"))
#     model.add(MaxPooling1D(2))

#     model.add(Conv1D(32, 5, activation="relu"))
#     model.add(GlobalMaxPooling1D())

#     model.add(Dense(1))
#     model.summary()

#     model.compile(optimizer=optimizer, loss="mae")

#     return model


# 6-48 Jenaデータセット用のより分解能の高いデータジェネレータの準備

# 6-49 1次元畳み込みベースとGRU層で構成されたモデル

# In[30]:


# See original document


# SimpleRNN - 詳解 ディープラーニング ~TensorFlow・Kerasによる時系列データ処理

# In[31]:


# def weight_variable(shape, name=None):
#     return np.random.normal(scale=.01, size=shape)


# In[32]:


# Simple RNN Model

def create_model_SimpleRNN(units=32, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), n_out=1):
    model = Sequential()

    model.add(SimpleRNN(units=units,
                        activation='tanh',
                        kernel_initializer='glorot_normal',
                        recurrent_initializer='orthogonal',
                        input_shape=(lookback // step, float_data.shape[-1]),  # input_shape=(240, 14)
                        ))

    model.add(Dense(n_out))

    model.add(Activation("linear"))

    model.summary()

    model.compile(optimizer=optimizer, loss="mae")
    #     model.compile(optimizer=optimizer, loss="mean_squared_error")  # 2乗平均誤差関数

    return model


# LSTM Long Short-Term Memory - 詳解 ディープラーニング ~TensorFlow・Kerasによる時系列データ処理

# In[33]:


# LSTM Model (Long Short-Term Memory)

def create_model_LSTM(units=32, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), n_out=1):
    model = Sequential()

    model.add(LSTM(units=units,
                   activation='tanh',
                   recurrent_activation='sigmoid',
                   kernel_initializer='glorot_normal',
                   recurrent_initializer='orthogonal',
                   input_shape=(lookback // step, float_data.shape[-1]),  # input_shape=(240, 14)
                   ))

    model.add(Dense(n_out))

    model.add(Activation("linear"))

    model.summary()

    model.compile(optimizer=optimizer, loss="mae")
    #     model.compile(optimizer=optimizer, loss="mean_squared_error")  # 2乗平均誤差関数

    return model


# GRU Gated Recurrent Unit- 詳解 ディープラーニング ~TensorFlow・Kerasによる時系列データ処理

# In[34]:


# GRU with kernel_initializer Model (Gated Recurrent Unit)

def create_model_GRU(units=32, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), n_out=1):
    model = Sequential()

    model.add(GRU(units=units,
                  activation='tanh',
                  recurrent_activation='sigmoid',
                  kernel_initializer='glorot_normal',
                  recurrent_initializer='orthogonal',
                  input_shape=(lookback // step, float_data.shape[-1]),  # input_shape=(240, 14)
                  ))

    model.add(Dense(n_out))

    model.add(Activation("linear"))

    model.summary()

    model.compile(optimizer=optimizer, loss="mae")
    #     model.compile(optimizer=optimizer, loss="mean_squared_error")  # 2乗平均誤差関数

    return model


# In[35]:


# create model

model_name = params["model_name"] if "model_name" in params else "GRU_Model"
print("model_name= ", model_name)

units = params["units"] if "units" in params else 32
print("units= ", units)

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# In[36]:


# Fully Connected Model
if "Fully_Connected_Model" in model_name:
    model = create_model_Dense(units=units, optimizer=optimizer)


# # GRU_Model
# elif "GRU_Model" in model_name:
#     model = create_model_GRU(units=units, optimizer=optimizer)

# GRU_with_Dropout_Model (21minutes)
elif "GRU_with_Dropout_Model" in model_name:
    model = create_model_GRUwithDropout(units=units, optimizer=optimizer)


# GRU_with_Dropout_and_Stacking_Model
elif "GRU_with_Dropout_and_Stacking_Model" in model_name:
    model = create_model_GRUwithDropoutAndStacking(units=units, optimizer=optimizer)


# GRU_based_Bidirectional_Model
elif "GRU_based_Bidirectional_Model" in model_name:
    model = create_model_GRUbasedBidirectional(units=units, optimizer=optimizer)


# # One_Dimensional_CNN_Model
# elif "One_Dimensional_CNN_Model" in model_name:
#     model = create_model_1dCNN(optimizer=optimizer)

# SimpleRNN_Model
elif "SimpleRNN_Model" in model_name:
    model = create_model_SimpleRNN(units=units, optimizer=optimizer)


# LSTM_Model
elif "LSTM_Model" in model_name:
    model = create_model_LSTM(units=units, optimizer=optimizer)


# GRU_Model
elif "GRU_Model" in model_name:
    model = create_model_GRU(units=units, optimizer=optimizer)


else:
    print("unsupported model")

# In[37]:


# draw model diagram

# plot_model(model, show_shapes=True, to_file=os.path.join(output_dir, "model_diagram.png"))

# #### training

# In[38]:


# set early-stopping

best_model_h5 = "best_model_" + tf_version + ".h5"

callbacks = [EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=5, mode="auto", verbose=0),
             ModelCheckpoint(filepath=os.path.join(output_dir, best_model_h5), monitor="val_loss", save_best_only=True,
                             save_weights_only=False),
             ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10),  # コールバックが起動したら学習率を10で割る
             ]

# In[39]:


epochs = params["epochs"] if "epochs" in params else 2

# step per epoch
spe_min = 100
spe_max = 500
spe = max(spe_min, int(TRAIN_GEN_MAX_INDEX / (lookback / step)))  # 200,000 ÷ 240
spe = min(spe_max, spe)
step_per_epoch = params["step_per_epoch"] if "step_per_epoch" in params else spe

msg = "epochs= {}\nstep_per_epoch= {}".format(epochs, step_per_epoch)
print(msg)
porec.append(msg)

# In[40]:


# model training with training data

begin_time = datetime.now()

msg = "begin_time: {}".format(begin_time.strftime("%Y/%m/%d %H:%M:%S"))
print(msg)
porec.append(msg)

# result = model.fit_generator(train_gen,
#                              steps_per_epoch=step_per_epoch,
#                              epochs=epochs,
#                              validation_data=val_gen,
#                              validation_steps=val_steps,
#                              callbacks=callbacks,
#                              verbose=0)

history = model.fit(train_gen,
                    steps_per_epoch=step_per_epoch,
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    callbacks=callbacks,
                    verbose=0)

msg = "Training Computation Time: {}".format(datetime.now() - begin_time)
print(msg)
porec.append(msg)

# In[41]:


msg = "Training result:"
print(msg)
porec.append(msg)

train_mean_val_loss = np.array(history.history["val_loss"]).mean()

msg = "train_mean_val_loss= {:.8f}".format(train_mean_val_loss)  # valの標準mae
print(msg)
porec.append(msg)


# In[42]:


# Training and validation loss history graph
# loss: mae  # mse

def draw_loss(history, ylim=(0.10, 0.30)):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")

    #     plt.plot(epochs[1:], loss[1:], "bo", label="Training loss")       # delete the first point because it's too big.
    #     plt.plot(epochs[1:], val_loss[1:], "b", label="Validation loss")  # delete the first point because it's too big.

    #     plt.ylim(ylim)

    plt.xlabel("epoch")
    plt.ylabel("loss[mae]")
    plt.title("Training and validation loss")
    plt.legend()

    # FIXME
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    return


plt.style.use("ggplot")

# draw_loss(history, ylim=(0.10, 0.30))
draw_loss(history)

# In[43]:


# load the best model

del model

model = load_model(os.path.join(output_dir, best_model_h5))  # defined at early-stopping

np.savez(os.path.join(output_dir, "RNN_GRU_parameters.npz"),
         lookback=lookback,
         step=step,
         delay=delay,
         batch_size=batch_size,
         target_mean=target_mean,
         target_std=target_std)

# outfile = np.load(output_dir + "/RNN_GRU_parameters.npz")
# print(outfile["lookback"])

if scaler:
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"), compress=True)
    scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))

# ### Step3: EVALUATION

# #### evaluation

# In[44]:


# model.summary()

with StringIO() as buf:
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    model_text = buf.getvalue()

msg = "model_def= " + model_text
print(msg)
porec.append(msg)

# In[45]:


print(history.history.keys())

# In[46]:


# model evaluation with test data

# result = model.evaluate_generator(test_gen,
#                                   steps=step_per_epoch,
#                                   max_queue_size=10,
#                                   workers=1,
#                                   use_multiprocessing=False,
#                                   verbose=1)

test_loss = model.evaluate(test_gen,
                           steps=step_per_epoch,
                           max_queue_size=10,
                           workers=1,
                           use_multiprocessing=False,
                           verbose=1)

mae_standard = test_loss  # 標準MAE, loss="mae"
mae = mae_standard * target_std  # MAE = 標準MAE × target_std

msg = "Test result:"
print(msg)
porec.append(msg)

msg = "target_std= {:.8f}".format(target_std)
print(msg)
porec.append(msg)

msg = "mae_standard= {:.8f}".format(mae_standard)
print(msg)
porec.append(msg)

msg = "mae= mae_standard * target_std= {:.8f}".format(mae)
print(msg)
porec.append(msg)

# #### additional report

# ### simulation case1:  to predict the N latest target values

# In[47]:


msg = "simulation case1:  to predict the N latest target values"
print(msg)
porec.append(msg)

# In[48]:


"""
example jena climate
 1.実データの最後の■(=shape(1440,14))を使って，1日後(delay=144)の温度☆(shape=(1,))を予測する
   ………□□■
 2.最後から２つ目の■を使って（＝1日分遡る），1日後の温度を予測する
   ………□■□
 3.繰り返す

"""

# In[49]:


# to predict the last N target values

pred = []
true = []

N_MAX = 150  # 300

N = min((len(float_data) - lookback) // delay + 1, N_MAX)  # 予測データポイント数

msg = "number of predicted data points= {}".format(N)
print(msg)
porec.append(msg)

pred_starting_index = len(float_data) - lookback - delay * (N - 1)
print("pred_starting_index= ", pred_starting_index)
print("end_index= ", len(float_data))

for n in range(N):
    begin = pred_starting_index + delay * n
    end = begin + lookback
    #     print(n, begin, end, end-begin)  # debug code

    x = float_data[begin:end]

    x = x[::step]
    x = x.flatten().reshape(1, int(lookback / step), float_data.shape[1])  # (1, 240, 14)

    mae_standard = model.predict(x).flatten()[0]
    pred.append(mae_standard * target_std + target_mean)

    if n < (N - 1):
        x = float_data[end][target_index]  # 温度項目を抽出する
        x = x * target_std + target_mean  # 標準化からの戻し
        true.append(x)

msg = "len(pred)= {}\tpred[:10]= {}".format(len(pred), np.round(pred, 2)[:10])
print(msg)
porec.append(msg)

msg = "len(true)= {}\ttrue[:10]= {}".format(len(true), np.round(true, 2)[:10])
print(msg)
porec.append(msg)

# In[50]:


msg = "#\tpred\ttrue\tabs(pred-true)\tabs((pred-true)/true)(%)"
print(msg)
porec.append(msg)

for i in range(len(pred) - 1):
    msg = "{}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.0f}".format(i, pred[i], true[i], np.abs(pred[i] - true[i]),
                                                        np.abs((pred[i] - true[i]) / true[i]) * 100)
    print(msg)
    porec.append(msg)

msg = "{}\t{:.2f}".format(len(true), pred[-1])
print(msg)
porec.append(msg)

# In[51]:


len(pred[:-1]), len(true)

# In[52]:


# draw prediction versus true value graph

plt.figure(figsize=(15, 5))

plt.plot(pred[:-1], label="prediction", linestyle="dashed")
plt.plot(true, label="true value", lw=3)

plt.xlabel("time")
plt.ylabel("target value")
plt.title("anomaly detection")
plt.legend()
plt.title("prediction and true value")
plt.legend()

plt.savefig(os.path.join(output_dir, "predictionvstrue.png"))
plt.close()

# In[53]:


# # draw differentiated prediction versus true value graph

# plt.figure(figsize=(10,5))

# L = len(pr)-1
# # L=100

# x = range(L)
# plt.plot(x, (np.diff(pr[::-1])[-L:]), label="prediction", linestyle="dashed")  # differentiate
# plt.bar(x,  (np.diff(tr[::-1])[-L:]), label="true value", lw=3)  # differentiate

# plt.axhline(color="k")
# plt.title("prediction vs. true value")
# plt.legend()

# plt.savefig(output_dir + "/differentiate.png")
# plt.show()
# plt.close()


# ### simulation case2:  anomaly detection

# In[54]:


msg = "simulation case2:  anomaly detection"
print(msg)
porec.append(msg)

# calculate MSE

# In[55]:


# calculate MSE

pr = pred[:-1]  # 最後の予測値(pred)は実データ(true)にはないのでマイナス１する
tr = true

len(pr), len(tr)
err_list = []

W = 10  # window
S = 5  # slide

for n in range(0, len(pr) - W + 1, S):
    #     print("n= {}\tp= {}\tt= {}".format(n, pr[n:n+W], tr[n:n+W]))

    err = ((np.array(pr[n:n + W]) - np.array(tr[n:n + W])) ** 2).sum() / W  # MSE
    err_list.append(err)

print(f"{err_list[:5]= }")

# calculate threshold

# In[56]:


# calculate threshold

idx_desc = np.argsort(err_list)[::-1]  # 降順インデックス
print("idx_desc= ", idx_desc)

err_list = np.array(err_list)
print("err_list[idx_desc]= ", err_list[idx_desc])

threshold = err_list[idx_desc[int(len(idx_desc) * 0.05 + 1)]]  # calculate the value for the index of top 5 percents.

msg = "threshold(top 5%)= {:.4f}".format(threshold)
print(msg)
porec.append(msg)

msg = "number of detected anomalies: {}".format((err_list > threshold).sum())
print(msg)
porec.append(msg)

msg = "detected anomalies:"
print(msg)
porec.append(msg)

for window_id, v in enumerate(err_list):
    if v > threshold:
        msg = "window_id= {}\tmse= {:.4f}".format(window_id, v)
        print(msg)
        porec.append(msg)

# In[57]:


# draw threshold graph

fig = plt.figure(figsize=(15, 5), facecolor="w")
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# line chart
ax1.set_xlabel("window(descend order)")
ax1.set_ylabel("mean square error")
ax1.set_title("threshold [top 5%]")

ax1.plot(err_list[idx_desc], label="mse")
ax1.axhline(y=threshold, linestyle="dashed")
ax1.text(10, threshold, np.round(threshold, 4), fontsize=15, va='center', ha='center', backgroundcolor='w')
ax1.legend()

# bar chart
ax2.set_xlabel("window(descend order)")
ax2.set_ylabel("mean square error")
ax2.set_title("threshold [top 5%] (bar chart)")

ax2.bar(range(len(err_list)), err_list[idx_desc], label="mse")
ax2.axhline(y=threshold, linestyle="dashed")
ax2.text(10, threshold, np.round(threshold, 4), fontsize=15, va='center', ha='center', backgroundcolor='w')
ax2.legend()

plt.savefig(os.path.join(output_dir, "threshold.png"))
plt.close()

# In[58]:


# draw anomaly detection graph

fig = plt.figure(figsize=(15, 5), facecolor="w")
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# line chart
ax1.set_xlabel("window(chronological order)")
ax1.set_ylabel("mean square error")
ax1.set_title("anomaly detection")

ax1.plot(err_list, label="mse")
ax1.axhline(y=threshold, linestyle="dashed")
ax1.text(10, threshold, np.round(threshold, 4), fontsize=15, va='center', ha='center', backgroundcolor='w')
ax1.legend()

# bar chart
ax2.set_xlabel("window(chronological order)")
ax2.set_ylabel("mean square error")
ax2.set_title("anomaly detection (bar chart)")

ax2.bar(range(len(err_list)), err_list, label="mse")
ax2.axhline(y=threshold, linestyle="dashed")
ax2.text(10, threshold, np.round(threshold, 4), fontsize=15, va='center', ha='center', backgroundcolor='w')
ax2.legend()

plt.savefig(os.path.join(output_dir, "anomaly_detection.png"))
plt.close()

# ## Further Analysis

# In[59]:


# None


# #### Post-Processing

# In[60]:


# open result file(list format)

result_dat = pathlib.Path(result_file).with_suffix(".dat")

if pathlib.Path(result_dat).exists():
    pathlib.Path(result_dat).unlink()

# save contents
rf = codecs.open(str(result_dat), "w", "utf-8")
print(porec, file=rf)

# close result file
rf.close()

# In[61]:


# open result file(text format)

if pathlib.Path(result_file).exists():
    pathlib.Path(result_file).unlink()

# list to string
result_str = ""
for line in porec:
    result_str += line + "\n"

# save contents
rf = codecs.open(result_file, "w", "utf-8")
print(result_str, file=rf)

# close result file
rf.close()

# In[62]:


print(result_str)

# In[63]:


print("Done! ", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

# ## Predict Test - Jena Climate

# In[64]:


# data_file = r"F:\OneDrive - Hitachi Group\Programming\Python\mlstarterkit\sampledataset\TimeSeriesAnalysis\Climate\predict\data\test1.txt"


# In[65]:


# import ast

# model = load_model(os.path.join(output_dir, best_model_h5))
# scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))


# lookback = scaler["lookback"]  #1440
# step = scaler["step"]  # 6

# mean = scaler["mean"]
# std = scaler["std"]

# target_mean = scaler["target_mean"]
# target_std = scaler["target_std"]

# print(f"{lookback= }\n{step= }\n{mean=}\n{std=}\n{target_mean= }\n{target_std= }")

# x = list()

# with open(data_file, encoding="utf-8") as f:
#     for text in f:
#         x.append(list(ast.literal_eval(text)))

# # standardization
# x = np.array(x)
# x -= mean
# x /= std

# x = x[::step]
# x = x.flatten().reshape(1, int(lookback/step), x.shape[1])  # (1, 240, 14)

# mae_standard = model.predict(x).flatten()[0]

# ans = mae_standard * target_std + target_mean
# print(f"{ans= }")


# In[ ]:


"""
Post Processing
"""
post_processing(output_dir, call_dict)
