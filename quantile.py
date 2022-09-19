import tensorflow as tf
import src.data.datasets as data
from sklearn.preprocessing import MinMaxScaler
import src.data.preprocessor as pre
import model.CNN as net 
import numpy as np
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from tensorflow.keras import backend as K
import os
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

dataset = data.get_dataset("CMAPSS1")
window_size = 30
calib = 50
data_num = 100


scaler = MinMaxScaler(feature_range=(-1, 1))
train = pre.apply_scaling_fn(scaler.fit_transform, dataset["train"])
test = pre.apply_scaling_fn(scaler.transform, dataset["test"])
removable_cols = ["sm01", "sm05", "sm06", "sm10", "sm16", "sm18", "sm19"]
cmapss_op_list = ["time", "os1", "os2", "os3"]
removable_cols += cmapss_op_list
train = train.drop(removable_cols, axis=1)

test = test.drop(removable_cols, axis=1)
train = pre.dataframe_to_supervised(train, n_in=window_size-1)
test = pre.dataframe_to_supervised(test, n_in=window_size-1)


X = np.vstack(train[0][:calib])
y = np.vstack(train[1][:calib])

X_calib = np.vstack(train[0][calib:])
y_calib = np.vstack(train[1][calib:])

X_test = []
y_test = []
for i in range(data_num):
    X_test.append(test[0][i][-1,:,:,:])
    y_test.append(test[1][i][-1,:])

X_test = np.array(X_test)
y_test = np.array(y_test)
# model = net.create_model(window_size=window_size, feature_dim=14, kernel_size=(10, 1), filter_num=10, dropout_rate=0)
# model.summary()


# model.compile(optimizer=Adam(learning_rate=1e-1), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])

# model.fit(x=X, y=y, batch_size = 512, epochs = 50)


def quantile_loss(q,y_true,y_pred):
    """
    q -- quantile level
    y_true -- true values
    y_pred -- predicted values
    """
    diff = (y_true - y_pred)
    mask = y_true >= y_pred
    mask_ = y_true < y_pred
    loss = (q * K.sum(tf.boolean_mask(diff, mask), axis=-1) - (1 - q) * K.sum(tf.boolean_mask(diff, mask_), axis=-1))
    return loss

model_low = net.simple_cnn(window_size=window_size, feature_dim=14, kernel_size=(10, 1), filter_num=10, dropout_rate=0.5)
model_high = net.simple_cnn(window_size=window_size, feature_dim=14, kernel_size=(10, 1), filter_num=10, dropout_rate=0.5)
model_mean = net.simple_cnn(window_size=window_size, feature_dim=14, kernel_size=(10, 1), filter_num=10, dropout_rate=0.5)

model_low.compile(optimizer=Adam(learning_rate=1e-3), loss=lambda y_true,y_pred: quantile_loss(0.05,y_true,y_pred), metrics=[RootMeanSquaredError()])
model_high .compile(optimizer=Adam(learning_rate=1e-3), loss=lambda y_true,y_pred: quantile_loss(0.95,y_true,y_pred), metrics=[RootMeanSquaredError()])
model_mean.compile(optimizer=Adam(learning_rate=1e-3), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])

model_low.fit(x=X, y=y, batch_size = 512, epochs = 50)
model_high.fit(x=X, y=y, batch_size = 512, epochs = 50)
model_mean.fit(x=X, y=y, batch_size = 512, epochs = 50)

plt.plot(model_low.predict(X_test),'-')
plt.plot(model_high.predict(X_test),'--')
plt.plot(model_mean.predict(X_test),'o')
plt.plot(y_test, '*')

