import src.data.datasets as data
from sklearn.preprocessing import MinMaxScaler
import src.data.preprocessor as pre
import src.model.network as net 
import numpy as np
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

dataset = data.get_dataset("CMAPSS2")
window_size = 20


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


X = np.vstack(train[0])
y = np.vstack(train[1])

X_test = []
y_test = []
for i in range(259):
    X_test.append(test[0][i][-1,:,:,:])
    y_test.append(test[1][i][-1,:])

X_test = np.array(X_test)
y_test = np.array(y_test)
# model = net.create_model(window_size=window_size, feature_dim=14, kernel_size=(10, 1), filter_num=10, dropout_rate=0)
# model.summary()


# model.compile(optimizer=Adam(learning_rate=1e-1), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])

# model.fit(x=X, y=y, batch_size = 512, epochs = 50)


model = net.simple_cnn(window_size=window_size, feature_dim=14, kernel_size=(10, 1), filter_num=10, dropout_rate=0.5)
model.summary()


model.compile(optimizer=Adam(learning_rate=1e-3), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])

model.fit(x=X, y=y, batch_size = 512, epochs = 50)
model.evaluate(x=X_test, y=y_test)

