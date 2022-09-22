from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os

import src.data.preprocessor2 as pre
import src.data.datasets2 as data
from src.model.network import create_DCNN, create_MQDCNN, MultiQuantileLoss
from src.utils import compute_coverage_len, compute_quantile, compute_quantiles_nex


#in order not to use GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

#CMAPSS data removable and ignorable cols
removable_cols = ["sm01", "sm05", "sm06", "sm10", "sm16", "sm18", "sm19"]
ignore_columns = ["time", "os1", "os2", "os3"]

epochs = 250 #training epochs for both DCNN and MQDCNN 
optimizer = Adam(learning_rate=1e-3) #model optimizer
alpha = 0.1 #miscoverage rate

#MQDCNN related parameters
quantiles = [alpha, 1 - alpha]
loss_func = MultiQuantileLoss(quantiles=quantiles)

for exp in range(4):
    #load, split, and preprocess a dataset from CMAPSS datasets
    dataset = data.get_dataset("CMAPSS"+str(exp), MinMaxScaler(feature_range=(-1, 1)))
    split_dataset = pre.split_dataset(dataset, calval_size=0.1, random_state=0)
    proc_dataset = pre.preprocess_split(split_dataset, scaler_factory=dataset["scaler_factory"], window_size=dataset["window_size"], removable_cols=removable_cols, ignore_columns=ignore_columns)

    


    X_train = proc_dataset["train"]["X"]
    y_train = proc_dataset["train"]["y"]
    #DCNN training
    DCNN = create_DCNN(window_size=dataset["window_size"], feature_dim=14, kernel_size=(10, 1), filter_num=10, dropout_rate=0.5)
    DCNN.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])
    DCNN.fit(x=X_train, y=y_train, batch_size = 512, epochs = epochs)
    #MQDCNN training
    MQDCNN = create_MQDCNN(quantiles=quantiles, window_size=dataset["window_size"], feature_dim=14, kernel_size=(10, 1), filter_num=10, dropout_rate=0.5)
    MQDCNN.compile(optimizer=optimizer, loss=loss_func, metrics=[RootMeanSquaredError()])
    MQDCNN.fit(x=X_train, y=y_train, batch_size = 512, epochs = epochs)