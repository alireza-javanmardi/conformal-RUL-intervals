import os
import sys
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

import src.data.preprocessor as pre
import src.data.datasets as data
from src.model.CNN import create_DCNN, create_MQDCNN, MultiQuantileLoss
import src.helper as h

#in order not to use GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

dataset_name = sys.argv[1]
cal_portion_str = sys.argv[2]
cal_portion = float(cal_portion_str)
exp_seed_str = sys.argv[3]
exp_seed = int(exp_seed_str)


#For TensorFlow program to run deterministically
#this sets the Python seed, the NumPy seed, and the TensorFlow seed.
tf.keras.utils.set_random_seed(exp_seed)
tf.config.experimental.enable_op_determinism()


#--------------------------------------------------------------------------------------------------------------------------------
#CMAPSS data removable and ignorable cols
removable_cols = ["sm01", "sm05", "sm06", "sm10", "sm16", "sm18", "sm19"]
ignore_columns = ["time", "os1", "os2", "os3"]

epochs = 250 #training epochs for both DCNN and MQDCNN 
epoch_th = 200 #after this threshold, the learning rate changes

# if dataset_name=="CMAPSS2":
#     epochs = 100 
#     epoch_th = 75 
# else:
#     epochs = 250 
#     epoch_th = 200

def scheduler(epoch, lr):
  if epoch <=epoch_th:
    return lr
  else:
    lr = 1e-4
    return lr
lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)
alpha_array = np.array([0.1, 0.15, 0.2, 0.25]) #array of miscoverage rates

#MQDCNN related parameters
#quantile 0.5 is for the median predictor
quantiles = np.concatenate((alpha_array, np.array([0.5]), 1-alpha_array))
loss_func = MultiQuantileLoss(quantiles=quantiles)

#load, split, and preprocess the specified dataset 
dataset = data.get_dataset(dataset_name, MinMaxScaler(feature_range=(-1, 1)))
split_dataset = pre.split_dataset(dataset, cal_size=cal_portion, random_state=exp_seed)
proc_dataset = pre.preprocess_split(split_dataset, scaler_factory=dataset["scaler_factory"], window_size=dataset["window_size"], removable_cols=removable_cols, ignore_columns=ignore_columns)
#X and y
X_train = proc_dataset["train"]["X"]
y_train = proc_dataset["train"]["y"]
X_cal = proc_dataset["cal"]["X"]
y_cal = proc_dataset["cal"]["y"]
idx_cal = proc_dataset["cal"]["index"]
X_test, y_test, idx_test = h.reform_test_data(proc_dataset["test"])
#piecewise RUL definition
y_train[y_train>125] = 125
y_cal[y_cal>125] = 125
y_test[y_test>125] = 125
#MQDCNN training
MQDCNN = create_MQDCNN(quantiles=quantiles, window_size=dataset["window_size"], feature_dim=14, kernel_size=(10, 1), filter_num=10, dropout_rate=0.5)
MQDCNN.compile(optimizer=Adam(learning_rate=1e-3), loss=loss_func, metrics=[RootMeanSquaredError()])
MQDCNN_hist = MQDCNN.fit(x=X_train, y=y_train, shuffle=False, batch_size = 512, epochs = epochs, callbacks=[lr_schedule], verbose=2)
#MQDCNN.save(os.path.join("saved_models", dataset_name, "cal_portion_"+cal_portion_str, "seed_"+exp_seed_str, "MQDCNN"))
#np.save(os.path.join("saved_models", dataset_name, "cal_portion_"+cal_portion_str, "seed_"+exp_seed_str,'MQDCNN_history.npy'), MQDCNN_hist.history)
#DCNN training
DCNN = create_DCNN(window_size=dataset["window_size"], feature_dim=14, kernel_size=(10, 1), filter_num=10, dropout_rate=0.5)
DCNN.compile(optimizer=Adam(learning_rate=1e-3), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])
DCNN_hist = DCNN.fit(x=X_train, y=y_train, shuffle=False, batch_size = 512, epochs = epochs,  callbacks=[lr_schedule], verbose=2)
#DCNN.save(os.path.join("saved_models", dataset_name, "cal_portion_"+cal_portion_str, "seed_"+exp_seed_str, "DCNN"))  
#np.save(os.path.join("saved_models", dataset_name, "cal_portion_"+cal_portion_str, "seed_"+exp_seed_str,'DCNN_history.npy'), DCNN_hist.history)
print("evaluation of MQDCNN:", MQDCNN.evaluate(X_test, y_test, verbose=2))
print("evaluation of DCNN::", DCNN.evaluate(X_test, y_test, verbose=2))
#--------------------------------------------------------------------------------------------------------------------------------
#Difficulty estimate of the datapoints using a simple random forest with default configurations of sklearn
y_hat_train = DCNN.predict(x=X_train, verbose=0)
y_hat_cal = DCNN.predict(x=X_cal, verbose=0)
y_hat_test = DCNN.predict(x=X_test, verbose=0)
res_train = np.abs(y_hat_train - y_train) 
res_cal = np.abs(y_hat_cal - y_cal) 



X_train_reshaped = X_train.reshape((-1,dataset["window_size"]*14))
X_cal_reshaped = X_cal.reshape((-1,dataset["window_size"]*14))
X_test_reshaped = X_test.reshape((-1,dataset["window_size"]*14))

RF = RandomForestRegressor(random_state=exp_seed) 
RF.fit(X_train_reshaped, res_train)
#joblib.dump(RF, os.path.join("saved_models", dataset_name, "cal_portion_"+cal_portion_str, "seed_"+exp_seed_str, "RF.joblib"))
print("Random forest details:")
print("average labels (training absolute residuals of DCNN):", res_train.mean())
print("training mean absolute error of RF:", mean_absolute_error(res_train, RF.predict(X_train_reshaped)))
print("calibration mean absolute error of RF:", mean_absolute_error(res_cal, RF.predict(X_cal_reshaped)))

#--------------------------------------------------------------------------------------------------------------------------------

rho = 0.99
sigma_cal = RF.predict(X_cal_reshaped).reshape((-1,1))
sigma_test = RF.predict(X_test_reshaped).reshape((-1,1))
y_hat_cal_CQR = MQDCNN.predict(x=X_cal, verbose=0)
y_hat_test_CQR = MQDCNN.predict(x=X_test, verbose=0)

scores = np.abs(y_cal - y_hat_cal) 
scores_normalized = scores/sigma_cal

 
intervals_dic = {}
for a, alpha in enumerate(alpha_array):
    q = h.compute_quantile(scores, alpha)
    q_array = h.compute_quantiles_nex(rho, scores, idx_test, idx_cal, alpha)
    q_normalized = h.compute_quantile(scores_normalized, alpha)
    q_array_normalized = h.compute_quantiles_nex(rho, scores_normalized, idx_test, idx_cal, alpha)

    scores_low = y_hat_cal_CQR[a] - y_cal
    scores_high = y_cal - y_hat_cal_CQR[a+len(alpha_array)+1] 
    scores_CQR = np.maximum(scores_low, scores_high)
    q_CQR = h.compute_quantile(scores_CQR, alpha)

    intervals_dic_alpha = {
        "SCP": (np.maximum(0,y_hat_test - q), y_hat_test + q),
        "nex-SCP": (np.maximum(0, y_hat_test  - q_array), y_hat_test  + q_array),
        "SCP+NNM": (np.maximum(0,y_hat_test - q_normalized*sigma_test), y_hat_test + q_normalized*sigma_test),
        "nex-SCP+NNM": (np.maximum(0, y_hat_test  - q_array_normalized*sigma_test), y_hat_test  + q_array_normalized*sigma_test),
        "CQR": (np.maximum(0, y_hat_test_CQR[a] - q_CQR), y_hat_test_CQR[a+len(alpha_array)+1] + q_CQR)
        }
    intervals_dic[alpha] = intervals_dic_alpha


results_dic = {
    "Ground truth RULs": y_test,
    "Single-point RUL predictions": y_hat_test,
    "Single-point RUL predictions CQR": y_hat_test_CQR[len(alpha_array)],
    "intervals": intervals_dic
    }    
os.makedirs(os.path.join("results_CNN", dataset_name, "cal_portion_"+cal_portion_str, "seed_"+exp_seed_str), exist_ok=True)      
with open(os.path.join("results_CNN", dataset_name, "cal_portion_"+cal_portion_str, "seed_"+exp_seed_str, "results.pkl"), 'wb') as f:
    pickle.dump(results_dic, f)
