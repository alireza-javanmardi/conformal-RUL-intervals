import os
import sys
import pickle 
import random
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor 
from sklearn.metrics import mean_pinball_loss, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

import src.data.preprocessor as pre
import src.data.datasets as data
import src.helper as h

dataset_name = sys.argv[1]
cal_portion_str = sys.argv[2]
cal_portion = float(cal_portion_str)
exp_seed_str = sys.argv[3]
exp_seed = int(exp_seed_str)

#Set the Python seed and the NumPy seed
os.environ['PYTHONHASHSEED']=str(exp_seed)
random.seed(exp_seed)
np.random.seed(exp_seed)

#--------------------------------------------------------------------------------------------------------------------------------
#CMAPSS data removable and ignorable cols
removable_cols = ["sm01", "sm05", "sm06", "sm10", "sm16", "sm18", "sm19"]
ignore_columns = ["time", "os1", "os2", "os3"]

alpha_array = np.array([0.1, 0.15, 0.2, 0.25]) #array of miscoverage rates
quantiles = np.concatenate((alpha_array, np.array([0.5]), 1-alpha_array))

#load, split, and preprocess the specified dataset 
dataset = data.get_dataset(dataset_name, MinMaxScaler(feature_range=(-1, 1)))
split_dataset = pre.split_dataset(dataset, cal_size=cal_portion, random_state=exp_seed)
proc_dataset = pre.preprocess_split(split_dataset, scaler_factory=dataset["scaler_factory"], window_size=1, removable_cols=removable_cols, ignore_columns=ignore_columns)
#X and y
X_train = proc_dataset["train"]["X"].reshape((-1,14))
y_train = proc_dataset["train"]["y"].reshape(-1)
X_cal = proc_dataset["cal"]["X"].reshape((-1,14))
y_cal = proc_dataset["cal"]["y"].reshape(-1)
idx_cal = proc_dataset["cal"]["index"]
X_test, y_test, idx_test = h.reform_test_data(proc_dataset["test"])
X_test = X_test.reshape((-1,14))
y_test = y_test.reshape(-1)
#piecewise RUL definition
y_train[y_train>125] = 125
y_cal[y_cal>125] = 125
y_test[y_test>125] = 125
all_models = {}
for q in quantiles:
    gbr = HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=exp_seed)
    all_models["q %1.2f" % q] = gbr.fit(X_train, y_train)
    print("q", q)
    print("training RMSE of GB:", np.sqrt(mean_squared_error(y_train, gbr.predict(X_train))))
    print("calibration RMSE of GB:", np.sqrt(mean_squared_error(y_cal, gbr.predict(X_cal))))
    print("test RMSE of GB:", np.sqrt(mean_squared_error(y_test, gbr.predict(X_test))))

gbr_ls = HistGradientBoostingRegressor(loss="squared_error", random_state=exp_seed)
all_models["mse"] = gbr_ls.fit(X_train, y_train)
print("mse")
print("training RMSE of GB:", np.sqrt(mean_squared_error(y_train, gbr_ls.predict(X_train))))
print("calibration RMSE of GB:", np.sqrt(mean_squared_error(y_cal, gbr_ls.predict(X_cal))))
print("test RMSE of GB:", np.sqrt(mean_squared_error(y_test, gbr_ls.predict(X_test))))
#--------------------------------------------------------------------------------------------------------------------------------
#Difficulty estimate of the datapoints using a simple random forest with default configurations of sklearn
y_hat_train = gbr_ls.predict(X_train)
y_hat_cal = gbr_ls.predict(X_cal)
y_hat_test = gbr_ls.predict(X_test)
res_train = np.abs(y_hat_train - y_train) 
res_cal = np.abs(y_hat_cal - y_cal) 

RF = RandomForestRegressor(random_state=exp_seed) 
RF.fit(X_train, res_train)
print("Random forest details:")
print("average labels (training absolute residuals of GB):", res_train.mean())
print("training mean absolute error of RF:", mean_absolute_error(res_train, RF.predict(X_train)))
print("calibration mean absolute error of RF:", mean_absolute_error(res_cal, RF.predict(X_cal)))
#--------------------------------------------------------------------------------------------------------------------------------
rho = 0.99
sigma_cal = RF.predict(X_cal)
sigma_test = RF.predict(X_test)
y_hat_cal_CQR ={}
y_hat_test_CQR ={}
for q in quantiles:
    y_hat_cal_CQR["q %1.2f" % q] = all_models["q %1.2f" % q].predict(X_cal)
    y_hat_test_CQR["q %1.2f" % q] = all_models["q %1.2f" % q].predict(X_test)

scores = np.abs(y_cal - y_hat_cal) 
scores_normalized = scores/sigma_cal

 
left_cvg_len_dic = {}
for i in range(len(alpha_array)):
    alpha = alpha_array[i]
    q = h.compute_quantile(scores, alpha)
    q_array = h.compute_quantiles_nex(rho, scores.reshape((-1,1)), idx_test, idx_cal, alpha).reshape(-1)
    q_normalized = h.compute_quantile(scores_normalized, alpha)
    q_array_normalized = h.compute_quantiles_nex(rho, scores_normalized.reshape((-1,1)), idx_test, idx_cal, alpha).reshape(-1)

    scores_low = y_hat_cal_CQR["q %1.2f" % alpha] - y_cal
    scores_high = y_cal - y_hat_cal_CQR["q %1.2f" % (1-alpha)] 
    scores_CQR = np.maximum(scores_low, scores_high)
    q_CQR = h.compute_quantile(scores_CQR, alpha)

    intervals_dic = {
        "SCP": (np.maximum(0,y_hat_test - q), y_hat_test + q),
        "nex-SCP": (np.maximum(0, y_hat_test  - q_array), y_hat_test  + q_array),
        "adaptive SCP": (np.maximum(0,y_hat_test - q_normalized*sigma_test), y_hat_test + q_normalized*sigma_test),
        "adaptive nex-SCP": (np.maximum(0, y_hat_test  - q_array_normalized*sigma_test), y_hat_test  + q_array_normalized*sigma_test),
        "CQR": (np.maximum(0, y_hat_test_CQR["q %1.2f" % alpha] - q_CQR), y_hat_test_CQR["q %1.2f" % (1-alpha)] + q_CQR)
        }

    colors_dic = {
        "SCP": "blue",
        "nex-SCP": "red",
        "adaptive SCP": "darkblue",
        "adaptive nex-SCP": "darkred",
        "CQR": "green"
        }    
    single_points_dic = {
        "SCP": y_hat_test,
        "nex-SCP": y_hat_test,
        "adaptive SCP": y_hat_test,
        "adaptive nex-SCP": y_hat_test,
        "CQR": y_hat_test_CQR["q %1.2f" % 0.5]
        } 
    os.makedirs(os.path.join("result_figs_GB", dataset_name, "cal_portion_"+cal_portion_str, "seed_"+exp_seed_str, "alpha_"+str(alpha)), exist_ok=True)
    LCL_alpha_dic = {}    
    for k in intervals_dic.keys():
        h.plot_sorted_targets_intervals(intervals_dic[k], single_points_dic[k], y_test, colors_dic[k], k+" intervals")
        plt.savefig(os.path.join("result_figs_GB", dataset_name, "cal_portion_"+cal_portion_str, "seed_"+exp_seed_str, "alpha_"+str(alpha), k+'.pdf'), format='pdf', bbox_inches='tight', pad_inches=0)
        LCL_alpha_dic[k] = h.compute_coverage_len(y_test, intervals_dic[k][0], intervals_dic[k][1])

    left_cvg_len_dic[alpha] = LCL_alpha_dic
os.makedirs(os.path.join("result_cvgs_GB", dataset_name, "cal_portion_"+cal_portion_str, "seed_"+exp_seed_str), exist_ok=True)      
with open(os.path.join("result_cvgs_GB", dataset_name, "cal_portion_"+cal_portion_str, "seed_"+exp_seed_str, "left_cvg_len_dic.pkl"), 'wb') as f:
    pickle.dump(left_cvg_len_dic, f)