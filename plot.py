import os
import sys
import pickle 
import numpy as np
import matplotlib.pyplot as plt


import src.helper as h

#this is an script for generating violinplots as well as separate prediction plots
#this file needs to be run for each (model, dataset, calibration_portion) pair: python plot.py CNN CMAPSS1 0.1 
model = sys.argv[1]
dataset_name = sys.argv[2]
cal_portion_str = sys.argv[3]
cal_portion = float(cal_portion_str)

#all seeds, alphas, and methods have to be brought here
seed_list = [10, 11, 15, 17, 18, 22, 24, 25, 26, 27, 28, 32, 33, 36, 37]
alpha_list = [0.1, 0.15, 0.2, 0.25]
method_list = ["SCP","SCP+NNM", "nex-SCP", "nex-SCP+NNM","CQR"]

colors_dic = {
    "SCP": "blue",
    "nex-SCP": "red",
    "SCP+NNM": "darkblue",
    "nex-SCP+NNM": "darkred",
    "CQR": "green"
} 



#-------------------------------------------------------------------------------------------------------------------------------


left_cvg_data = np.zeros((len(seed_list), len(alpha_list)*len(method_list)))
cvg_data = np.zeros((len(seed_list), len(alpha_list)*len(method_list)))
len_data = np.zeros((len(seed_list), len(alpha_list)*len(method_list)))

for s, seed in enumerate(seed_list):
    folder_addr = os.path.join("results_"+model, dataset_name, "cal_portion_"+cal_portion_str, "seed_"+str(seed))
    addr = os.path.join(folder_addr, "results.pkl")
    with open(addr, 'rb') as f:
        res = pickle.load(f)
    for a, alpha in enumerate(alpha_list):
        os.makedirs(os.path.join(folder_addr, "alpha_"+str(alpha)), exist_ok=True)
        for m, method in enumerate(method_list):
            if method=="CQR":
                y_hat_test = res["Single-point RUL predictions CQR"]
            else:
                y_hat_test = res["Single-point RUL predictions"]

            y_test = res["Ground truth RULs"]
            h.plot_sorted_targets_intervals(res["intervals"][alpha][method], y_hat_test, y_test, colors_dic[method], method+" intervals")
            plt.savefig(os.path.join(folder_addr, "alpha_"+str(alpha), method+'.pdf'), format='pdf', bbox_inches='tight', pad_inches=0)
            plt.close()

            left_cvg_data[s,len(alpha_list)*m +a], cvg_data[s,len(alpha_list)*m +a],  len_data[s,len(alpha_list)*m +a] = h.compute_coverage_len(y_test, res["intervals"][alpha][method][0], res["intervals"][alpha][method][1])

h.violinplot_cvg_len_data(cvg_data, len_data, alpha_list, method_list)
plt.savefig(os.path.join("results_"+model, dataset_name, "cal_portion_"+cal_portion_str, dataset_name+'.pdf'), format='pdf', bbox_inches='tight', pad_inches=0)