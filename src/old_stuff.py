def compute_mu(cal_errors, distances, indices):
    """difficulty estimate for calibration/test point

    Args:
        cal_errors (np.array): an array of absolute errors of calibration points
        distances (np.array): an array of distances of K nearest neighbors (of either calibration or test points) 
        in calibration data 
        indices (np.array): an array of indices of K nearest neighbors (of either calibration or test points)
        in calibration data

        distances and indices are the outputs of the KNN alg!
    Returns:
        mu(np.array): _description_
    """
    mu = []
    for dist, idx in zip(distances, indices):
        nomi = ((cal_errors[idx]).reshape(-1)/dist).sum()
        denomi = (1/dist).sum()
        mu.append(nomi/denomi)
    return np.array(mu).reshape((-1,1))



    #divide calval data into calibration and validation, calibrate using the calibration splita and test it on validation split
R = 50
rho = 0.99

res = [] #each element is a dictionary that contains SCP, nex-SCP, CQR results for each alpha
for r in range(R):
    val_idx, cal_idx = pre.split_by_group(X=proc_dataset["calval"]["X"], groups=proc_dataset["calval"]["id"], n_splits=1, test_size=0.5, random_state=r+exp_seed)
    res.append(h.validate_calibration_epoch(proc_dataset["calval"], val_idx, cal_idx, DCNN, MQDCNN, alpha_array, rho))
for alpha in alpha_array:
    SCP, nex_SCP, CQR = zip(*[d[alpha] for d in res])
    validate_calibration_res = {"SCP": SCP, "nex-SCP": nex_SCP, "CQR": CQR}
    h.plot_validate_calibration(validate_calibration_res)
    plt.savefig(os.path.join("result_figs", dataset_name, "cal_portion_"+cal_portion_str, "seed_"+exp_seed_str,'validate_calibration'+str(alpha)+'.pdf'), format='pdf', bbox_inches='tight', pad_inches=0)
#--------------------------------------------------------------------------------------------------------------------------------



def validate_calibration_epoch(calval_data, val_idx, cal_idx, DCNN, MQDCNN, RF, alpha_array, rho):   
    """checking whether calibration works appropriately

    Args:
        calval_data: calibration+validation data
        val_idx: index of validation data points
        cal_idx: index of calibration data points
        DCNN: trained DCNN
        MQDCNN: trained MQDCNN
        RF: trained random forest
        alpha_array (np.array): array of error rates in conformal prediction
        rho (float): for weight calculation in nex-SCP

    Returns:
        a dictionary of average left coverage, coverage, and prediction interval length for SCP, nex-SCP, and CQR for 
        different alpha
    """
    X_val, X_cal = calval_data["X"][val_idx], calval_data["X"][cal_idx]
    y_val, y_cal = calval_data["y"][val_idx], calval_data["y"][cal_idx]
    idx_val, idx_cal = calval_data["index"][val_idx], calval_data["index"][cal_idx]

    y_hat_cal = DCNN.predict(x=X_cal, verbose=0)
    y_hat_val = DCNN.predict(x=X_val, verbose=0)
    sigma_cal = DCNN.predict(x=X_cal, verbose=0)
    sigma_val = DCNN.predict(x=X_val, verbose=0)
    scores = np.abs(y_cal - y_hat_cal) 
    y_hat_cal_CQR = MQDCNN.predict(x=X_cal, verbose=0)
    y_hat_val_CQR = MQDCNN.predict(x=X_val, verbose=0)
    res = {}
    for i in range(len(alpha_array)):
        alpha = alpha_array[i]
        q = compute_quantile(scores, alpha)
        q_array = compute_quantiles_nex(rho, scores, idx_val, idx_cal, alpha)
        SCP = compute_coverage_len(y_val, y_hat_val-q, y_hat_val+q)
        nex_SCP = compute_coverage_len(y_val, y_hat_val-q_array, y_hat_val+q_array)

    
        scores_low = y_hat_cal_CQR[i] - y_cal
        scores_high = y_cal - y_hat_cal_CQR[i+len(alpha_array)] 
        scores_CQR = np.maximum(scores_low, scores_high)
        q_CQR = compute_quantile(scores_CQR, alpha)
        CQR = compute_coverage_len(y_val, y_hat_val_CQR[i] - q_CQR, y_hat_val_CQR[i+len(alpha_array)] + q_CQR)
        res[alpha] = SCP, nex_SCP, CQR
    return res




    
def plot_validate_calibration(results):
    
    fig = plt.figure(figsize=(40, 10))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 35
    methods = ["SCP", "nex-SCP", "CQR"]
    colors = ["blue", "red", "green"]
    Labels = ["Average lower bound coverage", "Average coverage", "Average interval length"]
    for i in range(3):
        ax = fig.add_subplot(1,3,i+1)
        for m, c in zip(methods, colors):
            ax.hist(np.stack(results[m])[:,i], color=c, alpha=0.2, label=m)
        plt.xlabel(Labels[i])
    plt.legend(loc='upper right')