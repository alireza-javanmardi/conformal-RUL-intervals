import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def compute_coverage_len(y, y_lower, y_upper):
    """Compute average coverage and length of prediction intervals
    Originally from: https://github.com/yromano/cqr.git

    Args:
        y_test (np.array): true labels
        y_lower (np.array): estimated lower bound for the labels
        y_upper (np.array): estimated upper bound for the labels

    Returns:
        left_coverage :  average left coverage
        coverage (float): average coverage
        avg_length :  average prediction interval length
    """ 
    if y.shape != y_lower.shape or y.shape != y_upper.shape:
        raise ValueError("y, y_lower, and y_upper must have the same shape")
    else:
        in_the_range = np.sum((y >= y_lower) & (y <= y_upper))
        above_lower_bound = np.sum(y >= y_lower)

        left_coverage = above_lower_bound / len(y)
        coverage = in_the_range / len(y)
        avg_length = np.mean(abs(y_upper - y_lower))

        return left_coverage, coverage, avg_length


def compute_quantile(scores, alpha):
    """compute quantile from the scores

    Args:
        scores (list or np.array): scores of calibration data
        alpha (float): error rate in conformal prediction
    """
    n = len(scores)

    return np.quantile(scores, np.ceil((n+1)*(1-alpha))/n, method="inverted_cdf")


def calculate_weight(rho, idx_test, idx_cal):
    """calculating weights for nex-SCP

    Args:
        rho (float)
        idx_test: index of test data points
        idx_cal: index od calibration data points

    Returns:
        array of weights
    """
    return rho**np.abs(idx_test - idx_cal)

def compute_quantiles_nex(rho, scores, idx_test, idx_cal, alpha):
    """compute quantile from the scores for nex-SCP 

    Args:
        rho: for weight calculation
        scores: scores of calibration data
        idx_test: index of test data points
        idx_cal: index od calibration data points
        alpha: error rate in conformal prediction

    Returns:
        array of quantiles corresponding to each test point
    """
    sorted_scores_idx = scores.argsort(axis=0)
    sorted_scores = scores[sorted_scores_idx]

    q_list = []
    pos_list = []
    for i in idx_test: 

        weights = calculate_weight(rho, i, idx_cal[sorted_scores_idx]) 
        weights_normalized = weights/(weights.sum()+1)
        pos_i = np.where(weights_normalized.cumsum() >= 1-alpha)[0][0]
        qi = sorted_scores[pos_i]
        q_list.append(qi[0][0])
        pos_list.append(pos_i)
    return np.array(q_list).reshape((-1,1))



def reform_test_data(test_data):
    """return the last datapoint (X, y, index) of each series in test data 

    Args:
        test_data: test data

    Returns:
        three lists of X's, y's, and indexes of last datapoints in test data
    """
    X_test, y_test, idx_test = [], [], []
    for id in np.unique(test_data["id"]):
        X_test.append(test_data["X"][test_data["id"]==id][-1])
        y_test.append(test_data["y"][test_data["id"]==id][-1])
        idx_test.append(test_data["index"][test_data["id"]==id][-1])
    return np.array(X_test), np.array(y_test), np.array(idx_test) 

def PHM_score(y, y_hat):
    """compute the score function defined in the PHM challenge 2008

    Args:
        y (np.array): true labels
        y_hat (np.array): estimated labels

    Returns:
        float: computed score
    """
    d = y_hat - y
    return np.sum(np.where(d < 0, np.expm1(-d/13.), np.expm1(d/10.)))


def plot_sorted_targets_intervals(intervals, single_point_predictions, y, color, label):
    """plot sorted actual test RUL labels alongside their prediction intervals 

    Args:
        intervals (tuple): containing lower and upper bound of intervals
        single_point_predictions (np.array): for SCP and nex-SCP, simply the result of the single point predictor
        for CQR, the median (0.5-quantile)
        y (np.array): ground truth of test RULS
        color (str): interval color
        label (str): plot label
    """
    sorted_y_idx = y.argsort(axis=0).reshape(-1)
    sorted_y = y[sorted_y_idx]
    lower = intervals[0].reshape(-1)
    upper = intervals[1].reshape(-1)
    sorted_lower = lower[sorted_y_idx]
    sorted_upper = upper[sorted_y_idx]
    
    fig = plt.figure(figsize=(13, 10))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 25
    plt.fill_between(range(len(sorted_y)), sorted_lower, sorted_upper, color=color, alpha=0.2, label=label)
    plt.plot(range(len(sorted_y)), sorted_y, 'ok', label="Ground truth RULs", alpha=0.6)
    plt.plot(range(len(sorted_y)), single_point_predictions[sorted_y_idx], '--k', label="Single-point RUL predictions", alpha=0.6)
    plt.ylim([0,140])
    plt.xlabel('Test instances with increasing RUL')
    plt.ylabel('RUL')
    # plt.legend(loc='lower right')
            
def plot_train_history(hist_dic):
    """plot the hsitory of training 

    Args:
        hist_dic (dict): a dictionary contatining history of training for each model (DCNN and MQDCNN)
    """
    fig = plt.figure(figsize=(30, 10))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 35
    i = 1
    for model in hist_dic.keys():
        ax = fig.add_subplot(1,2,i)
        ax.plot(hist_dic[model].history['loss'], label='train')
        ax.plot(hist_dic[model].history['val_loss'], label='val')
        plt.ylabel(model + " loss")
        i= i + 1
    plt.legend(loc='upper right')
    

def violinplot_cvg_len_data(cvg_data, len_data, alphas, methods):
    """violin plots of the coverage and lengths in one figure

    Args:
        cvg_data(np.array): coverage matrix where rows represents the seeds, each len(alphas) columns belong to a CP method (shape: (len(seeds), len(alphas)*len(methods)))
        len_data(np.array): length matrix where rows represents the seeds, each len(alphas) columns belong to a CP method (shape: (len(seeds), len(alphas)*len(methods)))
        alphas(list): list of alphas
        methods(list): list of CP methods
    """

    labels = [] #labels for legend
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 25
    fig, ax1 = plt.subplots(figsize=(40, 10))
    ax1.set_ylabel('Average Coverage')
    violin_parts1 = ax1.violinplot(cvg_data, showmeans=True)
    violinplot_set_color(violin_parts1, 'tab:blue')

    ax1.set_xticks(np.arange(1, len(alphas)*len(methods)+1), labels=alphas*len(methods))
    ax1.set_xlim(0.25, len(alphas)*len(methods) + 0.75)
    ax1.set_xlabel(r'Miscoverage rate ($\alpha$)')

    #vertical lines for separating methods from each other
    for i in range(len(methods)-1):
        ax1.axvline(x=len(alphas)*(i+1)+ 0.5, color='k')
    
    #Horizontal lines for nominal coverage 
    for a in alphas:
        ax1.axhline(y=1-a, color='k', linestyle=":")
    ax1.set_ylim([0,1])
    labels.append((mpatches.Patch(color=violin_parts1['bodies'][0].get_facecolor()), 'Average Coverage'))



    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Average Width')  
    violin_parts2 = ax2.violinplot(len_data, showmeans=True)
    violinplot_set_color(violin_parts2, 'r')
    ax2.set_ylim([10,100])
    secax = ax1.secondary_xaxis('top')
    secax.set_xticks(np.arange((len(alphas)+1)/2, len(alphas)*len(methods), len(alphas)), labels=methods)
    labels.append((mpatches.Patch(color=violin_parts2['bodies'][0].get_facecolor()), 'Average Width'))
    plt.legend(*zip(*labels), loc="lower left")
    
def violinplot_set_color(violin_parts, color):
    """set the color of the violin plot

    Args:
        violin_parts: return values of the violinplot
        color (str): the color
    """
    for pc in violin_parts['bodies']:
        pc.set_color(color)
        pc.set_alpha(0.3)
    violin_parts['cbars'].set_color('k')
    violin_parts['cmaxes'].set_color('k')
    violin_parts['cmins'].set_color('k')
    violin_parts['cmeans'].set_color('k')
