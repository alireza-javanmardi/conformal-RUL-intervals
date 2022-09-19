import numpy as np
import matplotlib.pyplot as plt

def compute_coverage_len(y, y_lower, y_upper):
    """Compute average coverage and length of prediction intervals
    Originally from: https://github.com/yromano/cqr.git

    Args:
        y_test (np.array): true labels
        y_lower (np.array): estimated lower bound for the labels
        y_upper (np.array): estimated upper bound for the labels

    Returns:
        coverage (float): average coverage
        left_coverage :  average left coverage
        avg_length :  average prediction interval length
    """
    in_the_range = np.sum((y >= y_lower) & (y <= y_upper))
    above_lower_bound = np.sum(y >= y_lower)

    left_coverage = above_lower_bound / len(y)
    coverage = in_the_range / len(y)
    avg_length = np.mean(abs(y_upper - y_lower))

    return left_coverage, coverage, avg_length


def plot_quantiles(y_test, y_lower, y_upper):

    plt.figure(figsize=(30, 10))
    plt.plot(y_lower, 'r', label='lower quantile')
    plt.plot(y_upper, 'b', label='upper quantile')
    plt.plot(y_test, 'k*', label='true')
    plt.legend(loc="upper left")  
    return 0




def compute_quantile(scores, alpha):
    """compute quantile from the scores

    Args:
        scores (list or np.array): scores of calibration data
        alpha (float): error rate in conformal prediction
    """
    n = len(scores)

    return np.quantile(scores, np.ceil((n+1)*(1-alpha))/n)


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