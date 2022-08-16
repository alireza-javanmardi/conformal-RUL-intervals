import numpy as np
import matplotlib.pyplot as plt

def compute_coverage_len(y_test, y_lower, y_upper):
    """ 
    https://github.com/yromano/cqr.git
    Compute average coverage and length of prediction intervals
    Parameters
    ----------
    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    Returns
    -------
    coverage : float, average coverage
    left_coverage : float, average correctness of the lower bounds
    avg_length : float, average length
    """
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    above_lower_bound = np.sum(y_test >= y_lower)
    coverage = in_the_range / len(y_test) * 100
    left_coverage = above_lower_bound / len(y_test) * 100
    avg_length = np.mean(abs(y_upper - y_lower))
    return left_coverage, coverage, avg_length


def plot_quantiles(y_test, y_lower, y_upper):

    plt.figure(figsize=(30, 10))
    plt.plot(y_lower, 'r', label='lower quantile')
    plt.plot(y_upper, 'b', label='upper quantile')
    plt.plot(y_test, 'k*', label='true')
    plt.legend(loc="upper left")  
    return 0


def compute_quantiles(y_test, y_lower, y_upper, alpha, n, indep=False, alpha_low=None, alpha_high=None):
    # nonconformity scores
    scores_low = y_lower - y_test
    scores_high = y_test - y_upper
    scores = np.maximum(scores_low, scores_high)
    # compute quantile of scores
    q = np.quantile(scores, np.ceil((n+1)*(1-alpha))/n)
    if not indep:
        return q
    else:
        if None in [alpha_low, alpha_high]:
            raise Exception("alpha_low and alpha_high has to be determined.")
        else:
            # compute different quantiles for upper and lower bounds
            q_low = np.quantile(scores_low, np.ceil((n+1)*(1-alpha_low))/n)
            q_high = np.quantile(scores_high, np.ceil((n+1)*(1-alpha_high))/n)
            return q, q_low, q_high
