import numpy as np
from scipy.stats import gaussian_kde, norm, t, lognorm
from scipy.special import rel_entr
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance


def prior_kde(prior_dist_data):
    kde1 = gaussian_kde(prior_dist_data)
    return kde1


def calculate_statistics(prior_dist_data, current_dist_data, kde1):
    kde2 = gaussian_kde(current_dist_data)

    # Define a range of points where we will evaluate the KDEs
    # This range should cover the support of both distributions adequately
    x_min = min(np.min(prior_dist_data), np.min(current_dist_data))
    x_max = max(np.max(prior_dist_data), np.max(current_dist_data))
    x = np.linspace(x_min, x_max, 1000)

    # Evaluate the PDFs on the range of points
    pdf1 = kde1(x)
    pdf2 = kde2(x)

    # To avoid division by zero, we add a small constant to the PDFs
    epsilon = 1e-10
    pdf1 += epsilon
    pdf2 += epsilon


    kl_divergence = np.sum(rel_entr(pdf1, pdf2) * (x[1] - x[0])) # Kullback–Leibler
    tv_distance =  0.5 * np.sum(np.abs(pdf1 - pdf2)) # Total Variational Distance
    hellinger_distance = np.sqrt(np.sum((np.sqrt(pdf1) - np.sqrt(pdf2))**2)) / np.sqrt(2) # Hellinger Distance
    ks_stat, ks_p_value = ks_2samp(prior_dist_data, current_dist_data) # Kolmogorov-Smirnov Test and p-value
    emd = wasserstein_distance(prior_dist_data, current_dist_data) # Earth mover distance

    input_features = np.array([kl_divergence, tv_distance, hellinger_distance, ks_stat, emd])

    return input_features