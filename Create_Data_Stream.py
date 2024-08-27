from scipy.stats import gaussian_kde, norm, t, lognorm
import numpy as np
import random
from Calculate_Statistics import calculate_statistics

def obtain_prior_dist_sample(dist_type):
    dist_a_data = None

    if dist_type == 'norm':
        dist_a_data = list(norm(loc=0, scale=1).rvs(size=25))

    if dist_type == 't':
        dist_a_data = list(t(df=2.5).rvs(size=25) / np.sqrt(5))

    if dist_type == 'lognorm':
        dist_a_data = list((lognorm(s=0.5, scale=np.exp(1)).rvs(size=25) - 3) / 1.6)

    return dist_a_data


def create_data_stream_stateful_roc(dist_type):
    data_stream = None

    if dist_type == 'norm':
        data_stream = norm(loc=0, scale=1).rvs(size=3333)
        return data_stream, np.ones(3333)

    if dist_type == 't':
        data_stream = t(df=2.5).rvs(size=3333) / np.sqrt(5)
        return data_stream, np.ones(3333)

    if dist_type == 'lognorm':
        data_stream = (lognorm(s=0.5, scale=np.exp(1)).rvs(size=3334) - 3) / 1.6
        return data_stream, np.ones(3334)


def create_data_stream_stateful(shift_point, mu_shift, sigma_shift, dist_type):
    # Define dist_a and dist_a_data based on dist_type
    if dist_type == 'norm':
        dist_a = norm(loc=0, scale=1).rvs(size=10000)
        dist_a_data = norm(loc=0, scale=1).rvs(size=shift_point)
    elif dist_type == 't':
        dist_a = t(df=2.5).rvs(size=10000) / np.sqrt(5)
        dist_a_data = t(df=2.5).rvs(size=shift_point) / np.sqrt(5)
    elif dist_type == 'lognorm':
        dist_a = (lognorm(s=0.5, scale=np.exp(1)).rvs(size=10000) - 3) / 1.6
        dist_a_data = (lognorm(s=0.5, scale=np.exp(1)).rvs(size=shift_point) - 3) / 1.6
    else:
        raise ValueError("Unsupported distribution type")

    # Apply shift and scaling to dist_a
    dist_b = (dist_a + mu_shift) * sigma_shift

    # Sample from dist_b_data
    dist_b_data = np.random.choice(dist_b, size=500, replace=False)

    # Combine dist_a_data and dist_b_data
    data_stream = np.concatenate([dist_a_data, dist_b_data])

    # Create target stream: 1 for dist_a_data, 0 for dist_b_data
    target_stream = np.concatenate([np.ones(shift_point), np.zeros(500)])

    return data_stream, target_stream
