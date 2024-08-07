import numpy as np
from itertools import product
from scipy import stats

def min_pdf(x, mu1, sigma1, mu2, sigma2):
    dist1_pdf = stats.norm.pdf(x, mu1, sigma1)
    dist2_pdf = stats.norm.pdf(x, mu2, sigma2)
    return np.minimum(dist1_pdf, dist2_pdf)


def overlap_area(mu1, sigma1, mu2, sigma2):

    lower_limit = min(mu1 - 4 * sigma1, mu2 - 4 * sigma2)
    upper_limit = max(mu1 + 4 * sigma1, mu2 + 4 * sigma2)

    x = np.linspace(lower_limit, upper_limit, 1000)
    y = min_pdf(x, mu1, sigma1, mu2, sigma2)
    area = np.trapezoid(y, x)
    return area

def normal_shifts(mu_A, sigma_A):

    muB_lower, muB_upper = -1, 1
    sigmaB_lower, sigmaB_upper = .1, 3

    overlap_lower, overlap_upper = .4, .7

    all_muB = [x/10 for x in range(int(muB_lower*10), int((muB_upper*10)+1), 1)]
    all_sigmaB = [x/10 for x in range(int(sigmaB_lower*10), int((sigmaB_upper*10)+1), 1)]
    mu_sigma_distB = [x for x in list(product(all_muB, all_sigmaB)) if not ((x[0] == 0.0) and (x[1] == 1.0))]

    all_muB_sigmaB_permutations = []

    for mu_B, sigma_B in mu_sigma_distB:
        overlap = overlap_area(mu_A, sigma_A, mu_B, sigma_B)
        if (overlap >= overlap_lower) and (overlap <= overlap_upper):
            all_muB_sigmaB_permutations.append((mu_B, sigma_B, overlap))

    # Contains all mu, sigma for distB that has a 40% to 70% overlap with distA
    all_muB_sigmaB_permutations.sort(key = lambda x:x[2])

