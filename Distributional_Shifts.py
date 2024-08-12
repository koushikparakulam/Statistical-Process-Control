
import numpy as np
from itertools import product
from scipy.stats import gaussian_kde, norm, t, lognorm
from scipy.integrate import simpson
import random


def compute_pdf(dist, x):
    """Compute the PDF of a distribution given an array of x-values."""
    if hasattr(dist, 'pdf'):
        return dist.pdf(x)
    elif isinstance(dist, gaussian_kde):
        return dist(x)
    else:
        raise ValueError("Unsupported distribution type")


def min_pdf(x, dist1, dist2):
    """Compute the minimum of two PDFs at given x-values."""
    pdf1 = compute_pdf(dist1, x)
    pdf2 = compute_pdf(dist2, x)
    return np.minimum(pdf1, pdf2)


def overlap_area(dist1, dist2, lower_limit, upper_limit, num_points=1000):
    """Compute the overlap area between two distributions."""
    x = np.linspace(lower_limit, upper_limit, num_points)
    y = min_pdf(x, dist1, dist2)
    area = simpson(y, x=x)
    return area, x


def total_area(dist, x):
    """Compute the total area under the PDF of a distribution over a range of x-values."""
    pdf = compute_pdf(dist, x)
    area = simpson(pdf, x=x)
    return area

#str(o_d)+'% to '+str(o_d +5)+'%'
def distributional_shifts(dist_a_transformed):
    all_shifts = random.sample([x for x in range(50, 300)], 5)
    overlap_discretization = [(o_d / 100, str((o_d+5)/10)) for o_d in range(5, 85, 10)]

    all_dist_b_overlaps_perms = {o_d[1]: [] for o_d in overlap_discretization}

    all_sigma_b = [sigma_shift / 10 for sigma_shift in range(1, 35, 1)]
    all_mu_b = [mu_shift / 100 for mu_shift in range(-220, 220, 10)]

    all_mu_sigma_b = [x for x in list(product(all_mu_b, all_sigma_b))]
    iter = 1

    #temp = []
    for mu, sigma in all_mu_sigma_b:
        print('\rSteps: '+ str(iter), '\tTotal: '+str(len(all_mu_sigma_b)), end='')

        dist_b_transformed = (dist_a_transformed + mu) * sigma

        dist_b = gaussian_kde(dist_b_transformed)
        dist_a = gaussian_kde(dist_a_transformed)

        lower_limit = min(np.percentile(dist_a_transformed, 0.1), np.percentile(dist_b_transformed, 0.1))
        upper_limit = max(np.percentile(dist_a_transformed, 99.9), np.percentile(dist_b_transformed, 99.9))

        # Compute the overlap area
        overlap, x = overlap_area(dist_a, dist_b, lower_limit, upper_limit)

        total_area1 = total_area(dist_a, x)
        total_area2 = total_area(dist_b, x)

        total_overlap = overlap / ((total_area1 + total_area2)/2)
        #temp.append((mu, sigma, total_overlap))

        for o_d in overlap_discretization:
            if (o_d[0] <= float(total_overlap) < o_d[0] + .1):
                all_dist_b_perms = all_dist_b_overlaps_perms[o_d[1]]
                all_dist_b_perms.append([(shift, mu, sigma) for shift in all_shifts])
                all_dist_b_overlaps_perms[o_d[1]] = all_dist_b_perms
                break

        iter+=1

    return all_dist_b_overlaps_perms







