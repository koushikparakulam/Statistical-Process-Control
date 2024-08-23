from Distributional_Shift.Test_LSTM import run_test
from Distributional_Shifts import distributional_shifts
from Data_Pickler import state_retriever, state_saver
import random
import numpy as np
from scipy.integrate import simpson
from Create_Train_Test_Validate import create_ttv
from scipy.stats import gaussian_kde, norm, t, lognorm
from scipy.special import rel_entr
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from All_Batch_Makers import make_roc, make_train_test_validate
import torch



def main():

    # Compute overlaps

    normal_dist_a = norm(loc=0, scale=1).rvs(size=1000)
    t_dist_a = t(df=2.5).rvs(size=1000) / np.sqrt(5)
    lognormal_dist_a = (lognorm(s=0.5, scale=np.exp(1)).rvs(size=1000) - 3) / 1.6

    print('\nCreating Normal Dist Set\n')
    normal_dist_b_overlaps_perms = distributional_shifts(normal_dist_a)
    state_saver('normal_dist_b_overlaps_perms', normal_dist_b_overlaps_perms)

    print('\nCreating T Dist Set\n')
    t_dist_b_overlaps_perms = distributional_shifts(t_dist_a)
    state_saver('t_dist_b_overlaps_perms', t_dist_b_overlaps_perms)

    print('\nCreating Lognorm Dist Set\n')
    lognormal_dist_b_overlaps_perms = distributional_shifts(lognormal_dist_a)
    state_saver('lognormal_dist_b_overlaps_perms', lognormal_dist_b_overlaps_perms)

    normal_dist_b_overlaps_perms = state_retriever('normal_dist_b_overlaps_perms')
    t_dist_b_overlaps_perms = state_retriever('t_dist_b_overlaps_perms')
    lognormal_dist_b_overlaps_perms = state_retriever('lognormal_dist_b_overlaps_perms')

    model_parameters = [5, 20, 1, 1, 20, 5]
    input_dim, hidden_dim, output_dim = model_parameters[0], model_parameters[1], model_parameters[2]
    num_layers, window_size, sequence_length = model_parameters[3], model_parameters[4], model_parameters[5]

    make_roc(model_parameters[4], model_parameters[5])
    make_train_test_validate(normal_dist_b_overlaps_perms, t_dist_b_overlaps_perms, lognormal_dist_b_overlaps_perms,
                             model_parameters)

    roc_path = input("Provide ROC Path File")
    lstm_state_file = input("Provide LSTM State File")

    run_test(roc_path, lstm_state_file)

if __name__=='__main__':
    main()
