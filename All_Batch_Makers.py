from Calculate_Statistics import calculate_statistics, prior_kde
from Create_Data_Stream import create_data_stream_stateful, obtain_prior_dist_sample, create_data_stream_stateful_roc
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from Data_Pickler import state_retriever, state_saver
from Create_Train_Test_Validate import create_ttv


def create_batch(shift_point, mu_shift, sigma_shift, dist_type, window_size, sequence_length, roc=False):
    # Pre-calculate the input and target stream
    if not roc:
        input_stream, target_stream = create_data_stream_stateful(shift_point, mu_shift, sigma_shift, dist_type)
    else:
        input_stream, target_stream = create_data_stream_stateful_roc(dist_type)
    prior_dist_data = obtain_prior_dist_sample(dist_type)

    # Precompute KDE for prior distribution
    p_kde = prior_kde(prior_dist_data)

    # Calculate the total number of windows and preallocate arrays
    num_windows = len(input_stream) - window_size + 1
    max_batches = (num_windows // sequence_length) + (1 if num_windows % sequence_length != 0 else 0)
    all_windows = np.empty((max_batches, sequence_length, 5))
    all_targets = np.empty(max_batches)

    temp_index = 0
    batch_index = 0
    start_index = 0

    for datapoint in range(num_windows):
        if datapoint+sequence_length+window_size-1 <= len(input_stream):
            if temp_index == 0:
                start_index = datapoint
            current_dist_data = input_stream[datapoint:datapoint + window_size]

            # Compute statistics for the current window
            input_features = calculate_statistics(prior_dist_data, current_dist_data, p_kde)
            all_windows[batch_index, temp_index] = input_features

            # If we've collected enough windows, process the batch
            if temp_index == sequence_length - 1:
                # Calculate the ratio of in-distribution data
                current_all_targets = target_stream[start_index:datapoint+window_size]
                cat_1 = np.sum(current_all_targets == 1)
                all_targets[batch_index] = cat_1 / len(current_all_targets)

                # Standardize features
                all_windows[batch_index] = StandardScaler().fit_transform(all_windows[batch_index])

                # Move to the next batch
                temp_index = 0
                batch_index += 1
            else:
                temp_index += 1
        else:
            break

    # Remove unused preallocated space if any
    all_windows = all_windows[:batch_index]
    all_targets = all_targets[:batch_index]

    return torch.from_numpy(all_windows), torch.from_numpy(all_targets)


# Doesn't account for input by input delivery, simply takes the whole input and..
# Calculates statistics, standardizes based on all windows, and does this for all batches
def create_all_batches(input_dist, window_size, sequence_length):
    all_batches_input = []
    all_batches_targets = []

    # for shift_point, mu_shift, sigma_shift, dist_type in input_dist:
    max_workers = min(20, len(input_dist))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        all_batches = [executor.submit(create_batch, shift_point, mu_shift, sigma_shift, dist_type, window_size,
                                       sequence_length) for
                       shift_point, mu_shift, sigma_shift, dist_type in input_dist]
        for current_batch in concurrent.futures.as_completed(all_batches):
            all_seq_windows, all_targets = current_batch.result()
            all_batches_input.append(all_seq_windows)
            all_batches_targets.append(all_targets)
    '''batch = 1
    for shift_point, mu_shift, sigma_shift, dist_type in input_dist:
        print('\rBatch: '+str(batch) +'\tTotal Batches: '+str(len(input_dist)), end='')
        all_seq_windows, all_targets = create_batch(shift_point, mu_shift, sigma_shift, dist_type, window_size,
                                                    sequence_length)
        all_batches_input.append(all_seq_windows)
        all_batches_targets.append(all_targets)
        batch+=1'''
    return all_batches_input, all_batches_targets


def train_validate_batch_maker(curr_train, curr_validate, model_parameters):
    window_size, sequence_length = model_parameters[4], model_parameters[5]

    print('\nCreating Training Set\n')
    all_training_x, all_training_y = create_all_batches(curr_train, window_size, sequence_length)

    print('\nCreating Validation Set\n')
    all_validate_x, all_validate_y = create_all_batches(curr_validate, window_size, sequence_length)

    return all_training_x, all_training_y, all_validate_x, all_validate_y


def test_batch_maker(curr_test, model_parameters):

    window_size, sequence_length = model_parameters[4], model_parameters[5]

    print('\nCreating Testing Set\n')
    all_test_x, all_test_y = create_all_batches(curr_test, window_size, sequence_length)

    return all_test_x, all_test_y


def make_roc(window_size, sequence_length):

    print('\nCreating Norm ROC Set\n')
    norm_roc_x, norm_roc_y = create_batch(None, None, None, 'norm', window_size, sequence_length, True)

    print('\nCreating T ROC Set\n')
    t_roc_x, t_roc_y = create_batch(None, None, None, 't', window_size, sequence_length, True)

    print('\nCreating LogNorm ROC Set\n')
    lognorm_roc_x, lognorm_roc_y = create_batch(None, None, None, 'lognorm', window_size, sequence_length, True)

    all_roc_x = [norm_roc_x, t_roc_x, lognorm_roc_x]
    all_roc_y = [norm_roc_y, t_roc_y, lognorm_roc_y]

    state_saver('all_roc', [all_roc_x, all_roc_y])


def make_train_test_validate(normal_dist_b_overlaps_perms, t_dist_b_overlaps_perms, lognormal_dist_b_overlaps_perms,
                             model_parameters):
    all_dist_test = []
    all_training_x, all_training_y, all_validate_x, all_validate_y = [], [], [], []
    all_test_x, all_test_y = [],[]

    for overlap in normal_dist_b_overlaps_perms.keys():
        print('\nOverlap: '+str(overlap)+'\n')
        curr_cirriculm_train, curr_cirriculm_test, curr_cirriculm_validate = \
            create_ttv(normal_dist_b_overlaps_perms, t_dist_b_overlaps_perms, lognormal_dist_b_overlaps_perms, overlap)
        training_x, training_y, validate_x, validate_y = train_validate_batch_maker(curr_cirriculm_train,
                                                                                    curr_cirriculm_validate,
                                                                                    model_parameters)
        test_x, test_y = test_batch_maker(curr_cirriculm_test, model_parameters)

        all_training_x.append(training_x)
        all_training_y.append(training_y)
        all_validate_x.append(validate_x)
        all_validate_y.append(validate_y)
        all_test_x.append(test_x)
        all_test_y.append(test_y)


        all_train_validate = [all_training_x, all_training_y, all_validate_x, all_validate_y]
        state_saver('all_train_validate', all_train_validate)

        all_test = [all_test_x, all_test_y]
        state_saver('all_test', all_test)

        #all_dist_test.extend(curr_cirriculm_test)


    all_train_validate = [all_training_x, all_training_y, all_validate_x, all_validate_y]
    state_saver('all_train_validate', all_train_validate)

    all_test = [all_test_x, all_test_y]
    state_saver('all_test', all_test)
