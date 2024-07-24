import random


def ttv_condense(train_unmod, test_unmod, validate_unmod, dist_type):
    train = [distribution+(dist_type,) for distribution_shifts in train_unmod for distribution in distribution_shifts]
    test = [distribution + (dist_type,) for distribution_shifts in test_unmod for distribution in distribution_shifts]
    validate = [distribution + (dist_type,) for distribution_shifts in validate_unmod for distribution in distribution_shifts]

    return train, test, validate


def ttv_split(train_test_validate, percentages, dist_type):
    """Split the list into parts based on the given percentages."""
    # Ensure the percentages add up to 100%
    if sum(percentages) != 100:
        raise ValueError("Percentages must sum to 100")

    # Shuffle the list to randomize the order
    random.shuffle(train_test_validate)

    # Calculate the split points
    total_length = len(train_test_validate)
    train = int(total_length * (percentages[0] / 100))
    test = train + int(total_length * (percentages[1] / 100))

    # Split the list
    train_data = train_test_validate[:train]
    test_data = train_test_validate[train:test]
    validate_data = train_test_validate[test:]

    train_data, test_data, validate_data = ttv_condense(train_data, test_data, validate_data, dist_type)

    return train_data, test_data, validate_data


def create_ttv(normal_dist_b_overlaps_perms, t_dist_b_overlaps_perms, lognormal_dist_b_overlaps_perms, overlap):
    min_length_overlaps = {}
    # Get the lists from each dictionary
    norm_perm = normal_dist_b_overlaps_perms[overlap]
    t_perm = t_dist_b_overlaps_perms[overlap]
    lognorm_perm = lognormal_dist_b_overlaps_perms[overlap]

    # Find the minimum length list
    min_length_perms = min([len(norm_perm), len(t_perm), len(lognorm_perm)])

    # Store the minimum length list in the new dictionary
    min_length_overlaps[overlap] = min_length_perms

    minimum_sample_size = min(list(min_length_overlaps.values()))

    norm_dist = random.sample(list(normal_dist_b_overlaps_perms[overlap]), minimum_sample_size)
    t_dist = random.sample(list(t_dist_b_overlaps_perms[overlap]), minimum_sample_size)
    lognorm_dist = random.sample(list(lognormal_dist_b_overlaps_perms[overlap]), minimum_sample_size)

    norm_train, norm_test, norm_validate = ttv_split(norm_dist, [40, 40, 20], 'norm')
    t_train, t_test, t_validate = ttv_split(t_dist, [40, 40, 20], 't')
    lognorm_train, lognorm_test, lognorm_validate = ttv_split(lognorm_dist, [40, 40, 20], 'lognorm')

    curr_cirriculm_train = norm_train + t_train + lognorm_train
    curr_cirriculm_test = norm_test + t_test + lognorm_test
    curr_cirriculm_validate = norm_validate + t_validate + lognorm_validate

    return curr_cirriculm_train, curr_cirriculm_test, curr_cirriculm_validate