import torch

def find_cutoff_index(tensor):
    # Find the number of 1. values
    num_ones = torch.sum(tensor == 1.0).item()

    # Find the first index where 0. values start
    first_zero_index = (tensor == 0.0).nonzero(as_tuple=True)[0][0].item()

    # Cutoff index: first_zero_index + num_ones - 1
    cutoff_index = first_zero_index + num_ones - 1

    return cutoff_index


def truncate_tensor_1d(tensor_1d):
    cutoff_index = find_cutoff_index(tensor_1d)
    return tensor_1d[:cutoff_index + 1], cutoff_index


def truncate_tensor_3d(tensor_3d, cutoff_index):
    return tensor_3d[:, :cutoff_index + 1]