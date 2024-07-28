from Calculate_Statistics import calculate_statistics, prior_kde
from Create_Data_Stream import create_data_stream_stateful, obtain_prior_dist_sample, create_data_stream_stateful_roc
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import concurrent.futures
from Data_Pickler import state_retriever, state_saver
from Truncate_Tensors import truncate_tensor_1d, truncate_tensor_3d
import torch.nn as nn
import torch.optim as optim
from LSTM import LSTMModel
from copy import deepcopy
import random


def train_validate_lstm(all_train_validate, model_parameters):
    input_dim, hidden_dim, output_dim = model_parameters[0], model_parameters[1], model_parameters[2]
    num_layers, window_size, sequence_length = model_parameters[3], model_parameters[4], model_parameters[5]

    all_curr_training_x, all_curr_training_y = all_train_validate[0], all_train_validate[1]
    all_curr_validate_x, all_curr_validate_y = all_train_validate[2], all_train_validate[3]



    LSTM_model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    optimizer = optim.Adam(LSTM_model.parameters(), lr=0.001)

    CE_binary_loss = nn.BCELoss()
    best_model = LSTM_model

    for all_training_x, all_training_y, all_validate_x, all_validate_y in zip(all_curr_training_x, all_curr_training_y,
                                                                              all_curr_validate_x, all_curr_validate_y):
        lowest_error_model = None
        epoch, validate_network_loss = 0, float('inf')
        integrity_count = 100

        while integrity_count:
            print('\rEpoch: ' + str(epoch) + ' Integrity Count: ' + str(integrity_count) + '\tValidation Loss: ' + str(
                validate_network_loss), end='')
            paired_training_x_y = list(zip(all_training_x, all_training_y))
            paired_validate_x_y = list(zip(all_validate_x, all_validate_y))

            random.shuffle(paired_training_x_y)
            random.shuffle(paired_validate_x_y)

            all_training_x, all_training_y = zip(*paired_training_x_y)
            all_validate_x, all_validate_y = zip(*paired_validate_x_y)

            all_training_x, all_training_y = list(all_training_x), list(all_training_y)
            all_validate_x, all_validate_y = list(all_validate_x), list(all_validate_y)

            train(best_model, optimizer, CE_binary_loss, all_training_x, all_training_y)
            validate_network_loss = validate_test(best_model, CE_binary_loss, all_validate_x, all_validate_y)

            if lowest_error_model:
                if validate_network_loss < lowest_error_model[0]:
                    lowest_error_model = (validate_network_loss, deepcopy(best_model), deepcopy(optimizer.state_dict()))
                    integrity_count = 100
                else:
                    integrity_count -= 1
            else:
                lowest_error_model = (validate_network_loss, deepcopy(best_model), deepcopy(optimizer.state_dict()))

            epoch += 1

        best_model = lowest_error_model[1]
        best_model_state_dict = lowest_error_model[2]

        optimizer = optim.Adam(best_model.parameters(), lr=0.001)
        optimizer.load_state_dict(best_model_state_dict)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*.1

    return best_model


def train(LSTM_model, optimizer, CE_binary_loss, training_x, training_y):
    LSTM_model.train()

    for batch_inputs, batch_targets in zip(training_x, training_y):
        # Cuts off extraneous targets for a balanced training set
        batch_targets, cutoff_index = truncate_tensor_1d(batch_targets)
        batch_inputs = truncate_tensor_3d(batch_inputs, cutoff_index)

        batch_inputs, batch_targets = batch_inputs.float(), batch_targets.float()

        batch_size = batch_inputs.size(0)
        hidden = LSTM_model.init_hidden(batch_size)

        optimizer.zero_grad()  # Clear gradients

        # Forward pass
        outputs, hidden = LSTM_model(batch_inputs, hidden)
        outputs = outputs.squeeze(1)

        # Detach hidden state to prevent backpropagation through entire history
        (hidden[0].detach(), hidden[1].detach())

        # Compute loss for the batch
        network_loss = CE_binary_loss(outputs, batch_targets)

        # Backward propagate the network loss calculating gradients for all network weights
        network_loss.backward()

        # Perform updates to the weights using the gradients
        optimizer.step()


def validate_test(LSTM_model, CE_binary_loss, vt_x, vt_y):
    LSTM_model.eval()  # Set the model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation
        for batch_inputs, batch_targets in zip(vt_x, vt_y):
            # Initialize hidden state for each batch based on the batch size

            batch_inputs, batch_targets = batch_inputs.float(), batch_targets.float()

            batch_size = batch_inputs.size(0)
            hidden = LSTM_model.init_hidden(batch_size)

            # Forward pass
            outputs, hidden = LSTM_model(batch_inputs, hidden)
            outputs = outputs.squeeze(1)

            # Calculate loss for the batch
            loss = CE_binary_loss(outputs, batch_targets)
            total_loss += loss.item()

    vt_network_loss = total_loss / len(vt_y)

    return vt_network_loss