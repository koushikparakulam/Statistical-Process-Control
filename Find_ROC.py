import torch


def false_negative_roc(LSTM_model, all_roc_x, all_roc_y, fn_rate):
    LSTM_model.eval()  # Set the model to evaluation mode
    total_target_size= sum([len(batch_t) for batch_t in all_roc_y])
    all_fn, all_cutoff = [],[]

    for threshold in range(10, 1000, 10):

        cutoff_threshold, total_fn = threshold/1000, 0

        with torch.no_grad():  # Disable gradient calculation
            for batch_inputs, batch_targets in zip(all_roc_x, all_roc_y):
                # Initialize hidden state for each batch based on the batch size

                batch_inputs, batch_targets = batch_inputs.float(), batch_targets.float()

                batch_size = batch_inputs.size(0)
                hidden = LSTM_model.init_hidden(batch_size)

                # Forward pass
                outputs, hidden = LSTM_model(batch_inputs, hidden)
                outputs = outputs.squeeze(1)
                all_false_negatives = outputs <= cutoff_threshold

                total_fn += len(all_false_negatives)

        current_fn_rate = (total_fn/total_target_size)

        if current_fn_rate == fn_rate:
            optimal_cutoff = cutoff_threshold
            return optimal_cutoff

        all_fn.append(current_fn_rate)
        all_cutoff.append(cutoff_threshold)

    closest_i = min(range(len(all_fn)), key=lambda i: abs((all_fn[i]) - fn_rate))

    return all_cutoff[closest_i]