
from Distributional_Shift.All_Batch_Makers import create_all_batches
from Distributional_Shift.Data_Pickler import state_retriever
from Distributional_Shift.Find_ROC import false_negative_roc
from Distributional_Shift.LSTM import LSTMModel
import torch

def run_test(roc_path, lstm_state_file):
    all_sigma = [.2, .33, .5, 3.0, 2.0, 1.5]
    all_mu = [.25, .5, .75, 1.0, 1.5, 2.0]
    all_dist = ['norm', 't', 'lognorm']
    all_shift_point = [50, 300]

    control_sigma_shifts = []
    control_mu_shifts = []

    for dist_type in all_dist:
        for shift_point in all_shift_point:
            control_sigma_shifts.extend([(shift_point, 0, sigma, dist_type) for sigma in all_sigma])
            control_mu_shifts.extend([(shift_point, mu, 1, dist_type) for mu in all_mu])

    model_parameters = [5, 1024, 1, 1, 20, 5]

    control_sigma_input_batches, control_sigma_target_batches = create_all_batches(control_sigma_shifts,
                                                                                   model_parameters[4],
                                                                                   model_parameters[5])
    control_mu_input_batches, control_mu_target_batches = create_all_batches(control_mu_shifts, model_parameters[4],
                                                                             model_parameters[5])


    all_roc = state_retriever(roc_path)

    lstm_state_dict = state_retriever(lstm_state_file)
    lstm_model = LSTMModel(model_parameters[0], model_parameters[1], model_parameters[2], model_parameters[3])
    lstm_model.load_state_dict(lstm_state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model.to(device)

    best_cutoff = false_negative_roc(lstm_model, all_roc[0], all_roc[1], 1 / 500, device)

    all_detection_points_sigma = test(lstm_model, control_sigma_input_batches, control_sigma_target_batches,
                                      best_cutoff, device, model_parameters[4], model_parameters[5],
                                      control_sigma_shifts)
    all_detection_points_mu = test(lstm_model, control_mu_input_batches, control_mu_target_batches, best_cutoff, device,
                                   model_parameters[4], model_parameters[5], control_mu_shifts)

    for dp, shift_mu_sigma_dist in zip(all_detection_points_mu, control_mu_shifts):
        shift_point = shift_mu_sigma_dist[0]
        if dp != float('inf'):
            print(shift_mu_sigma_dist, dp)
        else:
            print(shift_mu_sigma_dist, float('inf'))

def test(LSTM_model, test_x, test_y, cutoff, device, window_size, sequence_length, all_shifts):
    LSTM_model.eval()  # Set the model to evaluation mode

    # Initialize counters
    TP = 0  # True Positives
    TN = 0  # True Negatives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    all_detection_points = []
    with torch.no_grad():  # Disable gradient calculation
        for batch_inputs, batch_targets, shifts in zip(test_x, test_y, all_shifts):
            batch_inputs, batch_targets = batch_inputs.float().to(device), batch_targets.float().to(device)
            batch_targets_mod = torch.where((batch_targets > 0) & (batch_targets < 1), torch.tensor(0.0), batch_targets)
            batch_size = batch_inputs.size(0)

            hidden = LSTM_model.init_hidden(batch_size)
            hidden = tuple([h.to(device) for h in hidden])

            # Forward pass
            outputs, hidden = LSTM_model(batch_inputs, hidden)
            outputs = outputs.squeeze(1)

            # Apply the cutoff to the outputs to get predictions
            predictions = (outputs > cutoff).float()
            predicted = predictions.tolist()
            shift_point = shifts[0]
            detection_point = float('inf')
            if 0 in predicted:
                all_n_i = [index for index, value in enumerate(predicted) if value == 0]
                for i, n in enumerate(all_n_i):
                    detection_point = (window_size+sequence_length-1)+(n*sequence_length)
                    if detection_point >= shift_point:
                        break
                    elif detection_point < shift_point and i <len(all_n_i)-1:
                        continue
                    else:
                        detection_point = float('inf')
            else:
                detection_point = float('inf')
            all_detection_points.append(detection_point-shift_point)

            # Update TP, TN, FP, FN counts
            TP += ((predictions == 1) & (batch_targets_mod == 1)).sum().item()
            TN += ((predictions == 0) & (batch_targets_mod == 0)).sum().item()
            FP += ((predictions == 1) & (batch_targets_mod == 0)).sum().item()
            FN += ((predictions == 0) & (batch_targets_mod == 1)).sum().item()

    # Calculate Precision, Recall, and F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    TPR = recall

    # False Positive Rate
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    # True Negative Rate (Specificity)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    # False Negative Rate
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    return all_detection_points