import torch
import torch.nn as nn
import torch.optim as optim


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        # batch_first simply means the output will have the batch as the first element
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        # Initialize hidden state and cell state
        # .to(x.device) ensures that the initial states are on the same device (either CPU or GPU)
        out, hidden = self.lstm(x, hidden)
        out = self.linear(out[:, -1, :])

        return self.sigmoid(out), hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))