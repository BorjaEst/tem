# Standard modules
import torch

# Custom modules
import torch_tem.utils as utils


class LSTM(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=1, n_a=4):
        # First call super class init function to set up torch.nn.Module style model and inherit it's functionality
        super(LSTM, self).__init__()
        # LSTM layer
        self.lstm = torch.nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        # Hidden to output
        self.lin = torch.nn.Linear(hidden_dim, out_dim)
        # Copy number of actions, will be needed for input data vector
        self.n_a = n_a

    def forward(self, data, prev_hidden=None):
        # If previous hidden and cell state are not provided: initialise them randomly
        if prev_hidden is None:
            hidden_state = torch.randn(self.lstm.num_layers, data.shape[0], self.lstm.hidden_size)
            cell_state = torch.randn(self.lstm.num_layers, data.shape[0], self.lstm.hidden_size)
            prev_hidden = (hidden_state, cell_state)
        # Run input through lstm
        lstm_out, lstm_hidden = self.lstm(data, prev_hidden)
        # Apply linear network to lstm output to get output: prediction at each timestep
        lin_out = self.lin(lstm_out)
        # And since we want a one-hot prediciton: do softmax on top
        out = utils.softmax(lin_out)
        # Return output and hidden state
        return out, lstm_hidden

    def prepare_data(self, data_in):
        # Transform list of actions of each step into batch of one-hot row vectors
        actions = [torch.zeros((len(step[2]), self.n_a)).scatter_(1, torch.tensor(step[2]).unsqueeze(1), 1.0) for step in data_in]
        # Concatenate observation and action together along column direction in each step
        vectors = [torch.cat((step[1], action), dim=1) for step, action in zip(data_in, actions)]
        # Then stack all these together along the second dimension, which is sequence length
        data = torch.stack(vectors, dim=1)
        # Return data in [batch_size, seq_len, input_dim] dimension as expected by lstm
        return data
