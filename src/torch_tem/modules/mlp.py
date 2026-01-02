import numpy as np
import torch


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation=(torch.nn.functional.elu, None), hidden_dim=None, bias=(True, True)):
        # First call super class init function to set up torch.nn.Module style model and inherit it's functionality
        super(MLP, self).__init__()
        # Check if this network consists of module: are input and output dimensions lists? If not, make them (but remember it wasn't)
        if type(in_dim) is list:
            self.is_list = True
        else:
            in_dim = [in_dim]
            out_dim = [out_dim]
            self.is_list = False
        # Find number of modules
        self.N = len(in_dim)
        # Create weights (input->hidden, hidden->output) for each module
        self.w = torch.nn.ModuleList([])
        for n in range(self.N):
            # If number of hidden dimensions is not specified: mean of input and output
            if hidden_dim is None:
                hidden = int(np.mean([in_dim[n], out_dim[n]]))
            else:
                hidden = hidden_dim[n] if self.is_list else hidden_dim
            # Each module has two sets of weights: input->hidden and hidden->output
            self.w.append(torch.nn.ModuleList([torch.nn.Linear(in_dim[n], hidden, bias=bias[0]), torch.nn.Linear(hidden, out_dim[n], bias=bias[1])]))
        # Copy activation function for hidden layer and output layer
        self.activation = activation
        # Initialise all weights
        with torch.no_grad():
            for from_layer in range(2):
                for n in range(self.N):
                    # Set weights to xavier initalisation
                    torch.nn.init.xavier_normal_(self.w[n][from_layer].weight)
                    # Set biases to 0
                    if bias[from_layer]:
                        self.w[n][from_layer].bias.fill_(0.0)

    def set_weights(self, from_layer, value):
        # If single value is provided: copy it for each module
        if type(value) is not list:
            input_value = [value for n in range(self.N)]
        else:
            input_value = value
        # Run through all modules and set weights starting from requested layer to the specified value
        with torch.no_grad():
            # MLP is setup as follows: w[module][layer] is Linear object, w[module][layer].weight is Parameter object for linear weights, w[module][layer].weight.data is tensor of weight values
            for n in range(self.N):
                # If a tensor is provided: copy the tensor to the weights
                if type(input_value[n]) is torch.Tensor:
                    self.w[n][from_layer].weight.copy_(input_value[n])
                # If only a single value is provided: set that value everywhere
                else:
                    self.w[n][from_layer].weight.fill_(input_value[n])

    def forward(self, data):
        # Make input data into list, if this network doesn't consist of modules
        if self.is_list:
            input_data = data
        else:
            input_data = [data]
        # Run input through network for each module
        output = []
        for n in range(self.N):
            # Pass through first weights from input to hidden layer
            module_output = self.w[n][0](input_data[n])
            # Apply hidden layer activation
            if self.activation[0] is not None:
                module_output = self.activation[0](module_output)
            # Pass through second weights from hidden to output layer
            module_output = self.w[n][1](module_output)
            # Apply output layer activation
            if self.activation[1] is not None:
                module_output = self.activation[1](module_output)
            # Transpose output again to go back to column vectors instead of row vectors
            output.append(module_output)
        # If this network doesn't consist of modules: select output from first module to return
        if not self.is_list:
            output = output[0]
        # And return output
        return output
