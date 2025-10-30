from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron with support for multiple parallel modules."""

    def __init__(self, in_dim, out_dim, activation=(F.elu, None), hidden_dim=None, bias=(True, True)):
        super(MLP, self).__init__()

        # Handle list vs single module
        self.is_list = isinstance(in_dim, list)
        if not self.is_list:
            in_dim = [in_dim]
            out_dim = [out_dim]

        self.N = len(in_dim)
        self.w = nn.ModuleList([])

        for n in range(self.N):
            hidden = int(np.mean([in_dim[n], out_dim[n]])) if hidden_dim is None else (hidden_dim[n] if self.is_list else hidden_dim)
            self.w.append(nn.ModuleList([nn.Linear(in_dim[n], hidden, bias=bias[0]), nn.Linear(hidden, out_dim[n], bias=bias[1])]))

        self.activation = activation

        # Initialize weights
        with torch.no_grad():
            for from_layer in range(2):
                for n in range(self.N):
                    nn.init.xavier_normal_(self.w[n][from_layer].weight)
                    if bias[from_layer]:
                        self.w[n][from_layer].bias.fill_(0.0)

    def set_weights(self, from_layer, value):
        """Set weights for a specific layer."""
        input_value = value if isinstance(value, list) else [value for _ in range(self.N)]

        with torch.no_grad():
            for n in range(self.N):
                if isinstance(input_value[n], torch.Tensor):
                    self.w[n][from_layer].weight.copy_(input_value[n])
                else:
                    self.w[n][from_layer].weight.fill_(input_value[n])

    def forward(self, data):
        """Forward pass through the MLP."""
        input_data = data if self.is_list else [data]
        output = []

        for n in range(self.N):
            module_output = self.w[n][0](input_data[n])
            if self.activation[0] is not None:
                module_output = self.activation[0](module_output)

            module_output = self.w[n][1](module_output)
            if self.activation[1] is not None:
                module_output = self.activation[1](module_output)

            output.append(module_output)

        return output if self.is_list else output[0]
