from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron with support for multiple parallel modules.

    This MLP implementation allows creating multiple independent two-layer networks
    that can be processed in parallel. This is useful for multi-frequency or
    multi-scale architectures where the same operation needs to be applied
    independently at different resolutions.

    Architecture:
    =============
    Each module consists of two linear layers:
        h = activation[0](W1·x + b1)
        y = activation[1](W2·h + b2)

    Where:
    - W1: Input-to-hidden weights [hidden_dim, in_dim]
    - W2: Hidden-to-output weights [out_dim, hidden_dim]
    - b1, b2: Optional biases
    - activation[0]: Hidden layer activation (e.g., ELU, tanh)
    - activation[1]: Output layer activation (e.g., None, exp, softmax)

    Parallel Processing:
    ====================
    When initialized with lists of dimensions, creates N independent modules
    that share the same architecture but have separate parameters. This enables
    efficient batch processing of multi-scale representations.

    Args:
        in_dim: Input dimensionality (int or list of ints for parallel modules)
        out_dim: Output dimensionality (int or list of ints for parallel modules)
        activation: Tuple of (hidden_activation, output_activation) functions.
            Use None for linear layers. Default: (F.elu, None)
        hidden_dim: Hidden layer size. If None, uses mean of in_dim and out_dim.
            Can be int (shared) or list (per module). Default: None
        bias: Tuple of (input_bias, output_bias) booleans. Default: (True, True)

    Attributes:
        is_list: Whether this MLP contains multiple parallel modules
        N: Number of parallel modules
        w: ModuleList of ModuleLists, where w[n] contains [layer1, layer2] for module n
        activation: Tuple of activation functions for hidden and output layers

    Example:
        # Single module
        mlp = MLP(10, 5, activation=(torch.relu, None))

        # Three parallel modules with different dimensions
        mlp = MLP([10, 20, 30], [5, 8, 12], activation=(F.elu, torch.exp))
    """

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
        """Set weights for a specific layer.

        Allows manual initialization or modification of layer weights after
        construction. Useful for setting specific initialization schemes or
        loading pretrained weights.

        Args:
            from_layer: Layer index to modify (0 = input→hidden, 1 = hidden→output)
            value: Weight tensor or scalar to set. Can be:
                - Single value: Applied to all modules
                - List of values: One per module
                - Tensor: Directly copied to layer weights
                - Scalar: Fills entire weight matrix with this value
        """
        input_value = value if isinstance(value, list) else [value for _ in range(self.N)]

        with torch.no_grad():
            for n in range(self.N):
                if isinstance(input_value[n], torch.Tensor):
                    self.w[n][from_layer].weight.copy_(input_value[n])
                else:
                    self.w[n][from_layer].weight.fill_(input_value[n])

    def forward(self, data):
        """Forward pass through the MLP.

        Processes input through all parallel modules independently, applying
        the two-layer transformation with specified activations.

        Args:
            data: Input tensor or list of input tensors. If MLP was initialized
                with list of dimensions, expects list of tensors (one per module).
                Otherwise, expects single tensor.

        Returns:
            Output tensor or list of output tensors, matching input format.
            Each output has shape [batch, out_dim[n]] for module n.
        """
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
