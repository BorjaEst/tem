"""Multi-Layer Perceptron (MLP) utilities for torch_tem package."""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with support for multiple parallel modules.

    Can be used as a single MLP or multiple parallel MLPs (one per frequency module).
    Supports custom activation functions per layer and optional bias terms.
    """

    def __init__(
        self,
        in_dim: Union[int, List[int]],
        out_dim: Union[int, List[int]],
        activation: Tuple[Optional[Callable], Optional[Callable]] = (torch.nn.functional.elu, None),
        hidden_dim: Optional[Union[int, List[int]]] = None,
        bias: Tuple[bool, bool] = (True, True),
    ):
        """Initialize MLP.

        Args:
            in_dim: Input dimension(s). If list, creates multiple parallel modules.
            out_dim: Output dimension(s). Must match in_dim list structure.
            activation: Tuple of (hidden_activation, output_activation).
                       None means no activation.
            hidden_dim: Hidden layer dimension(s). If None, uses mean of in/out dims.
            bias: Tuple of (use_bias_hidden, use_bias_output) flags.
        """
        super().__init__()

        # Check if this network consists of modules
        if isinstance(in_dim, list):
            self.is_list = True
        else:
            in_dim = [in_dim]
            out_dim = [out_dim]
            self.is_list = False

        # Find number of modules
        self.N = len(in_dim)

        # Create weights (input->hidden, hidden->output) for each module
        self.w = nn.ModuleList([])
        for n in range(self.N):
            # If number of hidden dimensions is not specified: mean of input and output
            if hidden_dim is None:
                hidden = int(np.mean([in_dim[n], out_dim[n]]))
            else:
                hidden = hidden_dim[n] if self.is_list else hidden_dim

            # Each module has two sets of weights: input->hidden and hidden->output
            self.w.append(nn.ModuleList([nn.Linear(in_dim[n], hidden, bias=bias[0]), nn.Linear(hidden, out_dim[n], bias=bias[1])]))

        # Copy activation function for hidden layer and output layer
        self.activation = activation

        # Initialize all weights
        self._initialize_weights(bias)

    def _initialize_weights(self, bias: Tuple[bool, bool]):
        """Initialize weights with Xavier initialization and zero biases."""
        with torch.no_grad():
            for from_layer in range(2):
                for n in range(self.N):
                    # Set weights to Xavier initialization
                    nn.init.xavier_normal_(self.w[n][from_layer].weight)
                    # Set biases to 0
                    if bias[from_layer]:
                        self.w[n][from_layer].bias.fill_(0.0)

    def set_weights(self, from_layer: int, value: Union[float, torch.Tensor, List]):
        """Set weights of a specific layer.

        Args:
            from_layer: Layer index (0 for input->hidden, 1 for hidden->output).
                       Use -1 for the last layer.
            value: Value to set. Can be:
                  - float: fills all weights with this value
                  - Tensor: copies tensor to weights
                  - List[Tensor]: one tensor per module
        """
        # Handle negative indexing
        if from_layer < 0:
            from_layer = 2 + from_layer

        # If single value is provided: copy it for each module
        if not isinstance(value, list):
            input_value = [value for n in range(self.N)]
        else:
            input_value = value

        # Set weights for each module
        with torch.no_grad():
            for n in range(self.N):
                # If a tensor is provided: copy the tensor to the weights
                if isinstance(input_value[n], torch.Tensor):
                    self.w[n][from_layer].weight.copy_(input_value[n])
                # If only a single value is provided: set that value everywhere
                else:
                    self.w[n][from_layer].weight.fill_(input_value[n])

    def get_weights(self, from_layer: int) -> List[torch.Tensor]:
        """Get weights of a specific layer.

        Args:
            from_layer: Layer index (0 for input->hidden, 1 for hidden->output).
                       Use -1 for the last layer.

        Returns:
            List of weight tensors, one per module
        """
        # Handle negative indexing
        if from_layer < 0:
            from_layer = 2 + from_layer

        return [self.w[n][from_layer].weight for n in range(self.N)]

    def forward(self, data: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the MLP.

        Args:
            data: Input data. Single tensor or list of tensors (one per module).

        Returns:
            Output data. Same structure as input.
        """
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

            output.append(module_output)

        # If this network doesn't consist of modules: return single output
        if not self.is_list:
            output = output[0]

        return output
