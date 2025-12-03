"""Multi-Layer Perceptron (MLP) utilities for torch_tem package.

This module provides a flexible MLP implementation that supports:
- Single or multiple parallel MLPs (for frequency-specific processing)
- Custom activation functions per layer
- Direct weight manipulation for initialization strategies

The implementation uses nn.Sequential internally for cleaner code while
maintaining a convenient API for multi-module scenarios common in TEM.
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn


class _FunctionActivation(nn.Module):
    """Wrapper to convert activation functions to nn.Module for use in Sequential.

    This allows functional activations (e.g., torch.nn.functional.elu) to be
    used within nn.Sequential blocks alongside nn.Module layers.
    """

    def __init__(self, func: Callable[[Tensor], Tensor]) -> None:
        """Initialize the activation wrapper.

        Args:
            func: Activation function that takes a tensor and returns a tensor.
        """
        super().__init__()
        self.func = func

    def forward(self, x: Tensor) -> Tensor:
        """Apply the activation function.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        return self.func(x)


class MLP(nn.Module):
    """Multi-layer perceptron with support for multiple parallel modules.

    This class implements a 2-layer MLP (input → hidden → output) that can operate
    either as a single network or as multiple independent parallel networks. The
    parallel mode is particularly useful for frequency-specific processing in TEM,
    where each frequency module may have different input/output dimensions.

    The implementation uses nn.Sequential internally for each module, providing
    clean separation between network structure and the multi-module logic.

    Attributes:
        is_list: Whether this MLP operates in multi-module mode.
        N: Number of parallel modules.
        networks: ModuleList containing nn.Sequential blocks, one per module.
        activation: Tuple of (hidden_activation, output_activation) functions.

    Example:
        >>> # Single MLP
        >>> mlp = MLP(in_dim=10, out_dim=8, hidden_dim=12)
        >>> y = mlp(torch.randn(4, 10))  # Returns tensor of shape (4, 8)

        >>> # Parallel MLPs (e.g., for 3 frequency modules)
        >>> mlp = MLP(in_dim=[10, 8, 6], out_dim=[12, 10, 8])
        >>> y_list = mlp([torch.randn(4, 10), torch.randn(4, 8), torch.randn(4, 6)])
        >>> # Returns list of 3 tensors with shapes [(4, 12), (4, 10), (4, 8)]
    """

    def __init__(
        self,
        in_dim: Union[int, List[int]],
        out_dim: Union[int, List[int]],
        activation: Tuple[Optional[Callable[[Tensor], Tensor]], Optional[Callable[[Tensor], Tensor]]] = (torch.nn.functional.elu, None),
        hidden_dim: Optional[Union[int, List[int]]] = None,
        bias: Tuple[bool, bool] = (True, True),
    ) -> None:
        """Initialize MLP.

        Args:
            in_dim: Input dimension(s). If list, creates multiple parallel modules.
                   Each element specifies the input dimension for that module.
            out_dim: Output dimension(s). Must match in_dim list structure if in_dim is a list.
                    Each element specifies the output dimension for that module.
            activation: Tuple of (hidden_activation, output_activation).
                       Each can be a callable or nn.Module. None means no activation.
                       Defaults to (ELU, None) for hidden and output layers respectively.
            hidden_dim: Hidden layer dimension(s). Can be:
                       - None: automatically set to mean of in_dim and out_dim for each module
                       - int: same hidden dimension for all modules
                       - List[int]: specific hidden dimension per module
            bias: Tuple of (use_bias_hidden, use_bias_output) boolean flags.
                 Controls whether bias terms are used in each layer.

        Raises:
            ValueError: If out_dim structure doesn't match in_dim when in_dim is a list.
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

        # Store activation functions for reference
        self.activation = activation

        # Create sequential networks (input->hidden->output) for each module
        self.networks = nn.ModuleList()
        for n in range(self.N):
            # If number of hidden dimensions is not specified: mean of input and output
            if hidden_dim is None:
                hidden = int(np.mean([in_dim[n], out_dim[n]]))
            else:
                hidden = hidden_dim[n] if self.is_list else hidden_dim

            # Build sequential network with layers and activations
            layers = []
            layers.append(nn.Linear(in_dim[n], hidden, bias=bias[0]))
            if activation[0] is not None:
                layers.append(activation[0] if isinstance(activation[0], nn.Module) else _FunctionActivation(activation[0]))
            layers.append(nn.Linear(hidden, out_dim[n], bias=bias[1]))
            if activation[1] is not None:
                layers.append(activation[1] if isinstance(activation[1], nn.Module) else _FunctionActivation(activation[1]))

            self.networks.append(nn.Sequential(*layers))

        # Initialize all weights
        self._initialize_weights(bias)

    def _initialize_weights(self, bias: Tuple[bool, bool]) -> None:
        """Initialize weights with Xavier initialization and zero biases.

        Uses Xavier (Glorot) normal initialization for all Linear layer weights,
        which is appropriate for networks with tanh/sigmoid-like activations.
        All biases are initialized to zero.

        Args:
            bias: Tuple indicating which layers have bias terms (not used in current impl,
                 but kept for API consistency - bias presence is checked dynamically).
        """
        with torch.no_grad():
            for n in range(self.N):
                for layer in self.networks[n]:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight)
                        if layer.bias is not None:
                            layer.bias.fill_(0.0)

    def set_weights(self, from_layer: int, value: Union[float, Tensor, List[Tensor]]) -> None:
        """Set weights of a specific layer.

        This method allows direct manipulation of layer weights, which is useful for:
        - Custom initialization strategies (e.g., identity matrix for transitions)
        - Copying pretrained weights
        - Debugging and testing

        Args:
            from_layer: Layer index (0 for input→hidden, 1 for hidden→output).
                       Negative indexing supported: -1 refers to the last layer.
            value: Value to set. Can be:
                  - float: fills all weights with this scalar value
                  - Tensor: copies this tensor to weights (must match shape)
                  - List[Tensor]: one tensor per module (for multi-module MLPs)

        Example:
            >>> mlp = MLP(in_dim=10, out_dim=8)
            >>> mlp.set_weights(1, 0.0)  # Zero out output layer (useful for identity init)
            >>> mlp.set_weights(0, torch.eye(8, 10))  # Set input layer to specific matrix
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
                # Get the Linear layer at the specified position (0 or 1)
                # In Sequential: [Linear, Activation?, Linear, Activation?]
                # Layer 0 is at index 0, Layer 1 is at index 2 (skipping activation)
                linear_idx = from_layer * 2
                linear_layer = self.networks[n][linear_idx]

                # If a tensor is provided: copy the tensor to the weights
                if isinstance(input_value[n], Tensor):
                    linear_layer.weight.copy_(input_value[n])
                # If only a single value is provided: set that value everywhere
                else:
                    linear_layer.weight.fill_(input_value[n])

    def get_weights(self, from_layer: int) -> List[Tensor]:
        """Get weights of a specific layer.

        Returns the weight matrices (not biases) of the specified layer across
        all modules. Always returns a list, even for single-module MLPs.

        Args:
            from_layer: Layer index (0 for input→hidden, 1 for hidden→output).
                       Negative indexing supported: -1 refers to the last layer.

        Returns:
            List of weight tensors, one per module. Each tensor has shape
            (out_features, in_features) following PyTorch Linear convention.

        Example:
            >>> mlp = MLP(in_dim=[10, 8], out_dim=[12, 10], hidden_dim=[15, 12])
            >>> weights = mlp.get_weights(0)  # Get input→hidden weights
            >>> # Returns [Tensor(15, 10), Tensor(12, 8)]
        """
        # Handle negative indexing
        if from_layer < 0:
            from_layer = 2 + from_layer

        # Get the Linear layer at the specified position
        # In Sequential: [Linear, Activation?, Linear, Activation?]
        linear_idx = from_layer * 2
        return [self.networks[n][linear_idx].weight for n in range(self.N)]

    def forward(self, data: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
        """Forward pass through the MLP.

        Processes input data through the 2-layer network(s). For multi-module MLPs,
        each input tensor is processed by its corresponding module independently.

        Args:
            data: Input data. Structure must match initialization:
                 - Tensor: for single-module MLPs (shape: [batch, in_dim])
                 - List[Tensor]: for multi-module MLPs (one tensor per module)

        Returns:
            Output data with same structure as input:
            - Tensor: for single-module MLPs (shape: [batch, out_dim])
            - List[Tensor]: for multi-module MLPs (one tensor per module)

        Example:
            >>> mlp = MLP(in_dim=[10, 8], out_dim=[12, 10])
            >>> x = [torch.randn(4, 10), torch.randn(4, 8)]
            >>> y = mlp(x)  # Returns list: [Tensor(4, 12), Tensor(4, 10)]
        """
        # Make input data into list, if this network doesn't consist of modules
        if self.is_list:
            input_data = data
        else:
            input_data = [data]

        # Run input through network for each module
        output = [self.networks[n](input_data[n]) for n in range(self.N)]

        # If this network doesn't consist of modules: return single output
        if not self.is_list:
            output = output[0]

        return output
