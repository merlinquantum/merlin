# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from typing import List, Optional, Literal

import torch
from torch import nn as nn


@dataclass
class MLPConfig:
    """Configuration for Multi-Layer Perceptron"""
    hidden_sizes: List[int] = None
    dropout: float = 0.0
    activation: Literal['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu', 'selu'] = 'relu'
    normalization: Optional[Literal['batch', 'layer']] = None


class MLP(nn.Module):
    """
    An enhanced Multi-Layer Perceptron (MLP) implementation with customizable options.

    Features:
    - Multiple activation functions
    - Batch/Layer normalization options
    - Customizable weight initialization

    Args:
        input_size (int): Dimension of the input features
        output_size (int): Dimension of the output
        config (MLPConfig): Configuration for the network architecture
    """

    # Class-level mapping of available activation functions
    ACTIVATIONS = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
        'selu': nn.SELU(),
    }

    def __init__(self, input_size:int, output_size:int, config: MLPConfig):
        super().__init__()
        last_layer_size = input_size
        layers = []

        self.config = config

        # Validate activation function
        if config.activation not in self.ACTIVATIONS:
            raise ValueError(f"Unsupported activation function: {config.activation}")

        # Build network architecture
        for layer_size in config.hidden_sizes:
            # Add linear layer
            layers.append(nn.Linear(last_layer_size, layer_size))

            # Add normalization if specified
            if config.normalization == 'batch':
                layers.append(nn.BatchNorm1d(layer_size))
            elif config.normalization == 'layer':
                layers.append(nn.LayerNorm(layer_size))

            # Add activation function
            layers.append(self.ACTIVATIONS[config.activation])

            # Add dropout if specified
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))

            last_layer_size = layer_size

        # Add final output layer
        layers.append(nn.Linear(last_layer_size, output_size))

        # Create sequential model from layers
        self.network = nn.Sequential(*layers)

    def initialize_weights(self, method='xavier'):
        """
        Initialize network weights using the specified method.

        Args:
            method (str): Initialization method ('xavier' or 'kaiming') (default: 'xavier')
        """

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                if method == 'xavier':
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)
                else:
                    raise ValueError(f"Unsupported initialization method: {method}")

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, output_size)
        """
        return self.network(x)


# Example usage:
if __name__ == "__main__":
    classical_config = MLPConfig(
        hidden_sizes=[64, 32],
        dropout=0.1,
        activation='relu',
        normalization='batch'
    )

    model = MLP(input_size=10, output_size=2, config=classical_config)
    model.initialize_weights()
    # Test forward pass with random input
    x = torch.randn(32, 10)  # batch_size=32, input_size=10
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (32, 2)
