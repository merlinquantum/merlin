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

"""
Output mapping implementations for quantum-to-classical conversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .strategies import OutputMappingStrategy


class OutputMapper:
    """Handles mapping quantum probability distributions to classical outputs."""

    @staticmethod
    def create_mapping(strategy: OutputMappingStrategy, input_size: int, output_size: int):
        """Create an output mapping based on strategy."""
        if strategy == OutputMappingStrategy.LINEAR:
            return nn.Linear(input_size, output_size)
        elif strategy in [OutputMappingStrategy.GROUPING, OutputMappingStrategy.LEXGROUPING]:
            return LexGroupingMapper(input_size, output_size)
        elif strategy == OutputMappingStrategy.MODGROUPING:
            return ModGroupingMapper(input_size, output_size)
        elif strategy == OutputMappingStrategy.NONE:
            if input_size != output_size:
                raise ValueError(
                    f"Distribution size ({input_size}) must equal "
                    f"output size ({output_size}) when using 'none' strategy"
                )
            return nn.Identity()
        else:
            raise ValueError(f"Unknown output mapping strategy: {strategy}")


class LexGroupingMapper(nn.Module):
    """Maps probability distributions using lexicographical grouping."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, probability_distribution: torch.Tensor) -> torch.Tensor:
        """Group probability distribution into equal-sized buckets."""
        pad_size = (self.output_size - (self.input_size % self.output_size)) % self.output_size

        if pad_size > 0:
            padded = F.pad(probability_distribution, (0, pad_size))
        else:
            padded = probability_distribution

        if probability_distribution.dim() == 2:
            return padded.view(probability_distribution.shape[0], self.output_size, -1).sum(dim=-1)
        else:
            return padded.view(self.output_size, -1).sum(dim=-1)


class ModGroupingMapper(nn.Module):
    """Maps probability distributions using modulo-based grouping."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, probability_distribution: torch.Tensor) -> torch.Tensor:
        """Group probability distribution based on indices modulo output_size."""
        if self.output_size > self.input_size:
            if probability_distribution.dim() == 2:
                pad_size = self.output_size - self.input_size
                padded = F.pad(probability_distribution, (0, pad_size))
                return padded
            else:
                pad_size = self.output_size - self.input_size
                padded = F.pad(probability_distribution, (0, pad_size))
                return padded

        indices = torch.arange(self.input_size, device=probability_distribution.device)
        group_indices = indices % self.output_size

        if probability_distribution.dim() == 2:
            batch_size = probability_distribution.shape[0]
            result = torch.zeros(batch_size, self.output_size,
                                 device=probability_distribution.device,
                                 dtype=probability_distribution.dtype)
            for b in range(batch_size):
                result[b] = torch.zeros(self.output_size,
                                        device=probability_distribution.device,
                                        dtype=probability_distribution.dtype)
                result[b].index_add_(0, group_indices, probability_distribution[b])
            return result
        else:
            result = torch.zeros(self.output_size,
                                 device=probability_distribution.device,
                                 dtype=probability_distribution.dtype)
            result.index_add_(0, group_indices, probability_distribution)
            return result