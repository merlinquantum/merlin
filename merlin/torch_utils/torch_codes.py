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

import torch
from typing import Optional, Dict
import math as math
from ..core.generators import CircuitType

"""
Output mapping implementations for quantum-to-classical conversion.
"""

import torch.nn as nn
import torch.nn.functional as F
from ..sampling.strategies import OutputMappingStrategy


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
class FeatureEncoder:
    """Utility class for encoding classical features into quantum circuit parameters."""

    def __init__(self, feature_count: int):
        self.feature_count = feature_count

    def encode(self, X_norm: torch.Tensor, circuit_type: CircuitType, n_modes: int,
               bandwidth_coeffs: Optional[Dict[str, torch.Tensor]] = None,
               total_shifters: Optional[int] = None) -> torch.Tensor:
        """Encode normalized features into quantum circuit parameters."""
        batch_size, num_features = X_norm.shape

        def get_scale(key: str, idx: int = 0) -> torch.Tensor:
            """Get bandwidth tuning coefficient while preserving gradients."""
            if bandwidth_coeffs is None or key not in bandwidth_coeffs:
                return torch.tensor(1.0, dtype=X_norm.dtype, device=X_norm.device)

            v = bandwidth_coeffs[key]
            if not isinstance(v, torch.Tensor):
                return torch.tensor(float(v), dtype=X_norm.dtype, device=X_norm.device)

            if v.dim() == 0:
                return v

            if idx < len(v):
                return v[idx]
            else:
                return v[-1]

        # PARALLEL_COLUMNS: Cartesian product of features and modes
        if circuit_type == CircuitType.PARALLEL_COLUMNS:
            cols = []
            for dim_idx in range(num_features):
                for m_idx in range(n_modes):
                    scale = get_scale(f"dim_{dim_idx}", m_idx)
                    multiplier = (m_idx + 1) * math.pi
                    encoded = scale*multiplier * math.pi * X_norm[:, dim_idx]
                    cols.append(encoded.unsqueeze(1))
            return torch.cat(cols, dim=1)



        elif circuit_type == CircuitType.SERIES:
            cols = []

            # If there's only one feature, replicate it across (n_modes–1) slots
            if num_features == 1:
                # Get the bandwidth scale for that single feature (dim_0)
                scale = get_scale("dim_0")
                # For each mode index from 0 to (n_modes–2), multiply by (m_idx+1)·π
                for m_idx in range(n_modes - 1):
                    multiplier = (m_idx + 1) * math.pi
                    encoded = scale * multiplier * X_norm[:, 0]
                    cols.append(encoded.unsqueeze(1))
                return torch.cat(cols, dim=1)

            # Otherwise (num_features >= 2), generate all non-empty subsets
            # but never exceed n_modes - 1 encodings
            max_encodings = n_modes - 1
            max_subsets = min((1 << num_features) - 1, max_encodings)

            # Generate subsets (1 to max_subsets)
            for subset in range(1, max_subsets + 1):
                # Determine which features are in this subset
                features_in_subset = []
                for i in range(num_features):
                    if subset & (1 << i):
                        features_in_subset.append(i)

                # Calculate encoding for this subset
                if len(features_in_subset) == 1:
                    # Single feature
                    idx = features_in_subset[0]
                    scale = get_scale(f"dim_{idx}")
                    encoded = scale * math.pi * X_norm[:, idx]
                else:
                    # Multiple features - sum them
                    scales = [get_scale(f"dim_{i}") for i in features_in_subset]
                    avg_scale = torch.stack(scales).mean()

                    feature_sum = torch.zeros_like(X_norm[:, 0])
                    for idx in features_in_subset:
                        feature_sum = feature_sum + X_norm[:, idx]

                    encoded = avg_scale * math.pi * feature_sum

                cols.append(encoded.unsqueeze(1))

            # Should have exactly max_subsets encodings, no padding needed
            return torch.cat(cols, dim=1)


        # PARALLEL: Direct feature-to-parameter mapping
        elif circuit_type == CircuitType.PARALLEL:
            if num_features == 1:
                cols = []
                scale = get_scale("dim_0")
                for b in range(n_modes - 1):
                    encoded = scale * math.pi * X_norm[:, 0]
                    cols.append(encoded.unsqueeze(1))
                return torch.cat(cols, dim=1)
            else:
                cols = []
                for i in range(num_features):
                    scale = get_scale(f"dim_{i}")
                    encoded = scale * math.pi * X_norm[:, i]
                    cols.append(encoded.unsqueeze(1))
                return torch.cat(cols, dim=1)

        raise ValueError(f"Unknown circuit type: {circuit_type}")


class SamplingProcess:
    """Handles quantum measurement sampling with different methods."""

    def __init__(self):
        self.gradient_method = "exact"  # Always use exact for gradients

    def pcvl_sampler(self, distribution: torch.Tensor, shots: int,
                     method: str = 'multinomial') -> torch.Tensor:
        """Apply sampling noise to a probability distribution."""
        if shots <= 0:
            return distribution

        # Validate method
        valid_methods = ['multinomial', 'binomial', 'gaussian']
        if method not in valid_methods:
            raise ValueError(f"Invalid sampling method: {method}. Valid options are: {valid_methods}")

        if method == 'multinomial':
            if distribution.dim() == 1:
                sampled_counts = torch.multinomial(
                    distribution, num_samples=shots, replacement=True
                )
                noisy_dist = torch.zeros_like(distribution)
                for idx in sampled_counts:
                    noisy_dist[idx] += 1
                return noisy_dist / shots
            else:
                batch_size = distribution.shape[0]
                noisy_dists = []
                for i in range(batch_size):
                    sampled_counts = torch.multinomial(
                        distribution[i], num_samples=shots, replacement=True
                    )
                    noisy_dist = torch.zeros_like(distribution[i])
                    for idx in sampled_counts:
                        noisy_dist[idx] += 1
                    noisy_dists.append(noisy_dist / shots)
                return torch.stack(noisy_dists)

        elif method == 'binomial':
            return torch.distributions.Binomial(shots, distribution).sample() / shots

        elif method == 'gaussian':
            std_dev = torch.sqrt(distribution * (1 - distribution) / shots)
            noise = torch.randn_like(distribution) * std_dev
            noisy_dist = distribution + noise
            noisy_dist = torch.clamp(noisy_dist, 0, 1)
            noisy_dist = noisy_dist / noisy_dist.sum(dim=-1, keepdim=True)
            return noisy_dist


def validate_positive_int(value, name):
    """Validate that a value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def validate_non_negative_int(value, name):
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}")
    return value