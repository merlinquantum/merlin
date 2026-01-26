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

Quantum outputs are expected to be:
1. Per state amplitudes, if the processing was a simulation
2. Per state probabilities, if the processing was on hardware
"""

import torch
import torch.nn as nn

from merlin.core import ComputationSpace

from .strategies import (
    MeasurementKind,
    MeasurementStrategyLike,
    _resolve_measurement_kind,
)


class OutputMapper:
    """
    Handles mapping quantum state amplitudes or probabilities to classical outputs.

    This class provides factory methods for creating different types of output mappers
    that convert quantum state amplitudes or probabilities to classical outputs.
    """

    @staticmethod
    def create_mapping(
        strategy: MeasurementStrategyLike,
        computation_space: ComputationSpace = ComputationSpace.FOCK,
        keys: list[tuple[int, ...]] | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Create an output mapping based on the specified strategy.

        Args:
            strategy: The measurement mapping strategy to use
            computation_space: The computation space for the measurement.
            keys: (Only used for ModeExpectations measurement strategy) List of tuples that represent the possible quantum Fock states.
                  For example, keys = [(0,1,0,2), (1,0,1,0), ...]
            dtype: Target dtype for internal tensors. Defaults to torch.float32.

        Returns:
            A PyTorch module that maps the per state amplitudes or probabilities to the desired format.

        Raises:
            ValueError: If strategy is unknown
        """
        try:
            kind = _resolve_measurement_kind(strategy)
        except TypeError as exc:
            raise ValueError(f"Unknown measurement strategy: {strategy}") from exc
        if kind == MeasurementKind["PROBABILITIES"]:
            return Probabilities()
        elif kind == MeasurementKind.MODE_EXPECTATIONS:
            if keys is None:
                raise ValueError(
                    "When using ModeExpectations measurement strategy, keys must be provided."
                )
            return ModeExpectations(computation_space, keys, dtype=dtype)
        elif kind == MeasurementKind.AMPLITUDES:
            return Amplitudes()
        else:
            raise ValueError(f"Unknown measurement strategy: {strategy}")


class Probabilities(nn.Module):
    """Maps quantum state amplitudes or probabilities to the complete Fock state probability distribution."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Compute the probability distribution of possible Fock states from amplitudes or probabilities.

        Args:
            x: Input Fock states amplitudes or probabilities of shape (n_batch, num_states) or (num_states,)

        Returns:
            Fock states probability tensor of shape (batch_size, num_states) or (num_states,)
        """
        trailing_dim = x.shape[-1]
        # Collapse any leading batch dimensions so amplitude detection works uniformly for scalars, matrices or tensors.
        leading_shape = x.shape[:-1]
        reshaped = x.reshape(-1, trailing_dim)

        # Determine if x represents amplitudes (normalized squared norm)
        norm = torch.sum(reshaped.abs() ** 2, dim=1, keepdim=True)
        is_amplitude = torch.allclose(norm, torch.ones_like(norm), atol=1e-6)

        if is_amplitude:
            prob = reshaped.abs() ** 2
        else:
            prob = reshaped

        return prob.reshape(*leading_shape, trailing_dim)


class ModeExpectations(nn.Module):
    """Maps quantum state amplitudes or probabilities to the per mode expected number of photons."""

    def __init__(
        self,
        computation_space: ComputationSpace,
        keys: list[tuple[int, ...]],
        *,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the expectation grouping mapper.

        Args:
            computation_space: The computation space (FOCK, UNBUNCHED, DUAL_RAIL).
            keys: List of tuples describing the possible Fock states output from the circuit preceding the output
                  mapping. e.g., [(0,1,0,2), (1,0,1,0), ...]
            dtype: Target dtype for internal tensors. Defaults to torch.float32.
        """
        super().__init__()
        self.computation_space = computation_space
        self.keys = keys

        if not keys:
            raise ValueError("Keys list cannot be empty")

        if len({len(key) for key in keys}) > 1:
            raise ValueError("All keys must have the same length (number of modes)")

        # Resolve dtype (default to float32 for backward compatibility)
        resolved_dtype = dtype if dtype is not None else torch.float32

        # Create mask and register as buffer
        keys_tensor = torch.tensor(keys, dtype=torch.long)
        if computation_space in {
            ComputationSpace.UNBUNCHED,
            ComputationSpace.DUAL_RAIL,
        }:
            mask = (keys_tensor >= 1).T.to(dtype=resolved_dtype)
        else:
            mask = keys_tensor.T.to(dtype=resolved_dtype)

        # Make the expected type explicit for static analysers.
        self.mask: torch.Tensor
        self.register_buffer("mask", mask)

    def marginalize_per_mode(
        self, probability_distribution: torch.Tensor
    ) -> torch.Tensor:
        """
        Marginalize Fock state probabilities to get per-mode occupation expected values.

        Args:
            probability_distribution (torch.Tensor): Tensor of shape (N, num_keys) with probabilities
                for each Fock state, with requires_grad=True

        Returns:
            torch.Tensor: Shape (N, num_modes) with marginal per mode expected number of photons
        """
        marginalized = probability_distribution @ self.mask.T
        return marginalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert the per state amplitudes to per state probabilities if x are amplitudes. Then, marginalize the per state probability distribution into a per mode expected value.

        Args:
            x: Input Fock states amplitudes or probabilities of shape (n_batch, num_states) or (num_states,)

        Returns:
            Expected value tensor of shape (batch_size, num_modes)
        """
        # Validate input
        if x.dim() not in [1, 2]:
            raise ValueError("Input must be 1D or 2D tensor")

        # Get probabilities
        distribution_mapper = Probabilities()
        prob = distribution_mapper(x)

        # Handle both 1D and 2D inputs uniformly
        original_shape = prob.shape
        if prob.dim() == 1:
            prob = prob.unsqueeze(0)

        marginalized_probs = self.marginalize_per_mode(prob)

        if len(original_shape) == 1:
            marginalized_probs = marginalized_probs.squeeze(0)

        return marginalized_probs


class Amplitudes(nn.Module):
    """
    Output the Fock state vector (also called amplitudes) directly. This can only be done with a simulator because amplitudes cannot be retrieved
    from the per state probabilities obtained with a QPU.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the fock state vector amplitudes."""
        original_shape = x.shape
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if len(original_shape) == 1:
            x = x.squeeze(0)
        return x
