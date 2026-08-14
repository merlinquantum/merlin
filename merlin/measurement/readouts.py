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

"""Internal probability readouts used by measurement strategies."""

from collections.abc import Sequence
from numbers import Integral
from typing import cast

import torch
from torch import nn


class _OccupancyReadout(nn.Module):
    """Collapse count-resolved probability keys to binary occupancy keys.

    The readout is bound once to the layer output keys. At forward time it only
    applies the precomputed column-to-bin map to the current probability tensor.

    The reduction sums grouped probabilities without renormalizing, so total
    probability mass is preserved rather than rescaled back to 1. This keeps
    occupancy grouping consistent with the non-occupancy ``probs()`` path,
    which also never renormalizes after photon loss or lossy detectors.
    """

    input_size: int
    output_size: int
    output_keys: tuple[tuple[int, ...], ...]

    def __init__(self, output_keys: Sequence[Sequence[int]]) -> None:
        """Initialize an occupancy readout from count-resolved output keys.

        Parameters
        ----------
        output_keys : Sequence[Sequence[int]]
            Output keys whose order matches the input probability tensor.

        Raises
        ------
        TypeError
            If output keys contain non-integer entries.
        ValueError
            If output keys are empty, have inconsistent lengths, or contain
            negative values.
        """
        super().__init__()
        normalized_keys = _normalize_output_keys(output_keys)
        occupancy_keys = [_to_occupancy_key(key) for key in normalized_keys]
        grouped_keys = tuple(sorted(set(occupancy_keys)))
        key_to_group = {key: index for index, key in enumerate(grouped_keys)}

        self.input_size = len(normalized_keys)
        self.output_size = len(grouped_keys)
        self.output_keys = grouped_keys
        self.register_buffer(
            "_group_indices",
            torch.tensor(
                [key_to_group[key] for key in occupancy_keys],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Return probabilities over binary occupancy keys.

        The reduction is a mass-preserving column sum: each output bin is the
        sum of the input probabilities that collapse to it, and the total
        mass of ``probabilities`` is left unchanged. This intentionally
        matches the non-occupancy ``probs()`` path (see
        ``DistributionStrategy.process``), which never renormalizes either.
        Callers that apply photon loss or lossy detectors upstream will see
        that same sub-1 total mass carried through occupancy grouping rather
        than silently rescaled back to 1.

        Parameters
        ----------
        probabilities : torch.Tensor
            Probability tensor whose last dimension matches ``input_size``.

        Returns
        -------
        torch.Tensor
            Tensor with the same leading dimensions as ``probabilities`` and
            final dimension ``output_size``.

        Raises
        ------
        TypeError
            If ``probabilities`` is not a floating-point tensor.
        ValueError
            If the final tensor dimension does not match the bound input size.
        """
        if probabilities.shape[-1] != self.input_size:
            raise ValueError(
                "Occupancy readout expected probability width "
                f"{self.input_size}, received {probabilities.shape[-1]}."
            )
        if not torch.is_floating_point(probabilities):
            raise TypeError("Occupancy readout requires floating-point probabilities.")

        original_shape = probabilities.shape
        if probabilities.dim() == 1:
            matrix = probabilities.unsqueeze(0)
        else:
            matrix = probabilities.reshape(-1, self.input_size)

        group_indices = cast(torch.Tensor, self._group_indices).to(
            device=probabilities.device
        )
        grouped = torch.zeros(
            matrix.shape[0],
            self.output_size,
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        grouped.index_add_(1, group_indices, matrix)

        if probabilities.dim() == 1:
            return grouped.squeeze(0)
        return grouped.reshape(*original_shape[:-1], self.output_size)


def _normalize_output_keys(
    output_keys: Sequence[Sequence[int]],
) -> list[tuple[int, ...]]:
    """Validate and normalize output keys to integer tuples."""
    if len(output_keys) == 0:
        raise ValueError("output_keys must not be empty.")

    normalized: list[tuple[int, ...]] = []
    key_length: int | None = None
    for index, key in enumerate(output_keys):
        if isinstance(key, (str, bytes)) or not isinstance(key, Sequence):
            raise TypeError(f"output_keys[{index}] must be a sequence of integers.")

        normalized_key: list[int] = []
        for value in key:
            if not isinstance(value, Integral):
                raise TypeError("output keys must contain only integer values.")
            int_value = int(value)
            if int_value < 0:
                raise ValueError("output keys must contain non-negative integers.")
            normalized_key.append(int_value)

        if key_length is None:
            key_length = len(normalized_key)
        elif len(normalized_key) != key_length:
            raise ValueError("All output keys must have the same length.")
        normalized.append(tuple(normalized_key))

    return normalized


def _to_occupancy_key(key: tuple[int, ...]) -> tuple[int, ...]:
    """Convert a count-resolved key to binary occupied/unoccupied values."""
    return tuple(1 if count > 0 else 0 for count in key)
