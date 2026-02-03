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

from collections.abc import Callable
from dataclasses import dataclass

import torch

from merlin.core.state_vector import StateVector

DetectorTransformOutput = list[
    dict[tuple[int | None, ...], list[tuple[torch.Tensor, torch.Tensor]]]
]


@dataclass
class PartialMeasurementBranch:
    """
    Single branch of a partial measurement for a specific measured-mode outcome.

    Args:
        outcome: Outcome restricted to measured modes, in detector order.
        probability: Per-batch probability for this outcome.
        amplitudes: Conditional state vector on unmeasured modes.
    """

    def __init__(
        self,
        outcome: tuple[int, ...],
        probability: torch.Tensor,
        amplitudes: StateVector,
    ) -> None:
        self.outcome = outcome  # measured modes only
        self.probability = probability  # shape: batch or scalar
        self.amplitudes = amplitudes


class PartialMeasurement:
    """
    Collection of measurement branches along with measured/unmeasured mode metadata.

    Args:
        branches: Tuple of branches, ordered lexicographically by outcome.
        measured_modes: Indices of measured modes in the full system.
        unmeasured_modes: Indices of unmeasured modes in the full system.
    """

    def __init__(
        self,
        branches: tuple[PartialMeasurementBranch, ...],
        measured_modes: tuple[int, ...],
        unmeasured_modes: tuple[int, ...],
        grouping: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.branches = branches
        self.measured_modes = measured_modes
        self.unmeasured_modes = unmeasured_modes
        self._grouping = grouping

        self.verify_branches_order()

    def verify_branches_order(self) -> None:
        """Verify that branches are ordered lexicographically by their outcomes."""
        if len(self.branches) < 2:
            return
        previous_outcome = self.branches[0].outcome
        for branch in self.branches[1:]:
            if branch.outcome < previous_outcome:
                self.reorder_branches()
                return
            previous_outcome = branch.outcome

    def reorder_branches(self) -> None:
        """Reorder branches lexicographically by their outcomes."""
        self.branches = tuple(sorted(self.branches, key=lambda branch: branch.outcome))

    @property
    def probability_tensor_shape(self) -> tuple[int, int]:
        """Return the expected (batch, n_outcomes) shape for the probability tensor."""
        probas = self._probability_tensor()
        return (probas.shape[0], probas.shape[1])

    @property
    def n_measured_modes(self) -> int:
        return len(self.measured_modes)

    @property
    def n_unmeasured_modes(self) -> int:
        return len(self.unmeasured_modes)

    @property
    def tensor(self) -> torch.Tensor:
        """
        Return the probabilities of all branches as a tensor of shape (batch, n_outcomes).

        This property assumes that all branches are ordered lexicographically by their outcomes
        so the stacking of probabilities follows the same order.
        """
        return self._probability_tensor()

    def _probability_tensor(self) -> torch.Tensor:
        probas = torch.stack(
            [self._as_batch(branch.probability) for branch in self.branches], dim=1
        )
        if self._grouping is not None:
            probas = self._grouping(probas)
        return probas

    @staticmethod
    def _as_batch(probability: torch.Tensor) -> torch.Tensor:
        """Ensure probabilities are at least 1D (batch dimension)."""
        if probability.ndim == 0:
            return probability.unsqueeze(0)
        return probability

    def __repr__(self) -> str:
        return (
            f"PartialMeasurement(measured modes={self.measured_modes}, "
            f"number of branches={len(self.branches)}, "
            f"probability tensor shape={self.probability_tensor_shape}, "
            f"StateVector shape={self.branches[0].amplitudes.shape})"
        )

    @staticmethod
    def from_detector_transform_output(
        detector_output: DetectorTransformOutput,
        *,
        grouping: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> "PartialMeasurement":
        """
        Branch-based `PartialMeasurement` wrapper from DetectorTransform(partial_measurement=True) output.
        """
        branches: list[PartialMeasurementBranch] = []
        measured_modes: tuple[int, ...] = ()
        unmeasured_modes: tuple[int, ...] = ()
        modes_initialized = False

        for i in range(len(detector_output)):
            item = detector_output[i]
            n_photons = i  # Number of photons in unmeasured modes (to verify)
            for full_outcome, outputs in item.items():
                if not modes_initialized:
                    measured_modes = tuple(
                        idx for idx, elem in enumerate(full_outcome) if elem is not None
                    )
                    unmeasured_modes = tuple(
                        idx for idx, elem in enumerate(full_outcome) if elem is None
                    )
                    modes_initialized = True

                measured_only_outcome = tuple(
                    elem for elem in full_outcome if elem is not None
                )
                # For every (prob, amp) corresponding to this outcome, create a branch
                for output in outputs:
                    probs, amps = output
                    branches.append(
                        PartialMeasurementBranch(
                            outcome=measured_only_outcome,
                            probability=probs,
                            amplitudes=StateVector(
                                tensor=amps,
                                n_modes=len(unmeasured_modes),
                                n_photons=n_photons,
                            ),
                        )
                    )

        branches.sort(key=lambda branch: branch.outcome)
        return PartialMeasurement(
            branches=tuple(branches),
            measured_modes=measured_modes,
            unmeasured_modes=unmeasured_modes,
            grouping=grouping,
        )
