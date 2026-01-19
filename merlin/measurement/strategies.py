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

"""Measurement strategy definitions for quantum-to-classical conversion."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch

from merlin.core.computation_space import ComputationSpace
from merlin.utils.grouping import LexGrouping, ModGrouping


class MeasurementStrategy(Enum):
    """Strategy for measuring quantum states or counts and possibly apply mapping to classical outputs."""

    PROBABILITIES = "probabilities"
    MODE_EXPECTATIONS = "mode_expectations"
    AMPLITUDES = "amplitudes"


class BaseMeasurementStrategy:
    """Base interface for measurement post-processing strategies."""

    def supports_sampling(self) -> bool:
        """Return whether the strategy can apply sampling to distributions."""
        return False

    def process(
        self,
        *,
        distribution: torch.Tensor,
        amplitudes: torch.Tensor,
        apply_sampling: bool,
        effective_shots: int,
        sample_fn: Callable[[torch.Tensor, int], torch.Tensor],
        apply_photon_loss: Callable[[torch.Tensor], torch.Tensor],
        apply_detectors: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Return the processed result for the selected measurement strategy."""
        raise NotImplementedError


class DistributionStrategy(BaseMeasurementStrategy):
    """Shared logic for distribution-based strategies."""

    def supports_sampling(self) -> bool:
        return True

    def process(
        self,
        *,
        distribution: torch.Tensor,
        amplitudes: torch.Tensor,
        apply_sampling: bool,
        effective_shots: int,
        sample_fn: Callable[[torch.Tensor, int], torch.Tensor],
        apply_photon_loss: Callable[[torch.Tensor], torch.Tensor],
        apply_detectors: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        # Distribution strategies apply detector/noise transforms before sampling.
        distribution = apply_photon_loss(distribution)
        distribution = apply_detectors(distribution)

        if apply_sampling and effective_shots > 0:
            return sample_fn(distribution, effective_shots)
        return distribution


class ProbabilitiesStrategy(DistributionStrategy):
    """Return output probabilities (optionally sampled)."""

    pass


class ModeExpectationsStrategy(DistributionStrategy):
    """Return per-mode expectations (optionally sampled)."""

    pass


class AmplitudesStrategy(BaseMeasurementStrategy):
    """Return raw amplitudes (sampling is not supported)."""

    def process(
        self,
        *,
        distribution: torch.Tensor,
        amplitudes: torch.Tensor,
        apply_sampling: bool,
        effective_shots: int,
        sample_fn: Callable[[torch.Tensor, int], torch.Tensor],
        apply_photon_loss: Callable[[torch.Tensor], torch.Tensor],
        apply_detectors: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        # Amplitudes bypass detectors, photon loss, and sampling.
        if apply_sampling:
            raise RuntimeError(
                "Sampling cannot be applied when measurement_strategy=MeasurementStrategy.AMPLITUDES."
            )
        return amplitudes


def resolve_measurement_strategy(
    measurement_strategy: MeasurementStrategy,
) -> BaseMeasurementStrategy:
    """Return the concrete strategy implementation for the enum value."""
    if measurement_strategy == MeasurementStrategy.PROBABILITIES:
        return ProbabilitiesStrategy()
    if measurement_strategy == MeasurementStrategy.MODE_EXPECTATIONS:
        return ModeExpectationsStrategy()
    if measurement_strategy == MeasurementStrategy.AMPLITUDES:
        return AmplitudesStrategy()
    raise TypeError(f"Unknown measurement_strategy: {measurement_strategy}")


class MeasurementType(Enum):
    """Low-level measurement type used by strategy implementations."""

    PARTIAL = "partial"


@dataclass(frozen=True, slots=True)
class MeasurementStrategyV3:
    """Immutable definition of a measurement strategy for output post-processing."""

    type: MeasurementType
    measured_modes: tuple[int, ...]
    computation_space: ComputationSpace | None = None
    grouping: LexGrouping | ModGrouping | None = None

    @staticmethod
    def partial(
        modes: list[int],
        computation_space: ComputationSpace,
        grouping: LexGrouping | ModGrouping | None = None,
    ) -> "MeasurementStrategyV3":
        """Create a partial measurement on the given mode indices."""

        if len(modes) == 0:
            raise ValueError("modes cannot be empty")
        if len(set(modes)) != len(modes):
            raise ValueError("Duplicate mode indices")
        if any(m < 0 for m in modes):
            raise ValueError("Negative mode index")

        return MeasurementStrategyV3(
            type=MeasurementType.PARTIAL,
            measured_modes=tuple(modes),
            grouping=grouping,
            computation_space=computation_space,
        )

    def validate_modes(self, n_modes: int) -> None:
        """Validate mode indices and warn when the selection covers all modes."""
        for m in self.measured_modes:
            if m < 0 or m >= n_modes:
                raise ValueError(
                    f"Invalid mode indices {self.measured_modes} for circuit with {n_modes} modes"
                )
        if len(self.measured_modes) == n_modes:
            raise Warning(
                "All modes are measured; consider using .probs() instead of .partial()"
            )

    def get_unmeasured_modes(self, n_modes: int) -> tuple[int, ...]:
        """Return the complement of the measured modes after validation."""
        self.validate_modes(n_modes)
        return tuple(m for m in range(n_modes) if m not in self.measured_modes)
