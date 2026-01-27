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

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, TypeAlias, cast

import torch

from merlin.core.computation_space import ComputationSpace
from merlin.core.partial_measurement import DetectorTransformOutput, PartialMeasurement
from merlin.measurement.process import partial_measurement
from merlin.utils.deprecations import warn_deprecated_enum_access
from merlin.utils.grouping import LexGrouping, ModGrouping


class _LegacyMeasurementStrategy(Enum):
    """Legacy enum kept for backward compatibility with MeasurementStrategy.PROBABILITIES-style usage."""

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
        grouping: Callable[[torch.Tensor], torch.Tensor] | None = None,
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
        grouping: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Distribution strategies apply detector/noise transforms before sampling.
        distribution = apply_photon_loss(distribution)
        distribution = apply_detectors(distribution)

        if apply_sampling and effective_shots > 0:
            distribution = sample_fn(distribution, effective_shots)
        if grouping is not None:
            return grouping(distribution)
        return distribution


class ProbabilitiesStrategy(DistributionStrategy):
    """Return output probabilities (optionally sampled)."""

    pass


class ModeExpectationsStrategy(DistributionStrategy):
    """Return per-mode expectations (optionally sampled)."""

    pass


class AmplitudesStrategy(BaseMeasurementStrategy):
    """Return raw amplitudes (sampling is not supported)."""

    def process(self, *, amplitudes: torch.Tensor, **kwargs: object) -> torch.Tensor:
        # Amplitudes bypass detectors, photon loss, and sampling.
        apply_sampling = bool(kwargs.get("apply_sampling", False))
        if apply_sampling:
            raise RuntimeError(
                "Sampling cannot be applied when measurement_strategy=MeasurementStrategy.AMPLITUDES."
            )
        return amplitudes


class PartialMeasurementStrategy(BaseMeasurementStrategy):
    """Return a PartialMeasurement from detector partial-measurement output."""

    def __init__(self, measured_modes: tuple[int, ...]) -> None:
        self._measured_modes = measured_modes

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
        grouping: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> PartialMeasurement:
        if apply_sampling and effective_shots > 0:
            raise RuntimeError(
                "Sampling cannot be applied when measurement_strategy=MeasurementStrategy.partial()."
            )
        detector_output = apply_detectors(amplitudes)
        if not isinstance(detector_output, list):
            raise TypeError(
                "Partial measurement expects detector output in partial_measurement mode."
            )
        return partial_measurement(cast(DetectorTransformOutput, detector_output))


class MeasurementKind(Enum):
    """High-level measurement kinds used by MeasurementStrategy."""

    PROBABILITIES = "PROBABILITIES"
    MODE_EXPECTATIONS = "MODE_EXPECTATIONS"
    AMPLITUDES = "AMPLITUDES"
    PARTIAL = "PARTIAL"


class _MeasurementStrategyMeta(type):
    def __getattr__(cls, name: str) -> _LegacyMeasurementStrategy:
        if warn_deprecated_enum_access("MeasurementStrategy", name):
            return _LegacyMeasurementStrategy[name]
        raise AttributeError(
            f"type object 'MeasurementStrategy' has no attribute {name!r}"
        )


@dataclass(frozen=True, slots=True)
class MeasurementStrategy(metaclass=_MeasurementStrategyMeta):
    """Immutable definition of a measurement strategy for output post-processing."""

    type: MeasurementKind
    measured_modes: tuple[int, ...] = ()
    computation_space: ComputationSpace | None = None
    grouping: LexGrouping | ModGrouping | None = None
    if TYPE_CHECKING:
        PROBABILITIES: ClassVar[_LegacyMeasurementStrategy]
        MODE_EXPECTATIONS: ClassVar[_LegacyMeasurementStrategy]
        AMPLITUDES: ClassVar[_LegacyMeasurementStrategy]

    @staticmethod
    def probs(
        computation_space: ComputationSpace,
        grouping: LexGrouping | ModGrouping | None = None,
    ) -> MeasurementStrategy:
        return MeasurementStrategy(
            type=MeasurementKind["PROBABILITIES"],
            computation_space=computation_space,
            grouping=grouping,
        )

    @staticmethod
    def mode_expectations(
        computation_space: ComputationSpace,
    ) -> MeasurementStrategy:
        return MeasurementStrategy(
            type=MeasurementKind.MODE_EXPECTATIONS,
            computation_space=computation_space,
        )

    @staticmethod
    def amplitudes() -> MeasurementStrategy:
        return MeasurementStrategy(
            type=MeasurementKind.AMPLITUDES,
            computation_space=ComputationSpace.UNBUNCHED,
        )

    @staticmethod
    def partial(
        modes: list[int],
        computation_space: ComputationSpace,
        grouping: LexGrouping | ModGrouping | None = None,
    ) -> MeasurementStrategy:
        """Create a partial measurement on the given mode indices."""

        if len(modes) == 0:
            raise ValueError("modes cannot be empty")
        if len(set(modes)) != len(modes):
            raise ValueError("Duplicate mode indices")
        if any(m < 0 for m in modes):
            raise ValueError("Negative mode index")

        return MeasurementStrategy(
            type=MeasurementKind.PARTIAL,
            measured_modes=tuple(modes),
            grouping=grouping,
            computation_space=computation_space,
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MeasurementStrategy):
            return (
                self.type == other.type
                and self.measured_modes == other.measured_modes
                and self.computation_space == other.computation_space
                and self.grouping == other.grouping
            )
        if isinstance(other, _LegacyMeasurementStrategy):
            return self.type.name == other.name
        if isinstance(other, MeasurementKind):
            return self.type == other
        if isinstance(other, str):
            return self.type.name == other or self.type.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((
            self.type,
            self.measured_modes,
            self.computation_space,
            self.grouping,
        ))

    def validate_modes(self, n_modes: int) -> None:
        """Validate mode indices and warn when the selection covers all modes."""
        for m in self.measured_modes:
            if m < 0 or m >= n_modes:
                raise ValueError(
                    f"Invalid mode indices {self.measured_modes} for circuit with {n_modes} modes"
                )
        if len(self.measured_modes) == n_modes:
            warnings.warn(
                "All modes are measured; consider using .probs() instead of .partial()",
                UserWarning,
                stacklevel=2,
            )

    def get_unmeasured_modes(self, n_modes: int) -> tuple[int, ...]:
        """Return the complement of the measured modes after validation."""
        self.validate_modes(n_modes)
        return tuple(m for m in range(n_modes) if m not in self.measured_modes)


MeasurementType = MeasurementKind

MeasurementStrategyLike: TypeAlias = (
    MeasurementStrategy | _LegacyMeasurementStrategy | MeasurementKind
)


def _resolve_measurement_kind(
    measurement_strategy: MeasurementStrategyLike,
) -> MeasurementKind:
    if isinstance(measurement_strategy, MeasurementStrategy):
        return measurement_strategy.type
    if isinstance(measurement_strategy, _LegacyMeasurementStrategy):
        return MeasurementKind[measurement_strategy.name]
    if isinstance(measurement_strategy, MeasurementKind):
        return measurement_strategy
    raise TypeError(f"Unknown measurement_strategy: {measurement_strategy}")


def resolve_measurement_strategy(
    measurement_strategy: MeasurementStrategyLike,
) -> BaseMeasurementStrategy:
    """Return the concrete strategy implementation for the enum value."""
    kind = _resolve_measurement_kind(measurement_strategy)
    if kind == MeasurementKind["PROBABILITIES"]:
        return ProbabilitiesStrategy()
    if kind == MeasurementKind.MODE_EXPECTATIONS:
        return ModeExpectationsStrategy()
    if kind == MeasurementKind.AMPLITUDES:
        return AmplitudesStrategy()
    if kind == MeasurementKind.PARTIAL:
        if not isinstance(measurement_strategy, MeasurementStrategy):
            raise TypeError(
                "MeasurementStrategy.partial() must be used for partial measurement."
            )
        return PartialMeasurementStrategy(
            measured_modes=measurement_strategy.measured_modes
        )
    raise TypeError(f"Unknown measurement_strategy: {measurement_strategy}")
