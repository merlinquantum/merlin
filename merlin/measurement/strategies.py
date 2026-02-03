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

# Deprecation guide (target: v0.4):
# - Remove `_LegacyMeasurementStrategy`, `_MeasurementStrategyMeta`, and any
#   enum-style attribute access (`MeasurementStrategy.PROBABILITIES`, etc.).
# - Delete compatibility paths in `resolve_measurement_strategy` and
#   `_resolve_measurement_kind` that accept `_LegacyMeasurementStrategy`.
# - Drop `MeasurementStrategyLike` alias and any tests that rely on legacy enums.
# - Update all call sites to use the new factories (lots of tetsts to update!):
#     - `MeasurementStrategy.probs(computation_space)`
#     - `MeasurementStrategy.mode_expectations(computation_space)`
#     - `MeasurementStrategy.amplitudes()`
#     - `MeasurementStrategy.partial(...)`
# - Remove related deprecations in `merlin/utils/deprecations.py` that map legacy
#   enums to new factories, and update docs/examples accordingly.
# - If external compatibility is still needed, provide a separate shim module.


class _LegacyMeasurementStrategy(Enum):
    """Legacy enum kept only for backward compatibility (deprecated API)."""

    NONE = "none"
    PROBABILITIES = "probabilities"
    MODE_EXPECTATIONS = "mode_expectations"
    AMPLITUDES = "amplitudes"


class BaseMeasurementStrategy:
    """New API: internal strategy interface for post-processing implementations."""

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
    ) -> torch.Tensor | PartialMeasurement:
        """Return the processed result for the selected measurement strategy."""
        raise NotImplementedError


class DistributionStrategy(BaseMeasurementStrategy):
    """New API: shared logic for distribution-based strategies."""

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
    """New API: return output probabilities (optionally sampled)."""

    pass


class ModeExpectationsStrategy(DistributionStrategy):
    """New API: return per-mode expectations (optionally sampled)."""

    pass


class AmplitudesStrategy(BaseMeasurementStrategy):
    """New API: return raw amplitudes (sampling is not supported)."""

    def process(self, *, amplitudes: torch.Tensor, **kwargs: object) -> torch.Tensor:
        # Amplitudes bypass detectors, photon loss, and sampling.
        apply_sampling = bool(kwargs.get("apply_sampling", False))
        if apply_sampling:
            raise RuntimeError(
                "Sampling cannot be applied when measurement_strategy=MeasurementStrategy.AMPLITUDES."
            )
        return amplitudes


class PartialMeasurementStrategy(BaseMeasurementStrategy):
    """New API: return a PartialMeasurement from detector partial-measurement output."""

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
        # Apply photon loss before detectors to match detector basis configuration.
        amplitudes = apply_photon_loss(amplitudes)
        detector_output = apply_detectors(amplitudes)
        if not isinstance(detector_output, list):
            raise TypeError(
                "Partial measurement expects detector output in partial_measurement mode."
            )
        return partial_measurement(cast(DetectorTransformOutput, detector_output))


class MeasurementKind(Enum):
    """New API: internal measurement kinds used by MeasurementStrategy."""

    # This is an internal discriminator so runtime can route to the correct strategy.
    # Not meant to be user-facing

    PROBABILITIES = "PROBABILITIES"
    MODE_EXPECTATIONS = "MODE_EXPECTATIONS"
    AMPLITUDES = "AMPLITUDES"
    PARTIAL = "PARTIAL"


class _MeasurementStrategyMeta(type):
    def __getattr__(cls, name: str) -> MeasurementStrategy | _LegacyMeasurementStrategy:
        # Backward compatibility shim: allow MeasurementStrategy.NONE for amplitudes.
        if name == "NONE":
            return MeasurementStrategy.amplitudes()
        # All other enum-style access is deprecated; warn and return legacy enum.
        if warn_deprecated_enum_access("MeasurementStrategy", name):
            return _LegacyMeasurementStrategy[name]
        raise AttributeError(
            f"type object 'MeasurementStrategy' has no attribute {name!r}"
        )


@dataclass(frozen=True, slots=True)
class MeasurementStrategy(metaclass=_MeasurementStrategyMeta):
    """New API: immutable definition of a measurement strategy for output post-processing."""

    type: MeasurementKind
    measured_modes: tuple[int, ...] = ()
    computation_space: ComputationSpace | None = None
    grouping: LexGrouping | ModGrouping | None = None
    if TYPE_CHECKING:
        # Type-checker-only legacy/compat attributes. At runtime, the metaclass
        # resolves these names to either a new API instance (NONE) or legacy enums.
        NONE: ClassVar[MeasurementStrategy]
        # TODO: verify if we want NONE or method none()
        PROBABILITIES: ClassVar[_LegacyMeasurementStrategy]
        MODE_EXPECTATIONS: ClassVar[_LegacyMeasurementStrategy]
        AMPLITUDES: ClassVar[_LegacyMeasurementStrategy]

    @staticmethod
    def probs(
        computation_space: ComputationSpace,
        grouping: LexGrouping | ModGrouping | None = None,
    ) -> MeasurementStrategy:
        # Full measurement returning a probability distribution.
        return MeasurementStrategy(
            type=MeasurementKind["PROBABILITIES"],
            computation_space=computation_space,
            grouping=grouping,
        )

    @staticmethod
    def mode_expectations(
        computation_space: ComputationSpace,
    ) -> MeasurementStrategy:
        # Mode_expectations
        # Per-mode expectation values from the measured distribution.
        return MeasurementStrategy(
            type=MeasurementKind.MODE_EXPECTATIONS,
            computation_space=computation_space,
        )

    @staticmethod
    def amplitudes(
        computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
    ) -> MeasurementStrategy:
        # Raw amplitudes without detector/noise/sampling processing.
        return MeasurementStrategy(
            type=MeasurementKind.AMPLITUDES,
            computation_space=computation_space,
        )

    @staticmethod
    def partial(
        modes: list[int],
        computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
        grouping: LexGrouping | ModGrouping | None = None,
    ) -> MeasurementStrategy:
        """Create a partial measurement on the given mode indices."""

        if len(modes) == 0:
            raise ValueError("modes cannot be empty")
        if len(set(modes)) != len(modes):
            raise ValueError("Duplicate mode indices")
        if any(m < 0 for m in modes):
            raise ValueError("Negative mode index")

        # Partial measurement is explicit and validated; modes drive processing.
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
        # Hard validation for out-of-range indices; warn if equivalent to full measurement.
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


MeasurementStrategyLike: TypeAlias = MeasurementStrategy | _LegacyMeasurementStrategy


def _resolve_measurement_kind(
    measurement_strategy: MeasurementStrategyLike,
) -> MeasurementKind:
    # Accept new API objects or legacy enum aliases.
    if isinstance(measurement_strategy, MeasurementStrategy):
        return measurement_strategy.type
    if isinstance(measurement_strategy, _LegacyMeasurementStrategy):
        if measurement_strategy == _LegacyMeasurementStrategy.NONE:
            # Legacy NONE aliases amplitudes.
            return MeasurementKind.AMPLITUDES
        return MeasurementKind[measurement_strategy.name]
    raise TypeError(f"Unknown measurement_strategy: {measurement_strategy}")


def resolve_measurement_strategy(
    measurement_strategy: MeasurementStrategyLike,
) -> BaseMeasurementStrategy:
    """Return the concrete strategy implementation for the enum value."""
    # Map high-level kind to the concrete strategy implementation.
    kind = _resolve_measurement_kind(measurement_strategy)
    if kind == MeasurementKind["PROBABILITIES"]:
        return ProbabilitiesStrategy()
    if kind == MeasurementKind.MODE_EXPECTATIONS:
        return ModeExpectationsStrategy()
    if kind == MeasurementKind.AMPLITUDES:
        return AmplitudesStrategy()
    if kind == MeasurementKind.PARTIAL:
        # Partial measurement requires the new API instance to carry modes.
        if not isinstance(measurement_strategy, MeasurementStrategy):
            raise TypeError(
                "MeasurementStrategy.partial() must be used for partial measurement."
            )
        return PartialMeasurementStrategy(
            measured_modes=measurement_strategy.measured_modes
        )
    raise TypeError(f"Unknown measurement_strategy: {measurement_strategy}")
