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
from merlin.core.partial_measurement import PartialMeasurement
from merlin.core.sectored_distribution import (
    SectoredDistribution,
    clean_sectored_distribution,
)
from merlin.measurement.process import SamplingProcess, partial_measurement
from merlin.utils.deprecations import error_deprecated_enum_access
from merlin.utils.grouping import LexGrouping, ModGrouping

# Deprecation guide (target: v0.4):
# - Remove `_LegacyMeasurementStrategy`, `_MeasurementStrategyMeta`, and any
#   enum-style attribute access (`MeasurementStrategy.PROBABILITIES`, etc.).
# - Delete compatibility paths in `resolve_measurement_strategy` and
#   `_resolve_measurement_kind` that accept `_LegacyMeasurementStrategy`.
# - Drop :data:`~merlin.measurement.strategies.MeasurementStrategyLike` alias and any tests that rely on legacy enums.
# - Update all call sites to use the new factories (lots of tests to update!):
#     - `MeasurementStrategy.probs(computation_space)`
#     - `MeasurementStrategy.mode_expectations(computation_space)`
#     - `MeasurementStrategy.amplitudes()`
#     - `MeasurementStrategy.partial(...)`
# - Remove related deprecations in `merlin/utils/deprecations.py` that map legacy
#   enums to new factories, and update docs/examples accordingly.
# - If external compatibility is still needed, provide a separate shim module.


# Note: kept some Legacy to keep the None measurement strategy


class _LegacyMeasurementStrategy(Enum):
    """Legacy enum kept only for backward compatibility (deprecated API)."""

    NONE = "none"


class BaseMeasurementStrategy:
    """New API: internal strategy interface for post-processing implementations."""

    def supports_sampling(self) -> bool:
        """Return whether the strategy can apply sampling to distributions."""
        return False

    def process(
        self,
        *,
        distribution: torch.Tensor | SectoredDistribution,
        amplitudes: torch.Tensor | SectoredDistribution,
        apply_sampling: bool,
        effective_shots: int,
        sampler: SamplingProcess,
        apply_photon_loss: Callable[
            [torch.Tensor | SectoredDistribution], torch.Tensor | SectoredDistribution
        ],
        apply_detectors: Callable[
            [torch.Tensor | SectoredDistribution], torch.Tensor | SectoredDistribution
        ],
        grouping: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor | PartialMeasurement | SectoredDistribution:
        """Return the processed result for the selected measurement strategy.

        Parameters
        ----------
        distribution : torch.Tensor | SectoredDistribution
            Probability distribution before final post-processing, or a sectored
            distribution in the g2 noise case.
        amplitudes : torch.Tensor | SectoredDistribution
            Raw amplitudes before measurement-specific processing, or a sectored
            distribution in the g2 noise case.
        apply_sampling : bool
            Whether sampling should be applied.
        effective_shots : int
            Effective number of shots used for sampling.
        sampler : SamplingProcess
            Sampling process object providing sampling methods.
        apply_photon_loss : Callable[[torch.Tensor], torch.Tensor]
            Photon-loss transform.
        apply_detectors : Callable[[torch.Tensor], torch.Tensor]
            Detector transform.
        grouping : Callable[[torch.Tensor], torch.Tensor] | None
            Optional grouping applied to the resulting probabilities.

        Returns
        -------
        torch.Tensor | PartialMeasurement | SectoredDistribution
            Processed measurement result.
        """
        raise NotImplementedError


class DistributionStrategy(BaseMeasurementStrategy):
    """New API: shared logic for distribution-based strategies."""

    def supports_sampling(self) -> bool:
        return True

    def process(
        self,
        *,
        distribution: torch.Tensor | SectoredDistribution,
        amplitudes: torch.Tensor | SectoredDistribution,
        apply_sampling: bool,
        effective_shots: int,
        sampler: SamplingProcess,
        apply_photon_loss: Callable[
            [torch.Tensor | SectoredDistribution], torch.Tensor | SectoredDistribution
        ],
        apply_detectors: Callable[
            [torch.Tensor | SectoredDistribution], torch.Tensor | SectoredDistribution
        ],
        grouping: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Distribution strategies apply detector/noise transforms before sampling.
        distribution = apply_photon_loss(distribution)
        distribution = apply_detectors(distribution)
        # Change the sectored distribution to a tensor
        self.keys = None
        if isinstance(distribution, SectoredDistribution):
            distribution = clean_sectored_distribution(distribution)
            self.keys, distribution = distribution.to_tensor(return_keys=True)
        if apply_sampling and effective_shots > 0:
            distribution = sampler.pcvl_sampler(distribution, effective_shots)
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

    def process(
        self,
        *,
        amplitudes: torch.Tensor | SectoredDistribution,
        sampler: SamplingProcess | None = None,
        **kwargs: object,
    ) -> torch.Tensor | SectoredDistribution:
        # Amplitudes bypass detectors, photon loss, and sampling.
        apply_sampling = bool(kwargs.get("apply_sampling", False))
        if apply_sampling:
            raise RuntimeError(
                "Sampling cannot be applied when measurement_strategy=MeasurementStrategy.amplitudes()."
            )
        return amplitudes


class PartialMeasurementStrategy(BaseMeasurementStrategy):
    """New API: return a PartialMeasurement from detector partial-measurement output."""

    def __init__(self, measured_modes: tuple[int, ...]) -> None:
        """Initialize the partial-measurement strategy.

        Parameters
        ----------
        measured_modes : tuple[int, ...]
            Mode indices to measure.
        """
        self._measured_modes = measured_modes

    def process(
        self,
        *,
        distribution: torch.Tensor | SectoredDistribution,
        amplitudes: torch.Tensor | SectoredDistribution,
        apply_sampling: bool,
        effective_shots: int,
        sampler: SamplingProcess,
        apply_photon_loss: Callable[
            [torch.Tensor | SectoredDistribution], torch.Tensor | SectoredDistribution
        ],
        apply_detectors: Callable[
            [torch.Tensor | SectoredDistribution], torch.Tensor | SectoredDistribution
        ],
        grouping: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> PartialMeasurement:
        if apply_sampling and effective_shots > 0:
            raise RuntimeError(
                "Sampling cannot be applied when measurement_strategy=MeasurementStrategy.partial()."
            )
        # In partial measurement, amplitudes should always be Tensor, not SectoredDistribution.
        # Cast to ensure type narrowing for the apply_photon_loss and apply_detectors calls.
        amplitudes_tensor = cast(torch.Tensor, amplitudes)
        # Apply photon loss before detectors to match detector basis configuration.
        amplitudes_tensor = cast(
            torch.Tensor,
            apply_photon_loss(amplitudes_tensor),
        )
        detector_output = apply_detectors(amplitudes_tensor)
        if not isinstance(detector_output, list):
            raise TypeError(
                "Partial measurement expects detector output in partial_measurement mode."
            )
        partial_measurement_result = partial_measurement(
            detector_output, grouping=grouping
        )
        return partial_measurement_result


class MeasurementKind(Enum):
    """New API: internal measurement kinds used by MeasurementStrategy."""

    # This is an internal discriminator so runtime can route to the correct strategy.
    # Not meant to be user-facing

    PROBABILITIES = "PROBABILITIES"
    MODE_EXPECTATIONS = "MODE_EXPECTATIONS"
    AMPLITUDES = "AMPLITUDES"
    PARTIAL = "PARTIAL"


class _MeasurementStrategyMeta(type):
    def __getattr__(cls, name: str) -> MeasurementStrategy:
        # Backward compatibility shim: allow MeasurementStrategy.NONE for amplitudes.
        if name == "NONE":
            return MeasurementStrategy.amplitudes()
        # All other enum-style access is deprecated; Fail
        error_deprecated_enum_access("MeasurementStrategy", name)

        raise AttributeError(
            f"type object 'MeasurementStrategy' has no attribute {name!r}"
        )


@dataclass(frozen=True, slots=True)
class MeasurementStrategy(metaclass=_MeasurementStrategyMeta):
    """New API: immutable definition of a measurement strategy for output post-processing.

    Parameters
    ----------
    type : MeasurementKind
        Measurement strategy kind.
    measured_modes : tuple[int, ...]
        Measured modes for partial measurement.
    computation_space : ComputationSpace | None
        Computation space used by the strategy.
    grouping : LexGrouping | ModGrouping | None
        Optional grouping applied to probability outputs. If
        ``occupancy_readout`` is ``True``, grouping is applied after the
        occupancy readout.
    occupancy_readout : bool
        Whether probability outputs are collapsed to binary occupied/unoccupied
        output keys. If the distribution reaches the readout with sub-unit
        mass, raw tensor outputs preserve that mass without renormalizing. When
        ``return_object=True``, the resulting ``ProbabilityDistribution``
        normalizes on construction by design. Default value is ``False``.
    """

    type: MeasurementKind
    measured_modes: tuple[int, ...] = ()
    computation_space: ComputationSpace | None = None
    grouping: LexGrouping | ModGrouping | None = None
    occupancy_readout: bool = False
    if TYPE_CHECKING:
        # Type-checker-only legacy/compat attributes. At runtime, the metaclass
        # resolves these names to either a new API instance (NONE) or legacy enums.
        NONE: ClassVar[MeasurementStrategy]

    @staticmethod
    def probs(
        computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
        grouping: LexGrouping | ModGrouping | None = None,
        *,
        occupancy_readout: bool = False,
    ) -> MeasurementStrategy:
        """Create a probability-output measurement strategy.

        Parameters
        ----------
        computation_space : ComputationSpace
            Computation space used to enumerate the output basis.
        grouping : LexGrouping | ModGrouping | None
            Optional grouping applied to the resulting probabilities. If
            ``occupancy_readout`` is ``True``, grouping is applied after the
            occupancy readout.
        occupancy_readout : bool
            Whether to collapse count-resolved Fock output keys into binary
            occupied/unoccupied keys before returning probabilities. Only
            supported with ``ComputationSpace.FOCK``. Default value is
            ``False``. The collapse sums grouped probabilities without
            renormalizing, so any sub-unit mass present at the readout is
            preserved instead of rescaled to 1, consistent with the rest of
            this probability-output path. This applies to raw tensor outputs;
            when ``return_object=True``, the resulting
            ``ProbabilityDistribution`` normalizes on construction by design.

        Returns
        -------
        MeasurementStrategy
            Probability measurement strategy.

        Raises
        ------
        TypeError
            If ``occupancy_readout`` is not a bool.
        ValueError
            If occupancy readout is requested outside ``ComputationSpace.FOCK``.
        """
        # Full measurement returning a probability distribution.
        computation_space = ComputationSpace.coerce(computation_space)
        if type(occupancy_readout) is not bool:
            raise TypeError("occupancy_readout must be a bool.")
        if occupancy_readout:
            if computation_space is not ComputationSpace.FOCK:
                raise ValueError(
                    "occupancy_readout=True is only supported with "
                    "computation_space=ComputationSpace.FOCK."
                )
        return MeasurementStrategy(
            type=MeasurementKind["PROBABILITIES"],
            computation_space=computation_space,
            grouping=grouping,
            occupancy_readout=occupancy_readout,
        )

    @staticmethod
    def mode_expectations(
        computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
    ) -> MeasurementStrategy:
        """Create a per-mode expectation measurement strategy.

        Parameters
        ----------
        computation_space : ComputationSpace
            Computation space used to enumerate the output basis.

        Returns
        -------
        MeasurementStrategy
            Mode-expectation measurement strategy.
        """
        # Mode_expectations
        # Per-mode expectation values from the measured distribution.
        computation_space = ComputationSpace.coerce(computation_space)
        return MeasurementStrategy(
            type=MeasurementKind.MODE_EXPECTATIONS,
            computation_space=computation_space,
        )

    @staticmethod
    def amplitudes(
        computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
    ) -> MeasurementStrategy:
        """Create an amplitude-output measurement strategy.

        Parameters
        ----------
        computation_space : ComputationSpace
            Computation space used to enumerate the output basis.

        Returns
        -------
        MeasurementStrategy
            Amplitude measurement strategy.
        """
        # Raw amplitudes without detector/noise/sampling processing.
        computation_space = ComputationSpace.coerce(computation_space)
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
        """Create a partial measurement on the given mode indices.
        Note that the specified grouping only applies on the resulting probabilities, not on the amplitudes.

        Parameters
        ----------
        modes : list[int]
            Mode indices to measure.
        computation_space : ComputationSpace
            Computation space used to enumerate the output basis.
        grouping : LexGrouping | ModGrouping | None
            Optional grouping applied to the resulting probabilities only.

        Returns
        -------
        MeasurementStrategy
            Partial-measurement strategy.

        Raises
        ------
        ValueError
            If ``modes`` is empty, contains duplicates, or contains negative
            indices.
        """

        if len(modes) == 0:
            raise ValueError("modes cannot be empty")
        if len(set(modes)) != len(modes):
            raise ValueError("Duplicate mode indices")
        if any(m < 0 for m in modes):
            raise ValueError("Negative mode index")

        # Partial measurement is explicit and validated; modes drive processing.
        computation_space = ComputationSpace.coerce(computation_space)
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
                and self.occupancy_readout == other.occupancy_readout
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
            self.occupancy_readout,
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
    raise TypeError(f"Unknown measurement_strategy: {measurement_strategy}")


def resolve_measurement_strategy(
    measurement_strategy: MeasurementStrategyLike,
) -> BaseMeasurementStrategy:
    """Return the concrete strategy implementation for the enum value.

    Parameters
    ----------
    measurement_strategy : :data:`~merlin.measurement.strategies.MeasurementStrategyLike`
        Measurement strategy definition or legacy enum alias.

    Returns
    -------
    BaseMeasurementStrategy
        Concrete runtime strategy implementation.
    """
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
