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
Utilities and helpers for QuantumLayer initialization and configuration.
"""

# Usage outline (mirrors QuantumLayer.__init__ phases):
# 1) validate_and_resolve_circuit_source -> obtain prefixes/specs
# 2) validate_encoding_mode -> enforce amplitude/classical constraints
# 3) prepare_input_state -> normalize input state (incl. experiment override)
# 4) vet_experiment -> reject unsupported experiments
# 5) resolve_circuit -> build circuit/experiment wrapper
# 6) setup_noise_and_detectors -> extract noise/detectors + compatibility checks
# 7) Encoding/output utilities -> used during parameter prep & forward

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import exqalibur as xqlbr
import perceval as pcvl
import torch

from ..builder.circuit_builder import CircuitBuilder
from ..core.computation_space import ComputationSpace
from ..core.generators import StateGenerator, StatePattern
from ..core.state_vector import StateVector
from ..measurement.detectors import resolve_detectors
from ..measurement.photon_loss import resolve_photon_loss
from ..measurement.strategies import (
    MeasurementKind,
    MeasurementStrategyLike,
    _resolve_measurement_kind,
)
from ..pcvl_pytorch.utils import pcvl_to_tensor


@dataclass(frozen=True)
class EncodingModeConfig:
    """Validated encoding configuration."""

    amplitude_encoding: bool
    input_size: int | None
    n_photons: int | None
    input_parameters: list[str]


@dataclass(frozen=True)
class CircuitSource:
    """Resolved circuit source configuration."""

    source_type: Literal["builder", "circuit", "experiment"]
    builder: CircuitBuilder | None
    circuit: pcvl.Circuit | None
    experiment: pcvl.Experiment | None
    trainable_parameters: list[str]
    input_parameters: list[str]
    angle_encoding_specs: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ResolvedCircuit:
    """Resolved circuit and experiment."""

    circuit: pcvl.Circuit
    experiment: pcvl.Experiment
    noise_model: Any | None
    has_custom_noise: bool


@dataclass(frozen=True)
class NoiseAndDetectorConfig:
    """Extracted noise and detector configuration."""

    photon_survival_probs: list[float]
    has_custom_noise: bool
    detectors: list[pcvl.Detector]
    has_custom_detectors: bool
    detector_warnings: list[str]


@dataclass(frozen=True)
class InitializationContext:
    """Immutable context for QuantumLayer initialization."""

    device: torch.device | None
    dtype: torch.dtype
    complex_dtype: torch.dtype
    amplitude_encoding: bool
    input_size: int | None
    circuit: pcvl.Circuit
    experiment: pcvl.Experiment
    noise_model: Any | None
    has_custom_noise: bool
    input_state: StateVector | list[int] | torch.Tensor | None
    n_photons: int | None
    trainable_parameters: list[str]
    input_parameters: list[str]
    angle_encoding_specs: dict[str, dict[str, Any]]
    photon_survival_probs: list[float]
    detectors: list[pcvl.Detector]
    has_custom_detectors: bool
    computation_space: ComputationSpace
    measurement_strategy: MeasurementStrategyLike
    warnings: list[str]
    return_object: bool


def validate_encoding_mode(
    amplitude_encoding: bool,
    input_size: int | None,
    n_photons: int | None,
    input_parameters: list[str] | None,
) -> EncodingModeConfig:
    """Fail-fast validation for amplitude encoding constraints."""
    resolved_input_params = list(input_parameters) if input_parameters else []

    if amplitude_encoding:
        if input_size is not None:
            raise ValueError(
                "When amplitude_encoding is enabled, do not specify input_size; it "
                "is inferred from the computation space."
            )
        if n_photons is None:
            raise ValueError("n_photons must be provided when amplitude_encoding=True.")
        if resolved_input_params:
            raise ValueError(
                "Amplitude encoding cannot be combined with classical input parameters."
            )
        resolved_input_size = None
    else:
        resolved_input_size = int(input_size) if input_size is not None else None

    return EncodingModeConfig(
        amplitude_encoding=amplitude_encoding,
        input_size=resolved_input_size,
        n_photons=n_photons,
        input_parameters=resolved_input_params,
    )


def prepare_input_state(
    input_state: StateVector
    | pcvl.StateVector
    | pcvl.BasicState
    | list
    | tuple
    | torch.Tensor
    | None,
    n_photons: int | None,
    computation_space: ComputationSpace,
    device: torch.device | None,
    complex_dtype: torch.dtype,
    experiment: pcvl.Experiment | None = None,
    circuit_m: int | None = None,
    amplitude_encoding: bool = False,
) -> tuple[StateVector | list[int] | torch.Tensor | None, int | None]:
    """Normalize input_state to canonical form.

    Parameters
    ----------
    input_state : StateVector | pcvl.StateVector | pcvl.BasicState | list | tuple | torch.Tensor | None
        The input state in various formats. ``StateVector`` is the canonical type.
        Legacy formats are auto-converted with deprecation warnings where appropriate.
    n_photons : int | None
        Number of photons (used for default state generation).
    computation_space : ComputationSpace
        The computation space configuration.
    device : torch.device | None
        Target device for tensors.
    complex_dtype : torch.dtype
        Complex dtype for tensor conversion.
    experiment : pcvl.Experiment | None
        Optional experiment whose input_state takes precedence.
    circuit_m : int | None
        Number of modes in the circuit (for default state generation).
    amplitude_encoding : bool
        Whether amplitude encoding is enabled.

    Returns
    -------
    tuple[StateVector | list[int] | torch.Tensor | None, int | None]
        The normalized input state and resolved photon count.

    Raises
    ------
    ValueError
        If neither input_state nor n_photons is provided, or if StateVector is empty.

    Warns
    -----
    DeprecationWarning
        When ``torch.Tensor`` is passed as input_state (deprecated in favor of StateVector).
    UserWarning
        When both experiment.input_state and input_state are provided.
    """
    # Experiment input_state takes precedence
    if experiment is not None and experiment.input_state is not None:
        if input_state is not None and experiment.input_state != input_state:
            warnings.warn(
                "Both 'experiment.input_state' and 'input_state' are provided. "
                "'experiment.input_state' will be used.",
                UserWarning,
                stacklevel=2,
            )
        input_state = experiment.input_state

    # === Handle StateVector (canonical, preferred) ===
    if isinstance(input_state, StateVector):
        return input_state, input_state.n_photons

    # === Handle torch.Tensor (DEPRECATED) ===
    if isinstance(input_state, torch.Tensor):
        warnings.warn(
            "Passing torch.Tensor as input_state is deprecated and will be removed in 0.4. "
            "Use StateVector.from_tensor() instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        # Pass through as tensor for backward compatibility
        return input_state, n_photons

    # === Handle tuple (convert to list, same as list handling) ===
    if isinstance(input_state, tuple):
        input_state = list(input_state)

    # === Handle pcvl.BasicState ===
    if isinstance(input_state, pcvl.BasicState):
        if not isinstance(input_state, xqlbr.FockState):
            raise ValueError("BasicState with annotations is not supported")
        input_state = list(input_state)

    # === Handle pcvl.StateVector ===
    elif isinstance(input_state, pcvl.StateVector):
        if len(input_state) == 0:
            raise ValueError("input_state StateVector cannot be empty")
        sv_n_photons = input_state.n.pop()
        if n_photons is not None and sv_n_photons != n_photons:
            raise ValueError(
                "Inconsistent number of photons between input_state and n_photons."
            )
        return (
            pcvl_to_tensor(
                input_state,
                computation_space,
                device=device,
                dtype=complex_dtype,
            ),
            sv_n_photons,
        )

    # === Validation: need either input_state or n_photons ===
    if input_state is None and n_photons is None:
        raise ValueError("Either input_state or n_photons must be provided")

    # === Generate default state from n_photons ===
    if input_state is None and n_photons is not None:
        if computation_space is ComputationSpace.DUAL_RAIL:
            input_state = [1, 0] * n_photons
        elif amplitude_encoding:
            if circuit_m is None:
                raise ValueError(
                    "circuit_m must be provided to generate default state for amplitude encoding."
                )
            input_state = [1] * n_photons + [0] * (circuit_m - n_photons)
        else:
            if circuit_m is None:
                raise ValueError(
                    "circuit_m must be provided to generate default state when input_state is omitted."
                )
            input_state = StateGenerator.generate_state(
                circuit_m, n_photons, StatePattern.SPACED
            )

    # At this point input_state is always list[int]
    return cast(list[int], input_state), n_photons


def validate_and_resolve_circuit_source(
    builder: CircuitBuilder | None,
    circuit: pcvl.Circuit | None,
    experiment: pcvl.Experiment | None,
    trainable_parameters: list[str] | None,
    input_parameters: list[str] | None,
) -> CircuitSource:
    """Enforce exactly one of (builder, circuit, experiment) is provided."""
    if sum(x is not None for x in (circuit, builder, experiment)) != 1:
        raise ValueError(
            "Provide exactly one of 'circuit', 'builder', or 'experiment'."
        )

    if builder is not None and (
        trainable_parameters is not None or input_parameters is not None
    ):
        raise ValueError(
            "When providing a builder, do not also specify 'trainable_parameters' "
            "or 'input_parameters'. Those prefixes are derived from the builder."
        )

    if builder is not None:
        return CircuitSource(
            source_type="builder",
            builder=builder,
            circuit=None,
            experiment=None,
            trainable_parameters=list(builder.trainable_parameter_prefixes),
            input_parameters=list(builder.input_parameter_prefixes),
            angle_encoding_specs=builder.angle_encoding_specs,
        )

    resolved_trainable = list(trainable_parameters) if trainable_parameters else []
    resolved_input = list(input_parameters) if input_parameters else []
    source_type: Literal["circuit", "experiment"] = (
        "circuit" if circuit is not None else "experiment"
    )

    return CircuitSource(
        source_type=source_type,
        builder=None,
        circuit=circuit,
        experiment=experiment,
        trainable_parameters=resolved_trainable,
        input_parameters=resolved_input,
        angle_encoding_specs={},
    )


def vet_experiment(experiment: pcvl.Experiment) -> dict[str, bool]:
    """Check experiment constraints."""
    has_post_select = experiment.post_select_fn is not None
    has_heralding = bool(experiment.heralds) or bool(experiment.in_heralds)
    has_feedforward = bool(getattr(experiment, "has_feedforward", False))
    has_td_attr = getattr(experiment, "has_td", None)
    has_td = has_td_attr() if callable(has_td_attr) else bool(has_td_attr)
    has_min_photons_filter = bool(getattr(experiment, "min_photons_filter", False))
    has_noise = bool(getattr(experiment, "noise", None))

    if has_post_select or has_heralding:
        raise ValueError(
            "The provided experiment must not have post-selection or heralding."
        )
    if has_feedforward:
        raise ValueError(
            "Feed-forward components are not supported inside a QuantumLayer experiment."
        )
    if has_td:
        raise ValueError(
            "The provided experiment must be unitary, and must not have post-selection or heralding."
        )
    if has_min_photons_filter:
        raise ValueError("The provided experiment must not have a min_photons_filter.")

    return {
        "is_unitary": not has_td,
        "has_noise": has_noise,
        "has_post_select": has_post_select,
        "has_heralding": has_heralding,
        "has_feedforward": has_feedforward,
        "has_min_photons_filter": has_min_photons_filter,
    }


def resolve_circuit(
    circuit_source: CircuitSource,
    pcvl_module,
) -> ResolvedCircuit:
    """Convert builder/circuit/experiment to unified circuit form."""
    if circuit_source.source_type == "builder":
        if circuit_source.builder is None:
            raise RuntimeError("Builder must be provided for builder source type.")
        circuit = circuit_source.builder.to_pcvl_circuit(pcvl_module)
        experiment = pcvl_module.Experiment(circuit)
        noise_model = None
        has_custom_noise = False
    elif circuit_source.source_type == "circuit":
        if circuit_source.circuit is None:
            raise RuntimeError("Circuit must be provided for circuit source type.")
        circuit = circuit_source.circuit
        experiment = pcvl_module.Experiment(circuit)
        noise_model = None
        has_custom_noise = False
    elif circuit_source.source_type == "experiment":
        if circuit_source.experiment is None:
            raise RuntimeError(
                "Experiment must be provided for experiment source type."
            )
        experiment = circuit_source.experiment
        noise_model = getattr(experiment, "noise", None)
        circuit = experiment.unitary_circuit()
        has_custom_noise = noise_model is not None
    else:
        raise RuntimeError("Resolved circuit could not be determined.")

    return ResolvedCircuit(
        circuit=circuit,
        experiment=experiment,
        noise_model=noise_model,
        has_custom_noise=has_custom_noise,
    )


def setup_noise_and_detectors(
    experiment: pcvl.Experiment,
    circuit: pcvl.Circuit,
    computation_space: ComputationSpace,
    measurement_strategy: MeasurementStrategyLike,
) -> NoiseAndDetectorConfig:
    """Extract and validate noise/detectors."""
    photon_survival_probs, empty_noise_model = resolve_photon_loss(
        experiment, circuit.m
    )
    detectors, empty_detectors = resolve_detectors(experiment, circuit.m)

    has_custom_noise = not empty_noise_model
    has_custom_detectors = not empty_detectors
    detector_warnings: list[str] = []

    if has_custom_detectors and computation_space is not ComputationSpace.FOCK:
        detectors = [pcvl.Detector.pnr()] * circuit.m
        detector_warnings.append(
            f"Detectors are ignored in favor of ComputationSpace: {computation_space}"
        )

    amplitude_readout = (
        _resolve_measurement_kind(measurement_strategy) == MeasurementKind.AMPLITUDES
    )
    if amplitude_readout and has_custom_noise:
        raise RuntimeError(
            "measurement_strategy=MeasurementStrategy.AMPLITUDES cannot be used when the experiment defines a NoiseModel."
        )
    if amplitude_readout and has_custom_detectors:
        raise RuntimeError(
            "measurement_strategy=MeasurementStrategy.AMPLITUDES does not support experiments with detectors. "
            "Compute amplitudes without detectors and apply a Partial DetectorTransform manually if needed."
        )

    return NoiseAndDetectorConfig(
        photon_survival_probs=photon_survival_probs,
        has_custom_noise=has_custom_noise,
        detectors=detectors,
        has_custom_detectors=has_custom_detectors,
        detector_warnings=detector_warnings,
    )


def apply_angle_encoding(
    x: torch.Tensor,
    spec: dict[str, Any],
) -> torch.Tensor:
    """Apply custom angle encoding using stored metadata."""
    combos: list[tuple[int, ...]] = spec.get("combinations", [])
    scale_map: dict[int, float] = spec.get("scales", {})

    if x.dim() == 1:
        x_batch = x.unsqueeze(0)
        squeeze = True
    elif x.dim() == 2:
        x_batch = x
        squeeze = False
    else:
        raise ValueError(
            f"Angle encoding expects 1D or 2D tensors, got shape {tuple(x.shape)}"
        )

    if not combos:
        encoded = x_batch
        return encoded.squeeze(0) if squeeze else encoded

    encoded_cols: list[torch.Tensor] = []
    feature_dim = x_batch.shape[-1]

    for combo in combos:
        indices = list(combo)
        if any(idx >= feature_dim for idx in indices):
            raise ValueError(
                f"Input feature dimension {feature_dim} insufficient for angle encoding combination {combo}"
            )

        selected = x_batch[:, indices]
        scales = [scale_map.get(idx, 1.0) for idx in indices]
        scale_tensor = x_batch.new_tensor(scales)
        value = (selected * scale_tensor).sum(dim=1, keepdim=True)
        encoded_cols.append(value)

    encoded = (
        torch.cat(encoded_cols, dim=1)
        if encoded_cols
        else x_batch.new_zeros((x_batch.shape[0], 0))
    )

    return encoded.squeeze(0) if squeeze else encoded


def prepare_input_encoding(
    x: torch.Tensor,
    prefix: str | None = None,
    angle_encoding_specs: dict[str, dict[str, Any]] | None = None,
) -> torch.Tensor:
    """Prepare input encoding based on mode."""
    if not angle_encoding_specs:
        return x

    spec = None
    if prefix is not None:
        spec = angle_encoding_specs.get(prefix)
    elif len(angle_encoding_specs) == 1:
        spec = next(iter(angle_encoding_specs.values()))

    if spec:
        return apply_angle_encoding(x, spec)

    return x


def split_inputs_by_prefix(
    prefixes: list[str],
    tensor: torch.Tensor,
    angle_encoding_specs: dict[str, dict[str, Any]],
    spec_mappings: dict[str, list[str]] | None = None,
) -> list[torch.Tensor] | None:
    """Split a single logical input tensor into per-prefix chunks when possible."""
    counts: list[int] = []
    for prefix in prefixes:
        count = feature_count_for_prefix(prefix, angle_encoding_specs, spec_mappings)
        if count is None:
            return None
        counts.append(count)

    total_required = sum(counts)
    feature_dim = tensor.shape[-1] if tensor.dim() > 1 else tensor.shape[0]
    if total_required != feature_dim:
        return None

    slices: list[torch.Tensor] = []
    offset = 0
    for count in counts:
        end = offset + count
        slices.append(
            tensor[..., offset:end] if tensor.dim() > 1 else tensor[offset:end]
        )
        offset = end
    return slices


def feature_count_for_prefix(
    prefix: str,
    angle_encoding_specs: dict[str, dict[str, Any]],
    spec_mappings: dict[str, list[str]] | None = None,
) -> int | None:
    """Infer the number of raw features associated with an encoding prefix."""
    spec = angle_encoding_specs.get(prefix)
    if spec:
        combos = spec.get("combinations", [])
        feature_indices = {idx for combo in combos for idx in combo}
        if feature_indices:
            return len(feature_indices)

    mapping = (spec_mappings or {}).get(prefix, [])
    if mapping:
        return len(mapping)

    return None


def normalize_output_key(
    key: Iterable[int] | torch.Tensor | Sequence[int],
) -> tuple[int, ...]:
    """Normalize output key to tuple[int, ...]."""
    if isinstance(key, torch.Tensor):
        return tuple(int(v) for v in key.tolist())
    return tuple(int(v) for v in key)
