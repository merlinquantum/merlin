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

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import perceval as pcvl
import torch
from perceval.components.feed_forward_configurator import FFCircuitProvider
from perceval.components.linear_circuit import ACircuit
from perceval.utils import NoiseModel

from ..core.computation_space import ComputationSpace
from ..measurement.detectors import DetectorTransform
from ..measurement.mappers import OutputMapper
from ..measurement.strategies import (
    MeasurementKind,
    MeasurementStrategy,
    MeasurementStrategyLike,
    _resolve_measurement_kind,
)
from ..pcvl_pytorch.utils import pcvl_to_tensor
from .layer import QuantumLayer
from .module import MerlinModule


@dataclass
class FFStage:
    unitary: pcvl.Circuit
    active_modes: tuple[int, ...]
    measured_modes: tuple[int, ...]
    detectors: dict[int, pcvl.Detector | None]
    provider: FFCircuitProvider | None


@dataclass
class StageRuntime:
    circuit: pcvl.Circuit
    pre_layer: QuantumLayer | None
    detector_transform: DetectorTransform | None
    conditional_circuits: dict[tuple[int, ...], pcvl.Circuit]
    conditional_default_key: tuple[int, ...] | None
    measured_modes: tuple[int, ...]
    global_measured_modes: tuple[int, ...]
    active_modes: tuple[int, ...]
    detectors: dict[int, pcvl.Detector | None]
    provider: FFCircuitProvider | None
    pre_layers: dict[int, QuantumLayer] = field(default_factory=dict)
    detector_cache: dict[int, DetectorTransform] = field(default_factory=dict)
    conditional_layer_cache: dict[tuple[tuple[int, ...], int], QuantumLayer] = field(
        default_factory=dict
    )
    trainable_parameters: list[str] | None = None
    initial_amplitudes: torch.Tensor | None = None
    classical_input_size: int = 0


@dataclass
class BranchState:
    amplitudes: torch.Tensor
    weight: torch.Tensor
    remaining_n: int
    measurement_key: tuple[int | None, ...]
    basis_keys: tuple[tuple[int, ...], ...]


class FeedForwardBlock(MerlinModule):
    """
    Feed-forward photonic block constructed directly from a Perceval experiment.

    The block introspects the provided :class:`pcvl.Experiment`, splits it into
    unitary / detector / :class:`~perceval.components.feed_forward_configurator.FFCircuitProvider`
    stages and turns each segment into one or more :class:`~merlin.algorithms.layer.QuantumLayer`
    instances. At run time the block executes every stage, branching on every
    partial measurement outcome and accumulating the classical probability for
    each branch.

    Parameters
    ----------
    experiment:
        Perceval experiment containing the full feed-forward definition. The
        current implementation requires noise-free experiments (``NoiseModel()``
        or ``None``).
    input_state:
        Initial quantum state. May be provided as a Fock occupation list,
        :class:`pcvl.BasicState`, :class:`pcvl.StateVector`, or a tensor whose
        components represent amplitudes in the experiment Fock basis (the tensor
        is only required for amplitude-encoding inputs).
    trainable_parameters:
        Optional list of Perceval parameter prefixes that should remain
        learnable across all stages.
    input_parameters:
        Perceval parameter prefixes that receive classical inputs. They are
        consumed by the *first* stage only; once the first detection happens all
        branches switch to amplitude encoding and the classical tensor is
        ignored.
    computation_space:
        Currently restricted to :attr:`~merlin.core.computation_space.ComputationSpace.FOCK`.
    measurement_strategy:
        Controls how classical outputs are produced.

        - ``MeasurementStrategy.PROBABILITIES`` returns a tensor of
          shape ``(batch_size, num_output_keys)`` whose columns match the fully
          specified Fock states stored in :pyattr:`output_keys`.
        - ``MeasurementStrategy.MODE_EXPECTATIONS`` collapses every branch into
          a single tensor of shape ``(batch_size, num_modes)`` that contains the
          per-mode photon expectations aggregated across all measurement keys.
          The :pyattr:`output_keys` attribute is retained for metadata while
          :pyattr:`output_state_sizes` reports ``num_modes`` for every key.
        - ``MeasurementStrategy.AMPLITUDES`` yields a list of tuples
          ``(measurement_key, branch_probability, remaining_photons, amplitudes)``
          so callers can reason about the mixed state left by each branch.
    """

    def __init__(
        self,
        experiment: pcvl.Experiment,
        *,
        input_state: list[int]
        | pcvl.BasicState
        | pcvl.StateVector
        | torch.Tensor
        | None = None,
        trainable_parameters: list[str] | None = None,
        input_parameters: list[str] | None = None,
        computation_space: ComputationSpace = ComputationSpace.FOCK,
        measurement_strategy: MeasurementStrategyLike = MeasurementStrategy.probs(
            ComputationSpace.FOCK
        ),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32
        if computation_space is not ComputationSpace.FOCK:
            raise ValueError(
                "FeedForwardBlock currently only supports FOCK computation space."
            )
        self.computation_space = computation_space

        self.measurement_strategy = measurement_strategy
        self.stages = self._parse_experiment_stages(experiment)
        if not self.stages:
            raise ValueError(
                "FeedForwardBlock could not identify any feed-forward stage in the provided experiment."
            )

        self._ensure_noise_free(experiment)
        self.total_modes = experiment.circuit_size
        self._complex_dtype = (
            torch.complex128
            if self.dtype in (torch.float64, torch.double)
            else torch.complex64
        )
        resolved_input_state = self._resolve_input_state_from_experiment(
            experiment, input_state
        )
        (
            self._base_input_state,
            self.n_photons,
            self._initial_amplitudes,
        ) = self._prepare_initial_state(resolved_input_state, experiment)
        self._stage_runtimes: list[StageRuntime] = []
        self._output_keys: list[tuple[int, ...]] | None = None
        self._output_state_sizes: dict[tuple[int, ...], int] | None = None
        self._output_mapper_cache: dict[
            tuple[tuple[tuple[int, ...], ...], MeasurementStrategyLike], torch.nn.Module
        ] = {}
        self._layer_registry_counter = 0
        self._basis_cache: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {}
        for idx, stage in enumerate(self.stages):
            runtime = self._build_stage_runtime(
                stage,
                base_input_state=self._base_input_state,
                initial_amplitudes=self._initial_amplitudes,
                trainable_parameters=trainable_parameters,
                input_parameters=input_parameters,
                is_first=(idx == 0),
            )
            self._stage_runtimes.append(runtime)
            if runtime.pre_layer is not None:
                self._register_layer(f"stage{idx}_pre", runtime.pre_layer)
        if not self._stage_runtimes:
            raise ValueError("No executable stages were created for FeedForwardBlock.")

    def _register_layer(self, prefix: str, layer: QuantumLayer | None) -> None:
        if layer is None:
            return
        name = f"{prefix}_{self._layer_registry_counter}"
        self._layer_registry_counter += 1
        self.add_module(name, layer)

    def _resolve_input_state_from_experiment(
        self,
        experiment: pcvl.Experiment,
        provided_state: list[int]
        | pcvl.BasicState
        | pcvl.StateVector
        | torch.Tensor
        | None,
    ):
        experiment_state = getattr(experiment, "input_state", None)
        if experiment_state is None:
            return provided_state
        if provided_state is not None and not self._states_match(
            experiment_state, provided_state
        ):
            warnings.warn(
                "Both 'experiment.input_state' and 'input_state' are provided. "
                "'experiment.input_state' will be used.",
                UserWarning,
                stacklevel=2,
            )
        return experiment_state

    @staticmethod
    def _states_match(left, right) -> bool:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return torch.equal(left, right)
        try:
            result = left == right
            if isinstance(result, torch.Tensor):
                return bool(result.item())
            return bool(result)
        except Exception:
            return False

    def _ensure_noise_free(self, experiment: pcvl.Experiment) -> None:
        noise = getattr(experiment, "noise", None)
        if noise is None:
            return
        if isinstance(noise, NoiseModel) and noise == NoiseModel():
            return
        raise NotImplementedError(
            "FeedForwardBlock does not support experiments with a NoiseModel yet."
        )

    def _prepare_initial_state(
        self,
        input_state: list[int]
        | pcvl.BasicState
        | pcvl.StateVector
        | torch.Tensor
        | None,
        experiment: pcvl.Experiment,
    ) -> tuple[list[int], int, torch.Tensor | None]:
        source_state = input_state
        if source_state is None:
            source_state = getattr(experiment, "input_state", None)
            if source_state is None:
                raise ValueError(
                    "FeedForwardBlock requires an input_state matching the experiment."
                )

        total_modes = experiment.circuit_size
        amplitude_tensor: torch.Tensor | None = None

        if isinstance(source_state, torch.Tensor):
            amplitude_tensor = self._normalize_amplitude_tensor(source_state)
            n_photons = self._infer_photon_number_from_basis(
                amplitude_tensor.shape[-1], total_modes
            )
            base_state = self._default_input_state(total_modes, n_photons)
        elif isinstance(source_state, pcvl.StateVector):
            n_photons = self._extract_statevector_photons(source_state)
            amplitude_tensor = pcvl_to_tensor(
                source_state,
                computation_space=self.computation_space,
                dtype=self._complex_dtype,
                device=self.device,
            )
            base_state = self._default_input_state(total_modes, n_photons)
        else:
            base_state = self._normalize_fock_input(source_state, total_modes)
            n_photons = sum(base_state)

        if n_photons <= 0:
            raise ValueError("Input state must contain at least one photon.")

        return base_state, n_photons, amplitude_tensor

    def _normalize_fock_input(
        self,
        state_like: list[int] | tuple[int, ...] | pcvl.BasicState,
        total_modes: int,
    ) -> list[int]:
        if isinstance(state_like, pcvl.BasicState):
            values = [int(v) for v in state_like]
        else:
            values = [int(v) for v in state_like]
        if len(values) < total_modes:
            values = values + [0] * (total_modes - len(values))
        elif len(values) > total_modes:
            raise ValueError(
                f"Input state has {len(values)} modes but experiment expects {total_modes}."
            )
        if sum(values) <= 0:
            raise ValueError("Input state must contain at least one photon.")
        return values

    def _default_input_state(self, total_modes: int, n_photons: int) -> list[int]:
        if total_modes <= 0:
            raise ValueError("Experiment must contain at least one mode.")
        state = [0] * total_modes
        state[0] = n_photons
        return state

    def _normalize_amplitude_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim not in (1, 2):
            raise ValueError("Amplitude input tensors must be 1D or 2D.")
        target_dtype = self._complex_dtype
        if tensor.is_complex():
            normalized = tensor.to(dtype=target_dtype, device=self.device)
        elif tensor.is_floating_point():
            real = tensor.to(dtype=self.dtype, device=self.device)
            imag = torch.zeros_like(real)
            normalized = torch.complex(real, imag).to(dtype=target_dtype)
        else:
            raise TypeError("Amplitude tensors must be floating point or complex.")
        return normalized

    def _infer_photon_number_from_basis(self, num_states: int, total_modes: int) -> int:
        n = 1
        while n <= total_modes * 8 + 8:
            states = math.comb(n + total_modes - 1, n)
            if states == num_states:
                return n
            if states > num_states:
                break
            n += 1
        raise ValueError(
            "Unable to infer photon number from amplitude tensor length. "
            "Ensure it matches the Fock basis of the experiment."
        )

    @staticmethod
    def _extract_statevector_photons(state_vector: pcvl.StateVector) -> int:
        photon_counts = set(state_vector.n)
        if len(photon_counts) != 1:
            raise ValueError(
                "StateVector input must have a consistent photon number across components."
            )
        return next(iter(photon_counts))

    def _parse_experiment_stages(self, experiment: pcvl.Experiment) -> list[FFStage]:
        stages: list[FFStage] = []
        active_modes: list[int] = list(range(experiment.circuit_size))
        mode_positions: dict[int, int] = {
            mode: idx for idx, mode in enumerate(active_modes)
        }
        current_circuit = pcvl.Circuit(len(active_modes))
        measured_modes: list[int] = []
        detectors: dict[int, pcvl.Detector | None] = {}
        stage_has_detectors = False

        def remap_modes(target_modes: Sequence[int]) -> int | tuple[int, ...]:
            local_modes: list[int] = []
            for mode in target_modes:
                if mode not in mode_positions:
                    raise ValueError(
                        f"Component targets removed mode {mode}; available modes: {active_modes}."
                    )
                local_modes.append(mode_positions[mode])
            if not local_modes:
                raise ValueError("Component must reference at least one optical mode.")
            return tuple(local_modes) if len(local_modes) > 1 else local_modes[0]

        def finalize_stage(provider: FFCircuitProvider | None) -> None:
            nonlocal current_circuit, measured_modes, detectors, stage_has_detectors
            nonlocal active_modes, mode_positions

            has_circuit = current_circuit.ncomponents() > 0
            has_detectors = bool(measured_modes)
            if not has_circuit and not has_detectors and provider is None:
                return

            positions_snapshot = {mode: idx for idx, mode in enumerate(active_modes)}
            ordered_measured = tuple(
                sorted(measured_modes, key=lambda m: positions_snapshot[m])
            )
            stage = FFStage(
                unitary=current_circuit.copy(),
                active_modes=tuple(active_modes),
                measured_modes=ordered_measured,
                detectors=dict(detectors),
                provider=provider,
            )
            stages.append(stage)

            if ordered_measured:
                active_modes = [
                    mode for mode in active_modes if mode not in ordered_measured
                ]
            mode_positions = {mode: idx for idx, mode in enumerate(active_modes)}
            current_circuit = pcvl.Circuit(len(active_modes))
            measured_modes = []
            detectors = {}
            stage_has_detectors = False

        for modes, component in experiment.flatten():
            if isinstance(component, ACircuit):
                if component.__class__.__name__.lower() == "barrier":
                    continue
                local_mapping = remap_modes(modes)
                current_circuit.add(local_mapping, component.copy(), merge=True)
            elif isinstance(component, pcvl.Detector):
                stage_has_detectors = True
                for mode in modes:
                    if mode not in mode_positions:
                        raise ValueError(
                            f"Detector added on removed mode {mode}; current active modes: {active_modes}."
                        )
                    measured_modes.append(mode)
                    detectors[mode] = component
            elif isinstance(component, FFCircuitProvider):
                if not stage_has_detectors:
                    raise ValueError(
                        "Encountered a feed-forward configurator without preceding detectors."
                    )
                finalize_stage(component)
            else:
                raise TypeError(
                    f"Unsupported experiment component type: {type(component).__name__}"
                )

        finalize_stage(provider=None)
        return stages

    def _build_partial_detector(
        self,
        layer: QuantumLayer,
        *,
        stage_detectors: dict[int, pcvl.Detector | None],
        active_modes: Sequence[int],
    ) -> DetectorTransform:
        detectors: list[pcvl.Detector | None] = []
        for mode in active_modes:
            detectors.append(stage_detectors.get(mode))

        return DetectorTransform(
            layer.computation_process.simulation_graph.mapped_keys,
            detectors,
            dtype=layer.dtype,
            device=layer.device,
            partial_measurement=True,
        )

    def _build_stage_runtime(
        self,
        stage: FFStage,
        *,
        base_input_state: list[int] | pcvl.BasicState | None,
        initial_amplitudes: torch.Tensor | None,
        trainable_parameters: list[str] | None,
        input_parameters: list[str] | None,
        is_first: bool,
    ) -> StageRuntime:
        positions = {mode: idx for idx, mode in enumerate(stage.active_modes)}
        measured_local = tuple(positions[m] for m in stage.measured_modes)
        amplitude_encoding = False
        pre_layers: dict[int, QuantumLayer] = {}
        if is_first:
            if not stage.measured_modes:
                raise ValueError(
                    "FeedForwardBlock requires detectors preceding the first feed-forward provider."
                )
            if stage.provider is None:
                raise ValueError(
                    "FeedForwardBlock expects the first stage to contain a FFCircuitProvider."
                )
            amplitude_encoding = initial_amplitudes is not None
            if base_input_state is None:
                raise ValueError("Input state must be provided for the first stage.")
            if amplitude_encoding and input_parameters:
                raise ValueError(
                    "Amplitude-encoded input states cannot be combined with classical input parameters."
                )
            pre_layer = QuantumLayer(
                input_size=None if (amplitude_encoding or input_parameters) else 0,
                circuit=stage.unitary,
                input_state=base_input_state,
                n_photons=self.n_photons,
                trainable_parameters=trainable_parameters,
                input_parameters=input_parameters,
                measurement_strategy=MeasurementStrategy.amplitudes(
                    self.computation_space
                ),
                amplitude_encoding=amplitude_encoding,
                device=self.device,
                dtype=self.dtype,
            )
            detector_transform = self._build_partial_detector(
                pre_layer,
                stage_detectors=stage.detectors,
                active_modes=stage.active_modes,
            )
            if amplitude_encoding:
                pre_layers[self.n_photons] = pre_layer
        else:
            pre_layer = None
            detector_transform = None
            initial_amplitudes = None
            pre_layers = self._initialize_amplitude_pre_layers(
                stage, trainable_parameters
            )

        (
            conditional_circuits,
            default_key,
        ) = self._build_conditional_layers(stage.provider)
        classical_input_size = (
            pre_layer.input_size
            if (pre_layer is not None and not amplitude_encoding)
            else 0
        )
        return StageRuntime(
            circuit=stage.unitary,
            pre_layer=pre_layer,
            detector_transform=detector_transform,
            conditional_circuits=conditional_circuits,
            conditional_default_key=default_key,
            measured_modes=measured_local,
            global_measured_modes=stage.measured_modes,
            active_modes=stage.active_modes,
            detectors=stage.detectors,
            provider=stage.provider,
            pre_layers=pre_layers,
            trainable_parameters=trainable_parameters,
            initial_amplitudes=initial_amplitudes if is_first else None,
            classical_input_size=classical_input_size,
        )

    def _initialize_amplitude_pre_layers(
        self,
        stage: FFStage,
        trainable_parameters: list[str] | None,
    ) -> dict[int, QuantumLayer]:
        layers: dict[int, QuantumLayer] = {}
        max_photons = min(self.n_photons, len(stage.active_modes))
        for remaining in range(max_photons + 1):
            layer = QuantumLayer(
                input_size=None,
                circuit=stage.unitary.copy(),
                amplitude_encoding=True,
                n_photons=remaining,
                measurement_strategy=MeasurementStrategy.amplitudes(
                    self.computation_space
                ),
                device=self.device,
                dtype=self.dtype,
                trainable_parameters=trainable_parameters,
            )
            self._register_layer(
                f"stage{len(self._stage_runtimes)}_amp_{remaining}",
                layer,
            )
            layers[remaining] = layer
        return layers

    def _build_conditional_layers(
        self, provider: FFCircuitProvider | None
    ) -> tuple[dict[tuple[int, ...], pcvl.Circuit], tuple[int, ...] | None]:
        if provider is None:
            return {}, None
        # Normalize provider mappings: accept either Circuit or ACircuit entries.
        # When an ACircuit is provided, wrap it into a single-component Circuit
        # with matching mode count so downstream layers can consume a Circuit.
        configurations: dict[tuple[int, ...], pcvl.Circuit] = {}
        for state, element in provider._map.items():
            key = tuple(state)
            if isinstance(element, pcvl.Circuit):
                configurations[key] = element.copy()
            elif isinstance(element, ACircuit):
                # ACircuit represents a single photonic component; embed it into
                # a shell Circuit of the same width.
                width = getattr(element, "m", None)
                if width is None:
                    raise TypeError(
                        "Feed-forward configuration element exposes no mode count 'm'."
                    )
                wrapped = pcvl.Circuit(int(width))
                wrapped.add(tuple(range(int(width))), element.copy(), merge=True)
                configurations[key] = wrapped
            else:
                raise TypeError(
                    f"Unsupported feed-forward configuration element: {type(element).__name__}"
                )
        default_state = tuple([0] * provider.m)
        default_elem = provider.default_circuit
        if isinstance(default_elem, pcvl.Circuit):
            configurations.setdefault(default_state, default_elem.copy())
        elif isinstance(default_elem, ACircuit):
            width = getattr(default_elem, "m", None)
            if width is None:
                raise TypeError(
                    "Default feed-forward element exposes no mode count 'm'."
                )
            wrapped = pcvl.Circuit(int(width))
            wrapped.add(tuple(range(int(width))), default_elem.copy(), merge=True)
            configurations.setdefault(default_state, wrapped)
        else:
            raise TypeError(
                f"Unsupported default feed-forward element: {type(default_elem).__name__}"
            )
        return configurations, default_state

    def _prepare_classical_features(self, x: torch.Tensor | None) -> torch.Tensor:
        """
        Normalize/validate the classical input tensor expected by the first stage.
        """
        runtime = self._stage_runtimes[0]
        classical_size = int(runtime.classical_input_size or 0)

        def _ensure_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim == 0:
                raise ValueError(
                    "Classical feature tensors must expose at least one dimension."
                )
            return tensor.to(device=self.device, dtype=self.dtype)

        if classical_size == 0:
            if x is None:
                return torch.zeros(
                    (1, 0),
                    device=self.device,
                    dtype=self.dtype,
                )
            tensor = _ensure_tensor(x)
            if tensor.shape[-1] != 0:
                raise ValueError(
                    "The underlying experiment does not accept classical inputs, "
                    "so the provided tensor must have an empty feature dimension."
                )
            return tensor

        if x is None:
            raise ValueError(
                f"This experiment exposes {classical_size} classical inputs for the first stage; "
                "please provide a feature tensor."
            )
        tensor = _ensure_tensor(x)
        if tensor.shape[-1] != classical_size:
            raise ValueError(
                f"Expected classical input dimension {classical_size}, "
                f"received {tensor.shape[-1]}."
            )
        return tensor

    def forward(
        self, x: torch.Tensor | None = None
    ) -> torch.Tensor | list[tuple[tuple[int, ...], torch.Tensor, int, torch.Tensor]]:
        """
        Execute the feed-forward experiment.

        Parameters
        ----------
        x:
            Classical feature tensor. Only the first stage consumes classical
            inputs; subsequent stages operate purely in amplitude-encoding mode.
            When the experiment does not expose classical inputs this argument
            may be omitted (or ``None``), in which case an empty tensor is
            automatically supplied.

        Returns
        -------
        torch.Tensor | list
            ``PROBABILITIES`` returns a tensor of shape
            ``(batch_size, len(output_keys))`` aligned with the fully specified
            Fock states in :pyattr:`output_keys`. ``MODE_EXPECTATIONS`` produces
            a tensor of shape ``(batch_size, total_modes)`` where the columns
            already encode the per-mode expectations aggregated across all
            measurement keys (:pyattr:`output_state_sizes` stores
            ``total_modes`` for every key). ``AMPLITUDES`` yields a list of
            tuples ``(measurement_key, branch_probability, remaining_photons,
            amplitudes)`` describing every branch of the resulting mixed state.
        """
        if not self._stage_runtimes:
            raise RuntimeError("FeedForwardBlock has no stage runtimes to execute.")

        feature_tensor = self._prepare_classical_features(x)
        branches = self._run_stage(self._stage_runtimes[0], feature_tensor)

        for runtime in self._stage_runtimes[1:]:
            branches = self._propagate_future_stage(branches, runtime)

        return self._branches_to_outputs(branches)

    @property
    def output_keys(self) -> list[tuple[int, ...]]:
        """
        Return the measurement keys associated with the most recent classical forward pass.

        The list is populated after :meth:`forward` completes. For the
        ``PROBABILITIES`` strategy the list lines up with the tensor columns. For
        ``MODE_EXPECTATIONS`` it is retained for reference even though the
        returned tensor already aggregates all measurement outcomes. Calling the
        property before running the block raises ``RuntimeError``.
        """
        if self._output_keys is None:
            raise RuntimeError(
                "FeedForwardBlock output keys are undefined until forward() is executed."
            )
        return list(self._output_keys)

    @property
    def output_state_sizes(self) -> dict[tuple[int, ...], int]:
        """
        Return the number of remaining Fock states represented by each entry in ``output_keys``.

        Only available when ``measurement_strategy`` is ``PROBABILITIES`` or
        ``MODE_EXPECTATIONS``. For ``PROBABILITIES`` the value is always ``1``
        because each key now denotes a fully specified Fock state, while for
        ``MODE_EXPECTATIONS`` it equals the total number of modes contributing
        to the expectation vector.
        """
        if self._output_state_sizes is None:
            raise RuntimeError(
                "output_state_sizes is only available after a classical forward pass."
            )
        return dict(self._output_state_sizes)

    def describe(self) -> str:
        """
        Return a multi-line description of the feed-forward stages.

        The summary lists, in order, the global modes that remain active at each
        step, the subset of measured modes, and the type of feed-forward
        configurator attached to the stage. It is primarily intended for
        debugging or for logging experiment structure.
        """
        lines: list[str] = []
        for idx, stage in enumerate(self.stages):
            provider_label = (
                stage.provider.__class__.__name__ if stage.provider else "None"
            )
            lines.append(
                f"Stage {idx + 1}: measured_modes={stage.measured_modes or 'None'}, provider={provider_label}"
            )
        return "\n".join(lines)

    def _run_stage(
        self,
        runtime: StageRuntime,
        x: torch.Tensor,
    ) -> dict[tuple[int, ...], list[BranchState]]:
        if runtime.pre_layer is None or runtime.detector_transform is None:
            raise RuntimeError("Stage runtime is not fully initialised.")
        call_args: list[torch.Tensor] = []
        if runtime.initial_amplitudes is not None:
            call_args.append(runtime.initial_amplitudes)
        if runtime.classical_input_size:
            call_args.append(x)
        amplitudes = runtime.pre_layer(*call_args) if call_args else runtime.pre_layer()
        detector = runtime.detector_transform
        measurement_data = detector(amplitudes)
        branches: dict[tuple[int, ...], list[BranchState]] = {}

        measured_modes = runtime.measured_modes
        for remaining_n, bucket in enumerate(measurement_data):
            if not bucket:
                continue
            for measurement_key, entries in bucket.items():
                global_key = self._merge_measurement_key(None, runtime, measurement_key)
                reduced_key = self._reduce_measurement_values(
                    measurement_key, measured_modes
                )
                for probabilities, branch_amplitudes in entries:
                    conditional_output, basis_keys = self._apply_conditional_layer(
                        runtime,
                        reduced_key,
                        branch_amplitudes,
                        remaining_n,
                        detector,
                    )
                    branch = BranchState(
                        amplitudes=conditional_output,
                        basis_keys=basis_keys,
                        weight=probabilities,
                        remaining_n=remaining_n,
                        measurement_key=global_key,
                    )
                    branches.setdefault(global_key, []).append(branch)
        return branches

    def _propagate_future_stage(
        self,
        current_branches: dict[tuple[int, ...], list[BranchState]],
        runtime: StageRuntime,
    ) -> dict[tuple[int, ...], list[BranchState]]:
        if not current_branches:
            return {}

        new_branches: dict[tuple[int, ...], list[BranchState]] = {}
        for branch_list in current_branches.values():
            for branch in branch_list:
                if branch.remaining_n == 0:
                    stage_layer = None
                    stage_output = branch.amplitudes
                    stage_basis = branch.basis_keys
                else:
                    stage_layer = self._get_stage_amplitude_layer(
                        runtime, branch.remaining_n
                    )
                    stage_output = stage_layer(branch.amplitudes)
                    stage_basis = self._layer_basis(stage_layer)

                if branch.remaining_n == 0:
                    if runtime.measured_modes:
                        stage_key = tuple(
                            0 if idx in runtime.measured_modes else None
                            for idx in range(len(runtime.active_modes))
                        )
                        merged_key = self._merge_measurement_key(
                            branch.measurement_key, runtime, stage_key
                        )
                        reduced_key = tuple(0 for _ in runtime.measured_modes)
                        detector = runtime.detector_transform
                        conditional_output, basis_keys = self._apply_conditional_layer(
                            runtime,
                            reduced_key,
                            stage_output,
                            0,
                            detector,
                        )
                        new_branch = BranchState(
                            amplitudes=conditional_output,
                            basis_keys=basis_keys,
                            weight=branch.weight,
                            remaining_n=0,
                            measurement_key=merged_key,
                        )
                        new_branches.setdefault(merged_key, []).append(new_branch)
                    else:
                        propagated = BranchState(
                            amplitudes=stage_output,
                            basis_keys=stage_basis,
                            weight=branch.weight,
                            remaining_n=branch.remaining_n,
                            measurement_key=branch.measurement_key,
                        )
                        new_branches.setdefault(branch.measurement_key, []).append(
                            propagated
                        )
                    continue

                if runtime.measured_modes:
                    detector = self._get_detector_for_remaining(
                        runtime, branch.remaining_n, stage_layer
                    )
                    measurement_data = detector(stage_output)
                    for remaining_n, bucket in enumerate(measurement_data):
                        if not bucket:
                            continue
                        for measurement_key, entries in bucket.items():
                            merged_key = self._merge_measurement_key(
                                branch.measurement_key, runtime, measurement_key
                            )
                            reduced_key = self._reduce_measurement_values(
                                measurement_key, runtime.measured_modes
                            )
                            for probabilities, branch_amplitudes in entries:
                                conditional_output, basis_keys = (
                                    self._apply_conditional_layer(
                                        runtime,
                                        reduced_key,
                                        branch_amplitudes,
                                        remaining_n,
                                        detector,
                                    )
                                )
                                parent_weight = torch.nan_to_num(branch.weight, nan=0.0)
                                child_weight = torch.nan_to_num(probabilities, nan=0.0)
                                weight = parent_weight * child_weight
                                new_branch = BranchState(
                                    amplitudes=conditional_output,
                                    basis_keys=basis_keys,
                                    weight=weight,
                                    remaining_n=remaining_n,
                                    measurement_key=merged_key,
                                )
                                new_branches.setdefault(merged_key, []).append(
                                    new_branch
                                )
                else:
                    new_branch = BranchState(
                        amplitudes=stage_output,
                        basis_keys=stage_basis,
                        weight=branch.weight,
                        remaining_n=branch.remaining_n,
                        measurement_key=branch.measurement_key,
                    )
                    new_branches.setdefault(branch.measurement_key, []).append(
                        new_branch
                    )

        return new_branches

    def _get_stage_amplitude_layer(
        self, runtime: StageRuntime, remaining_n: int
    ) -> QuantumLayer:
        if remaining_n < 0:
            raise ValueError("Remaining photon count cannot be negative.")
        layer = runtime.pre_layers.get(remaining_n)
        if layer is None:
            layer = QuantumLayer(
                input_size=None,
                circuit=runtime.circuit.copy(),
                amplitude_encoding=True,
                n_photons=remaining_n,
                measurement_strategy=MeasurementStrategy.amplitudes(
                    self.computation_space
                ),
                device=self.device,
                dtype=self.dtype,
                trainable_parameters=runtime.trainable_parameters,
            )
            runtime.pre_layers[remaining_n] = layer
            self._register_layer(
                f"stage_amp_{id(runtime)}_{remaining_n}",
                layer,
            )
        return layer

    def _layer_basis(self, layer: QuantumLayer) -> tuple[tuple[int, ...], ...]:
        return tuple(layer.computation_process.simulation_graph.mapped_keys)

    def _get_detector_for_remaining(
        self,
        runtime: StageRuntime,
        remaining_n: int,
        stage_layer: QuantumLayer,
    ) -> DetectorTransform:
        if runtime.detector_transform is not None:
            return runtime.detector_transform
        detector = runtime.detector_cache.get(remaining_n)
        if detector is None:
            detector = self._build_partial_detector(
                stage_layer,
                stage_detectors=runtime.detectors,
                active_modes=runtime.active_modes,
            )
            runtime.detector_cache[remaining_n] = detector
        return detector

    def _detector_basis(
        self, detector: DetectorTransform | None, remaining_n: int
    ) -> list[tuple[int, ...]]:
        if detector is None:
            return []
        return detector.remaining_basis(remaining_n)

    def _reduce_measurement_values(
        self,
        measurement_key: tuple[int | None, ...],
        measured_modes: tuple[int, ...],
    ) -> tuple[int, ...]:
        if not measured_modes:
            return ()
        values: list[int] = []
        for idx in measured_modes:
            value = measurement_key[idx] if idx < len(measurement_key) else None
            values.append(0 if value is None else int(value))
        return tuple(values)

    def _merge_measurement_key(
        self,
        base_key: tuple[int | None, ...] | None,
        runtime: StageRuntime,
        stage_key: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        if base_key is None:
            merged: list[int | None] = [None] * self.total_modes
        else:
            merged = list(base_key)
        for local_idx, value in enumerate(stage_key):
            if local_idx >= len(runtime.active_modes):
                continue
            global_mode = runtime.active_modes[local_idx]
            if value is not None:
                merged[global_mode] = value
        return tuple(merged)

    def _select_conditional_layer(
        self,
        runtime: StageRuntime,
        measurement_key: tuple[int, ...],
        remaining_n: int,
    ) -> QuantumLayer | None:
        circuits = runtime.conditional_circuits
        if not circuits:
            return None

        actual_key = measurement_key if measurement_key in circuits else None
        if actual_key is None:
            fallback = runtime.conditional_default_key
            if fallback is not None and fallback in circuits:
                actual_key = fallback
            else:
                actual_key = next(iter(circuits.keys()))

        cache_key = (actual_key, remaining_n)
        layer = runtime.conditional_layer_cache.get(cache_key)
        if layer is None:
            circuit = circuits[actual_key]
            layer = QuantumLayer(
                input_size=None,
                circuit=circuit.copy(),
                amplitude_encoding=True,
                n_photons=remaining_n,
                measurement_strategy=MeasurementStrategy.amplitudes(
                    self.computation_space
                ),
                device=self.device,
                dtype=self.dtype,
                trainable_parameters=runtime.trainable_parameters,
            )
            runtime.conditional_layer_cache[cache_key] = layer
            self._register_layer(
                f"stage_cond_{id(runtime)}_{hash(actual_key)}_{remaining_n}",
                layer,
            )
        return layer

    def _apply_conditional_layer(
        self,
        runtime: StageRuntime,
        measurement_key: tuple[int, ...],
        branch_amplitudes: torch.Tensor,
        remaining_n: int,
        detector: DetectorTransform | None,
    ) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...]]:
        if remaining_n == 0:
            basis_keys = tuple(
                self._detector_basis(detector, remaining_n) if detector else []
            )
            return branch_amplitudes, basis_keys
        layer = self._select_conditional_layer(runtime, measurement_key, remaining_n)
        if layer is None:
            basis_keys = tuple(self._detector_basis(detector, remaining_n))
            return branch_amplitudes, basis_keys
        expected_dim = len(layer.computation_process.simulation_graph.mapped_keys)
        if branch_amplitudes.shape[-1] != expected_dim:
            basis_keys = tuple(layer.computation_process.simulation_graph.mapped_keys)
            return branch_amplitudes, basis_keys
        output = layer(branch_amplitudes)
        basis_keys = tuple(layer.computation_process.simulation_graph.mapped_keys)
        return output, basis_keys

    def _branches_to_outputs(
        self, branches: dict[tuple[int, ...], list[BranchState]]
    ) -> torch.Tensor | list[tuple[tuple[int, ...], torch.Tensor, int, torch.Tensor]]:
        if (
            _resolve_measurement_kind(self.measurement_strategy)
            == MeasurementKind.AMPLITUDES
        ):
            return self._branches_to_mixed_states(branches)
        return self._branches_to_classical(branches)

    def _branches_to_classical(
        self, branches: dict[tuple[int, ...], list[BranchState]]
    ) -> torch.Tensor:
        entries: list[
            tuple[
                tuple[int, ...],
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                tuple[tuple[int, ...], ...],
                int,
            ]
        ] = []
        for key, branch_list in branches.items():
            if not branch_list:
                continue
            # Partition branches by remaining_n to avoid mixing basis dimensions
            by_remaining: dict[int, list[BranchState]] = {}
            for b in branch_list:
                by_remaining.setdefault(b.remaining_n, []).append(b)
            for remaining_n, subgroup in by_remaining.items():
                (
                    probability_total,
                    amplitude_total,
                    weight_total,
                    basis_keys,
                    _rn,
                ) = self._aggregate_branch_list(subgroup)
                if probability_total is None:
                    continue
                if (
                    weight_total is None
                    and _resolve_measurement_kind(self.measurement_strategy)
                    != MeasurementKind["PROBABILITIES"]
                ):
                    continue
                entries.append((
                    key,
                    probability_total,
                    amplitude_total,
                    weight_total,
                    basis_keys,
                    remaining_n,
                ))

        if not entries:
            self._output_keys = []
            self._output_state_sizes = {}
            if (
                _resolve_measurement_kind(self.measurement_strategy)
                == MeasurementKind.MODE_EXPECTATIONS
            ):
                return torch.zeros(
                    (0, self.total_modes), dtype=self.dtype, device=self.device
                )
            return torch.zeros(0, device=self.device)

        strategy = self.measurement_strategy
        if _resolve_measurement_kind(strategy) == MeasurementKind["PROBABILITIES"]:
            return self._build_probability_tensor(entries)
        if _resolve_measurement_kind(strategy) == MeasurementKind.MODE_EXPECTATIONS:
            return self._build_mode_expectations(entries)

        keys: list[tuple[int, ...]] = []
        tensors: list[torch.Tensor] = []
        state_sizes: dict[tuple[int, ...], int] = {}
        for (
            key,
            probability_total,
            amplitude_total,
            weight_total,
            basis_keys,
            _remaining_n,
        ) in entries:
            classical = self._finalize_branch_output(
                probability_total,
                amplitude_total,
                weight_total,
                basis_keys,
            )
            keys.append(key)
            tensors.append(classical)
            size = classical.shape[-1] if classical.ndim else 1
            state_sizes[key] = size

        sample = tensors[0]
        if sample.ndim == 1:
            tensors = [t.unsqueeze(0) for t in tensors]
        max_dim = max(t.shape[-1] for t in tensors)
        if max_dim > 0:
            aligned: list[torch.Tensor] = []
            for t in tensors:
                if t.shape[-1] == max_dim:
                    aligned.append(t)
                    continue
                pad_shape = list(t.shape[:-1]) + [max_dim - t.shape[-1]]
                pad = torch.zeros(
                    *pad_shape,
                    dtype=t.dtype,
                    device=t.device,
                )
                aligned.append(torch.cat([t, pad], dim=-1))
            tensors = aligned
        stacked = torch.stack(tensors, dim=1)
        self._output_keys = keys
        self._output_state_sizes = state_sizes
        return stacked

    def _build_probability_tensor(
        self,
        entries: list[
            tuple[
                tuple[int, ...],
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                tuple[tuple[int, ...], ...],
                int,
            ]
        ],
    ) -> torch.Tensor:
        formatted: list[
            tuple[tuple[int, ...], torch.Tensor, tuple[tuple[int, ...], ...], int]
        ] = []
        for key, probability, _, _, basis_keys, remaining_n in entries:
            if probability is None:
                continue
            formatted.append((key, probability, basis_keys, remaining_n))
        stacked, flat_keys = self._flatten_probability_entries(formatted)
        self._output_keys = list(flat_keys)
        self._output_state_sizes = dict.fromkeys(flat_keys, 1)
        return stacked

    def _build_mode_expectations(
        self,
        entries: list[
            tuple[
                tuple[int, ...],
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                tuple[tuple[int, ...], ...],
                int,
            ]
        ],
    ) -> torch.Tensor:
        formatted: list[
            tuple[tuple[int, ...], torch.Tensor, tuple[tuple[int, ...], ...], int]
        ] = []
        for key, probability, _, _, basis_keys, remaining_n in entries:
            if probability is None:
                continue
            formatted.append((key, probability, basis_keys, remaining_n))
        probability_tensor, flat_keys = self._flatten_probability_entries(formatted)
        if probability_tensor.numel() == 0:
            self._output_keys = list(flat_keys)
            self._output_state_sizes = dict.fromkeys(flat_keys, self.total_modes)
            return torch.zeros(
                (0, self.total_modes),
                dtype=probability_tensor.dtype
                if probability_tensor.ndim
                else self.dtype,
                device=probability_tensor.device
                if probability_tensor.ndim
                else self.device,
            )

        basis_tuple = tuple(flat_keys)
        mapper = self._get_output_mapper(basis_tuple)
        expectations = mapper(probability_tensor)
        self._output_keys = list(flat_keys)
        self._output_state_sizes = dict.fromkeys(flat_keys, self.total_modes)
        return expectations

    def _flatten_probability_entries(
        self,
        entries: list[
            tuple[
                tuple[int, ...],
                torch.Tensor,
                tuple[tuple[int, ...], ...],
                int,
            ]
        ],
    ) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
        flat_keys: list[tuple[int, ...]] = []
        flat_tensors: list[torch.Tensor] = []
        for measurement_key, probability, basis_keys, remaining_n in entries:
            unmeasured_indices = [
                idx for idx, value in enumerate(measurement_key) if value is None
            ]
            if basis_keys:
                state_list = basis_keys
            elif unmeasured_indices:
                state_list = self._basis_states_for(
                    len(unmeasured_indices), remaining_n
                )
            else:
                state_list = ((),)
            expected_dim = len(state_list)
            expanded = probability
            if expanded.ndim == 0:
                expanded = expanded.unsqueeze(0).unsqueeze(-1)
            elif expanded.ndim == 1:
                expanded = expanded.unsqueeze(0)
            if expanded.shape[-1] < max(1, expected_dim):
                raise RuntimeError(
                    "Probability tensor does not cover all remaining basis states."
                )
            if expected_dim == 0:
                expected_dim = 1
                state_list = ((),)
            if expanded.shape[-1] > expected_dim:
                expanded = expanded[..., :expected_dim]
            for idx, basis_state in enumerate(state_list):
                if len(basis_state) != len(unmeasured_indices):
                    raise ValueError(
                        "Basis state dimension mismatch for measurement outcome."
                    )
                full_key = list(measurement_key)
                for mode_idx, value in zip(
                    unmeasured_indices, basis_state, strict=False
                ):
                    full_key[mode_idx] = value
                if any(entry is None for entry in full_key):
                    raise ValueError(
                        "Incomplete measurement key encountered while expanding probabilities."
                    )
                finalized = expanded[..., idx]
                flat_keys.append(tuple(full_key))
                flat_tensors.append(finalized)

        if not flat_tensors:
            empty = torch.zeros(0, device=self.device)
            return empty, []

        aligned: list[torch.Tensor] = []
        reference_shape: torch.Size | None = None
        for tensor in flat_tensors:
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            if reference_shape is None:
                reference_shape = tensor.shape
            elif tensor.shape != reference_shape:
                raise RuntimeError(
                    "Inconsistent probability tensor shapes across measurement keys."
                )
            aligned.append(tensor)

        stacked = torch.stack(aligned, dim=1)
        return stacked, flat_keys

    def _basis_states_for(
        self, n_modes: int, n_photons: int
    ) -> tuple[tuple[int, ...], ...]:
        cache_key = (n_modes, n_photons)
        basis = self._basis_cache.get(cache_key)
        if basis is None:
            layer = QuantumLayer(
                input_size=0,
                circuit=pcvl.Circuit(n_modes),
                n_photons=n_photons,
                computation_space=self.computation_space,
                device=self.device,
                dtype=self.dtype,
            )
            basis = tuple(layer.computation_process.simulation_graph.mapped_keys)
            self._basis_cache[cache_key] = basis
        return basis

    def _branches_to_mixed_states(
        self, branches: dict[tuple[int, ...], list[BranchState]]
    ) -> list[tuple[tuple[int, ...], torch.Tensor, int, torch.Tensor]]:
        mixed_states: list[tuple[tuple[int, ...], torch.Tensor, int, torch.Tensor]] = []
        for key, branch_list in branches.items():
            if not branch_list:
                continue
            (
                probability_total,
                amplitude_total,
                weight_total,
                basis_keys,
                remaining_n,
            ) = self._aggregate_branch_list(branch_list)
            if probability_total is None or weight_total is None:
                continue
            branch_probability = probability_total.sum(dim=-1)
            normalized_amplitudes = self._normalize_amplitudes(
                amplitude_total, weight_total
            )
            # Reindex amplitudes to the standard basis order expected by clients.
            # Determine unmeasured modes from the measurement key (positions with None).
            unmeasured_indices = [idx for idx, v in enumerate(key) if v is None]
            standard_basis = self._basis_states_for(
                len(unmeasured_indices), remaining_n
            )
            if basis_keys and basis_keys != standard_basis:
                # Map from current basis_keys to standard_basis
                index_of = {state: i for i, state in enumerate(basis_keys)}
                src = normalized_amplitudes
                out_shape = list(src.shape[:-1]) + [len(standard_basis)]
                reordered = torch.zeros(*out_shape, dtype=src.dtype, device=src.device)
                for tgt_idx, state in enumerate(standard_basis):
                    src_idx = index_of.get(state)
                    if src_idx is not None:
                        reordered[..., tgt_idx] = src[..., src_idx]
                normalized_amplitudes = reordered
            mixed_states.append((
                key,
                branch_probability,
                remaining_n,
                normalized_amplitudes,
            ))
        self._output_keys = [entry[0] for entry in mixed_states]
        self._output_state_sizes = None
        return mixed_states

    def _aggregate_branch_list(
        self, branch_list: list[BranchState]
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        tuple[tuple[int, ...], ...],
        int,
    ]:
        probability_total: torch.Tensor | None = None
        amplitude_total: torch.Tensor | None = None
        weight_total: torch.Tensor | None = None
        if not branch_list:
            # Empty canonical basis (typed) – use an explicit tuple() to satisfy mypy.
            return None, None, None, (), 0
        # Fast path: if all branches expose the *same* ordered basis we can
        # accumulate without reindexing (preserves original probability math).
        first_basis = branch_list[0].basis_keys
        uniform_ordered_basis = all(b.basis_keys == first_basis for b in branch_list)
        remaining_n = branch_list[0].remaining_n
        if uniform_ordered_basis:
            basis_keys = first_basis
            for branch in branch_list:
                weight_tensor = torch.nan_to_num(branch.weight, nan=0.0)
                if not torch.any(weight_tensor):
                    continue
                prob_contrib = self._branch_probability_contribution(
                    branch, weight_tensor
                )
                probability_total = (
                    prob_contrib
                    if probability_total is None
                    else probability_total + prob_contrib
                )
                amp_contrib = self._branch_amplitude_contribution(branch, weight_tensor)
                amplitude_total = (
                    amp_contrib
                    if amplitude_total is None
                    else amplitude_total + amp_contrib
                )
                weight_total = (
                    weight_tensor
                    if weight_total is None
                    else weight_total + weight_tensor
                )
            return (
                probability_total,
                amplitude_total,
                weight_total,
                basis_keys,
                remaining_n,
            )

        # General path: build a canonical basis as the sorted union of all branch
        # bases to align heterogeneous detector enumerations.
        # Collect all basis states across branches. Each state is a tuple[int, ...].
        all_states: set[tuple[int, ...]] = set()
        for b in branch_list:
            all_states.update(b.basis_keys)
        # Canonical basis is the sorted union; make sure it is always typed as
        # tuple[tuple[int, ...], ...] (even when empty) to avoid mypy inferring tuple[()].
        if not all_states:
            canonical_basis: tuple[tuple[int, ...], ...] = ()
        else:
            canonical_basis = tuple(sorted(all_states))
        basis_keys: tuple[tuple[int, ...], ...] = canonical_basis

        for branch in branch_list:
            weight_tensor = torch.nan_to_num(branch.weight, nan=0.0)
            if not torch.any(weight_tensor):
                continue
            # Reindex branch amplitudes to the canonical basis if necessary,
            # padding missing states with zeros.
            amplitudes = self._sanitize_amplitudes(branch.amplitudes)
            if branch.basis_keys != basis_keys:
                # Build source index map
                index_of: dict[tuple[int, ...], int] = {
                    basis_state: idx
                    for idx, basis_state in enumerate(branch.basis_keys)
                }
                tgt_len = len(basis_keys)
                src = amplitudes
                # Prepare output tensor filled with zeros
                out_shape = list(src.shape[:-1]) + [tgt_len]
                reindexed = torch.zeros(*out_shape, dtype=src.dtype, device=src.device)
                # Copy matching indices
                for tgt_idx, basis_state in enumerate(basis_keys):
                    src_idx = index_of.get(basis_state)
                    if src_idx is not None:
                        reindexed[..., tgt_idx] = src[..., src_idx]
                amplitudes = reindexed
            # Compute contributions in the canonical basis
            distribution = amplitudes.abs().pow(2)
            prob_contrib = weight_tensor.unsqueeze(-1) * distribution
            probability_total = (
                prob_contrib
                if probability_total is None
                else probability_total + prob_contrib
            )
            # Scale amplitudes by sqrt(weight) before accumulation
            scale = torch.sqrt(torch.clamp(weight_tensor, min=0.0)).to(amplitudes.dtype)
            while scale.ndim < amplitudes.ndim:
                scale = scale.unsqueeze(-1)
            amp_contrib = amplitudes * scale
            amplitude_total = (
                amp_contrib
                if amplitude_total is None
                else amplitude_total + amp_contrib
            )
            weight_total = (
                weight_tensor if weight_total is None else weight_total + weight_tensor
            )

        return (
            probability_total,
            amplitude_total,
            weight_total,
            basis_keys,
            remaining_n,
        )

    def _branch_probability_contribution(
        self, branch: BranchState, weight_override: torch.Tensor | None = None
    ) -> torch.Tensor:
        sanitized = self._sanitize_amplitudes(branch.amplitudes)
        distribution = sanitized.abs().pow(2)
        if weight_override is None:
            weight = torch.nan_to_num(branch.weight, nan=0.0)
        else:
            weight = weight_override
        while weight.ndim < distribution.ndim:
            weight = weight.unsqueeze(-1)
        return weight * distribution

    def _branch_amplitude_contribution(
        self, branch: BranchState, weight_override: torch.Tensor | None = None
    ) -> torch.Tensor:
        amplitudes = self._sanitize_amplitudes(branch.amplitudes)
        if weight_override is None:
            weight = torch.nan_to_num(branch.weight, nan=0.0)
        else:
            weight = weight_override
        while weight.ndim < amplitudes.ndim:
            weight = weight.unsqueeze(-1)
        scale = torch.sqrt(torch.clamp(weight, min=0.0)).to(amplitudes.dtype)
        return amplitudes * scale

    def _finalize_branch_output(
        self,
        probability: torch.Tensor | None,
        amplitude_sum: torch.Tensor | None,
        weight_sum: torch.Tensor | None,
        basis_keys: tuple[tuple[int, ...], ...],
    ) -> torch.Tensor:
        strategy = self.measurement_strategy
        if _resolve_measurement_kind(strategy) == MeasurementKind.AMPLITUDES:
            return self._normalize_amplitudes(amplitude_sum, weight_sum)

        if probability is None:
            raise RuntimeError("Probability data missing for feed-forward branch.")

        if (
            _resolve_measurement_kind(strategy) == MeasurementKind.MODE_EXPECTATIONS
            and not basis_keys
        ):
            shape = probability.shape[:-1] + (0,)
            return torch.zeros(
                shape, dtype=probability.dtype, device=probability.device
            )

        mapper = self._get_output_mapper(basis_keys)
        return mapper(probability)

    def _normalize_amplitudes(
        self,
        amplitude_sum: torch.Tensor | None,
        weight_sum: torch.Tensor | None,
    ) -> torch.Tensor:
        if amplitude_sum is None:
            raise RuntimeError("Amplitude data missing for feed-forward branch.")

        if weight_sum is None:
            weight_sum = torch.zeros(
                amplitude_sum.shape[:-1],
                dtype=amplitude_sum.real.dtype,
                device=amplitude_sum.device,
            )

        norm = torch.nan_to_num(weight_sum, nan=0.0)
        while norm.ndim < amplitude_sum.ndim:
            norm = norm.unsqueeze(-1)
        safe_norm = torch.sqrt(torch.clamp(norm, min=1e-30)).to(amplitude_sum.dtype)
        zero_mask = safe_norm == 0
        amplitude = amplitude_sum / torch.where(
            zero_mask, torch.ones_like(safe_norm), safe_norm
        )
        amplitude = torch.where(zero_mask, torch.zeros_like(amplitude), amplitude)
        return amplitude

    @staticmethod
    def _sanitize_amplitudes(tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(tensor):
            real = torch.where(
                torch.isnan(tensor.real), torch.zeros_like(tensor.real), tensor.real
            )
            imag = torch.where(
                torch.isnan(tensor.imag), torch.zeros_like(tensor.imag), tensor.imag
            )
            return torch.complex(real, imag)
        return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)

    def _get_output_mapper(
        self, basis_keys: tuple[tuple[int, ...], ...]
    ) -> torch.nn.Module:
        cache_key = (basis_keys, self.measurement_strategy)
        mapper = self._output_mapper_cache.get(cache_key)
        if mapper is None:
            keys = list(basis_keys) if basis_keys else None
            mapper = OutputMapper.create_mapping(
                self.measurement_strategy,
                self.computation_space,
                keys,
            )
            self._output_mapper_cache[cache_key] = mapper
        return mapper
