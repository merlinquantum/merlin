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
Main QuantumLayer implementation
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from typing import Any, cast

import perceval as pcvl
import torch
import torch.nn as nn

from ..builder.circuit_builder import (
    CircuitBuilder,
)
from ..core.computation_space import ComputationSpace
from ..core.generators import StateGenerator, StatePattern
from ..core.process import ComputationProcessFactory
from ..core.state_vector import StateVector
from ..measurement import OutputMapper
from ..measurement.autodiff import AutoDiffProcess
from ..measurement.detectors import DetectorTransform
from ..measurement.photon_loss import PhotonLossTransform
from ..measurement.strategies import (
    MeasurementKind,
    MeasurementStrategy,
    MeasurementStrategyLike,
    _resolve_measurement_kind,
    resolve_measurement_strategy,
)
from ..utils.deprecations import (
    normalize_measurement_strategy,
    sanitize_parameters,
)
from ..utils.grouping import ModGrouping
from .layer_utils import (
    InitializationContext,
    apply_angle_encoding,
    feature_count_for_prefix,
    prepare_input_encoding,
    prepare_input_state,
    resolve_circuit,
    setup_noise_and_detectors,
    split_inputs_by_prefix,
    validate_and_resolve_circuit_source,
    validate_encoding_mode,
    vet_experiment,
)
from .module import MerlinModule


class QuantumLayer(MerlinModule):
    """
    Quantum Neural Network Layer with factory-based architecture.

    This layer can be created either from a :class:`CircuitBuilder` instance, a pre-compiled :class:`pcvl.Circuit`,
    or an :class:Experiment`.
    """

    @sanitize_parameters
    def __init__(
        self,
        input_size: int | None = None,
        # Builder-based construction
        builder: CircuitBuilder | None = None,
        # Custom circuit construction
        circuit: pcvl.Circuit | None = None,
        # Custom experiment construction
        experiment: pcvl.Experiment | None = None,
        # For both custom circuits and builder
        input_state: StateVector
        | pcvl.StateVector
        | pcvl.BasicState
        | list
        | tuple
        | torch.Tensor
        | None = None,
        n_photons: int | None = None,
        # only for custom circuits and experiments
        trainable_parameters: list[str] | None = None,
        input_parameters: list[str] | None = None,
        # Common parameters
        amplitude_encoding: bool = False,
        computation_space: ComputationSpace | str | None = None,
        measurement_strategy: MeasurementStrategyLike = MeasurementKind[
            "PROBABILITIES"
        ],
        # device and dtype
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize a QuantumLayer from a builder, a Perceval circuit, or an experiment.

        This constructor wires the selected photonic circuit (or experiment) into a
        trainable PyTorch module and configures the computation space, input state,
        encoding, and measurement strategy. Exactly one of ``builder``, ``circuit``,
        or ``experiment`` must be provided.

        Parameters
        ----------
        input_size : int | None, optional
            Size of the classical input vector when angle encoding is used
            (``amplitude_encoding=False``). If omitted, it is inferred from the
            circuit metadata (input parameter prefixes and/or encoding specs).
            Must be omitted when ``amplitude_encoding=True``.
        builder : CircuitBuilder | None, optional
            High-level circuit builder that defines trainable structure, input
            encoders and their prefixes. Mutually exclusive with ``circuit`` and
            ``experiment``.
        circuit : pcvl.Circuit | None, optional
            A fully defined Perceval circuit. Mutually exclusive with ``builder``
            and ``experiment``.
        experiment : pcvl.Experiment | None, optional
            A Perceval experiment. Must be unitary and without post-selection or
            heralding. Mutually exclusive with ``builder`` and ``circuit``.
        input_state : StateVector | pcvl.StateVector | pcvl.BasicState | list | tuple | torch.Tensor | None, optional
            Logical input state of the circuit. Accepted forms:
            - ``StateVector`` (preferred, canonical type),
            - ``pcvl.StateVector`` (converted via ``StateVector.from_perceval()``),
            - ``pcvl.BasicState`` (converted via ``StateVector.from_basic_state()``),
            - list/tuple of occupations (converted via ``StateVector.from_basic_state()``),
            - ``torch.Tensor`` (DEPRECATED - will be removed in 0.4).
            If QuantumLayer is built from an experiment, the experiment's input state is used.
            If omitted, ``n_photons`` must be provided to derive a default state.
        n_photons : int | None, optional
            Number of photons used to infer a default input state and to size the
            computation space when amplitude encoding is enabled.
        trainable_parameters : list[str] | None, optional
            For custom circuits/experiments, the list of Perceval parameter
            prefixes to expose as trainable PyTorch parameters. When a
            ``builder`` is provided, these are taken from the builder and this
            argument must be omitted.
        input_parameters : list[str] | None, optional
            Perceval parameter prefixes used for classical (angle) encoding. For
            amplitude encoding, this must be empty/None.
        amplitude_encoding : bool, default: False
            DEPRECATED - will be removed in 0.4. Pass a ``StateVector`` to
            ``forward()`` for amplitude encoding instead.
            When True, the forward call expects an amplitude vector (or batch) on
            the first positional argument and propagates it through the quantum
            layer; ``input_size`` must not be set in this mode and
            ``n_photons`` must be provided.
        computation_space : ComputationSpace | str, optional
            Logical computation subspace to use: one of ``{"fock", "unbunched",
            "dual_rail"}``. If omitted, defaults to ``UNBUNCHED``. This argument
            is deprecated; move it into ``MeasurementStrategy.probs(...)``.
        measurement_strategy : MeasurementStrategy, default: PROBABILITIES
            Output mapping strategy. Supported values include ``PROBABILITIES``,
            ``MODE_EXPECTATIONS`` and ``AMPLITUDES``.
        device : torch.device | None, optional
            Target device for internal tensors (e.g., ``torch.device("cuda")``).
        dtype : torch.dtype | None, optional
            Precision for internal tensors (e.g., ``torch.float32``). The matching
            complex dtype is chosen automatically.

        Raises
        ------
        ValueError
            If an unexpected keyword argument is provided; if both or none of
            ``builder``, ``circuit``, ``experiment`` are provided; if
            ``amplitude_encoding=True`` and ``input_size`` is set; if
            ``amplitude_encoding=True`` and ``n_photons`` is not provided; if
            classical ``input_parameters`` are combined with
            ``amplitude_encoding=True``; if ``no_bunching`` conflicts with the
            selected ``computation_space``; if an ``experiment`` is not unitary or
            uses post-selection/heralding; if neither ``input_state`` nor
            ``n_photons`` is provided when required; or if an annotated
            ``BasicState`` is passed (annotations are not supported).
        TypeError
            If an unknown measurement strategy is selected during setup.

        Warns
        -----
        UserWarning
            When ``experiment.min_photons_filter`` or ``experiment.detectors`` are
            present (currently ignored).
        DeprecationWarning
            When ``amplitude_encoding=True`` is passed (deprecated in favor of
            passing ``StateVector`` to ``forward()``).
            When ``torch.Tensor`` is passed as ``input_state`` (deprecated in favor
            of ``StateVector``).

        """
        super().__init__()

        # === DEPRECATION WARNING: amplitude_encoding ===
        if amplitude_encoding:
            warnings.warn(
                "amplitude_encoding=True is deprecated and will be removed in 0.4. "
                "Pass a StateVector to forward() for amplitude encoding instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Phase 1: device + dtype normalization
        device, dtype, complex_dtype = MerlinModule.setup_device_and_dtype(
            device, dtype
        )
        # Phase 2: computation space resolution (legacy vs strategy-driven)
        measurement_strategy, computation_space = normalize_measurement_strategy(
            measurement_strategy, computation_space
        )

        # Phase 3: circuit source resolution (builder/circuit/experiment)
        circuit_source = validate_and_resolve_circuit_source(
            builder, circuit, experiment, trainable_parameters, input_parameters
        )
        # Phase 4: encoding validation (post-resolution)
        encoding_config = validate_encoding_mode(
            amplitude_encoding,
            input_size,
            n_photons,
            circuit_source.input_parameters,
        )
        # Phase 5: input state normalization
        # Phase 6: experiment vetting (if provided)
        if experiment is not None:
            vet_experiment(experiment)

        # Phase 7: circuit resolution
        resolved_circuit = resolve_circuit(circuit_source, pcvl)
        # Phase 8: input state normalization
        input_state, resolved_n_photons = prepare_input_state(
            input_state,
            n_photons,
            computation_space,
            device,
            complex_dtype,
            resolved_circuit.experiment,
            circuit_m=resolved_circuit.circuit.m,
            amplitude_encoding=amplitude_encoding,
        )
        # Phase 9: noise + detector setup
        noise_and_detectors = setup_noise_and_detectors(
            resolved_circuit.experiment,
            resolved_circuit.circuit,
            computation_space,
            measurement_strategy,
        )

        # Phase 10: build initialization context
        context = InitializationContext(
            device=device,
            dtype=dtype,
            complex_dtype=complex_dtype,
            amplitude_encoding=encoding_config.amplitude_encoding,
            input_size=encoding_config.input_size,
            circuit=resolved_circuit.circuit,
            experiment=resolved_circuit.experiment,
            noise_model=resolved_circuit.noise_model,
            has_custom_noise=resolved_circuit.has_custom_noise,
            input_state=input_state,
            n_photons=resolved_n_photons,
            trainable_parameters=circuit_source.trainable_parameters,
            input_parameters=circuit_source.input_parameters,
            angle_encoding_specs=circuit_source.angle_encoding_specs,
            photon_survival_probs=noise_and_detectors.photon_survival_probs,
            detectors=noise_and_detectors.detectors,
            has_custom_detectors=noise_and_detectors.has_custom_detectors,
            computation_space=computation_space,
            measurement_strategy=measurement_strategy,
            warnings=noise_and_detectors.detector_warnings,
        )

        # Phase 11: assign context to self + warnings
        self._finalize_from_context(context)
        # Phase 12: downstream setup
        # Defaults/validation handled in this method:
        # - Generate default input_state from n_photons when missing.
        # - Infer/validate input_size against encoder metadata.
        # - Setup parameters, measurement strategy, and output sizing.
        self._init_from_custom_circuit(context)

    def _finalize_from_context(self, context: InitializationContext) -> None:
        """Assign initialization context to instance attributes."""
        self.device = context.device
        self.dtype = context.dtype
        self.complex_dtype = context.complex_dtype
        self.input_size = context.input_size
        self.measurement_strategy = context.measurement_strategy
        self.experiment = context.experiment
        self.noise_model = context.noise_model
        self.amplitude_encoding = context.amplitude_encoding
        self.computation_space = context.computation_space
        self.angle_encoding_specs = context.angle_encoding_specs
        self.circuit = context.circuit
        self.has_custom_noise_model = context.has_custom_noise
        self.trainable_parameters = context.trainable_parameters
        self.input_parameters = context.input_parameters
        self.input_state = context.input_state
        self.n_photons = context.n_photons
        self._photon_survival_probs = context.photon_survival_probs
        self._detectors = context.detectors
        self._has_custom_detectors = context.has_custom_detectors
        self.detectors = self._detectors
        self._detector_transform: DetectorTransform | None = None
        self._photon_loss_transform: PhotonLossTransform | None = None
        self._detector_keys: list[tuple[int, ...]] = []
        self._raw_output_keys: list[tuple[int, ...]] = []
        self._detector_is_identity = True
        self._output_size = 0
        self._current_params: dict[str, Any] = {}

        for warning_msg in context.warnings:
            warnings.warn(warning_msg, UserWarning, stacklevel=3)

    # ---------------- core init paths ----------------

    def _init_from_custom_circuit(self, context: InitializationContext):
        """Initialize from custom circuit (backward compatible mode)."""
        circuit = context.circuit
        input_state = context.input_state
        n_photons = context.n_photons
        trainable_parameters = context.trainable_parameters
        input_parameters = context.input_parameters
        measurement_strategy = context.measurement_strategy

        if input_state is not None:
            self.input_state = input_state
        elif n_photons is not None:
            # Default behavior: place [1,0,1,0,...] in dual-rail, else distribute photons across modes
            if self.computation_space is ComputationSpace.DUAL_RAIL:
                self.input_state = [1, 0] * n_photons
            elif not self.amplitude_encoding:
                self.input_state = StateGenerator.generate_state(
                    circuit.m, n_photons, StatePattern.SPACED
                )
            else:
                self.input_state = [1] * n_photons + [0] * (circuit.m - n_photons)
        else:
            raise ValueError("Either input_state or n_photons must be provided")

        # Resolve n_photons and prepare input_state for ComputationProcess
        # Note: StateVector bypasses computation_space validation by using a placeholder list
        # during initialization; the actual tensor is set afterwards.
        process_input_state: list[int] | torch.Tensor
        statevector_input: StateVector | None = None
        if isinstance(self.input_state, StateVector):
            resolved_n_photons = (
                n_photons if n_photons is not None else self.input_state.n_photons
            )
            # Pass a placeholder list to ComputationProcess to avoid tensor dimension validation
            process_input_state = [1] * resolved_n_photons + [0] * (
                circuit.m - resolved_n_photons
            )
            statevector_input = self.input_state
        elif isinstance(self.input_state, torch.Tensor):
            resolved_n_photons = (
                n_photons  # n_photons must be provided for tensor input
            )
            process_input_state = self.input_state
        else:
            # list[int]
            resolved_n_photons = (
                n_photons if n_photons is not None else sum(self.input_state)
            )
            process_input_state = self.input_state

        self.computation_process = ComputationProcessFactory.create(
            circuit=circuit,
            input_state=process_input_state,
            trainable_parameters=trainable_parameters,
            input_parameters=input_parameters,
            n_photons=resolved_n_photons,
            device=self.device,
            dtype=self.dtype,
            computation_space=self.computation_space,
        )

        # If input_state was a StateVector, set the actual tensor now (after init to bypass validation)
        if statevector_input is not None:
            sv_tensor = statevector_input.to_dense()
            if sv_tensor.device != self.device:
                sv_tensor = sv_tensor.to(self.device)
            if sv_tensor.dtype != self.complex_dtype:
                sv_tensor = sv_tensor.to(self.complex_dtype)
            self.computation_process.input_state = sv_tensor

        # Setup PhotonLossTransform & DetectorTransform
        self.n_photons = self.computation_process.n_photons
        raw_keys = cast(
            list[tuple[int, ...]], self.computation_process.simulation_graph.mapped_keys
        )
        self._raw_output_keys = [self._normalize_output_key(key) for key in raw_keys]
        self._initialize_photon_loss_transform()
        self._initialize_detector_transform()

        # Validate that the declared input size matches encoder parameters
        spec_mappings = self.computation_process.converter.spec_mappings
        total_input_params = 0
        if input_parameters is not None:
            total_input_params = sum(
                len(spec_mappings.get(prefix, [])) for prefix in input_parameters
            )

        # Prefer metadata from angle encoding specs when available to deduce feature count
        expected_features: int | None = None
        if self.angle_encoding_specs:
            expected_features = 0
            specs_provided = False
            for metadata in self.angle_encoding_specs.values():
                # Each prefix maintains its own logical feature indices; count them separately
                # so distinct encoders do not collide when they reuse low-order indices.
                combos = metadata.get("combinations", [])
                prefix_indices = {idx for combo in combos for idx in combo}
                if not prefix_indices:
                    continue
                specs_provided = True
                expected_features += len(prefix_indices)
            if not specs_provided:
                expected_features = None

        if not self.amplitude_encoding:
            inferred_size = (
                expected_features
                if expected_features is not None
                else total_input_params
            )

            if self.input_size is None:
                # When the caller omits input_size, take the size the circuit exposes via its metadata.
                self.input_size = inferred_size
            elif inferred_size != self.input_size:
                if expected_features is not None:
                    raise ValueError(
                        f"Input size ({self.input_size}) must equal the number of encoded input features "
                        f"generated by the circuit ({expected_features})."
                    )
                else:
                    raise ValueError(
                        f"Input size ({self.input_size}) must equal the number of input parameters "
                        f"generated by the circuit ({total_input_params})."
                    )

        # Setup parameters and measurement strategy
        self._setup_parameters_from_custom(trainable_parameters)
        self._setup_measurement_strategy_from_custom(measurement_strategy)

        if self.amplitude_encoding:
            self._init_amplitude_metadata()

    def _setup_parameters_from_custom(self, trainable_parameters: list[str] | None):
        """Setup parameters from custom circuit configuration."""
        spec_mappings = self.computation_process.converter.spec_mappings
        self.thetas = []
        self.theta_names = []

        if trainable_parameters is None:
            return

        for tp in trainable_parameters:
            if tp in spec_mappings:
                theta_list = spec_mappings[tp]
                self.theta_names += theta_list
                parameter = nn.Parameter(
                    torch.randn(
                        (len(theta_list),), dtype=self.dtype, device=self.device
                    )
                    * torch.pi
                )
                self.register_parameter(tp, parameter)
                self.thetas.append(parameter)

    def _setup_measurement_strategy_from_custom(
        self, measurement_strategy: MeasurementStrategyLike
    ):
        """Setup output mapping for custom circuit construction."""
        if self._photon_loss_transform is None:
            raise RuntimeError(
                "Photon loss transform must be initialised before sizing."
            )
        if self._detector_transform is None:
            raise RuntimeError("Detector transform must be initialised before sizing.")

        kind = _resolve_measurement_kind(measurement_strategy)

        if kind == MeasurementKind.AMPLITUDES:
            keys = list(self._raw_output_keys)
        else:
            keys = (
                list(self._photon_loss_keys)
                if self._detector_is_identity
                else list(self._detector_keys)
            )

        dist_size = len(keys)

        # Determine output size (upstream model)
        if kind == MeasurementKind["PROBABILITIES"]:
            self._output_size = dist_size
        elif kind == MeasurementKind.MODE_EXPECTATIONS:
            # be defensive: `self.circuit` may be None or an untyped external object
            if self.circuit is not None and hasattr(self.circuit, "m"):
                self._output_size = self.circuit.m
            else:
                raise TypeError(f"Unknown circuit type: {type(self.circuit)}")
        elif kind == MeasurementKind.AMPLITUDES:
            self._output_size = dist_size
        elif kind == MeasurementKind.PARTIAL:
            if self._detector_transform is None:
                raise RuntimeError(
                    "Detector transform must be initialised before sizing."
                )
            self._output_size = self._detector_transform.output_size
        else:
            raise TypeError(f"Unknown measurement_strategy: {measurement_strategy}")

        # Create measurement mapping
        if kind == MeasurementKind.PARTIAL:
            self.measurement_mapping = nn.Identity()
        else:
            self.measurement_mapping = OutputMapper.create_mapping(
                measurement_strategy,
                self.computation_process.computation_space,
                keys,
                dtype=self.dtype,
            )

    def _init_amplitude_metadata(self) -> None:
        logical_keys = getattr(
            self.computation_process,
            "logical_keys",
            list(self.computation_process.simulation_graph.mapped_keys),
        )
        # TODO: here, the input_size corresponds to the size of the computation space
        # In future, we might want to decouple those two concepts
        self.input_size = len(logical_keys)

    def _create_dummy_parameters(self) -> list[torch.Tensor]:
        """Create dummy parameters for initialization."""
        spec_mappings = self.computation_process.converter.spec_mappings
        trainable_prefixes = list(
            getattr(self.computation_process, "trainable_parameters", [])
        )
        input_prefixes = list(self.computation_process.input_parameters)

        params: list[torch.Tensor] = []

        def _zeros(count: int) -> torch.Tensor:
            return torch.zeros(count, dtype=self.dtype, device=self.device)

        # Feed the true trainable parameters first, preserving converter order.
        theta_iter = iter(self.thetas)
        for prefix in trainable_prefixes:
            param = next(theta_iter, None)
            if param is not None:
                params.append(param)
                continue

            # Fall back to zero tensors only if no nn.Parameter exists yet.
            param_count = len(spec_mappings.get(prefix, []))
            params.append(_zeros(param_count))

        # Append any additional trainable parameters not covered by prefixes (defensive guard).
        params.extend(list(theta_iter))

        # Generate placeholder tensors for every declared input prefix in order. Encoders
        # sometimes omit converter specs ->  we fall
        # back to their stored combination metadata to deduce tensor length.
        for prefix in input_prefixes:
            # Counting parameters using their prefix
            param_count = self._feature_count_for_prefix(prefix) or 0
            if prefix in self.angle_encoding_specs:
                combos = self.angle_encoding_specs[prefix].get("combinations", [])
                if combos:
                    param_count = max(param_count, len(combos))
            params.append(_zeros(param_count))

        return params  # type: ignore[return-value]

    def _feature_count_for_prefix(self, prefix: str) -> int | None:
        """Infer the number of raw features associated with an encoding prefix."""
        spec_mappings = getattr(self.computation_process.converter, "spec_mappings", {})
        return feature_count_for_prefix(
            prefix, self.angle_encoding_specs, spec_mappings
        )

    def _split_inputs_by_prefix(
        self, prefixes: list[str], tensor: torch.Tensor
    ) -> list[torch.Tensor] | None:
        """Split a single logical input tensor into per-prefix chunks when possible."""
        spec_mappings = getattr(self.computation_process.converter, "spec_mappings", {})
        return split_inputs_by_prefix(
            prefixes, tensor, self.angle_encoding_specs, spec_mappings
        )

    def _prepare_input_encoding(
        self, x: torch.Tensor, prefix: str | None = None
    ) -> torch.Tensor:
        """Prepare input encoding based on mode."""
        return prepare_input_encoding(x, prefix, self.angle_encoding_specs)

    def _apply_angle_encoding(
        self, x: torch.Tensor, spec: dict[str, Any]
    ) -> torch.Tensor:
        """Apply custom angle encoding using stored metadata."""
        return apply_angle_encoding(x, spec)

    def _validate_amplitude_input(self, amplitude: torch.Tensor) -> torch.Tensor:
        if not isinstance(amplitude, torch.Tensor):
            raise TypeError(
                "Amplitude-encoded inputs must be provided as torch.Tensor instances"
            )

        if amplitude.dim() not in (1, 2):
            raise ValueError(
                "Amplitude-encoded inputs must be 1D (single state) or 2D (batch of states) tensors"
            )

        expected_dim = len(self.output_keys)
        feature_dim = amplitude.shape[-1]
        if feature_dim != expected_dim:
            raise ValueError(
                f"Amplitude input expects {expected_dim} components, received {feature_dim}."
            )
            # TODO: suggest/implement zero-padding or sparsity tensor format

        if amplitude.dtype not in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ):
            raise TypeError(
                "Amplitude-encoded inputs must use float32/float64 or complex64/complex128 dtype"
            )

        if self.device is not None and amplitude.device != self.device:
            amplitude = amplitude.to(self.device)

        if amplitude.is_complex():
            amplitude = amplitude.to(self.complex_dtype)
        else:
            amplitude = amplitude.to(self.dtype)

        return amplitude

    def set_input_state(self, input_state):
        self.input_state = input_state
        self.computation_process.input_state = input_state

    def prepare_parameters(
        self, input_parameters: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Prepare parameter list for circuit evaluation."""
        # Handle batching
        if input_parameters and input_parameters[0].dim() > 1:
            batch_size = input_parameters[0].shape[0]
            params = [theta.expand(batch_size, -1) for theta in self.thetas]
        else:
            params = list(self.thetas)

        # Apply input encoding
        prefixes = getattr(self.computation_process, "input_parameters", [])

        # Automatically split a single logical input across multiple prefixes when possible.
        # Builder circuits that define several encoders typically expose one logical tensor
        # to the user, while the converter expects separate tensors per prefix.
        if len(prefixes) > 1 and len(input_parameters) == 1:
            split_inputs = self._split_inputs_by_prefix(prefixes, input_parameters[0])
            if split_inputs is not None:
                input_parameters = split_inputs

        # Custom mode or multiple parameters
        for idx, x in enumerate(input_parameters):
            prefix = (
                prefixes[idx]
                if prefixes and idx < len(prefixes)
                else (prefixes[-1] if prefixes else None)
            )
            encoded = self._prepare_input_encoding(x, prefix)
            params.append(encoded)

        return params

    def forward(
        self,
        *input_parameters: torch.Tensor | StateVector,
        shots: int | None = None,
        sampling_method: str | None = None,
        simultaneous_processes: int | None = None,
    ) -> torch.Tensor:
        """Forward pass through the quantum layer.

        Encoding is inferred from the input type:

        - ``torch.Tensor`` (float): angle encoding (compatible with ``nn.Sequential``)
        - ``torch.Tensor`` (complex): amplitude encoding
        - ``StateVector``: amplitude encoding (preferred for quantum state injection)

        Parameters
        ----------
        *input_parameters : torch.Tensor | StateVector
            Input data. For angle encoding, pass float tensors. For amplitude
            encoding, pass a single ``StateVector`` or complex tensor.
        shots : int | None, optional
            Number of samples; if 0 or None, return exact amplitudes/probabilities.
        sampling_method : str | None, optional
            Sampling method, e.g. "multinomial".
        simultaneous_processes : int | None, optional
            Batch size hint for parallel computation.

        Returns
        -------
        torch.Tensor
            Output tensor after measurement mapping.

        Raises
        ------
        TypeError
            If inputs mix ``torch.Tensor`` and ``StateVector``, or if an
            unsupported input type is provided.
        ValueError
            If multiple ``StateVector`` inputs are provided.
        """
        # Phase 1: Input classification and validation
        tensor_inputs: list[torch.Tensor] = []
        amplitude_input: torch.Tensor | None = None
        original_input_state = None

        # Check for unsupported input types
        unsupported = [
            x
            for x in input_parameters
            if not isinstance(x, (torch.Tensor, StateVector))
        ]
        if unsupported:
            raise TypeError(
                f"Unsupported input types: {[type(x).__name__ for x in unsupported]}. "
                "Expected torch.Tensor or StateVector."
            )

        # Check for StateVector input → amplitude encoding
        if input_parameters and isinstance(input_parameters[0], StateVector):
            if len(input_parameters) > 1 and any(
                isinstance(x, StateVector) for x in input_parameters[1:]
            ):
                raise ValueError(
                    "Only one StateVector input is allowed per forward() call."
                )
            if len(input_parameters) > 1 and any(
                isinstance(x, torch.Tensor) for x in input_parameters[1:]
            ):
                raise TypeError(
                    "Cannot mix torch.Tensor and StateVector inputs in the same forward() call. "
                    "Use either tensor inputs (angle encoding) or StateVector (amplitude encoding)."
                )
            sv = input_parameters[0]
            # Convert to dense for computation pipeline (sparse not supported downstream).
            # StateVector's sparse representation is still valuable for memory-efficient
            # construction and manipulation; we only densify at computation time.
            amplitude_tensor = sv.to_dense()
            if amplitude_tensor.device != self.device:
                amplitude_tensor = amplitude_tensor.to(self.device)
            if amplitude_tensor.dtype != self.complex_dtype:
                amplitude_tensor = amplitude_tensor.to(self.complex_dtype)
            amplitude_input = self._validate_amplitude_input(amplitude_tensor)
            original_input_state = getattr(
                self.computation_process, "input_state", None
            )
            # tensor_inputs stays empty

        # Check for complex tensor input → amplitude encoding
        elif (
            input_parameters
            and len(input_parameters) == 1
            and isinstance(input_parameters[0], torch.Tensor)
            and input_parameters[0].is_complex()
        ):
            amplitude_input = self._validate_amplitude_input(input_parameters[0])
            original_input_state = getattr(
                self.computation_process, "input_state", None
            )
            # tensor_inputs stays empty

        # Legacy amplitude_encoding=True flag
        elif self.amplitude_encoding:
            tensor_inputs = [x for x in input_parameters if isinstance(x, torch.Tensor)]
            if not tensor_inputs:
                raise ValueError(
                    "QuantumLayer configured with amplitude_encoding=True expects an amplitude tensor input."
                )
            # Warn if using real tensor with amplitude_encoding (internal conversion is deprecated)
            if tensor_inputs and not tensor_inputs[0].is_complex():
                warnings.warn(
                    "Passing real-valued tensor with amplitude_encoding=True is deprecated and will be "
                    "removed in 0.4. Pass a StateVector or complex tensor to forward() instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            amplitude_input, tensor_inputs, original_input_state = (
                self._prepare_amplitude_input(tensor_inputs)
            )

        # Float tensor(s) → angle encoding
        else:
            tensor_inputs = [x for x in input_parameters if isinstance(x, torch.Tensor)]
            if any(isinstance(x, StateVector) for x in input_parameters):
                raise TypeError(
                    "Cannot mix torch.Tensor and StateVector inputs in the same forward() call. "
                    "Use either tensor inputs (angle encoding) or StateVector (amplitude encoding). "
                    "To use a custom input state with angle encoding, set it via the constructor or set_input_state()."
                )

        # Phase 2: Parameter assembly for circuit execution
        params, parameter_batch_dim = self._prepare_classical_parameters(tensor_inputs)

        # Phase 3: Compute amplitudes
        with self._temporary_input_state(amplitude_input, original_input_state):
            raw_inferred_state = getattr(self.computation_process, "input_state", None)
            inferred_state: torch.Tensor | None
            if isinstance(raw_inferred_state, torch.Tensor):
                inferred_state = raw_inferred_state
            else:
                inferred_state = None
            # Override inferred_state if amplitude encoding via new input types
            if amplitude_input is not None and original_input_state is not None:
                inferred_state = amplitude_input
            amplitudes = self._compute_amplitudes(
                params,
                inferred_state=inferred_state,
                parameter_batch_dim=parameter_batch_dim,
                simultaneous_processes=simultaneous_processes,
            )

        # Phase 4: Configure sampling/autodiff
        needs_gradient = (
            self.training
            and torch.is_grad_enabled()
            and any(p.requires_grad for p in self.parameters())
        )

        local_sampling_method = sampling_method or "multinomial"
        adp = AutoDiffProcess(local_sampling_method)

        requested_shots = int(shots or 0)
        apply_sampling = requested_shots > 0

        apply_sampling, effective_shots = adp.autodiff_backend(
            needs_gradient, apply_sampling, requested_shots
        )

        # Phase 5: Convert and normalize amplitudes
        if isinstance(amplitudes, tuple):
            amplitudes = amplitudes[1]
        elif not isinstance(amplitudes, torch.Tensor):
            raise TypeError(f"Unexpected amplitudes type: {type(amplitudes)}")

        distribution, amplitudes = self._renormalize_distribution_and_amplitudes(
            amplitudes
        )

        # Phase 6: Measurement strategy dispatch and output mapping
        strategy = resolve_measurement_strategy(self.measurement_strategy)
        # Handle backward compatibility for backpropagation - will be removed in future
        grouping = None
        if isinstance(self.measurement_strategy, MeasurementStrategy):
            if self.measurement_strategy.type == MeasurementKind.PROBABILITIES:
                grouping = self.measurement_strategy.grouping

        results = strategy.process(
            distribution=distribution,
            amplitudes=amplitudes,
            apply_sampling=apply_sampling,
            effective_shots=effective_shots,
            sample_fn=adp.sampling_noise.pcvl_sampler,
            apply_photon_loss=self._apply_photon_loss_transform,
            apply_detectors=self._apply_detector_transform,
            grouping=grouping,
        )
        # If partial measurement, return raw results
        if (
            _resolve_measurement_kind(self.measurement_strategy)
            == MeasurementKind.PARTIAL
        ):
            return results

        return self.measurement_mapping(results)

    def _compute_amplitudes(
        self,
        params: list[torch.Tensor],
        *,
        inferred_state: torch.Tensor | None,
        parameter_batch_dim: int,
        simultaneous_processes: int | None,
    ) -> torch.Tensor:
        """Select the computation path based on the encoding mode and input state."""
        if self.amplitude_encoding:
            if inferred_state is None:
                raise TypeError(
                    "Amplitude encoding requires the computation process input_state to be a tensor."
                )
            batch_size = (
                simultaneous_processes
                if simultaneous_processes is not None
                else (1 if inferred_state.dim() == 1 else inferred_state.shape[0])
            )
            return self.computation_process.compute_ebs_simultaneously(
                params, simultaneous_processes=batch_size
            )
        if isinstance(inferred_state, torch.Tensor):
            if parameter_batch_dim:
                chunk = simultaneous_processes or inferred_state.shape[-1]
                return self.computation_process.compute_ebs_simultaneously(
                    params, simultaneous_processes=chunk
                )
            return self.computation_process.compute_superposition_state(params)
        return self.computation_process.compute(params)

    def _renormalize_distribution_and_amplitudes(
        self, amplitudes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return probability distribution and renormalized amplitudes."""
        # even in amplitude mode, we do need to calculation distribution for renormalization
        # of the amplitudes
        distribution = amplitudes.real**2 + amplitudes.imag**2

        # renormalize distribution and amplitudes for UNBUNCHED and DUAL_RAIL spaces
        if (
            self.computation_space is ComputationSpace.UNBUNCHED
            or self.computation_space is ComputationSpace.DUAL_RAIL
        ):
            sum_probs = distribution.sum(dim=-1, keepdim=True)

            # Only normalize when sum > 0 to avoid division by zero
            valid_entries = sum_probs > 0
            if valid_entries.any():
                distribution = torch.where(
                    valid_entries,
                    distribution
                    / torch.where(valid_entries, sum_probs, torch.ones_like(sum_probs)),
                    distribution,
                )
                amplitudes = torch.where(
                    valid_entries,
                    amplitudes
                    / torch.where(
                        valid_entries, sum_probs.sqrt(), torch.ones_like(sum_probs)
                    ),
                    amplitudes,
                )

        return distribution, amplitudes

    def _prepare_amplitude_input(
        self, inputs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
        """Validate amplitude-encoded input and return remaining inputs."""
        if not inputs:
            raise ValueError(
                "QuantumLayer configured with amplitude_encoding=True expects an amplitude tensor input."
            )
        amplitude_input = self._validate_amplitude_input(inputs[0])
        original_input_state = getattr(self.computation_process, "input_state", None)
        return amplitude_input, inputs[1:], original_input_state

    @contextmanager
    def _temporary_input_state(
        self,
        amplitude_input: torch.Tensor | None,
        original_input_state: torch.Tensor | None,
    ):
        if amplitude_input is None:
            yield
            return
        self.set_input_state(amplitude_input)
        try:
            yield
        finally:
            if original_input_state is not None:
                self.set_input_state(original_input_state)

    def _prepare_classical_parameters(
        self, inputs: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], int]:
        """Prepare parameter list and return inferred batch dimension for classical inputs."""
        params = self.prepare_parameters(inputs)
        # Track batch width across classical inputs so we can route superposed tensors through the batched path.
        parameter_batch_dim = 0
        for tensor in params:
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 1:
                batch = tensor.shape[0]
                if parameter_batch_dim and batch != parameter_batch_dim:
                    raise ValueError(
                        "Inconsistent batch dimensions across classical input parameters."
                    )
                parameter_batch_dim = batch
        return params, parameter_batch_dim

    @sanitize_parameters
    def set_sampling_config(
        self, shots: int | None = None, sampling_method: str | None = None
    ):
        """Deprecated: sampling configuration must be provided at call time in `forward`."""
        # Fatal deprecation is handled by the sanitize_parameters decorator via registry.
        return None

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Manually move any additional tensors
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            device = args[0]
        if device is not None:
            self.device = device
            self.computation_process.simulation_graph = (
                self.computation_process.simulation_graph.to(device)
            )
            self.computation_process.converter = self.computation_process.converter.to(
                self.dtype, device
            )

            # Photon loss Module
            if self._photon_loss_transform is not None:
                self._photon_loss_transform = self._photon_loss_transform.to(device)
            # Detector Module
            if self._detector_transform is not None:
                self._detector_transform = self._detector_transform.to(device)

        return self

    @property
    def output_keys(self):
        """Return the Fock basis associated with the layer outputs."""
        if (
            getattr(self, "_photon_loss_transform", None) is None
            or getattr(self, "_detector_transform", None) is None
        ):
            return [self._normalize_output_key(key) for key in self._raw_output_keys]
        if (
            _resolve_measurement_kind(self.measurement_strategy)
            == MeasurementKind.AMPLITUDES
        ):
            return list(self._raw_output_keys)
        if self._detector_is_identity:
            return list(self._photon_loss_keys)
        return list(self._detector_keys)

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def has_custom_detectors(self) -> bool:
        return self._has_custom_detectors

    def _initialize_photon_loss_transform(self) -> None:
        self._photon_loss_transform = PhotonLossTransform(
            self._raw_output_keys,
            self._photon_survival_probs,
            dtype=self.dtype,
            device=self.device,
        )
        self._photon_loss_keys = self._photon_loss_transform.output_keys
        self._photon_loss_is_identity = self._photon_loss_transform.is_identity

    def _initialize_detector_transform(self) -> None:
        detectors = self._detectors
        partial = False
        if (
            _resolve_measurement_kind(self.measurement_strategy)
            == MeasurementKind.PARTIAL
        ):
            if not isinstance(self.measurement_strategy, MeasurementStrategy):
                raise TypeError(
                    "MeasurementStrategy.partial() must be used for partial measurement."
                )
            if not self.measurement_strategy.measured_modes:
                raise ValueError(
                    "Partial measurement requires at least one measured mode."
                )
            n_modes = len(self._photon_loss_keys[0])
            self.measurement_strategy.validate_modes(n_modes)
            measured = set(self.measurement_strategy.measured_modes)
            detectors = [
                det if idx in measured else None
                for idx, det in enumerate(self._detectors)
            ]
            partial = True
        detector_transform = DetectorTransform(
            self._photon_loss_keys,
            detectors,
            dtype=self.dtype,
            device=self.device,
            partial_measurement=partial,
        )
        self._detector_transform = detector_transform
        self._detector_keys = detector_transform.output_keys
        self._detector_is_identity = detector_transform.is_identity

    @staticmethod
    def _normalize_output_key(
        key: Iterable[int] | torch.Tensor | Sequence[int],
    ) -> tuple[int, ...]:
        if isinstance(key, torch.Tensor):
            return tuple(int(v) for v in key.tolist())
        return tuple(int(v) for v in key)

    def _apply_photon_loss_transform(self, distribution: torch.Tensor) -> torch.Tensor:
        if self._photon_loss_transform is None:
            raise RuntimeError(
                "Photon loss transform must be initialised before applying photon loss."
            )
        if self._photon_loss_is_identity:
            return distribution
        return self._photon_loss_transform(distribution)

    def _apply_detector_transform(self, distribution: torch.Tensor) -> torch.Tensor:
        if self._detector_transform is None:
            raise RuntimeError(
                "Detector transform must be initialised before applying detectors."
            )
        if self._detector_is_identity:
            return distribution
        return self._detector_transform(distribution)

    # =====================  EXPORT API FOR REMOTE PROCESSORS  =====================

    def _update_current_params(self) -> None:
        self._current_params.clear()
        for name, param in self.named_parameters():
            if param.requires_grad:
                self._current_params[name] = param.detach().cpu().numpy()

    def export_config(self) -> dict:
        """
        Export a standalone configuration for remote execution.
        """
        # TODO: to be revisited - not all options seems to be exported
        self._update_current_params()

        if self.experiment is not None:
            exported_circuit = self.experiment.unitary_circuit()
        else:
            exported_circuit = (
                self.circuit.copy() if hasattr(self.circuit, "copy") else self.circuit
            )

        spec_mappings = getattr(self.computation_process.converter, "spec_mappings", {})
        torch_params: dict[str, torch.Tensor] = {
            n: p for n, p in self.named_parameters() if p.requires_grad
        }

        for p in exported_circuit.get_parameters():
            pname: str = getattr(p, "name", "")
            for tp_prefix in self.trainable_parameters:
                names_for_prefix = spec_mappings.get(tp_prefix, [])
                if pname in names_for_prefix:
                    idx = names_for_prefix.index(pname)
                    tparam = torch_params.get(tp_prefix, None)
                    if tparam is None:
                        break
                    value = float(tparam.detach().cpu().view(-1)[idx].item())
                    p.set_value(value)
                    break

        config = {
            "circuit": exported_circuit,
            "experiment": self.experiment,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "input_state": getattr(self, "input_state", None),
            "n_modes": exported_circuit.m,
            "n_photons": (
                sum(getattr(self, "input_state", []) or [])
                if hasattr(self, "input_state")
                else None
            ),
            "trainable_parameters": list(self.trainable_parameters),
            "input_parameters": list(self.input_parameters),
            "noise_model": self.noise_model,
        }
        return config

    # ============================================================================

    @classmethod
    @sanitize_parameters
    def simple(
        cls,
        input_size: int,
        output_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        computation_space: ComputationSpace | str = ComputationSpace.UNBUNCHED,
    ):
        """Create a ready-to-train layer with a input_size-mode, (input_size//2)-photon architecture.

        The circuit is assembled via :class:`CircuitBuilder` with the following layout:

        1. A fully trainable entangling layer acting on all modes;
        2. A full input encoding layer spanning all encoded features;
        3. A fully trainable entangling layer acting on all modes.

        Args:
            input_size: Size of the classical input vector. Must be 20 or lower.
            output_size: Optional classical output width.
            device: Optional target device for tensors.
            dtype: Optional tensor dtype.
            computation_space: Logical computation subspace; one of {"fock", "unbunched", "dual_rail"}.

        Returns:
            QuantumLayer configured with the described architecture.
        """
        if input_size > 20:
            raise ValueError(
                "Input size too large for the simple layer construction. For large inputs (with larger size than 20), please use the CircuitBuilder. Here is a quick tutorial on how to use it: https://merlinquantum.ai/quickstart/first_quantum_layer.html#circuitbuilder-walkthrough"
            )
        if input_size < 1:
            raise ValueError(f"input_size must be at least 1, got {input_size}")

        if input_size == 1:
            n_photons = 1
            input_state = [0, 1]
            builder = CircuitBuilder(n_modes=2)

            # Trainable entangling layer before encoding
            builder.add_entangling_layer(trainable=True, name="LI_simple")

            # Angle encoding
            builder.add_angle_encoding(
                modes=[1], name="input", subset_combinations=False
            )

            # Trainable entangling layer after encoding
            builder.add_entangling_layer(trainable=True, name="RI_simple")

        else:
            n_photons = input_size // 2

            input_state = input_size * [0]
            for i in range(input_size):
                if i % 2 == 1:
                    input_state[i] = 1
            input_state = pcvl.BasicState(input_state)

            builder = CircuitBuilder(n_modes=input_size)

            # Trainable entangling layer before encoding
            builder.add_entangling_layer(trainable=True, name="LI_simple")

            # Angle encoding
            builder.add_angle_encoding(
                name="input",
                subset_combinations=False,
            )

            # Trainable entangling layer after encoding
            builder.add_entangling_layer(trainable=True, name="RI_simple")

        quantum_layer_kwargs = {
            "input_size": input_size,
            "input_state": input_state,
            "builder": builder,
            "n_photons": n_photons,
            "measurement_strategy": MeasurementKind["PROBABILITIES"],
            "device": device,
            "dtype": dtype,
            "computation_space": computation_space,
        }

        # mypy: quantum_layer_kwargs is constructed dynamically; cast to satisfy
        # the type checker that keys match the constructor signature.
        quantum_layer = cls(**cast(dict[str, Any], quantum_layer_kwargs))

        class SimpleSequential(nn.Module):
            """Simple Sequential Module that contains the quantum layer as well as the post processing"""

            def __init__(self, quantum_layer: QuantumLayer, post_processing: nn.Module):
                super().__init__()
                self.quantum_layer = quantum_layer
                self.post_processing = post_processing
                self.add_module("quantum_layer", quantum_layer)
                self.add_module("post_processing", post_processing)
                self.circuit = quantum_layer.circuit
                if hasattr(post_processing, "output_size"):
                    self._output_size = cast(int, post_processing.output_size)
                else:
                    self._output_size = quantum_layer.output_size

            @property
            def output_size(self):
                return self._output_size

            def forward(
                self,
                x: torch.Tensor,
                *,
                shots: int | None = None,
                sampling_method: str | None = "multinomial",
            ) -> torch.Tensor:
                q_out = self.quantum_layer(
                    x,
                    shots=shots,
                    sampling_method=sampling_method,
                )
                return self.post_processing(q_out)

        if output_size is not None:
            if not isinstance(output_size, int):
                raise TypeError("output_size must be an integer.")
            if output_size <= 0:
                raise ValueError("output_size must be a positive integer.")
            if output_size != quantum_layer.output_size:
                model = SimpleSequential(
                    quantum_layer, ModGrouping(quantum_layer.output_size, output_size)
                )
            else:
                model = SimpleSequential(quantum_layer, nn.Identity())
        else:
            model = SimpleSequential(quantum_layer, nn.Identity())

        return model

    def __str__(self) -> str:
        """String representation of the quantum layer."""
        n_modes = None
        circuit = getattr(self, "circuit", None)
        if circuit is not None and getattr(circuit, "m", None) is not None:
            n_modes = circuit.m

        modes_fragment = f", modes={n_modes}" if n_modes is not None else ""
        base_str = (
            f"QuantumLayer(custom_circuit{modes_fragment}, input_size={self.input_size}, "
            f"output_size={self.output_size}"
        )

        return base_str + ")"
