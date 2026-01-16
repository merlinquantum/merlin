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
from typing import Any, cast

import perceval as pcvl
import torch
import torch.nn as nn

from ..builder.circuit_builder import (
    ANGLE_ENCODING_MODE_ERROR,
    CircuitBuilder,
)
from ..core.computation_space import ComputationSpace
from ..core.generators import StateGenerator, StatePattern
from ..core.process import ComputationProcessFactory
from ..measurement import OutputMapper
from ..measurement.autodiff import AutoDiffProcess
from ..measurement.detectors import DetectorTransform
from ..measurement.photon_loss import PhotonLossTransform
from ..measurement.strategies import MeasurementStrategy
from ..utils.deprecations import sanitize_parameters
from ..utils.grouping import ModGrouping
from .layer_utils import (
    InitializationContext,
    prepare_input_state,
    resolve_circuit,
    setup_noise_and_detectors,
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
        input_state: list[int] | pcvl.BasicState | pcvl.StateVector | None = None,
        n_photons: int | None = None,
        # only for custom circuits and experiments
        trainable_parameters: list[str] | None = None,
        input_parameters: list[str] | None = None,
        # Common parameters
        amplitude_encoding: bool = False,
        computation_space: ComputationSpace | str = ComputationSpace.UNBUNCHED,
        measurement_strategy: MeasurementStrategy = MeasurementStrategy.PROBABILITIES,
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
        input_state : list[int] | pcvl.BasicState | pcvl.StateVector | None, optional
            Logical input state of the circuit. Accepted forms:
            - list of occupations (length = number of modes),
            - ``pcvl.BasicState`` without annotations (plain FockState only),
            - ``pcvl.StateVector`` (converted to a tensor according to
              ``computation_space``).
            If QuantumLayer is built from an experiment, the experiment's input state is used.
            If omitted, ``n_photons`` must be provided to derive a default state.
            The dual-rail space defaults to ``[1,0,1,0,...]`` while other spaces
            evenly distribute the photons across the available modes.
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
            When True, the forward call expects an amplitude vector (or batch) on
            the first positional argument and propagates it through the quantum
            layer; ``input_size`` must not be set in this mode and
            ``n_photons`` must be provided.
        computation_space : ComputationSpace | str, optional
            Logical computation subspace to use: one of ``{"fock", "unbunched",
            "dual_rail"}``. If omitted, defaults to ``UNBUNCHED``.
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

        """
        super().__init__()

        device, dtype, complex_dtype = MerlinModule.setup_device_and_dtype(
            device, dtype
        )
        computation_space = ComputationSpace.coerce(computation_space)

        circuit_source = validate_and_resolve_circuit_source(
            builder, circuit, experiment, trainable_parameters, input_parameters
        )
        encoding_config = validate_encoding_mode(
            amplitude_encoding,
            input_size,
            n_photons,
            circuit_source.input_parameters,
        )
        input_state, resolved_n_photons = prepare_input_state(
            input_state, n_photons, computation_space, device, complex_dtype, experiment
        )

        if experiment is not None:
            vet_experiment(experiment)

        resolved_circuit = resolve_circuit(circuit_source, pcvl)
        noise_and_detectors = setup_noise_and_detectors(
            resolved_circuit.experiment,
            resolved_circuit.circuit,
            computation_space,
            measurement_strategy,
        )

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

        self._finalize_from_context(context)

        self._init_from_custom_circuit(
            self.circuit,
            self.input_state,
            self.n_photons,
            self.trainable_parameters,
            self.input_parameters,
            self.measurement_strategy,
        )

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
        self._detector_transform = None
        self._detector_keys = []
        self._raw_output_keys = []
        self._detector_is_identity = True
        self._output_size = 0
        self._current_params = {}

        for warning_msg in context.warnings:
            warnings.warn(warning_msg, UserWarning, stacklevel=3)

    # ---------------- core init paths ----------------

    def _init_from_custom_circuit(
        self,
        circuit: pcvl.Circuit,
        input_state: list[int] | None,
        n_photons: int | None,
        trainable_parameters: list[str],
        input_parameters: list[str],
        measurement_strategy: MeasurementStrategy,
    ):
        """Initialize from custom circuit (backward compatible mode)."""
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

        resolved_n_photons = (
            n_photons if n_photons is not None else sum(self.input_state)
        )

        self.computation_process = ComputationProcessFactory.create(
            circuit=circuit,
            input_state=self.input_state,
            trainable_parameters=trainable_parameters,
            input_parameters=input_parameters,
            n_photons=resolved_n_photons,
            device=self.device,
            dtype=self.dtype,
            computation_space=self.computation_space,
        )

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

        # set input_size for amplitude encoding
        if self.amplitude_encoding:
            self.input_size = len(self.output_keys)

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
        self, measurement_strategy: MeasurementStrategy
    ):
        """Setup output mapping for custom circuit construction."""
        if self._photon_loss_transform is None:
            raise RuntimeError(
                "Photon loss transform must be initialised before sizing."
            )
        if self._detector_transform is None:
            raise RuntimeError("Detector transform must be initialised before sizing.")

        if measurement_strategy == MeasurementStrategy.AMPLITUDES:
            keys = list(self._raw_output_keys)
        else:
            keys = (
                list(self._photon_loss_keys)
                if self._detector_is_identity
                else list(self._detector_keys)
            )

        dist_size = len(keys)

        # Determine output size (upstream model)
        if measurement_strategy == MeasurementStrategy.PROBABILITIES:
            self._output_size = dist_size
        elif measurement_strategy == MeasurementStrategy.MODE_EXPECTATIONS:
            # be defensive: `self.circuit` may be None or an untyped external object
            if self.circuit is not None and hasattr(self.circuit, "m"):
                self._output_size = self.circuit.m
            else:
                raise TypeError(f"Unknown circuit type: {type(self.circuit)}")
        elif measurement_strategy == MeasurementStrategy.AMPLITUDES:
            self._output_size = dist_size
        else:
            raise TypeError(f"Unknown measurement_strategy: {measurement_strategy}")

        # Create measurement mapping
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
        spec = self.angle_encoding_specs.get(prefix)
        if spec:
            combos = spec.get("combinations", [])
            feature_indices = {idx for combo in combos for idx in combo}
            if feature_indices:
                return len(feature_indices)

        spec_mappings = getattr(self.computation_process.converter, "spec_mappings", {})
        mapping = spec_mappings.get(prefix, [])
        if mapping:
            return len(mapping)

        return None

    def _split_inputs_by_prefix(
        self, prefixes: list[str], tensor: torch.Tensor
    ) -> list[torch.Tensor] | None:
        """Split a single logical input tensor into per-prefix chunks when possible."""

        counts: list[int] = []
        for prefix in prefixes:
            count = self._feature_count_for_prefix(prefix)
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

    def _prepare_input_encoding(
        self, x: torch.Tensor, prefix: str | None = None
    ) -> torch.Tensor:
        """Prepare input encoding based on mode."""
        spec = None
        if prefix is not None:
            spec = self.angle_encoding_specs.get(prefix)
        elif len(self.angle_encoding_specs) == 1:
            spec = next(iter(self.angle_encoding_specs.values()))

        if spec:
            return self._apply_angle_encoding(x, spec)

        return x

    def _apply_angle_encoding(
        self, x: torch.Tensor, spec: dict[str, Any]
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

            # Select per-combo features and scale
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

        if squeeze:
            return encoded.squeeze(0)
        return encoded

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
        *input_parameters: torch.Tensor,
        shots: int | None = None,
        sampling_method: str | None = None,
        simultaneous_processes: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass through the quantum layer.

        When ``self.amplitude_encoding`` is ``True`` the first positional argument
        must contain the amplitude-encoded input state (either ``[num_states]`` or
        ``[batch_size, num_states]``). Remaining positional arguments are treated
        as classical inputs and processed via the standard encoding pipeline.

        Sampling is controlled by:
            - shots (int): number of samples; if 0 or None, return exact amplitudes/probabilities.
            - sampling_method (str): e.g. "multinomial".
        """

        inputs = list(input_parameters)
        amplitude_input: torch.Tensor | None = None
        original_input_state = None

        if self.amplitude_encoding:
            if not inputs:
                raise ValueError(
                    "QuantumLayer configured with amplitude_encoding=True expects an amplitude tensor input."
                )
            # verify that inputs is of the shape of layer.compute_graph.mapped_keys
            amplitude_input = self._validate_amplitude_input(inputs.pop(0))
            original_input_state = getattr(
                self.computation_process, "input_state", None
            )
            # amplitude_input becomes the new input_state
            self.set_input_state(amplitude_input)

        # classical_inputs = [
        #    tensor for tensor in inputs if isinstance(tensor, torch.Tensor)
        # ]

        # Prepare circuit parameters and any remaining classical inputs
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
        # TODO: input_state should support StateVector
        raw_inferred_state = getattr(self.computation_process, "input_state", None)
        # normalize the retrieved input_state to an optional tensor an
        inferred_state: torch.Tensor | None
        if isinstance(raw_inferred_state, torch.Tensor):
            inferred_state = raw_inferred_state
        else:
            inferred_state = None
        amplitudes: torch.Tensor

        # TODO: challenge the need for trying/finally here
        try:
            if self.amplitude_encoding:
                # raise error if amplitude encoding finds a non tensor state
                if inferred_state is None:
                    raise TypeError(
                        "Amplitude encoding requires the computation process input_state to be a tensor."
                    )
                # we always use the parallel ebs computation path for amplitude encoding to enable batching
                if simultaneous_processes is not None:
                    batch_size = simultaneous_processes
                else:
                    batch_size = (
                        inferred_state.dim() == 1 and 1 or inferred_state.shape[0]
                    )
                amplitudes = self.computation_process.compute_ebs_simultaneously(
                    params, simultaneous_processes=batch_size
                )
            elif isinstance(inferred_state, torch.Tensor):
                # otherwise the incremental EBS path allowing batch on input parameters
                if parameter_batch_dim:
                    # Classical inputs are batched: reuse the EBS batching kernel to propagate all coefficients at once.
                    chunk = simultaneous_processes or inferred_state.shape[-1]
                    amplitudes = self.computation_process.compute_ebs_simultaneously(
                        params, simultaneous_processes=chunk
                    )
                else:
                    amplitudes = self.computation_process.compute_superposition_state(
                        params
                    )
            else:
                amplitudes = self.computation_process.compute(params)
        finally:
            if amplitude_input is not None and original_input_state is not None:
                self.set_input_state(original_input_state)

        # Determine gradient needs
        needs_gradient = (
            self.training
            and torch.is_grad_enabled()
            and any(p.requires_grad for p in self.parameters())
        )

        # Per-call autodiff/sampling backend
        local_sampling_method = sampling_method or "multinomial"
        adp = AutoDiffProcess(local_sampling_method)

        # Derive apply_sampling from shots > 0
        requested_shots = int(shots or 0)
        apply_sampling = requested_shots > 0

        # Backend may override shots/sampling if gradients are required
        apply_sampling, effective_shots = adp.autodiff_backend(
            needs_gradient, apply_sampling, requested_shots
        )

        # Convert amplitudes to probabilities if needed
        if isinstance(amplitudes, tuple):
            amplitudes = amplitudes[1]
        elif not isinstance(amplitudes, torch.Tensor):
            raise TypeError(f"Unexpected amplitudes type: {type(amplitudes)}")

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
        if self.measurement_strategy in (
            MeasurementStrategy.PROBABILITIES,
            MeasurementStrategy.MODE_EXPECTATIONS,
        ):
            distribution = self._apply_photon_loss_transform(distribution)
            distribution = self._apply_detector_transform(distribution)

            # Apply sampling if requested
            if apply_sampling and effective_shots > 0:
                results = adp.sampling_noise.pcvl_sampler(distribution, effective_shots)
            else:
                results = distribution

        # For MeasurementStrategy.AMPLITUDES, bypass detectors and sampling
        else:
            if apply_sampling:
                raise RuntimeError(
                    "Sampling cannot be applied when measurement_strategy=MeasurementStrategy.AMPLITUDES."
                )
            results = amplitudes

        # Apply measurement mapping (returns tensor of shape [B, output_size])
        return self.measurement_mapping(results)

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
        if self.measurement_strategy == MeasurementStrategy.AMPLITUDES:
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
        self._detector_transform = DetectorTransform(
            self._photon_loss_keys,
            self._detectors,
            dtype=self.dtype,
            device=self.device,
        )
        self._detector_keys = self._detector_transform.output_keys
        self._detector_is_identity = self._detector_transform.is_identity

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
            "n_photons": sum(getattr(self, "input_state", []) or [])
            if hasattr(self, "input_state")
            else None,
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
        n_params: int = 90,
        output_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        computation_space: ComputationSpace | str = ComputationSpace.UNBUNCHED,
    ):
        """Create a ready-to-train layer with a 10-mode, 5-photon architecture.

        The circuit is assembled via :class:`CircuitBuilder` with the following layout:

        1. A fully trainable entangling layer acting on all modes;
        2. A full input encoding layer spanning all encoded features;
        3. A non-trainable entangling layer that redistributes encoded information;
        4. Optional trainable Mach-Zehnder blocks (two parameters each) to reach the requested ``n_params`` budget;
        5. A final entangling layer prior to measurement.

        Args:
            input_size: Size of the classical input vector.
            n_params: Number of trainable parameters to allocate across the additional MZI blocks. Values
                below the default entangling budget trigger a warning; values above it must differ by an even
                amount because each added MZI exposes two parameters.
            output_size: Optional classical output width.
            device: Optional target device for tensors.
            dtype: Optional tensor dtype.
            computation_space: Logical computation subspace; one of {"fock", "unbunched", "dual_rail"}.

        Returns:
            QuantumLayer configured with the described architecture.
        """

        n_modes = 10
        n_photons = 5

        builder = CircuitBuilder(n_modes=n_modes)

        # Trainable entangling layer before encoding
        builder.add_entangling_layer(trainable=True, name="gi_simple")
        entangling_params = n_modes * (n_modes - 1)

        requested_params = max(int(n_params), 0)
        if entangling_params > requested_params:
            warnings.warn(
                "Entangling layer introduces "
                f"{entangling_params} trainable parameters, exceeding the requested "
                f"budget of {requested_params}. The simple layer will expose "
                f"{entangling_params} trainable parameters.",
                RuntimeWarning,
                stacklevel=2,
            )

        if input_size > n_modes:
            raise ValueError(ANGLE_ENCODING_MODE_ERROR)

        input_modes = list(range(input_size))
        builder.add_angle_encoding(
            modes=input_modes,
            name="input",
            subset_combinations=False,
        )

        # Allocate additional trainable MZIs only if the budget exceeds the entangling layer
        remaining = max(requested_params - entangling_params, 0)
        if remaining % 2 != 0:
            raise ValueError(
                "Additional parameter budget must be even: each extra MZI exposes "
                "two trainable parameters."
            )

        mzi_idx = 0
        added_mzi_params = 0

        while remaining > 0:
            if n_modes < 2:
                raise ValueError("At least two modes are required to place an MZI.")

            start_mode = mzi_idx % (n_modes - 1)
            span_modes = [start_mode, start_mode + 1]
            prefix = f"mzi_extra{mzi_idx}"

            builder.add_entangling_layer(
                modes=span_modes,
                trainable=True,
                name=prefix,
            )

            remaining -= 2
            added_mzi_params += 2
            mzi_idx += 1

        # Post-MZI entanglement
        builder.add_superpositions()

        total_trainable = entangling_params + added_mzi_params
        expected_trainable = max(requested_params, entangling_params)
        if total_trainable != expected_trainable:
            raise ValueError(
                "Constructed circuit exposes "
                f"{total_trainable} trainable parameters but {expected_trainable} were expected."
            )

        quantum_layer_kwargs = {
            "input_size": input_size,
            "builder": builder,
            "n_photons": n_photons,
            "measurement_strategy": MeasurementStrategy.PROBABILITIES,
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
