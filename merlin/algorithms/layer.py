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
Main QuantumLayer implementation with bug fixes and index_photons support.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn

from ..builder.ansatz import Ansatz
from ..builder.circuit_builder import (
    ANGLE_ENCODING_MODE_ERROR,
    CircuitBuilder,
)
from ..core.components import GenericInterferometer
from ..core.process import ComputationProcessFactory
from ..sampling.autodiff import AutoDiffProcess
from ..sampling.strategies import OutputMappingStrategy
from ..torch_utils.torch_codes import OutputMapper


class QuantumLayer(nn.Module):
    """
    Enhanced Quantum Neural Network Layer with factory-based architecture.

    This layer can be created either from:
    1. An Ansatz object (from AnsatzFactory) - for auto-generated circuits
    2. Direct parameters - for custom circuits (backward compatible)

    Args:
        index_photons (List[Tuple[int, int]], optional): List of tuples (min_mode, max_mode)
            constraining where each photon can be placed. The first_integer is the lowest
            index layer a photon can take and the second_integer is the highest index.
            If None, photons can be placed in any mode from 0 to m-1.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        # Ansatz-based construction
        ansatz: Ansatz | None = None,
        # Custom circuit construction (backward compatible)
        circuit: pcvl.Circuit | CircuitBuilder | None = None,
        input_state: list[int] | None = None,
        n_photons: int | None = None,
        trainable_parameters: list[str] = None,
        input_parameters: list[str] = None,
        # Common parameters
        output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.LINEAR,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        shots: int = 0,
        sampling_method: str = "multinomial",
        no_bunching: bool = True,
        # New parameter for constrained photon placement
        index_photons: list[tuple[int, int]] | None = None,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype or torch.float32
        self.input_size = input_size
        self.no_bunching = no_bunching
        self.index_photons = index_photons

        builder_trainable: list[str] = []
        builder_input: list[str] = []

        self.angle_encoding_specs: dict[str, dict[str, Any]] = {}

        # convert CircuitBuilder to pcvl.Circuit if needed, otherwise use Circuit
        if isinstance(circuit, CircuitBuilder):
            builder_trainable = circuit.trainable_parameter_prefixes
            builder_input = circuit.input_parameter_prefixes
            self.angle_encoding_specs = circuit.angle_encoding_specs
            circuit = circuit.to_pcvl_circuit(pcvl)

        # Fix trainable and input parameters from builder or circuit, can also be empty lists
        if trainable_parameters is None:
            trainable_parameters = list(builder_trainable)
        else:
            trainable_parameters = list(trainable_parameters)

        if input_parameters is None:
            input_parameters = list(builder_input)
        else:
            input_parameters = list(input_parameters)

        # Determine construction mode
        # TODO: can be deprectated once Builder is fully supported
        if ansatz is not None:
            self._init_from_ansatz(ansatz, output_size, output_mapping_strategy)

        elif circuit is not None:
            self.circuit = circuit
            self._init_from_custom_circuit(
                circuit,
                input_state,
                n_photons,
                trainable_parameters,
                input_parameters,
                output_size,
                output_mapping_strategy,
            )
        else:
            raise ValueError("Either 'ansatz' or 'circuit' must be provided")

        # Setup sampling
        self.autodiff_process = AutoDiffProcess(sampling_method)
        self.shots = shots
        self.sampling_method = sampling_method

    def _init_from_ansatz(
        self,
        ansatz: Ansatz,
        output_size: int | None,
        output_mapping_strategy: OutputMappingStrategy,
    ):
        """Initialize from ansatz (auto-generated mode)."""
        self.ansatz = ansatz
        self.auto_generation_mode = True

        # For ansatz mode, we need to create a new computation process with correct device
        if self.index_photons is not None:
            # Create a new computation process with index_photons support or correct device
            self.computation_process = ComputationProcessFactory.create(
                circuit=ansatz.circuit,
                input_state=ansatz.input_state,
                trainable_parameters=ansatz.trainable_parameters,
                input_parameters=ansatz.input_parameters,
                reservoir_mode=ansatz.experiment.reservoir_mode,
                device=self.device,
                dtype=self.dtype,
                no_bunching=self.no_bunching,
                index_photons=self.index_photons,
            )
        else:
            # Use the ansatz's computation process as before
            # Set ansatz device to be the same as the QuantumLayer
            if self.device is not None:
                ansatz.device = self.device
            # Build computation process from ansatz on the correct device
            self.computation_process = ansatz._build_computation_process()

        self.feature_encoder = ansatz.feature_encoder

        # Use the ansatz's output mapping strategy - it should take precedence!
        actual_strategy = ansatz.output_mapping_strategy
        actual_output_size = output_size or ansatz.output_size

        # Setup bandwidth tuning if enabled
        if ansatz.experiment.use_bandwidth_tuning:
            self.bandwidth_coeffs = nn.ParameterDict()
            for d in range(self.input_size):
                init = torch.linspace(
                    0.0,
                    2.0,
                    steps=ansatz.experiment.n_modes,
                    dtype=self.dtype,
                    device=self.device,
                )
                self.bandwidth_coeffs[f"dim_{d}"] = nn.Parameter(
                    init.clone(), requires_grad=True
                )
        else:
            self.bandwidth_coeffs = None

        # Setup parameters
        self._setup_parameters_from_ansatz(ansatz)

        # Setup output mapping using ansatz configuration
        self._setup_output_mapping(ansatz, actual_output_size, actual_strategy)

    def _init_from_custom_circuit(
        self,
        circuit: pcvl.Circuit,
        input_state: list[int] | None,
        n_photons: int | None,
        trainable_parameters: list[str],
        input_parameters: list[str],
        output_size: int | None,
        output_mapping_strategy: OutputMappingStrategy,
    ):
        """Initialize from custom circuit (backward compatible mode)."""
        self.auto_generation_mode = False
        self.bandwidth_coeffs = None
        # Handle state - with index_photons consideration
        if input_state is not None:
            self.input_state = input_state
            # Validate input_state against index_photons constraints if provided
            if self.index_photons is not None:
                self._validate_input_state_with_index_photons(input_state)
        elif n_photons is not None:
            if self.index_photons is not None:
                # Create input state respecting index_photons constraints
                self.input_state = self._create_input_state_from_index_photons(
                    n_photons, circuit.m
                )
            else:
                # Default behavior: place photons in first n_photons modes
                self.input_state = [1] * n_photons + [0] * (circuit.m - n_photons)
        else:
            raise ValueError("Either input_state or n_photons must be provided")

        # Create computation process with index_photons support

        self.computation_process = ComputationProcessFactory.create(
            circuit=circuit,
            input_state=self.input_state,
            trainable_parameters=trainable_parameters,
            input_parameters=input_parameters,
            n_photons=n_photons,
            device=self.device,
            dtype=self.dtype,
            no_bunching=self.no_bunching,
            index_photons=self.index_photons,
        )

        # Setup parameters
        self._setup_parameters_from_custom(trainable_parameters)

        # Setup output mapping
        self._setup_output_mapping_from_custom(output_size, output_mapping_strategy)

    def _validate_input_state_with_index_photons(self, input_state: list[int]):
        """Validate that input_state respects index_photons constraints."""
        if self.index_photons is None:
            return  # No constraints to validate

        photon_idx = 0
        for mode_idx, photon_count in enumerate(input_state):
            for _ in range(photon_count):
                if photon_idx >= len(self.index_photons):
                    raise ValueError(
                        f"Input state has more photons than index_photons constraints. "
                        f"Found {sum(input_state)} photons but only {len(self.index_photons)} constraints."
                    )

                min_mode, max_mode = self.index_photons[photon_idx]
                if not (min_mode <= mode_idx <= max_mode):
                    raise ValueError(
                        f"Photon {photon_idx} is in mode {mode_idx} but index_photons constrains it to "
                        f"modes [{min_mode}, {max_mode}]"
                    )
                photon_idx += 1

    def _create_input_state_from_index_photons(
        self, n_photons: int, n_modes: int
    ) -> list[int]:
        """Create input state respecting index_photons constraints."""
        if self.index_photons is None or len(self.index_photons) != n_photons:
            raise ValueError(
                f"index_photons must specify constraints for exactly {n_photons} photons. "
                f"Got {len(self.index_photons) if self.index_photons else 0} constraints."
            )

        input_state = [0] * n_modes

        for photon_idx, (min_mode, max_mode) in enumerate(self.index_photons):
            # Validate constraint bounds
            if not (0 <= min_mode <= max_mode < n_modes):
                raise ValueError(
                    f"Invalid index_photons constraint for photon {photon_idx}: "
                    f"[{min_mode}, {max_mode}] must be within [0, {n_modes - 1}]"
                )

            # Place photon in the minimum allowed mode (simplest strategy)
            # Users can override by providing explicit input_state
            input_state[min_mode] += 1

        return input_state

    def _setup_parameters_from_ansatz(self, ansatz: Ansatz):
        """Setup parameters from ansatz configuration."""
        spec_mappings = self.computation_process.converter.spec_mappings
        self.thetas = []
        self.theta_names = []

        # Setup trainable parameters - FIXED: Only add if not in reservoir mode
        if not ansatz.experiment.reservoir_mode:
            for tp in ansatz.trainable_parameters:
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

        # Setup reservoir parameters if needed
        if ansatz.experiment.reservoir_mode and "phi_" in spec_mappings:
            phi_list = spec_mappings["phi_"]
            if phi_list:
                phi_values = []
                for _param_name in phi_list:
                    # For reservoir mode, just use random values
                    phi_values.append(2 * math.pi * np.random.rand())

                phi_tensor = torch.tensor(
                    phi_values, dtype=self.dtype, device=self.device
                )
                self.register_buffer("phi_static", phi_tensor)

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

    def _setup_output_mapping(
        self,
        ansatz: Ansatz,
        output_size: int | None,
        output_mapping_strategy: OutputMappingStrategy,
    ):
        """Setup output mapping for ansatz-based construction."""
        # Get distribution size
        dummy_params = self._create_dummy_parameters()
        distribution = self.computation_process.compute(dummy_params)
        dist_size = distribution.shape[-1]

        # Determine output size
        if output_size is None:
            if output_mapping_strategy == OutputMappingStrategy.NONE:
                self.output_size = dist_size
            else:
                raise ValueError(
                    "output_size must be specified for non-NONE strategies"
                )
        else:
            self.output_size = output_size

        # Validate NONE strategy
        if (
            output_mapping_strategy == OutputMappingStrategy.NONE
            and self.output_size != dist_size
        ):
            raise ValueError(
                f"Distribution size ({dist_size}) must equal output size ({self.output_size}) "
                f"when using 'none' strategy"
            )

        # Create output mapping
        self.output_mapping = OutputMapper.create_mapping(
            output_mapping_strategy, dist_size, self.output_size
        )

        # Ensure output mapping has correct dtype and device
        if hasattr(self.output_mapping, "weight"):
            self.output_mapping = self.output_mapping.to(
                dtype=self.dtype, device=self.device
            )

    def _setup_output_mapping_from_custom(
        self, output_size: int | None, output_mapping_strategy: OutputMappingStrategy
    ):
        """Setup output mapping for custom circuit construction."""
        # Get distribution size
        dummy_params = self._create_dummy_parameters()
        distribution = self.computation_process.compute(dummy_params)
        dist_size = distribution.shape[-1]

        # Determine output size
        if output_size is None:
            if output_mapping_strategy == OutputMappingStrategy.NONE:
                self.output_size = dist_size
            else:
                raise ValueError(
                    "output_size must be specified for non-NONE strategies"
                )
        else:
            self.output_size = output_size

        # Create output mapping
        self.output_mapping = OutputMapper.create_mapping(
            output_mapping_strategy, dist_size, self.output_size
        )

        # Ensure output mapping has correct dtype and device
        if hasattr(self.output_mapping, "weight"):
            self.output_mapping = self.output_mapping.to(
                dtype=self.dtype, device=self.device
            )

    def _create_dummy_parameters(self) -> list[torch.Tensor]:
        """Create dummy parameters for initialization."""
        params = list(self.thetas)

        # Add dummy input parameters - FIXED: Use correct parameter count
        if self.auto_generation_mode:
            dummy_input = torch.zeros(
                self.ansatz.total_shifters, dtype=self.dtype, device=self.device
            )
            params.append(dummy_input)  # type: ignore[arg-type]
        else:
            # For custom circuits, create dummy based on input parameter count
            spec_mappings = self.computation_process.converter.spec_mappings
            input_params = self.computation_process.input_parameters
            for ip in input_params:
                if ip in spec_mappings:
                    param_count = len(spec_mappings[ip])
                    dummy_input = torch.zeros(
                        param_count, dtype=self.dtype, device=self.device
                    )
                    params.append(dummy_input)  # type: ignore[arg-type]

        # Add static phi parameters if in reservoir mode
        if hasattr(self, "phi_static"):
            params.append(self.phi_static)  # type: ignore[arg-type]

        return params  # type: ignore[return-value]

    def _prepare_input_encoding(
        self, x: torch.Tensor, prefix: str | None = None
    ) -> torch.Tensor:
        """Prepare input encoding based on mode."""
        if self.auto_generation_mode:
            # Use FeatureEncoder for auto-generated circuits
            x_norm = torch.clamp(x, 0, 1)  # Ensure inputs are in valid range

            return self.feature_encoder.encode(
                x_norm,
                self.ansatz.experiment.circuit_type,
                self.ansatz.experiment.n_modes,
                self.bandwidth_coeffs,  # type: ignore[arg-type]
            )

        spec = None
        if prefix is not None:
            spec = self.angle_encoding_specs.get(prefix)
        elif len(self.angle_encoding_specs) == 1:
            spec = next(iter(self.angle_encoding_specs.values()))

        if spec:
            return self._apply_angle_encoding(x, spec)

        # For custom circuits without explicit encoding metadata, apply π scaling
        return x * torch.pi

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
            encoded = x_batch * torch.pi
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

        if squeeze:
            return encoded.squeeze(0)
        return encoded

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

        if self.auto_generation_mode and len(input_parameters) == 1:
            prefix = prefixes[0] if prefixes else None
            encoded = self._prepare_input_encoding(input_parameters[0], prefix)
            params.append(encoded)
        else:
            # Custom mode or multiple parameters
            for idx, x in enumerate(input_parameters):
                prefix = None
                if prefixes:
                    prefix = prefixes[idx] if idx < len(prefixes) else prefixes[-1]
                encoded = self._prepare_input_encoding(x, prefix)
                params.append(encoded)

        # Add static phi parameters if in reservoir mode
        if hasattr(self, "phi_static"):
            if input_parameters and input_parameters[0].dim() > 1:
                batch_size = input_parameters[0].shape[0]
                params.append(self.phi_static.expand(batch_size, -1))  # type: ignore[operator]
            else:
                params.append(self.phi_static)  # type: ignore[arg-type]

        return params

    def forward(
        self,
        *input_parameters: torch.Tensor,
        apply_sampling: bool | None = None,
        shots: int | None = None,
        return_amplitudes: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass through the quantum layer."""
        # Prepare parameters
        params = self.prepare_parameters(list(input_parameters))
        # Get quantum output
        if type(self.computation_process.input_state) is torch.Tensor:
            amplitudes = self.computation_process.compute_superposition_state(params)
        else:
            amplitudes = self.computation_process.compute(params)

        # Handle sampling
        needs_gradient = (
            self.training
            and torch.is_grad_enabled()
            and any(p.requires_grad for p in self.parameters())
        )
        # TODO/CAUTION: if needs_gradient is True and shots>0, the code raises a warning and casts apply_sampling = False and shots = 0
        apply_sampling, shots = self.autodiff_process.autodiff_backend(
            needs_gradient, apply_sampling or False, shots or self.shots
        )
        distribution = amplitudes.real**2 + amplitudes.imag**2
        if self.no_bunching:
            sum_probs = distribution.sum(dim=1, keepdim=True)

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
        if apply_sampling and shots > 0:
            distribution = self.autodiff_process.sampling_noise.pcvl_sampler(
                distribution, shots
            )
        if return_amplitudes:
            return self.output_mapping(distribution), amplitudes
        # Apply output mapping

        return self.output_mapping(distribution)

    def set_sampling_config(self, shots: int | None = None, method: str | None = None):
        """Update sampling configuration."""
        if shots is not None:
            if not isinstance(shots, int) or shots < 0:
                raise ValueError(f"shots must be a non-negative integer, got {shots}")
            self.shots = shots
        if method is not None:
            valid_methods = ["multinomial", "binomial", "gaussian"]
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid sampling method: {method}. Valid options are: {valid_methods}"
                )
            self.sampling_method = method

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
            if hasattr(self.output_mapping, "weight"):
                self.output_mapping = self.output_mapping.to(
                    dtype=self.dtype, device=self.device
                )
        return self

    def get_index_photons_info(self) -> dict:
        """
        Get information about index_photons constraints.

        Returns:
            dict: Information about photon placement constraints
        """
        if self.index_photons is None:
            return {
                "constrained": False,
                "message": "No photon placement constraints (photons can be placed in any mode)",
            }

        info: dict[str, Any] = {
            "constrained": True,
            "n_photons": len(self.index_photons),
            "constraints": [],
        }

        for i, (min_mode, max_mode) in enumerate(self.index_photons):
            info["constraints"].append({
                "photon_index": i,
                "allowed_modes": f"[{min_mode}, {max_mode}]",
                "n_allowed_modes": max_mode - min_mode + 1,
            })

        return info

    @classmethod
    def simple(
        cls,
        input_size: int,
        n_params: int = 100,
        shots: int = 0,
        reservoir_mode: bool = False,
        output_size: int | None = None,
        output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.NONE,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        no_bunching: bool = True,
    ):
        """Create a ready-to-train layer with a 10-mode, 5-photon architecture.

        The circuit is assembled via :class:`CircuitBuilder` with the following layout:

        1. A fully trainable generic interferometer acting on all modes;
        2. A full input encoding layer spanning all encoded features;
        3. A non-trainable entangling layer that redistributes encoded information;
        4. Optional trainable rotation layers to reach the requested ``n_params`` budget;
        5. A final entangling layer prior to measurement.

        Args:
            input_size: Size of the classical input vector.
            n_params: Number of trainable parameters to allocate across rotation layers.
            shots: Number of sampling shots for stochastic evaluation.
            reservoir_mode: Reserved for API compatibility (unused in builder mode).
            output_size: Optional classical output width.
            output_mapping_strategy: Strategy used to post-process the quantum distribution.
            device: Optional target device for tensors.
            dtype: Optional tensor dtype.
            no_bunching: Whether to restrict to states without photon bunching.

        Returns:
            QuantumLayer configured with the described architecture.
        """
        _ = reservoir_mode  # Reserved for API compatibility; builder path ignores this flag.

        n_modes = 10
        n_photons = 5

        builder = CircuitBuilder(n_modes=n_modes)

        # Trainable generic interferometer before encoding
        builder.add_generic_interferometer(trainable=True, name="gi_simple")
        generic_component = builder.circuit.components[-1]
        generic_params = 0
        if (
            isinstance(generic_component, GenericInterferometer)
            and generic_component.trainable
            and generic_component.span >= 2
        ):
            mzi_count = generic_component.span * (generic_component.span - 1) // 2
            generic_params = 2 * mzi_count

        requested_params = max(int(n_params), 0)
        if generic_params > requested_params:
            warnings.warn(
                "Generic interferometer introduces "
                f"{generic_params} trainable parameters, exceeding the requested "
                f"budget of {requested_params}. The simple layer will expose "
                f"{generic_params} trainable parameters.",
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

        builder.add_entangling_layer(depth=1)

        # Allocate additional trainable rotations only if the budget exceeds the interferometer
        remaining = max(requested_params - generic_params, 0)
        layer_idx = 0
        added_rotation_params = 0

        while remaining > 0:
            prefix = f"theta_layer{layer_idx}"
            if remaining >= n_modes:
                builder.add_rotation_layer(trainable=True, name=prefix)
                remaining -= n_modes
                added_rotation_params += n_modes
            else:
                modes = list(range(remaining))
                builder.add_rotation_layer(modes=modes, trainable=True, name=prefix)
                added_rotation_params += remaining
                remaining = 0
            layer_idx += 1

        # Post-rotation entanglement
        builder.add_entangling_layer(depth=1)

        total_trainable = generic_params + added_rotation_params
        expected_trainable = max(requested_params, generic_params)
        if total_trainable != expected_trainable:
            raise ValueError(
                "Constructed circuit exposes "
                f"{total_trainable} trainable parameters but {expected_trainable} were expected."
            )

        return cls(
            input_size=input_size,
            output_size=output_size,
            circuit=builder,
            n_photons=n_photons,
            output_mapping_strategy=output_mapping_strategy,
            shots=shots,
            no_bunching=no_bunching,
            device=device,
            dtype=dtype,
        )

    def __str__(self) -> str:
        """String representation of the quantum layer."""
        base_str = ""
        if self.auto_generation_mode:
            base_str = (
                f"QuantumLayer(ansatz={self.ansatz.experiment.circuit_type.value}, "
                f"modes={self.ansatz.experiment.n_modes}, "
                f"input_size={self.input_size}, output_size={self.output_size}"
            )
        else:
            base_str = (
                f"QuantumLayer(custom_circuit, input_size={self.input_size}, "
                f"output_size={self.output_size}"
            )

        # Add index_photons info if present
        if self.index_photons is not None:
            base_str += f", index_photons={self.index_photons}"

        return base_str + ")"
