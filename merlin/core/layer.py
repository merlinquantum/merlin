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

import math
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import perceval as pcvl

from .ansatz import Ansatz, AnsatzFactory
from ..sampling.strategies import OutputMappingStrategy
from ..sampling.mappers import OutputMapper
from ..sampling.autodiff import AutoDiffProcess
from ..core.process import ComputationProcessFactory

from ..core.photonicbackend import PhotonicBackend as Experiment
from ..core.generators import CircuitType, StatePattern

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

    def __init__(self, input_size: int, output_size: Optional[int] = None,
                 # Ansatz-based construction
                 ansatz: Optional[Ansatz] = None,
                 # Custom circuit construction (backward compatible)
                 circuit: Optional[pcvl.Circuit] = None,
                 input_state: Optional[List[int]] = None,
                 n_photons: Optional[int] = None,
                 trainable_parameters: List[str] = [],
                 input_parameters: List[str] = [],
                 # Common parameters
                 output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.LINEAR,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 shots: int = 0,
                 sampling_method: str = 'multinomial',
                 no_bunching: bool = True,
                 # New parameter for constrained photon placement
                 index_photons: Optional[List[Tuple[int, int]]] = None):

        super().__init__()

        self.device = device
        self.dtype = dtype or torch.float32
        self.input_size = input_size
        self.no_bunching = no_bunching
        self.index_photons = index_photons

        # Determine construction mode
        if ansatz is not None:
            self._init_from_ansatz(ansatz, output_size, output_mapping_strategy)
        elif circuit is not None:
            self._init_from_custom_circuit(
                circuit, input_state, n_photons, trainable_parameters,
                input_parameters, output_size, output_mapping_strategy
            )
        else:
            raise ValueError("Either 'ansatz' or 'circuit' must be provided")

        # Setup sampling
        self.autodiff_process = AutoDiffProcess()
        self.shots = shots
        self.sampling_method = sampling_method

    def _init_from_ansatz(self, ansatz: Ansatz, output_size: Optional[int],
                          output_mapping_strategy: OutputMappingStrategy):
        """Initialize from ansatz (auto-generated mode)."""
        self.ansatz = ansatz
        self.auto_generation_mode = True

        # For ansatz mode, we need to create a new computation process with index_photons
        if self.index_photons is not None:
            # Create a new computation process with index_photons support
            self.computation_process = ComputationProcessFactory.create(
                circuit=ansatz.circuit,
                input_state=ansatz.input_state,
                trainable_parameters=ansatz.trainable_parameters,
                input_parameters=ansatz.input_parameters,
                reservoir_mode=ansatz.experiment.reservoir_mode,
                device=self.device,
                dtype=self.dtype,
                no_bunching=self.no_bunching,
                index_photons=self.index_photons
            )
        else:
            # Use the ansatz's computation process as before
            self.computation_process = ansatz.computation_process

        self.feature_encoder = ansatz.feature_encoder

        # Use the ansatz's output mapping strategy - it should take precedence!
        actual_strategy = ansatz.output_mapping_strategy
        actual_output_size = output_size or ansatz.output_size

        # Setup bandwidth tuning if enabled
        if ansatz.experiment.use_bandwidth_tuning:
            self.bandwidth_coeffs = nn.ParameterDict()
            for d in range(self.input_size):
                init = torch.linspace(0.0, 2.0, steps=ansatz.experiment.n_modes,
                                      dtype=self.dtype, device=self.device)
                self.bandwidth_coeffs[f"dim_{d}"] = nn.Parameter(init.clone(), requires_grad=True)
        else:
            self.bandwidth_coeffs = None

        # Setup parameters
        self._setup_parameters_from_ansatz(ansatz)

        # Setup output mapping using ansatz configuration
        self._setup_output_mapping(ansatz, actual_output_size, actual_strategy)

    def _init_from_custom_circuit(self, circuit: pcvl.Circuit, input_state: Optional[List[int]],
                                  n_photons: Optional[int], trainable_parameters: List[str],
                                  input_parameters: List[str], output_size: Optional[int],
                                  output_mapping_strategy: OutputMappingStrategy):
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
                self.input_state = self._create_input_state_from_index_photons(n_photons, circuit.m)
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
            device=self.device,
            dtype=self.dtype,
            no_bunching=self.no_bunching,
            index_photons=self.index_photons
        )

        # Setup parameters
        self._setup_parameters_from_custom(trainable_parameters)

        # Setup output mapping
        self._setup_output_mapping_from_custom(output_size, output_mapping_strategy)

    def _validate_input_state_with_index_photons(self, input_state: List[int]):
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

    def _create_input_state_from_index_photons(self, n_photons: int, n_modes: int) -> List[int]:
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
                        torch.randn((len(theta_list),), dtype=self.dtype, device=self.device) * torch.pi
                    )
                    self.register_parameter(tp, parameter)
                    self.thetas.append(parameter)

        # Setup reservoir parameters if needed
        if ansatz.experiment.reservoir_mode and "phi_" in spec_mappings:
            phi_list = spec_mappings["phi_"]
            if phi_list:
                phi_values = []
                for param_name in phi_list:
                    # For reservoir mode, just use random values
                    phi_values.append(2 * math.pi * np.random.rand())

                phi_tensor = torch.tensor(phi_values, dtype=self.dtype, device=self.device)
                self.register_buffer("phi_static", phi_tensor)

    def _setup_parameters_from_custom(self, trainable_parameters: List[str]):
        """Setup parameters from custom circuit configuration."""
        spec_mappings = self.computation_process.converter.spec_mappings
        self.thetas = []
        self.theta_names = []

        for tp in trainable_parameters:
            if tp in spec_mappings:
                theta_list = spec_mappings[tp]
                self.theta_names += theta_list
                parameter = nn.Parameter(
                    torch.randn((len(theta_list),), dtype=self.dtype, device=self.device) * torch.pi
                )
                self.register_parameter(tp, parameter)
                self.thetas.append(parameter)

    def _setup_output_mapping(self, ansatz: Ansatz, output_size: Optional[int],
                              output_mapping_strategy: OutputMappingStrategy):
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
                raise ValueError("output_size must be specified for non-NONE strategies")
        else:
            self.output_size = output_size

        # Validate NONE strategy
        if output_mapping_strategy == OutputMappingStrategy.NONE and self.output_size != dist_size:
            raise ValueError(
                f"Distribution size ({dist_size}) must equal output size ({self.output_size}) "
                f"when using 'none' strategy"
            )

        # Create output mapping
        self.output_mapping = OutputMapper.create_mapping(
            output_mapping_strategy, dist_size, self.output_size
        )

        # Ensure output mapping has correct dtype and device
        if hasattr(self.output_mapping, 'weight'):
            self.output_mapping = self.output_mapping.to(dtype=self.dtype, device=self.device)

    def _setup_output_mapping_from_custom(self, output_size: Optional[int],
                                          output_mapping_strategy: OutputMappingStrategy):
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
                raise ValueError("output_size must be specified for non-NONE strategies")
        else:
            self.output_size = output_size

        # Create output mapping
        self.output_mapping = OutputMapper.create_mapping(
            output_mapping_strategy, dist_size, self.output_size
        )

        # Ensure output mapping has correct dtype and device
        if hasattr(self.output_mapping, 'weight'):
            self.output_mapping = self.output_mapping.to(dtype=self.dtype, device=self.device)

    def _create_dummy_parameters(self) -> List[torch.Tensor]:
        """Create dummy parameters for initialization."""
        params = [theta for theta in self.thetas]

        # Add dummy input parameters - FIXED: Use correct parameter count
        if self.auto_generation_mode:
            dummy_input = torch.zeros(self.ansatz.total_shifters, dtype=self.dtype, device=self.device)
            params.append(dummy_input)
        else:
            # For custom circuits, create dummy based on input parameter count
            spec_mappings = self.computation_process.converter.spec_mappings
            input_params = self.computation_process.input_parameters
            for ip in input_params:
                if ip in spec_mappings:
                    param_count = len(spec_mappings[ip])
                    dummy_input = torch.zeros(param_count, dtype=self.dtype, device=self.device)
                    params.append(dummy_input)

        # Add static phi parameters if in reservoir mode
        if hasattr(self, "phi_static"):
            params.append(self.phi_static)

        return params

    def _prepare_input_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare input encoding based on mode."""
        if self.auto_generation_mode:
            # Use FeatureEncoder for auto-generated circuits
            x_norm = torch.clamp(x, 0, 1)  # Ensure inputs are in valid range

            return self.feature_encoder.encode(
                x_norm,
                self.ansatz.experiment.circuit_type,
                self.ansatz.experiment.n_modes,
                self.bandwidth_coeffs
            )
        else:
            # For custom circuits, apply 2π scaling directly
            return x * torch.pi

    def prepare_parameters(self, input_parameters: List[torch.Tensor]) -> List[torch.Tensor]:
        """Prepare parameter list for circuit evaluation."""
        # Handle batching
        if input_parameters and input_parameters[0].dim() > 1:
            batch_size = input_parameters[0].shape[0]
            params = [theta.expand(batch_size, -1) for theta in self.thetas]
        else:
            params = [theta for theta in self.thetas]

        # Apply input encoding
        if self.auto_generation_mode and len(input_parameters) == 1:
            encoded = self._prepare_input_encoding(input_parameters[0])
            params.append(encoded)
        else:
            # Custom mode or multiple parameters
            for x in input_parameters:
                encoded = self._prepare_input_encoding(x)
                params.append(encoded)

        # Add static phi parameters if in reservoir mode
        if hasattr(self, "phi_static"):
            if input_parameters and input_parameters[0].dim() > 1:
                batch_size = input_parameters[0].shape[0]
                params.append(self.phi_static.expand(batch_size, -1))
            else:
                params.append(self.phi_static)

        return params

    def forward(self, *input_parameters: torch.Tensor,
                apply_sampling: Optional[bool] = None,
                shots: Optional[int] = None) -> torch.Tensor:
        """Forward pass through the quantum layer."""
        # Prepare parameters
        params = self.prepare_parameters(list(input_parameters))

        # Get quantum output
        distribution = self.computation_process.compute(params)

        # Handle sampling
        needs_gradient = self.training and torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters())
        apply_sampling, shots = self.autodiff_process.autodiff_backend(
            needs_gradient, apply_sampling or False, shots or self.shots
        )

        if apply_sampling and shots > 0:
            distribution = self.autodiff_process.sampling_noise.pcvl_sampler(
                distribution, shots, self.sampling_method
            )

        # Apply output mapping
        return self.output_mapping(distribution)

    def set_sampling_config(self, shots: Optional[int] = None, method: Optional[str] = None):
        """Update sampling configuration."""
        if shots is not None:
            if not isinstance(shots, int) or shots < 0:
                raise ValueError(f"shots must be a non-negative integer, got {shots}")
            self.shots = shots
        if method is not None:
            valid_methods = ['multinomial', 'binomial', 'gaussian']
            if method not in valid_methods:
                raise ValueError(f"Invalid sampling method: {method}. Valid options are: {valid_methods}")
            self.sampling_method = method

    def to(self, *args, **kwargs):

        super().to(*args, **kwargs)
        # Manually move any additional tensors
        device = kwargs.get('device', None)
        if device is None and len(args) > 0:
            device = args[0]
        if device is not None:
            self.device = device
            self.computation_process.simulation_graph = self.computation_process.simulation_graph.to(self.dtype, device)
            self.computation_process.converter = self.computation_process.converter.to(self.dtype, device)
        return self

    def get_index_photons_info(self) -> dict:
        """
        Get information about index_photons constraints.

        Returns:
            dict: Information about photon placement constraints
        """
        if self.index_photons is None:
            return {
                'constrained': False,
                'message': 'No photon placement constraints (photons can be placed in any mode)'
            }

        info = {
            'constrained': True,
            'n_photons': len(self.index_photons),
            'constraints': []
        }

        for i, (min_mode, max_mode) in enumerate(self.index_photons):
            info['constraints'].append({
                'photon_index': i,
                'allowed_modes': f'[{min_mode}, {max_mode}]',
                'n_allowed_modes': max_mode - min_mode + 1
            })

        return info

    @classmethod
    def simple(cls, input_size: int, n_params: int = 100,
               shots: int = 0, reservoir_mode: bool = False,
               output_size: Optional[int] = None,
               output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.NONE,
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None,
               no_bunching: bool = True):
        """
        Simplified interface for creating a QuantumLayer.

        Uses SERIES circuit type with PERIODIC state pattern as defaults.
        Automatically calculates the number of modes based on n_params.

        Args:
            input_size (int): Size of the input vector
            n_params (int): Total number of parameters desired (default: 100)
                Formula: n_params = 2 * n_modes^2
            shots (int): Number of shots for sampling (default: 0)
            reservoir_mode (bool): Whether to use reservoir mode (default: False)
            output_size (int, optional): Output dimension. If None, uses distribution size
            output_mapping_strategy: How to map quantum output to classical output
            device: PyTorch device
            dtype: PyTorch dtype
            no_bunching: Whether to exclude states with multiple photons per mode

        Returns:
            QuantumLayer: Configured quantum layer instance
        """
        # Calculate minimum modes needed based on n_params
        # n_params = 2 * n_modes^2, so n_modes = sqrt(n_params / 2)
        min_modes_from_params = int(math.ceil(math.sqrt(n_params / 2)))

        # Ensure we have at least input_size + 1 modes
        n_modes = max(min_modes_from_params, input_size + 1)

        # Number of photons equals input_size
        n_photons = input_size

        # Create experiment configuration
        experiment = Experiment(
            circuit_type=CircuitType.SERIES,  # Default to SERIES
            n_modes=n_modes,
            n_photons=n_photons,
            reservoir_mode=reservoir_mode,
            use_bandwidth_tuning=False,  # Keep simple by default
            state_pattern=StatePattern.PERIODIC  # Default to PERIODIC
        )

        # Create ansatz using AnsatzFactory
        ansatz = AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=input_size,
            output_size=output_size,  # Can be None for automatic calculation
            output_mapping_strategy=output_mapping_strategy,
            dtype=dtype,
            device= device
        )

        # IMPORTANT: Override the ansatz's output_mapping_strategy to ensure our parameter is used
        # This is necessary because _init_from_ansatz uses ansatz.output_mapping_strategy
        ansatz.output_mapping_strategy = output_mapping_strategy

        # Create and return the QuantumLayer instance
        return cls(
            input_size=input_size,
            output_size=output_size,
            ansatz=ansatz,
            output_mapping_strategy=output_mapping_strategy,
            shots=shots,
            no_bunching=no_bunching,
            device= device,
            dtype= dtype
        )
    def __str__(self) -> str:
        """String representation of the quantum layer."""
        base_str = ""
        if self.auto_generation_mode:
            base_str = f"QuantumLayer(ansatz={self.ansatz.experiment.circuit_type.value}, " \
                       f"modes={self.ansatz.experiment.n_modes}, " \
                       f"input_size={self.input_size}, output_size={self.output_size}"
        else:
            base_str = f"QuantumLayer(custom_circuit, input_size={self.input_size}, " \
                       f"output_size={self.output_size}"

        # Add index_photons info if present
        if self.index_photons is not None:
            base_str += f", index_photons={self.index_photons}"

        return base_str + ")"
