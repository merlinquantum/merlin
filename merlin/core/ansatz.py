"""
Updated Ansatz class with proper no_bunching support.
Replace the existing Ansatz class with this version.
"""

from typing import Optional
import torch
from .photonicbackend import PhotonicBackend
from ..sampling.strategies import OutputMappingStrategy
from ..torch_utils.torch_codes import FeatureEncoder
from ..core.generators import CircuitGenerator
from ..core.generators import StateGenerator
from ..core.process import ComputationProcessFactory


class Ansatz:
    """Complete configuration for a quantum neural network layer."""

    def __init__(self, PhotonicBackend: PhotonicBackend, input_size: int,
                 output_size: Optional[int] = None,
                 output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.LINEAR,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 no_bunching: bool = True):  # ADD no_bunching parameter
        self.experiment = PhotonicBackend
        self.input_size = input_size
        self.output_size = output_size
        self.output_mapping_strategy = output_mapping_strategy
        self.device = device
        self.dtype = dtype or torch.float32
        self.no_bunching = no_bunching  # STORE no_bunching

        # Create feature encoder
        self.feature_encoder = FeatureEncoder(input_size)

        # Generate circuit and state - PASS RESERVOIR MODE TO CIRCUIT GENERATOR
        self.circuit, self.total_shifters = CircuitGenerator.generate_circuit(
            PhotonicBackend.circuit_type,
            PhotonicBackend.n_modes,
            input_size,
            reservoir_mode=PhotonicBackend.reservoir_mode
        )

        self.input_state = StateGenerator.generate_state(
            PhotonicBackend.n_modes,
            PhotonicBackend.n_photons,
            PhotonicBackend.state_pattern
        )

        # Setup parameter patterns
        self.input_parameters = ["pl"]

        # Get circuit parameters once
        circuit_params = self.circuit.get_parameters()

        # In reservoir mode, the circuit has no trainable parameters
        # because interferometers use fixed random values
        if PhotonicBackend.reservoir_mode:
            self.trainable_parameters = []
        else:
            # Only add phi_ if the circuit actually has phi_ parameters
            has_phi_params = any(p.name.startswith('phi_') for p in circuit_params)
            self.trainable_parameters = ["phi_"] if has_phi_params else []

        # Create computation process with proper dtype
        # Only pass parameter specs that actually exist in the circuit
        parameter_specs = self.trainable_parameters + self.input_parameters

        # Filter out specs that don't match any circuit parameters
        circuit_param_names = [p.name for p in circuit_params]
        valid_specs = []
        for spec in parameter_specs:
            if any(name.startswith(spec) for name in circuit_param_names):
                valid_specs.append(spec)

        self.computation_process = ComputationProcessFactory.create(
            circuit=self.circuit,
            input_state=self.input_state,
            trainable_parameters=[s for s in valid_specs if s != "pl"],
            input_parameters=[s for s in valid_specs if s == "pl"],
            reservoir_mode=PhotonicBackend.reservoir_mode,
            no_bunching=self.no_bunching,  # PASS no_bunching here
            dtype=self.dtype,
            device=self.device
        )


class AnsatzFactory:
    """Factory for creating quantum layer ansatzes (complete configurations)."""

    @staticmethod
    def create(PhotonicBackend: PhotonicBackend, input_size: int,
               output_size: Optional[int] = None,
               output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.LINEAR,
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None,
               no_bunching: bool = True) -> Ansatz:  # ADD no_bunching parameter
        """Create a complete ansatz configuration."""
        return Ansatz(
            PhotonicBackend=PhotonicBackend,
            input_size=input_size,
            output_size=output_size,
            output_mapping_strategy=output_mapping_strategy,
            device=device,
            dtype=dtype,
            no_bunching=no_bunching  # PASS it through
        )