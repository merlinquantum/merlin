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
Ansatz configuration and factory for quantum layers.
"""

import torch

from ..core.generators import CircuitGenerator, StateGenerator
from ..core.photonicbackend import PhotonicBackend
from ..core.process import ComputationProcessFactory
from ..sampling.strategies import OutputMappingStrategy
from ..torch_utils.torch_codes import FeatureEncoder


class Ansatz:
    """Complete configuration for a quantum neural network layer."""

    def __init__(
        self,
        PhotonicBackend: PhotonicBackend,
        input_size: int,
        output_size: int | None = None,
        output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.LINEAR,
        dtype: torch.dtype | None = None,
    ):
        r"""Initialize the Ansatz with the given configuration.

        Args:
            PhotonicBackend (PhotonicBackend): The backend configuration to use.
            input_size (int): Size of the input feature vector.
            output_size (int | None): Size of the output vector. If None, it is defined by the backend.
            output_mapping_strategy (OutputMappingStrategy): Strategy for mapping outputs.
            dtype (torch.dtype | None): Data type for computations.
        """
        self.experiment = PhotonicBackend
        self.input_size = input_size
        self.output_size = output_size
        self.output_mapping_strategy = output_mapping_strategy
        self.dtype = dtype or torch.float32
        self.device: torch.device | None = None

        # Create feature encoder
        self.feature_encoder = FeatureEncoder(input_size)

        # Generate circuit and state - PASS RESERVOIR MODE TO CIRCUIT GENERATOR
        self.circuit, self.total_shifters = CircuitGenerator.generate_circuit(
            PhotonicBackend.circuit_type,
            PhotonicBackend.n_modes,
            input_size,
            reservoir_mode=PhotonicBackend.reservoir_mode,
        )

        self.input_state = StateGenerator.generate_state(
            PhotonicBackend.n_modes,
            PhotonicBackend.n_photons,
            PhotonicBackend.state_pattern,
        )

        # Setup parameter patterns
        self.input_parameters = ["pl"]
        self.trainable_parameters = [] if PhotonicBackend.reservoir_mode else ["phi_"]
        # self.trainable_parameters= ["phi"]

        # Get circuit parameters once
        circuit_params = self.circuit.get_parameters()

        # In reservoir mode, the circuit has no trainable parameters
        # because interferometers use fixed random values
        if PhotonicBackend.reservoir_mode:
            self.trainable_parameters = []
        else:
            # Only add phi_ if the circuit actually has phi_ parameters
            has_phi_params = any(p.name.startswith("phi_") for p in circuit_params)
            self.trainable_parameters = ["phi_"] if has_phi_params else []
        self.reservoir_mode = PhotonicBackend.reservoir_mode
        # Create computation process with proper dtype

    def _build_computation_process(self):
        self.computation_process = ComputationProcessFactory.create(
            circuit=self.circuit,
            input_state=self.input_state,
            trainable_parameters=self.trainable_parameters,
            input_parameters=self.input_parameters,
            reservoir_mode=self.reservoir_mode,
            dtype=self.dtype,
            device=self.device,
        )
        return self.computation_process


class AnsatzFactory:
    """Factory for creating quantum layer ansatzes (complete configurations)."""

    @staticmethod
    def create(
        PhotonicBackend: PhotonicBackend,
        input_size: int,
        output_size: int | None = None,
        output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.LINEAR,
        dtype: torch.dtype | None = None,
    ) -> Ansatz:
        r"""Create a complete ansatz configuration.

        Args:
            PhotonicBackend (PhotonicBackend): The backend configuration to use.
            input_size (int): Size of the input feature vector.
            output_size (int | None): Size of the output vector. If None, it is defined by the backend.
            output_mapping_strategy (OutputMappingStrategy): Strategy for mapping outputs.
            dtype (torch.dtype | None): Data type for computations.

        Returns:
            Ansatz: A complete ansatz configuration for the quantum layer.
        """
        return Ansatz(
            PhotonicBackend=PhotonicBackend,
            input_size=input_size,
            output_size=output_size,
            output_mapping_strategy=output_mapping_strategy,
            dtype=dtype,
        )
