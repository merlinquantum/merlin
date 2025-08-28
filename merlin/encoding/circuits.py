"""Pre-built encoding circuit templates using CircuitBuilder.

These templates show how to combine encoding strategies with the 
platform-agnostic CircuitBuilder.
"""

from typing import Optional, List, Dict, Any
import torch
from ..builder import CircuitBuilder
from ..core.circuit import Circuit
from .strategies import (
    EncodingStrategy,
    FourierEncoding,
    DataReuploadingEncoding,
    AmplitudeEncoding
)


class EncodingCircuitBuilder:
    """Builder for creating encoding circuits with various strategies."""

    @staticmethod
    def build_fourier_circuit(n_modes: int,
                             n_features: int,
                             n_layers: int = 2,
                             entangling_pattern: str = 'nearest_neighbor',
                             n_photons: Optional[int] = None) -> Circuit:
        """
        Build a Fourier encoding circuit.

        Args:
            n_modes: Number of modes/qubits
            n_features: Number of input features
            n_layers: Number of encoding layers
            entangling_pattern: Pattern for entangling gates
            n_photons: Number of photons (for photonic circuits)

        Returns:
            Circuit with Fourier encoding
        """
        builder = CircuitBuilder(n_modes=n_modes, n_photons=n_photons)

        # Create encoding strategy
        encoding = FourierEncoding(
            n_features=n_features,
            n_modes=n_modes,
            n_layers=n_layers
        )

        # Build encoding circuit
        encoding.build_encoding_circuit(builder)

        # Add final processing layer
        builder.add_rotation_layer(axis='z')
        builder.add_entangling_layer(pattern=entangling_pattern)
        builder.add_rotation_layer(axis='y')

        return builder.build()

    @staticmethod
    def build_iqp_circuit(n_modes: int,
                         n_features: int,
                         n_photons: Optional[int] = None) -> Circuit:
        """
        Build an IQP (Instantaneous Quantum Polynomial) encoding circuit.

        This is particularly suited for photonic platforms.

        Args:
            n_modes: Number of modes
            n_features: Number of features
            n_photons: Number of photons

        Returns:
            IQP-style circuit
        """
        builder = CircuitBuilder(n_modes=n_modes, n_photons=n_photons)

        # First layer: encode features as phases
        for i in range(n_features):
            mode = i % n_modes
            builder.circuit.rotation(mode, angle=f'iqp_feat_{i}', axis='z')

        # Entangling layer
        builder.add_entangling_layer(pattern='all_to_all', depth=1)

        # Second encoding layer with cross-terms
        pair_idx = 0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                mode = pair_idx % n_modes
                builder.circuit.rotation(
                    mode,
                    angle=f'iqp_cross_{i}_{j}',
                    axis='z'
                )
                pair_idx += 1

        # Final entangling
        builder.add_entangling_layer(pattern='all_to_all', depth=1)

        return builder.build()

    @staticmethod
    def build_qcnn_encoding(n_modes: int,
                           n_features: int,
                           n_conv_layers: int = 2,
                           n_photons: Optional[int] = None) -> Circuit:
        """
        Build a Quantum Convolutional Neural Network encoding.

        Args:
            n_modes: Number of modes
            n_features: Number of features
            n_conv_layers: Number of convolutional layers
            n_photons: Number of photons

        Returns:
            QCNN-style encoding circuit
        """
        builder = CircuitBuilder(n_modes=n_modes, n_photons=n_photons)

        # Initial feature encoding
        for i in range(n_features):
            mode = i % n_modes
            builder.circuit.rotation(mode, f'qcnn_input_{i}', 'z')

        # Convolutional layers
        for layer in range(n_conv_layers):
            # Convolution: nearest-neighbor interactions only
            for i in range(0, n_modes - 1, 2):
                if i + 1 < n_modes:
                    # Use adjacent modes only
                    builder.circuit.beam_splitter((i, i + 1), f'qcnn_conv{layer}_bs{i}')

            # Non-linearity (rotation on each mode)
            for i in range(n_modes):
                builder.circuit.rotation(i, f'qcnn_nl{layer}_rot{i}', 'y')

            # Second set of convolutions (offset by 1)
            for i in range(1, n_modes - 1, 2):
                if i + 1 < n_modes:
                    builder.circuit.beam_splitter((i, i + 1), f'qcnn_conv{layer}_bs_off{i}')

        return builder.build()
    @staticmethod
    def build_variational_circuit(n_modes: int,
                                 n_features: int,
                                 ansatz_layers: int = 3,
                                 encoding_type: str = 'fourier',
                                 n_photons: Optional[int] = None) -> Circuit:
        """
        Build a variational quantum circuit with encoding.

        Args:
            n_modes: Number of modes
            n_features: Number of features
            ansatz_layers: Number of variational layers
            encoding_type: Type of encoding ('fourier', 'amplitude', 'reupload')
            n_photons: Number of photons

        Returns:
            Variational circuit with encoding
        """
        builder = CircuitBuilder(n_modes=n_modes, n_photons=n_photons)

        # Choose encoding strategy
        if encoding_type == 'fourier':
            encoding = FourierEncoding(n_features, n_modes, n_layers=1)
        elif encoding_type == 'amplitude':
            encoding = AmplitudeEncoding(n_features, n_modes)
        elif encoding_type == 'reupload':
            encoding = DataReuploadingEncoding(
                n_features, n_modes, n_uploads=ansatz_layers
            )
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

        # Build encoding part
        encoding.build_encoding_circuit(builder)

        # Add variational ansatz layers
        for layer in range(ansatz_layers):
            # Parameterized rotations
            for mode in range(n_modes):
                builder.circuit.rotation(
                    mode,
                    angle=f'var_l{layer}_m{mode}_z',
                    axis='z'
                )
                builder.circuit.rotation(
                    mode,
                    angle=f'var_l{layer}_m{mode}_y',
                    axis='y'
                )

            # Entangling layer
            builder.add_entangling_layer(pattern='nearest_neighbor')

        # Final layer
        builder.add_rotation_layer(axis='z')

        return builder.build()


class QuantumFeatureMap:
    """
    Quantum feature map for kernel-based methods.

    Creates encoding circuits suitable for quantum kernel estimation.
    """

    def __init__(self,
                 n_modes: int,
                 n_features: int,
                 feature_map_type: str = 'pauli'):
        """
        Initialize quantum feature map.

        Args:
            n_modes: Number of modes
            n_features: Number of features
            feature_map_type: Type of feature map
        """
        self.n_modes = n_modes
        self.n_features = n_features
        self.feature_map_type = feature_map_type

    def build_feature_map(self, n_photons: Optional[int] = None) -> Circuit:
        """Build the feature map circuit."""
        builder = CircuitBuilder(n_modes=self.n_modes, n_photons=n_photons)

        if self.feature_map_type == 'pauli':
            # Pauli feature map
            for rep in range(2):  # Typically repeated twice
                # Encode features
                for i in range(self.n_features):
                    mode = i % self.n_modes
                    builder.circuit.rotation(
                        mode,
                        angle=f'pauli_feat_{i}_rep{rep}',
                        axis='z'
                    )

                # Entangle
                builder.add_entangling_layer(pattern='ring')

        elif self.feature_map_type == 'iqp':
            # IQP feature map
            return EncodingCircuitBuilder.build_iqp_circuit(
                self.n_modes, self.n_features, n_photons
            )

        elif self.feature_map_type == 'hardware_efficient':
            # Hardware-efficient feature map
            builder.add_hardware_efficient_ansatz(n_layers=2)

        return builder.build()
