"""Base classes for quantum data encoding strategies.

Provides flexible encoding of classical data into quantum circuits using
platform-agnostic components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union, Callable
import torch
import torch.nn as nn
import numpy as np
import math
from dataclasses import dataclass


class EncodingStrategy(ABC):
    """Abstract base class for data encoding strategies."""

    @abstractmethod
    def encode_parameters(self,
                         features: torch.Tensor,
                         bandwidth: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Encode classical features into quantum circuit parameters.

        Args:
            features: Input features (batch_size, n_features)
            bandwidth: Optional bandwidth/scaling parameters

        Returns:
            Dictionary mapping parameter names to values
        """
        pass

    @abstractmethod
    def build_encoding_circuit(self, builder: 'CircuitBuilder') -> 'CircuitBuilder':
        """
        Add encoding gates to circuit using builder.

        Args:
            builder: CircuitBuilder instance

        Returns:
            Modified builder
        """
        pass

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """Number of parameters this encoding uses."""
        pass

    @property
    @abstractmethod
    def encoding_type(self) -> str:
        """String identifier for encoding type."""
        pass


class FourierEncoding(EncodingStrategy):
    """
    Fourier series encoding of classical data.

    Encodes features as phases in rotation gates with Fourier frequencies.
    """

    def __init__(self,
                 n_features: int,
                 n_modes: int,
                 n_layers: int = 1,
                 frequencies: Optional[List[float]] = None,
                 trainable_bandwidth: bool = True):
        """
        Initialize Fourier encoding.

        Args:
            n_features: Number of input features
            n_modes: Number of quantum modes/qubits
            n_layers: Number of encoding layers
            frequencies: Custom Fourier frequencies (default: [1, 2, 3, ...])
            trainable_bandwidth: Whether bandwidth is trainable
        """
        self.n_features = n_features
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.trainable_bandwidth = trainable_bandwidth

        # Default frequencies: 1, 2, 3, ...
        if frequencies is None:
            self.frequencies = [float(i + 1) for i in range(n_modes)]
        else:
            self.frequencies = frequencies

        # Initialize bandwidth parameters if trainable
        if trainable_bandwidth:
            self.bandwidth = nn.Parameter(torch.ones(n_features))
        else:
            self.bandwidth = None

    def encode_parameters(self,
                         features: torch.Tensor,
                         bandwidth: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Encode features using Fourier series."""
        batch_size, n_feat = features.shape

        if bandwidth is None and self.trainable_bandwidth:
            bandwidth = self.bandwidth
        elif bandwidth is None:
            bandwidth = torch.ones(n_feat, device=features.device)

        params = {}

        for layer in range(self.n_layers):
            for feat_idx in range(min(n_feat, self.n_features)):
                for mode_idx in range(self.n_modes):
                    # Fourier encoding: bandwidth * frequency * pi * feature
                    freq = self.frequencies[mode_idx % len(self.frequencies)]
                    param_name = f'fourier_l{layer}_f{feat_idx}_m{mode_idx}'

                    # Apply bandwidth scaling
                    scale = bandwidth[feat_idx] if feat_idx < len(bandwidth) else 1.0

                    # Encode with Fourier frequency
                    encoded = scale * freq * math.pi * features[:, feat_idx]
                    params[param_name] = encoded

        return params

    def build_encoding_circuit(self, builder: 'CircuitBuilder') -> 'CircuitBuilder':
        """Build Fourier encoding circuit using platform-agnostic components."""
        from ..builder import CircuitBuilder

        for layer in range(self.n_layers):
            # Add rotation gates for each feature-mode combination
            for feat_idx in range(self.n_features):
                for mode_idx in range(self.n_modes):
                    param_name = f'fourier_l{layer}_f{feat_idx}_m{mode_idx}'
                    # This will be an input parameter (not trainable)
                    builder.circuit.rotation(mode_idx, angle=param_name, axis='z')

            # Add entangling layer between encoding layers
            if layer < self.n_layers - 1:
                builder.add_entangling_layer(pattern='nearest_neighbor')

        return builder

    @property
    def num_parameters(self) -> int:
        """Total number of encoding parameters."""
        return self.n_layers * self.n_features * self.n_modes

    @property
    def encoding_type(self) -> str:
        return "fourier"


class DataReuploadingEncoding(EncodingStrategy):
    """
    Data reuploading encoding strategy.

    Repeatedly encodes the same data at different circuit depths with
    trainable processing between uploads.
    """

    def __init__(self,
                 n_features: int,
                 n_modes: int,
                 n_uploads: int = 3,
                 trainable_processing: bool = True):
        """
        Initialize data reuploading encoding.

        Args:
            n_features: Number of input features
            n_modes: Number of quantum modes/qubits
            n_uploads: Number of times to reupload data
            trainable_processing: Add trainable gates between uploads
        """
        self.n_features = n_features
        self.n_modes = n_modes
        self.n_uploads = n_uploads
        self.trainable_processing = trainable_processing

    def encode_parameters(self,
                         features: torch.Tensor,
                         bandwidth: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Encode features with reuploading."""
        batch_size, n_feat = features.shape
        params = {}

        for upload in range(self.n_uploads):
            # Each upload encodes all features
            for feat_idx in range(min(n_feat, self.n_features)):
                # Map feature to modes (round-robin)
                mode_idx = feat_idx % self.n_modes
                param_name = f'reupload_u{upload}_f{feat_idx}'

                # Simple linear encoding for reuploading
                params[param_name] = math.pi * features[:, feat_idx]

        return params

    def build_encoding_circuit(self, builder: 'CircuitBuilder') -> 'CircuitBuilder':
        """Build data reuploading circuit."""
        from ..builder import CircuitBuilder

        for upload in range(self.n_uploads):
            # Upload data
            for feat_idx in range(self.n_features):
                mode_idx = feat_idx % self.n_modes
                param_name = f'reupload_u{upload}_f{feat_idx}'
                builder.circuit.rotation(mode_idx, angle=param_name, axis='y')

            # Add processing layer
            if self.trainable_processing:
                # Trainable rotations
                for mode in range(self.n_modes):
                    builder.circuit.rotation(
                        mode, 
                        angle=f'proc_u{upload}_m{mode}', 
                        axis='z'
                    )

                # Entangling
                builder.add_entangling_layer(pattern='nearest_neighbor')

        return builder

    @property
    def num_parameters(self) -> int:
        """Number of encoding parameters."""
        base = self.n_uploads * self.n_features
        if self.trainable_processing:
            base += self.n_uploads * self.n_modes
        return base

    @property
    def encoding_type(self) -> str:
        return "data_reuploading"


class AmplitudeEncoding(EncodingStrategy):
    """
    Amplitude encoding of classical data.

    Encodes features as amplitudes in the quantum state.
    """

    def __init__(self, n_features: int, n_modes: int):
        """
        Initialize amplitude encoding.

        Args:
            n_features: Number of input features
            n_modes: Number of quantum modes
        """
        self.n_features = n_features
        self.n_modes = n_modes

    def encode_parameters(self,
                         features: torch.Tensor,
                         bandwidth: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Encode features as amplitudes."""
        batch_size, n_feat = features.shape

        # Normalize features to create valid amplitudes
        amplitudes = features / torch.norm(features, dim=1, keepdim=True)

        # Convert to angles for state preparation
        params = {}
        for i in range(min(n_feat, self.n_features)):
            # Use arcsin for amplitude encoding
            angle = torch.asin(torch.clamp(amplitudes[:, i], -1, 1))
            params[f'amp_f{i}'] = angle

        return params

    def build_encoding_circuit(self, builder: 'CircuitBuilder') -> 'CircuitBuilder':
        """Build amplitude encoding circuit."""
        # Amplitude encoding typically requires special state preparation
        # For now, approximate with rotations
        for i in range(self.n_features):
            mode_idx = i % self.n_modes
            builder.circuit.rotation(mode_idx, angle=f'amp_f{i}', axis='y')

        return builder

    @property
    def num_parameters(self) -> int:
        return self.n_features

    @property
    def encoding_type(self) -> str:
        return "amplitude"


class HybridEncoding(EncodingStrategy):
    """
    Hybrid encoding combining multiple strategies.

    Allows combination of different encoding strategies in one circuit.
    """

    def __init__(self, encodings: List[EncodingStrategy]):
        """
        Initialize hybrid encoding.

        Args:
            encodings: List of encoding strategies to combine
        """
        self.encodings = encodings

    def encode_parameters(self,
                         features: torch.Tensor,
                         bandwidth: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Combine parameters from all encodings."""
        all_params = {}

        for encoding in self.encodings:
            params = encoding.encode_parameters(features, bandwidth)
            # Prefix with encoding type to avoid conflicts
            for key, value in params.items():
                prefixed_key = f'{encoding.encoding_type}_{key}'
                all_params[prefixed_key] = value

        return all_params

    def build_encoding_circuit(self, builder: 'CircuitBuilder') -> 'CircuitBuilder':
        """Build circuit combining all encodings."""
        for encoding in self.encodings:
            builder = encoding.build_encoding_circuit(builder)
            # Add barrier/processing between different encodings
            builder.add_entangling_layer(pattern='nearest_neighbor', depth=1)

        return builder

    @property
    def num_parameters(self) -> int:
        return sum(enc.num_parameters for enc in self.encodings)

    @property
    def encoding_type(self) -> str:
        return "hybrid"
