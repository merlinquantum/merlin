"""
Backend-agnostic QuantumModel interface.
Routes to appropriate backend (PhotonicBackend, future QiskitBackend, etc.)
Now with amplitude output support and measurement integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class QuantumConfig:
    # Circuit
    n_modes: Optional[int] = None
    n_qubits: Optional[int] = None
    n_photons: Optional[int] = None

    # Backend
    backend_type: str = "photonic"
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None

    # Execution
    shots: int = 0
    return_amplitudes: bool = False  # Option for amplitude output

    # Backend-specific options (passed through)
    backend_options: Dict[str, Any] = None

    # Input/Output
    n_features: Optional[int] = None
    output_size: Optional[int] = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dtype is None:
            self.dtype = torch.float32
        if self.backend_options is None:
            self.backend_options = {}
        # Set default dimensions
        if self.n_modes is None and self.n_qubits is None:
            self.n_modes = 4


class QuantumModel(nn.Module):
    """
    Backend-agnostic quantum model.
    Delegates all quantum-specific operations to the backend.
    Now supports amplitude output and measurements.
    """

    def __init__(
            self,
            circuit: Any,  # Can be Circuit, pcvl.Circuit, qiskit.QuantumCircuit, etc.
            config: Optional[QuantumConfig] = None,
            **kwargs: Any,
    ):
        super().__init__()

        # Create config
        if config is None:
            # Extract backend options from kwargs
            backend_options = {}
            for key in ['no_bunching', 'index_photons', 'reservoir_mode', 'input_state',
                        'trainable_parameters', 'input_parameters']:
                if key in kwargs:
                    backend_options[key] = kwargs.pop(key)

            config = QuantumConfig(backend_options=backend_options, **kwargs)
        else:
            # Merge kwargs into config
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        self.config = config
        self.circuit = circuit

        # Create appropriate backend
        self._create_backend()

        # Setup circuit through backend
        self._setup_circuit()

        # Setup measurements if present
        self._setup_measurements()

    def _create_backend(self):
        """Create the appropriate backend based on config."""
        if self.config.backend_type == "photonic":
            from ..backends.photonic import PhotonicBackend

            # Extract photonic-specific options
            photonic_kwargs = {
                'n_modes': self.config.n_modes,
                'n_photons': self.config.n_photons,
                'device': self.config.device,
                'dtype': self.config.dtype,
            }
            photonic_kwargs.update(self.config.backend_options)

            self.backend = PhotonicBackend(**photonic_kwargs)

        elif self.config.backend_type == "qiskit":
            # Future: QiskitBackend
            raise NotImplementedError("QiskitBackend not yet implemented")
        else:
            raise ValueError(f"Unknown backend type: {self.config.backend_type}")

    def _setup_circuit(self):
        """Setup circuit through the backend."""
        # Let backend handle circuit setup
        param_info = self.backend.setup_circuit(self.circuit)

        # Register parameters based on backend's info
        self._register_parameters(param_info)

    def _register_parameters(self, param_info: Dict[str, Any]):
        """Register trainable parameters as nn.Parameters."""
        if 'trainable' not in param_info:
            return

        # Each trainable parameter group becomes an nn.Parameter
        for param_name, param_size in param_info['trainable'].items():
            if isinstance(param_size, int):
                size = param_size
            else:
                size = len(param_size) if hasattr(param_size, '__len__') else 1

            # Initialize with small random values
            parameter = nn.Parameter(
                torch.randn(size, dtype=self.config.dtype, device=self.config.device) * 0.1
            )
            self.register_parameter(param_name, parameter)

    def _setup_measurements(self):
        """Setup measurement handling."""
        self.measurement_specs = {}
        self.measurement_dims = {}

        if hasattr(self.circuit, 'metadata') and 'measurements' in self.circuit.metadata:
            for meas_dict in self.circuit.metadata['measurements']:
                name = meas_dict['name']
                observable = meas_dict['observable']

                # Store spec by name
                self.measurement_specs[name] = observable

                # All measurements return single values for now
                self.measurement_dims[name] = 1

    def _collect_parameters(self) -> Dict[str, torch.Tensor]:
        """Collect all parameters for backend."""
        params = {}
        for name, param in self.named_parameters():
            if not name.startswith('output_layer.'):
                params[name] = param
        return params

    def forward(
            self,
            x: Optional[torch.Tensor] = None,
            input_params: Optional[Dict[str, torch.Tensor]] = None,
            shots: Optional[int] = None,
            return_amplitudes: Optional[bool] = None,
            return_dict: bool = False,
            **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through quantum circuit.

        Args:
            x: Input tensor (will be passed to backend for encoding)
            input_params: Explicit parameter values
            shots: Number of measurement shots
            return_amplitudes: If True, return (probabilities, amplitudes)
            return_dict: If True and measurements present, return dict
            **kwargs: Additional backend-specific arguments

        Returns:
            Output from quantum circuit (measurements, probabilities, or amplitudes)
        """
        # Determine whether to return amplitudes
        if return_amplitudes is None:
            return_amplitudes = self.config.return_amplitudes

        # Collect all parameters
        params = self._collect_parameters()

        # Add explicit input params (override)
        if input_params:
            params.update(input_params)

        # Execute through backend
        backend_result = self.backend.execute(
            params=params,
            input_data=x,
            shots=shots or self.config.shots,
            return_amplitudes=return_amplitudes and not self.backend.measurements,
            **kwargs
        )

        # If backend handled measurements, format output
        if self.backend.measurements:
            if return_dict and self.measurement_specs:
                # Convert to named dict
                result_dict = {}
                offset = 0
                for name in self.measurement_specs:
                    dim = self.measurement_dims[name]
                    if backend_result.dim() == 1:
                        result_dict[name] = backend_result[offset:offset + dim]
                    else:
                        result_dict[name] = backend_result[:, offset:offset + dim]
                    offset += dim
                return result_dict
            return backend_result

        # Handle amplitude return (no measurements)
        if return_amplitudes:
            probabilities, amplitudes = backend_result

            # Map output size if needed
            if self.config.output_size and probabilities.shape[-1] != self.config.output_size:
                probabilities = self._map_output(probabilities)

            return probabilities, amplitudes
        else:
            output = backend_result

            # Map output size if needed
            if self.config.output_size and output.shape[-1] != self.config.output_size:
                output = self._map_output(output)

            return output

    def _map_output(self, out: torch.Tensor) -> torch.Tensor:
        """Map quantum output to desired size."""
        if not hasattr(self, 'output_layer'):
            self.output_layer = nn.Linear(
                out.shape[-1],
                self.config.output_size,
                device=self.config.device,
                dtype=self.config.dtype
            )
        return self.output_layer(out)

    def iter_measurements(self):
        """Iterate over measurements like parameters."""
        for name, observable in self.measurement_specs.items():
            yield name, observable

    @classmethod
    def from_builder(
            cls,
            builder: "CircuitBuilder",
            backend_type: str = "photonic",
            config: Optional[QuantumConfig] = None,
            **kwargs
    ) -> "QuantumModel":
        """Create model from CircuitBuilder."""
        circuit = builder.build()

        if config is None:
            # Extract circuit metadata
            n_modes = builder.n_modes
            n_photons = getattr(builder, 'n_photons', None)

            config = QuantumConfig(
                n_modes=n_modes,
                n_photons=n_photons,
                backend_type=backend_type,
                **kwargs
            )

        return cls(circuit, config=config)

    @classmethod
    def simple(
            cls,
            input_size: int,
            n_params: int = 100,
            backend_type: str = "photonic",
            shots: int = 0,
            output_size: Optional[int] = None,
            return_amplitudes: bool = False,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            **backend_kwargs
    ) -> "QuantumModel":
        """
        Create a simple quantum model.

        Args:
            input_size: Number of input features
            n_params: Approximate number of parameters
            backend_type: Which backend to use
            shots: Number of measurement shots
            output_size: Output dimension
            return_amplitudes: Whether to return amplitudes
        device: PyTorch device
                    dtype: PyTorch dtype
                    **backend_kwargs: Backend-specific options

                Returns:
                    QuantumModel instance
                """
        if backend_type == "photonic":
            # Calculate photonic circuit size
            n_modes = max(int(math.ceil(math.sqrt(n_params / 2))), input_size + 1)
            n_photons = input_size

            # Use CircuitBuilder for simplicity
            from ..builder import CircuitBuilder

            builder = CircuitBuilder(n_modes=n_modes, n_photons=n_photons)

            # Add input layer
            builder.add_rotation_layer(role="input")

            # Add entangling (creates interferometer)
            builder.add_entangling_layer(trainable=not backend_kwargs.get('reservoir_mode', False))

            # Add output layer if not reservoir
            if not backend_kwargs.get('reservoir_mode', False):
                builder.add_rotation_layer(role="trainable")

            circuit = builder.build()

        elif backend_type == "qiskit":
            # Future: Build qiskit circuit
            raise NotImplementedError("Qiskit simple circuit not yet implemented")
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

        config = QuantumConfig(
            n_modes=n_modes if backend_type == "photonic" else None,
            n_qubits=input_size if backend_type == "qiskit" else None,
            n_photons=n_photons if backend_type == "photonic" else None,
            n_features=input_size,
            backend_type=backend_type,
            shots=shots,
            return_amplitudes=return_amplitudes,
            device=device,
            dtype=dtype,
            output_size=output_size,
            backend_options=backend_kwargs
        )

        return cls(circuit, config=config)

    def get_info(self) -> Dict[str, Any]:
        """Get information about the model and circuit."""
        info = {
            'backend': self.config.backend_type,
            'device': str(self.config.device),
            'dtype': str(self.config.dtype),
            'supports_amplitudes': getattr(self.backend, 'supports_amplitudes', False),
            'has_measurements': len(self.backend.measurements) > 0 if hasattr(self.backend, 'measurements') else False,
        }

        # Add backend-specific info
        if hasattr(self.backend, 'get_info'):
            info.update(self.backend.get_info())

        # Add measurement info
        if self.measurement_specs:
            info['measurements'] = list(self.measurement_specs.keys())

        return info