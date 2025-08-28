"""Platform-agnostic backend base classes."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import torch


@dataclass
class CompiledCircuit:
    """
    Compiled circuit ready for execution.

    Contains platform-specific representation.
    """
    operations: Any  # Platform-specific (e.g., Perceval circuit, Qiskit circuit)
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Initialize platform-specific attributes."""
        # These will be set by specific backends
        self.pcvl_circuit = None  # For PhotonicBackend
        self.qiskit_circuit = None  # For future QiskitBackend
        self.platform = self.metadata.get('platform', 'unknown')


class Backend(ABC):
    """
    Abstract base class for quantum execution backends.

    Platform-agnostic interface that different quantum platforms implement.
    """

    def __init__(self,
                 platform_type: str,
                 n_elements: int,  # modes for photonic, qubits for qubit platforms
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None):
        self.platform_type = platform_type
        self.n_elements = n_elements
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32

        # For compatibility with existing code
        if platform_type == 'photonic':
            self.n_modes = n_elements
        elif platform_type in ['qubit', 'ion_trap']:
            self.n_qubits = n_elements

        # Track measurements for mode selectivity
        self.measured_elements = set()
        self.active_elements = set(range(n_elements))

    @abstractmethod
    def compile(self, components: Union[List['Component'], 'Circuit']) -> CompiledCircuit:
        """
        Compile components or circuit to platform-specific representation.

        Args:
            components: List of components or Circuit object

        Returns:
            Platform-specific compiled circuit
        """
        pass

    @abstractmethod
    def execute(self,
                circuit: CompiledCircuit,
                params: Dict[str, torch.Tensor],
                shots: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Execute compiled circuit with parameters.

        Args:
            circuit: Compiled circuit
            params: Parameter values
            shots: Number of measurement shots
            **kwargs: Platform-specific options

        Returns:
            Execution results (probabilities, amplitudes, samples, etc.)
        """
        pass

    def mark_measured(self, elements: List[int]):
        """Mark elements as measured."""
        self.measured_elements.update(elements)
        self.active_elements -= set(elements)

    def get_active_elements(self) -> List[int]:
        """Get list of active (non-measured) elements."""
        return sorted(list(self.active_elements))

    def get_measured_elements(self) -> List[int]:
        """Get list of measured elements."""
        return sorted(list(self.measured_elements))

    def reset_measurements(self):
        """Reset measurement tracking."""
        self.measured_elements = set()
        self.active_elements = set(range(self.n_elements))

    @property
    def supports_gradients(self) -> bool:
        """Whether backend supports gradient computation."""
        return False

    @property
    def supports_gpu(self) -> bool:
        """Whether backend supports GPU acceleration."""
        return self.device.type == 'cuda' if self.device else False
