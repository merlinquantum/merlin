"""
Classical measurement layer for post-processing quantum outputs.
Can be used as a standalone layer after quantum models.
"""

from typing import Union, List, Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

from ..core.observables import PauliObservable, CompositeObservable, parse_observable
from .processor import MeasurementProcessor, PhotonicMeasurementProcessor, QubitMeasurementProcessor


class MeasurementLayer(nn.Module):
    """
    Classical layer that processes quantum outputs into measurements.

    This is a pure post-processing layer that can be added after
    any quantum model to extract expectation values.
    """

    def __init__(
            self,
            observables: Union[str, List[str], List[PauliObservable]],
            backend_type: str = "photonic",
            n_modes: Optional[int] = None,
            processor: Optional[MeasurementProcessor] = None
    ):
        """
        Initialize measurement layer.

        Args:
            observables: Observable(s) to measure
            backend_type: Type of quantum backend
            n_modes: Number of modes/qubits
            processor: Optional pre-configured processor
        """
        super().__init__()

        self.backend_type = backend_type
        self.n_modes = n_modes

        # Parse observables
        if isinstance(observables, str):
            observables = [observables]

        self.observables = []
        self.observable_names = []

        for i, obs in enumerate(observables):
            if isinstance(obs, str):
                parsed = parse_observable(obs, n_modes)
                self.observables.append(parsed)
                self.observable_names.append(f"obs_{i}")
            else:
                self.observables.append(obs)
                self.observable_names.append(f"obs_{i}")

        # Store processor if provided (will be set by backend)
        self.processor = processor

    def set_processor(self, processor: MeasurementProcessor):
        """Set the measurement processor (called by backend)."""
        self.processor = processor

    def forward(
            self,
            quantum_output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process quantum output into measurements.

        Args:
            quantum_output: Either probabilities or (probabilities, amplitudes) tuple
            return_dict: If True, return dict with named measurements

        Returns:
            Measurement results as tensor or dict
        """
        # Extract probabilities
        if isinstance(quantum_output, tuple):
            probabilities, amplitudes = quantum_output
        else:
            probabilities = quantum_output

        if self.processor is None:
            raise RuntimeError(
                "No measurement processor set. "
                "This should be set automatically by the quantum backend."
            )

        results = []
        for observable in self.observables:
            if isinstance(observable, CompositeObservable):
                result = self.processor.process_composite(probabilities, observable)
            else:
                result = self.processor.process_observable(probabilities, observable)
            results.append(result)

        if return_dict:
            return {
                name: result
                for name, result in zip(self.observable_names, results)
            }

        # Concatenate results
        if len(results) == 1:
            return results[0]
        return torch.cat(results, dim=-1)