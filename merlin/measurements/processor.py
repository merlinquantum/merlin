"""
Measurement processors for different quantum backends.
Handles the classical post-processing of quantum states.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import warnings

from ..core.observables import PauliObservable, CompositeObservable, NumberOperator


class MeasurementProcessor(ABC):
    """Abstract base class for measurement processing."""

    @abstractmethod
    def process_observable(
            self,
            probabilities: torch.Tensor,
            observable: Union[PauliObservable, NumberOperator]
    ) -> torch.Tensor:
        """
        Process single observable.

        Args:
            probabilities: Probability distribution (batch_size, n_states)
            observable: Observable to measure

        Returns:
            Expectation values (batch_size, 1)
        """
        pass

    @abstractmethod
    def can_measure(self, observable: Union[PauliObservable, NumberOperator]) -> bool:
        """Check if backend can measure this observable."""
        pass


class PhotonicMeasurementProcessor(MeasurementProcessor):
    """
    Measurement processor for photonic backend.

    For no_bunching=True: Maps to ±1 eigenvalues (like qubits)
    - Empty mode (n=0) → eigenvalue +1
    - Occupied mode (n=1) → eigenvalue -1

    For no_bunching=False: Number counting mode
    - Returns actual photon number expectation ⟨n̂⟩
    """

    def __init__(self, computation_process):
        """
        Initialize with computation process from backend.

        Args:
            computation_process: PhotonicBackend's computation_process
        """
        self.comp_process = computation_process
        self.n_modes = computation_process.m
        self.no_bunching = computation_process.no_bunching

        # Get Fock states from SLOS graph
        self.fock_states = computation_process.simulation_graph.mapped_keys
        self.n_states = len(self.fock_states)

        # Pre-compute eigenvalue matrices
        self._build_eigenvalue_matrices()

    def _build_eigenvalue_matrices(self):
        """Build eigenvalue matrices for each operator type."""
        # Get dtype from computation process
        dtype = self.comp_process.dtype if hasattr(self.comp_process, 'dtype') else torch.float32

        self.z_eigenvalues = torch.zeros(self.n_states, self.n_modes, dtype=dtype)

        # Also build number operator eigenvalues
        self.n_eigenvalues = torch.zeros(self.n_states, self.n_modes, dtype=dtype)
        if not self.no_bunching:
            self.n2_eigenvalues = torch.zeros(self.n_states, self.n_modes, dtype=dtype)

        for state_idx, fock_state in enumerate(self.fock_states):
            for mode_idx in range(self.n_modes):
                occupation = fock_state[mode_idx]

                if self.no_bunching:
                    # Binary occupation: Z eigenvalue = (1 - 2n)
                    # n=0 → +1, n=1 → -1
                    self.z_eigenvalues[state_idx, mode_idx] = 1 - 2 * occupation
                    # Number operator is just occupation
                    self.n_eigenvalues[state_idx, mode_idx] = occupation
                else:
                    # For bunched states:
                    # Z still maps as (1 - 2n) but n can be > 1
                    self.z_eigenvalues[state_idx, mode_idx] = 1 - 2 * occupation
                    # Number operator eigenvalue is just n
                    self.n_eigenvalues[state_idx, mode_idx] = occupation
                    # Number squared for variance calculations
                    self.n2_eigenvalues[state_idx, mode_idx] = occupation ** 2

    def can_measure(self, observable: Union[PauliObservable, NumberOperator]) -> bool:
        """Check if photonic backend can measure this observable."""
        if isinstance(observable, NumberOperator):
            return True
        # Standard Pauli: only Z and I supported
        return all(op in 'ZI' for op in observable.pauli_string)

    def process_observable(
            self,
            probabilities: torch.Tensor,
            observable: Union[PauliObservable, NumberOperator]
    ) -> torch.Tensor:
        """
        Process observable measurement.

        Args:
            probabilities: (batch_size, n_states)
            observable: Observable to measure

        Returns:
            Expectation values (batch_size, 1)
        """
        # Handle number operator
        if isinstance(observable, NumberOperator):
            return self._process_number_operator(probabilities, observable)

        # Standard Pauli measurement
        if not self.can_measure(observable):
            unsupported = [op for op in observable.pauli_string if op not in 'ZI']
            raise ValueError(
                f"Photonic backend cannot measure {observable.pauli_string}. "
                f"Unsupported operators: {set(unsupported)}. "
                "Only Z and I operators are supported."
            )

        # Compute eigenvalue for each Fock state
        eigenvalues = torch.ones(self.n_states, dtype=probabilities.dtype, device=probabilities.device)

        for mode_idx, op in enumerate(observable.pauli_string):
            if op == 'Z':
                z_vals = self.z_eigenvalues[:, mode_idx].to(dtype=probabilities.dtype, device=probabilities.device)
                eigenvalues *= z_vals
            # 'I' contributes factor of 1 (identity)

        # Compute expectation value
        expectation = probabilities @ eigenvalues.unsqueeze(-1)

        return expectation * observable.coefficient

    def _process_number_operator(
            self,
            probabilities: torch.Tensor,
            observable: NumberOperator
    ) -> torch.Tensor:
        """
        Process number operator measurement.

        Returns actual photon number expectation ⟨n̂⟩ or ⟨n̂^k⟩.
        """
        mode_idx = observable.mode_index

        if mode_idx >= self.n_modes:
            raise ValueError(f"Mode index {mode_idx} out of range for {self.n_modes} modes")

        if observable.power == 1:
            # Standard number operator
            n_vals = self.n_eigenvalues[:, mode_idx].to(dtype=probabilities.dtype, device=probabilities.device)
        elif observable.power == 2 and not self.no_bunching:
            # Number squared (for variance calculations)
            n_vals = self.n2_eigenvalues[:, mode_idx].to(dtype=probabilities.dtype, device=probabilities.device)
        else:
            # General power - compute on the fly
            n_vals = torch.zeros(self.n_states, dtype=probabilities.dtype, device=probabilities.device)
            for state_idx, fock_state in enumerate(self.fock_states):
                n_vals[state_idx] = fock_state[mode_idx] ** observable.power

        # Compute expectation value
        expectation = probabilities @ n_vals.unsqueeze(-1)

        # If no_bunching and user asks for number operator, inform them
        if self.no_bunching and observable.power == 1:
            # With no_bunching, n can only be 0 or 1
            # This is equivalent to (1-Z)/2 mapping
            pass  # Silent - the values are correct (0 or 1)

        return expectation * observable.coefficient

    def process_composite(
            self,
            probabilities: torch.Tensor,
            observable: CompositeObservable
    ) -> torch.Tensor:
        """Process composite observable as sum of terms."""
        result = torch.zeros(
            probabilities.shape[0], 1,
            dtype=probabilities.dtype,
            device=probabilities.device
        )

        for term in observable:
            result += self.process_observable(probabilities, term)

        return result

    def compute_number_statistics(
            self,
            probabilities: torch.Tensor,
            mode_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full number statistics for a mode.

        Returns:
            Dictionary with 'mean', 'variance', 'fano_factor'
        """
        # Mean
        n_vals = self.n_eigenvalues[:, mode_idx].to(dtype=probabilities.dtype, device=probabilities.device)
        mean_n = probabilities @ n_vals.unsqueeze(-1)

        # Variance
        if self.no_bunching:
            # For no_bunching: Var(n) = p(1-p) where p = P(n=1)
            variance = mean_n * (1 - mean_n)
        else:
            # For bunched: need ⟨n²⟩ - ⟨n⟩²
            n2_vals = self.n2_eigenvalues[:, mode_idx].to(dtype=probabilities.dtype, device=probabilities.device)
            mean_n2 = probabilities @ n2_vals.unsqueeze(-1)
            variance = mean_n2 - mean_n ** 2

        # Fano factor (variance/mean) - measure of non-Poissonianity
        # Avoid division by zero
        fano = torch.where(
            mean_n > 1e-8,
            variance / mean_n,
            torch.zeros_like(variance)
        )

        return {
            'mean': mean_n,
            'variance': variance,
            'fano_factor': fano
        }


class QubitMeasurementProcessor(MeasurementProcessor):
    """
    Measurement processor for qubit backends.
    Placeholder for future qubit backend implementation.
    """

    def __init__(self, n_qubits: int):
        """
        Initialize qubit measurement processor.

        Args:
            n_qubits: Number of qubits
        """
        self.n_qubits = n_qubits
        warnings.warn(
            "QubitMeasurementProcessor is a placeholder. "
            "Full implementation pending qubit backend."
        )

    def can_measure(self, observable: Union[PauliObservable, NumberOperator]) -> bool:
        """Qubit backends can measure all Pauli operators, but not number operators."""
        if isinstance(observable, NumberOperator):
            return False
        return all(op in 'IXYZ' for op in observable.pauli_string)

    def process_observable(
            self,
            probabilities: torch.Tensor,
            observable: Union[PauliObservable, NumberOperator]
    ) -> torch.Tensor:
        """
        Process Pauli observable for qubit system.

        Note: This is a placeholder. Full implementation requires:
        - Basis rotations for X and Y measurements
        - Proper eigenvalue computation
        """
        if isinstance(observable, NumberOperator):
            raise ValueError("Qubit backend cannot measure number operators")

        # Placeholder: only handle Z for now
        if not all(op in 'ZI' for op in observable.pauli_string):
            warnings.warn(
                f"X and Y measurements not yet implemented for qubits. "
                f"Returning zeros for {observable.pauli_string}"
            )
            return torch.zeros(probabilities.shape[0], 1, dtype=probabilities.dtype, device=probabilities.device)

        # Simple Z measurement (placeholder logic)
        n_states = probabilities.shape[1]
        eigenvalues = torch.ones(n_states, dtype=probabilities.dtype, device=probabilities.device)

        for qubit_idx, op in enumerate(observable.pauli_string):
            if op == 'Z':
                # Create Z eigenvalues based on computational basis
                for state_idx in range(n_states):
                    bit = (state_idx >> (self.n_qubits - 1 - qubit_idx)) & 1
                    eigenvalues[state_idx] *= (1 - 2 * bit)  # 0→+1, 1→-1

        expectation = probabilities @ eigenvalues.unsqueeze(-1)
        return expectation * observable.coefficient