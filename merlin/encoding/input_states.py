"""Input state generation for quantum circuits.

Provides various strategies for preparing input quantum states including
superposition states and photonic Fock states.
"""

from typing import List, Dict, Optional, Union, Tuple
import torch
import numpy as np
from enum import Enum


class StatePattern(Enum):
    """Input state preparation patterns."""

    DEFAULT = "default"
    SPACED = "spaced"
    SEQUENTIAL = "sequential"
    PERIODIC = "periodic"
    SUPERPOSITION = "superposition"
    CUSTOM = "custom"


class InputStateGenerator:
    """Generate input states for quantum circuits."""

    @staticmethod
    def generate_fock_state(n_modes: int,
                           n_photons: int,
                           pattern: StatePattern = StatePattern.DEFAULT,
                           index_photons: Optional[List[Tuple[int, int]]] = None) -> List[int]:
        """
        Generate a Fock state for photonic circuits.

        Args:
            n_modes: Number of optical modes
            n_photons: Number of photons
            pattern: State preparation pattern
            index_photons: Constraints on photon placement

        Returns:
            List representing Fock state occupation
        """
        if n_photons < 0 or n_photons > n_modes:
            raise ValueError(f"Cannot place {n_photons} photons in {n_modes} modes")

        if index_photons:
            # Apply constraints
            state = [0] * n_modes
            for i, (min_mode, max_mode) in enumerate(index_photons):
                if i < n_photons:
                    # Place photon in allowed range
                    mode = min(min_mode, n_modes - 1)
                    state[mode] += 1
            return state

        if pattern == StatePattern.SPACED:
            return InputStateGenerator._spaced_state(n_modes, n_photons)
        elif pattern == StatePattern.SEQUENTIAL:
            return InputStateGenerator._sequential_state(n_modes, n_photons)
        elif pattern == StatePattern.PERIODIC:
            return InputStateGenerator._periodic_state(n_modes, n_photons)
        else:
            # Default: photons in first modes
            return [1 if i < n_photons else 0 for i in range(n_modes)]

    @staticmethod
    def generate_superposition_state(n_modes: int,
                                    n_photons: int,
                                    components: List[Tuple[List[int], complex]]) -> Dict[str, complex]:
        """
        Generate a superposition of Fock states.

        Args:
            n_modes: Number of modes
            n_photons: Number of photons per component
            components: List of (state, amplitude) pairs

        Returns:
            Dictionary mapping state strings to amplitudes
        """
        superposition = {}

        # Normalize amplitudes
        total = sum(abs(amp)**2 for _, amp in components)
        norm = np.sqrt(total)

        for state, amplitude in components:
            # Validate state
            if len(state) != n_modes:
                raise ValueError(f"State {state} doesn't match n_modes={n_modes}")
            if sum(state) != n_photons:
                raise ValueError(f"State {state} doesn't have {n_photons} photons")

            # Convert to string key
            state_key = ','.join(str(x) for x in state)
            superposition[state_key] = amplitude / norm

        return superposition

    @staticmethod
    def generate_qubit_state(n_qubits: int,
                            pattern: str = "zero") -> List[int]:
        """
        Generate initial qubit state.

        Args:
            n_qubits: Number of qubits
            pattern: State pattern ('zero', 'plus', 'random')

        Returns:
            Initial qubit state
        """
        if pattern == "zero":
            return [0] * n_qubits
        elif pattern == "one":
            return [1] * n_qubits
        elif pattern == "plus":
            # Plus state needs special handling
            return [0] * n_qubits  # Will add Hadamards
        elif pattern == "random":
            return [np.random.randint(0, 2) for _ in range(n_qubits)]
        else:
            return [0] * n_qubits

    @staticmethod
    def _spaced_state(n_modes: int, n_photons: int) -> List[int]:
        """Generate evenly spaced photon state."""
        if n_photons == 0:
            return [0] * n_modes

        spacing = n_modes // n_photons
        state = [0] * n_modes

        for i in range(n_photons):
            pos = i * spacing
            if pos < n_modes:
                state[pos] = 1

        return state

    @staticmethod
    def _sequential_state(n_modes: int, n_photons: int) -> List[int]:
        """Generate sequential photon state."""
        return [1 if i < n_photons else 0 for i in range(n_modes)]

    @staticmethod
    def _periodic_state(n_modes: int, n_photons: int) -> List[int]:
        """Generate periodic photon pattern."""
        state = [0] * n_modes
        period = max(2, n_modes // n_photons)

        placed = 0
        for i in range(n_modes):
            if i % period == 0 and placed < n_photons:
                state[i] = 1
                placed += 1

        # Fill remaining
        while placed < n_photons:
            for i in range(n_modes):
                if state[i] == 0 and placed < n_photons:
                    state[i] = 1
                    placed += 1
                    break

        return state


class AdaptiveStatePreparation:
    """Adaptive state preparation based on input features."""

    def __init__(self, n_modes: int, n_photons: int):
        """
        Initialize adaptive state preparation.

        Args:
            n_modes: Number of modes
            n_photons: Number of photons
        """
        self.n_modes = n_modes
        self.n_photons = n_photons

    def prepare_from_features(self,
                             features: torch.Tensor,
                             threshold: float = 0.5) -> List[List[int]]:
        """
        Prepare input states based on feature values.

        Args:
            features: Input features (batch_size, n_features)
            threshold: Threshold for state selection

        Returns:
            Batch of input states
        """
        batch_size = features.shape[0]
        states = []

        for b in range(batch_size):
            state = [0] * self.n_modes
            feature_vals = features[b].cpu().numpy()

            # Place photons based on feature magnitudes
            sorted_indices = np.argsort(np.abs(feature_vals))[::-1]

            for i in range(min(self.n_photons, len(sorted_indices))):
                mode_idx = sorted_indices[i] % self.n_modes
                state[mode_idx] += 1

            states.append(state)

        return states
