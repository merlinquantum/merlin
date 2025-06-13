"""
Updated CircuitGenerator with reservoir mode support.
Replace the existing CircuitGenerator class with this version.
"""

import perceval as pcvl
import numpy as np
import random
from enum import Enum


class CircuitType(Enum):
    """Quantum circuit topology types."""
    PARALLEL_COLUMNS = "parallel_columns"
    SERIES = "series"
    PARALLEL = "parallel"


class StatePattern(Enum):
    """Input photon state patterns."""
    DEFAULT = "default"
    SPACED = "spaced"
    SEQUENTIAL = "sequential"
    PERIODIC = "periodic"


class CircuitGenerator:
    """Utility class for generating quantum photonic circuits."""

    @staticmethod
    def generate_circuit(circuit_type, n_modes, n_features, reservoir_mode=False):
        """Generate a quantum circuit based on specified type."""
        # Validate inputs
        if n_modes <= 0:
            raise ValueError(f"n_modes must be positive, got {n_modes}")
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")

        if circuit_type == CircuitType.PARALLEL_COLUMNS:
            return CircuitGenerator._build_parallel_columns_circuit(n_modes, n_features, reservoir_mode), n_features * n_modes
        elif circuit_type == CircuitType.SERIES:
            if n_features == 1:
                return CircuitGenerator._build_series_simple_circuit(n_modes, reservoir_mode), n_modes - 1
            else:
                num_params = min((1 << n_features) - 1, n_modes - 1)
                return CircuitGenerator._build_series_multi_circuit(n_modes, n_features, reservoir_mode), num_params
        elif circuit_type == CircuitType.PARALLEL:
            if n_features == 1:
                num_blocks = n_modes - 1
                return CircuitGenerator._build_parallel_simple_circuit(n_modes, num_blocks, reservoir_mode), num_blocks
            return CircuitGenerator._build_parallel_multi_circuit(n_modes, n_features, reservoir_mode), n_features
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")

    @staticmethod
    def _generate_interferometer(n_modes, stage_idx, reservoir_mode=False):
        """Generate a rectangular interferometer based on mode."""
        if reservoir_mode:
            # For reservoir mode: use fixed random values instead of parameters
            return pcvl.GenericInterferometer(
                n_modes,
                lambda idx: pcvl.BS(theta=np.pi * 2 * random.random()) // (0, pcvl.PS(phi=np.pi * 2 * random.random())),
                shape=pcvl.InterferometerShape.RECTANGLE,
                depth=2 * n_modes,
                phase_shifter_fun_gen=lambda idx: pcvl.PS(phi=np.pi * 2 * random.random())
            )
        else:
            # Original implementation with named parameters
            def mzi(P1, P2):
                return (pcvl.Circuit(2)
                        .add((0, 1), pcvl.BS())
                        .add(0, pcvl.PS(P1))
                        .add((0, 1), pcvl.BS())
                        .add(0, pcvl.PS(P2)))

            offset = stage_idx * (n_modes * (n_modes - 1) // 2)
            shape = pcvl.InterferometerShape.RECTANGLE

            return pcvl.GenericInterferometer(
                n_modes,
                fun_gen=lambda idx: mzi(
                    pcvl.P(f"phi_0{offset + idx}"),
                    pcvl.P(f"phi_1{offset + idx}")
                ),
                shape=shape,
                phase_shifter_fun_gen=lambda idx: pcvl.PS(
                    phi=pcvl.P(f"phi_02{stage_idx}_{idx}")
                )
            )

    @staticmethod
    def _build_parallel_columns_circuit(n_modes, n_features, reservoir_mode=False):
        """Build a PARALLEL_COLUMNS type circuit."""
        circuit = pcvl.Circuit(n_modes)
        ps_idx = 0
        for stage in range(n_features + 1):
            circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, stage, reservoir_mode))
            if stage < n_features:
                for m_idx in range(n_modes):
                    circuit.add(m_idx, pcvl.PS(pcvl.P(f"pl{ps_idx}x")))
                    ps_idx += 1
        return circuit

    @staticmethod
    def _build_series_simple_circuit(n_modes, reservoir_mode=False):
        """Build a SERIES type circuit for a single feature."""
        circuit = pcvl.Circuit(n_modes)
        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, 0, reservoir_mode))
        for m_idx in range(n_modes - 1):
            circuit.add(m_idx, pcvl.PS(pcvl.P(f"pl_{m_idx}")))

        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, 1, reservoir_mode))
        return circuit

    @staticmethod
    def _build_series_multi_circuit(n_modes, n_features, reservoir_mode=False):
        """Build a SERIES type circuit for multiple features."""
        circuit = pcvl.Circuit(n_modes)
        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, 0, reservoir_mode))

        # Based on the paper: we need 2^n_features - 1 phase shifters
        # but limited by n_modes - 1
        num_phase_shifters = min((1 << n_features) - 1, n_modes - 1)

        # Create exactly num_phase_shifters phase shifters
        for i in range(num_phase_shifters):
            circuit.add(i, pcvl.PS(pcvl.P(f"pl_{i}")))

        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, 1, reservoir_mode))
        return circuit

    @staticmethod
    def _build_parallel_simple_circuit(n_modes, num_blocks, reservoir_mode=False):
        """Build a PARALLEL type circuit for a single feature."""
        circuit = pcvl.Circuit(n_modes)
        for b in range(num_blocks):
            circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, b, reservoir_mode))
            circuit.add(0, pcvl.PS(pcvl.P(f"pl{b}x")))
        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, num_blocks + 1, reservoir_mode))

        return circuit

    @staticmethod
    def _build_parallel_multi_circuit(n_modes, n_features, reservoir_mode=False):
        """Build a PARALLEL type circuit for multiple features."""
        circuit = pcvl.Circuit(n_modes)
        for i in range(n_features):
            circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, i * 2, reservoir_mode))
            circuit.add(0, pcvl.PS(pcvl.P(f"pl{i}x")))
        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, n_features + 1, reservoir_mode))
        return circuit


# Keep StateGenerator as is...
class StateGenerator:
    """Utility class for generating photonic input states."""

    @staticmethod
    def generate_state(n_modes, n_photons, state_pattern):
        """Generate an input state based on specified pattern."""
        if n_photons < 0 or n_photons > n_modes:
            raise ValueError(f"Cannot place {n_photons} photons into {n_modes} modes.")

        if state_pattern == StatePattern.SPACED:
            return StateGenerator._generate_spaced_state(n_modes, n_photons)
        elif state_pattern == StatePattern.SEQUENTIAL:
            return StateGenerator._generate_sequential_state(n_modes, n_photons)
        elif state_pattern in [StatePattern.PERIODIC, StatePattern.DEFAULT]:
            return StateGenerator._generate_periodic_state(n_modes, n_photons)
        else:
            print(f"Warning: Unknown state pattern '{state_pattern}'. Using PERIODIC.")
            return StateGenerator._generate_periodic_state(n_modes, n_photons)

    @staticmethod
    def _generate_spaced_state(n_modes, n_photons):
        """Generate a state with evenly spaced photons."""
        if n_photons == 0:
            return [0] * n_modes
        if n_photons == 1:
            pos = n_modes // 2
            occ = [1 if i == pos else 0 for i in range(n_modes)]
            return occ
        positions = [int(i * n_modes / n_photons) for i in range(n_photons)]
        positions = [min(pos, n_modes - 1) for pos in positions]
        occ = [0] * n_modes
        for pos in positions:
            occ[pos] += 1
        return occ

    @staticmethod
    def _generate_periodic_state(n_modes, n_photons):
        """Generate a state with periodically placed photons."""
        bits = [1 if i % 2 == 0 else 0 for i in range(min(n_photons * 2, n_modes))]
        count = sum(bits)
        i = 0
        while count < n_photons and i < n_modes:
            if i >= len(bits):
                bits.append(0)
            if bits[i] == 0:
                bits[i] = 1
                count += 1
            i += 1
        padding = [0] * (n_modes - len(bits))
        return bits + padding

    @staticmethod
    def _generate_sequential_state(n_modes, n_photons):
        """Generate a state with sequentially placed photons."""
        occ = [1 if i < n_photons else 0 for i in range(n_modes)]
        return occ