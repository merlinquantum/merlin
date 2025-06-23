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
Quantum circuit generation utilities.
"""

import perceval as pcvl


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
    def generate_circuit(circuit_type, n_modes, n_features):
        """Generate a quantum circuit based on specified type."""
        # Validate inputs
        if n_modes <= 0:
            raise ValueError(f"n_modes must be positive, got {n_modes}")
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")

        if circuit_type == CircuitType.PARALLEL_COLUMNS:
            return CircuitGenerator._build_parallel_columns_circuit(n_modes, n_features), n_features * n_modes
        elif circuit_type == CircuitType.SERIES:
            if n_features == 1:
                return CircuitGenerator._build_series_simple_circuit(n_modes), n_modes - 1
            else:
                num_params = min((1 << n_features) - 1, n_modes - 1)
                return CircuitGenerator._build_series_multi_circuit(n_modes, n_features), num_params
        elif circuit_type == CircuitType.PARALLEL:
            if n_features == 1:
                num_blocks = n_modes - 1
                return CircuitGenerator._build_parallel_simple_circuit(n_modes, num_blocks), num_blocks
            return CircuitGenerator._build_parallel_multi_circuit(n_modes, n_features), n_features
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")

    @staticmethod
    def _generate_interferometer(n_modes, stage_idx):
        """Generate a rectangular interferometer."""

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
    def _build_parallel_columns_circuit(n_modes, n_features):
        """Build a PARALLEL_COLUMNS type circuit."""
        circuit = pcvl.Circuit(n_modes)
        ps_idx = 0
        for stage in range(n_features + 1):
            circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, stage))
            if stage < n_features:
                for m_idx in range(n_modes):
                    circuit.add(m_idx, pcvl.PS(pcvl.P(f"pl{ps_idx}x")))
                    ps_idx += 1
        return circuit

    @staticmethod
    def _build_series_simple_circuit(n_modes):
        """Build a SERIES type circuit for a single feature."""
        circuit = pcvl.Circuit(n_modes)
        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, 0))
        for m_idx in range(n_modes-1):
            circuit.add(m_idx, pcvl.PS(pcvl.P(f"pl{m_idx}x")))

        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, 1))
        return circuit

    @staticmethod
    def _build_series_multi_circuit(n_modes, n_features):
        """Build a SERIES type circuit for multiple features."""
        circuit = pcvl.Circuit(n_modes)
        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, 0))
        max_ps = min((1 << n_features) - 1, n_modes)
        #max_ps = n_modes-1
        ps_idx = 0

        for i in range(min(n_features, max_ps)):
            circuit.add(i, pcvl.PS(pcvl.P(f"pl{ps_idx}x")))
            #circuit.add(i, pcvl.PS(pcvl.P(f"pl{i}x")))  # Use loop variable i instead
            ps_idx += 1

        if n_features >= 2 and ps_idx < max_ps:
            for i in range(ps_idx, max_ps):
                circuit.add(i, pcvl.PS(pcvl.P(f"pl{ps_idx}x")))
             #   circuit.add(i, pcvl.PS(pcvl.P(f"pl{i}x")))  # Use loop variable i instead

                ps_idx += 1

        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, 1))
        return circuit

    @staticmethod
    def _build_series_multi_circuit(n_modes, n_features):
        """Build a SERIES type circuit for multiple features."""
        circuit = pcvl.Circuit(n_modes)
        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, 0))

        # Based on the paper: we need 2^n_features - 1 phase shifters
        # but limited by n_modes - 1 (can't have more phase shifters than modes allow)
        num_phase_shifters = min((1 << n_features) - 1, n_modes - 1)

        # Create exactly num_phase_shifters phase shifters
        for i in range(num_phase_shifters):
            circuit.add(i, pcvl.PS(pcvl.P(f"pl{i}x")))

        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, 1))
        return circuit
    @staticmethod
    def _build_parallel_simple_circuit(n_modes, num_blocks):
        """Build a PARALLEL type circuit for a single feature."""
        circuit = pcvl.Circuit(n_modes)
        for b in range(num_blocks):
            circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, b))
            circuit.add(0, pcvl.PS(pcvl.P(f"pl{b}x")))
        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, num_blocks+1))

        return circuit

    @staticmethod
    def _build_parallel_multi_circuit(n_modes, n_features):
        """Build a PARALLEL type circuit for multiple features."""
        circuit = pcvl.Circuit(n_modes)
        for i in range(n_features):
            circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, i * 2))
            circuit.add(0, pcvl.PS(pcvl.P(f"pl{i}x")))
           # circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, i * 2 + 1))
        circuit.add(0, CircuitGenerator._generate_interferometer(n_modes, n_features  + 1))
        return circuit





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