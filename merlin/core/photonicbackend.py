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
Experiment configuration for quantum layer setups.
"""

from ..core.generators import CircuitType, StatePattern


class PhotonicBackend:
    """Configuration container for quantum layer experiments."""

    def __init__(
        self,
        circuit_type: CircuitType,
        n_modes: int,
        n_photons: int,
        state_pattern: StatePattern = StatePattern.PERIODIC,
        use_bandwidth_tuning: bool = False,
        reservoir_mode: bool = False,
    ):
        r"""Initialize the PhotonicBackend with the given configuration.

        Args:
            circuit_type: The circuit type to use.
            n_modes: Number of modes in the circuit.
            n_photons: Number of photons to place in the circuit.
            state_pattern: The state pattern to use (default is periodic).
            use_bandwidth_tuning: Whether to use bandwidth tuning (default is False).
            reservoir_mode: Whether to use reservoir mode (default is False).
        """
        # Validate circuit_type
        if isinstance(circuit_type, str):
            try:
                circuit_type = CircuitType(circuit_type.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid circuit_type: {circuit_type}. "
                    f"Valid options are: {[e.value for e in CircuitType]}"
                )
        elif not isinstance(circuit_type, CircuitType):
            raise TypeError(
                f"circuit_type must be CircuitType enum or string, got {type(circuit_type)}"
            )

        # Validate n_modes
        if not isinstance(n_modes, int) or n_modes <= 0:
            raise ValueError(f"n_modes must be a positive integer, got {n_modes}")

        # Validate n_photons
        if not isinstance(n_photons, int) or n_photons < 0:
            raise ValueError(
                f"n_photons must be a non-negative integer, got {n_photons}"
            )

        if n_photons > n_modes:
            raise ValueError(f"Cannot place {n_photons} photons into {n_modes} modes")

        # Validate state_pattern
        if isinstance(state_pattern, str):
            try:
                state_pattern = StatePattern(state_pattern.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid state_pattern: {state_pattern}. "
                    f"Valid options are: {[e.value for e in StatePattern]}"
                )
        elif not isinstance(state_pattern, StatePattern):
            raise TypeError(
                f"state_pattern must be StatePattern enum or string, got {type(state_pattern)}"
            )

        self.circuit_type = circuit_type
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.state_pattern = state_pattern
        self.use_bandwidth_tuning = use_bandwidth_tuning
        self.reservoir_mode = reservoir_mode
