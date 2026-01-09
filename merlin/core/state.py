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

"""Input photon state patterns and helpers."""

from enum import Enum


class StatePattern(Enum):
    """Input photon state patterns."""

    DEFAULT = "default"
    SPACED = "spaced"
    SEQUENTIAL = "sequential"
    PERIODIC = "periodic"


def generate_state(n_modes: int, n_photons: int, state_pattern: StatePattern):
    """Generate an input occupation list for the requested pattern."""
    if n_photons < 0 or n_photons > n_modes:
        raise ValueError(f"Cannot place {n_photons} photons into {n_modes} modes.")

    if state_pattern == StatePattern.SPACED:
        return _generate_spaced_state(n_modes, n_photons)
    if state_pattern == StatePattern.SEQUENTIAL:
        return _generate_sequential_state(n_modes, n_photons)
    if state_pattern in (StatePattern.PERIODIC, StatePattern.DEFAULT):
        return _generate_periodic_state(n_modes, n_photons)

    # Fallback to periodic with a warning printed for visibility.
    print(f"Warning: Unknown state pattern '{state_pattern}'. Using PERIODIC.")
    return _generate_periodic_state(n_modes, n_photons)


def _generate_spaced_state(n_modes: int, n_photons: int):
    if n_photons == 0:
        return [0] * n_modes
    if n_photons == 1:
        pos = n_modes // 2
        return [1 if i == pos else 0 for i in range(n_modes)]

    positions = [int(i * n_modes / n_photons) for i in range(n_photons)]
    positions = [min(pos, n_modes - 1) for pos in positions]
    occ = [0] * n_modes
    for pos in positions:
        occ[pos] += 1
    return occ


def _generate_periodic_state(n_modes: int, n_photons: int):
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


def _generate_sequential_state(n_modes: int, n_photons: int):
    return [1 if i < n_photons else 0 for i in range(n_modes)]


__all__ = ["StatePattern", "generate_state"]
