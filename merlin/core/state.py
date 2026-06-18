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

"""Photon input-state helpers.

This module intentionally stays lightweight and avoids classes that only expose
static methods.
"""

from __future__ import annotations

from enum import Enum

import perceval as pcvl  # type: ignore


class StatePattern(str, Enum):
    """Enumeration of supported input photon state patterns."""

    DEFAULT = "default"
    SPACED = "spaced"
    SEQUENTIAL = "sequential"
    PERIODIC = "periodic"


def generate_state(
    n_modes: int,
    n_photons: int,
    state_pattern: StatePattern | str = StatePattern.DEFAULT,
) -> pcvl.BasicState:
    """Generate a Perceval Fock input state as a `pcvl.BasicState <https://perceval.quandela.net/docs/v1.2/reference/utils/states.html>`_.

    Parameters
    ----------
    n_modes : int
        Number of photonic modes.
    n_photons : int
        Total number of photons.
    state_pattern : StatePattern | str
        Placement strategy for the photons.

    Returns
    -------
    pcvl.BasicState
        Generated Perceval basic state.

    Raises
    ------
    ValueError
        If the inputs are inconsistent or the pattern is unknown.
    """

    occ = _generate_occupation(n_modes, n_photons, state_pattern)
    return pcvl.BasicState(tuple(occ))


def _generate_occupation(
    n_modes: int,
    n_photons: int,
    state_pattern: StatePattern | str = StatePattern.DEFAULT,
) -> list[int]:
    if n_modes <= 0:
        raise ValueError(f"n_modes must be positive, got {n_modes}")

    if n_photons < 0 or n_photons > n_modes:
        raise ValueError(f"Cannot place {n_photons} photons into {n_modes} modes.")

    if isinstance(state_pattern, str):
        try:
            pattern = StatePattern(state_pattern)
        except ValueError as exc:
            raise ValueError(
                f"Unknown state pattern '{state_pattern}'. "
                f"Valid values: {[p.value for p in StatePattern]}"
            ) from exc
    else:
        pattern = state_pattern

    if pattern == StatePattern.SPACED:
        return _generate_spaced_state(n_modes, n_photons)

    if pattern == StatePattern.SEQUENTIAL:
        return _generate_sequential_state(n_modes, n_photons)

    if pattern in (StatePattern.PERIODIC, StatePattern.DEFAULT):
        return _generate_periodic_state(n_modes, n_photons)

    raise ValueError(
        f"Unknown state pattern '{pattern}'. Valid values: {[p.value for p in StatePattern]}"
    )


def _generate_spaced_state(n_modes: int, n_photons: int) -> list[int]:
    if n_photons == 0:
        return [0] * n_modes

    if n_photons == 1:
        pos = n_modes // 2
        return [1 if idx == pos else 0 for idx in range(n_modes)]

    positions = [int(idx * n_modes / n_photons) for idx in range(n_photons)]
    positions = [min(pos, n_modes - 1) for pos in positions]

    occ = [0] * n_modes
    for pos in positions:
        occ[pos] += 1

    return occ


def _generate_periodic_state(n_modes: int, n_photons: int) -> list[int]:
    if n_photons == 0:
        return [0] * n_modes

    bits = [1 if idx % 2 == 0 else 0 for idx in range(min(n_photons * 2, n_modes))]
    count = sum(bits)
    idx = 0

    while count < n_photons and idx < n_modes:
        if idx >= len(bits):
            bits.append(0)

        if bits[idx] == 0:
            bits[idx] = 1
            count += 1

        idx += 1

    return bits + [0] * (n_modes - len(bits))


def _generate_sequential_state(n_modes: int, n_photons: int) -> list[int]:
    return [1 if idx < n_photons else 0 for idx in range(n_modes)]
