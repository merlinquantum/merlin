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

"""Core quantum layer components."""

from .base import AbstractComputationProcess
from .circuit import Circuit
from .components import BeamSplitter, Component, EntanglingBlock, Rotation
from .generators import CircuitGenerator, CircuitType, StateGenerator, StatePattern
from .photonicbackend import PhotonicBackend
from .process import ComputationProcess, ComputationProcessFactory


def __getattr__(name):
    """Lazy import for backward compatibility"""
    if name in ("Ansatz", "AnsatzFactory"):
        import warnings

        warnings.warn(
            f"Importing {name} from merlin.core is deprecated. "
            f"Use 'from merlin.builder.ansatz import {name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PhotonicBackend",
    "AbstractComputationProcess",
    "ComputationProcess",
    "ComputationProcessFactory",
    "CircuitType",
    "StatePattern",
    "CircuitGenerator",
    "StateGenerator",
    "Component",
    "Rotation",
    "BeamSplitter",
    "EntanglingBlock",
    "Circuit",
    "Ansatz",  # noqa: F401, F822
    "AnsatzFactory",  # noqa: F401, F822
]
