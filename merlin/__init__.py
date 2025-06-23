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
Merlin - Photonic Quantum Neural Networks for PyTorch

A comprehensive framework for integrating photonic quantum circuits
into PyTorch neural networks with automatic differentiation support.
"""

# Core API - Most users will only need these
from .core.layer import QuantumLayer
from .core.photonicbackend import PhotonicBackend
from .core.ansatz import Ansatz, AnsatzFactory

# Essential enums
from .core.generators import CircuitType, StatePattern
from .sampling.strategies import OutputMappingStrategy

# Advanced components (for power users)
from .core.generators import CircuitGenerator
from .core.generators import StateGenerator
from .torch_utils.torch_codes import FeatureEncoder
from .torch_utils.torch_codes import SamplingProcess
from .sampling.autodiff import AutoDiffProcess
from .sampling.mappers import OutputMapper, LexGroupingMapper, ModGroupingMapper
from .pcvl_pytorch import CircuitConverter, build_slos_distribution_computegraph

# Version and metadata
__version__ = "0.1.0"
__author__ = "Merlin Team"
__description__ = "Photonic Quantum Machine Learning Framework"

# Public API - what users see with `import merlin as ML`
__all__ = [
    # Core classes (most common usage)
    "QuantumLayer",
    "PhotonicBackend",
    "Ansatz",
    "AnsatzFactory",

    # Configuration enums
    "CircuitType",
    "StatePattern",
    "OutputMappingStrategy",

    # Advanced components
    "CircuitGenerator",
    "StateGenerator",
    "FeatureEncoder",
    "SamplingProcess",
    "AutoDiffProcess",
    "OutputMapper",
    "LexGroupingMapper",
    "ModGroupingMapper",

    "CircuitConverter",
    "build_slos_distribution_computegraph"
]