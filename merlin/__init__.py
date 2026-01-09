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
from .algorithms.feed_forward import FeedForwardBlock
from .algorithms.feed_forward_legacy import (
    FeedForwardBlockLegacy,
    PoolingFeedForwardLegacy,
    create_circuit,
    define_layer_no_input,
    define_layer_with_input,
)
from .algorithms.kernels import FeatureMap, FidelityKernel
from .algorithms.layer import QuantumLayer
from .algorithms.loss import NKernelAlignment
from .bridge.quantum_bridge import QuantumBridge
from .builder.circuit_builder import CircuitBuilder

# Essential enums
# Advanced components (for power users)
from .core.computation_space import ComputationSpace
from .core.merlin_processor import MerlinProcessor
from .core.process import ComputationProcess
from .core.state import StatePattern, generate_state
from .measurement import (
    Amplitudes,
    DetectorTransform,
    ModeExpectations,
    OutputMapper,
    Probabilities,
    resolve_detectors,
)
from .measurement.autodiff import AutoDiffProcess
from .measurement.process import SamplingProcess
from .measurement.strategies import MeasurementStrategy
from .pcvl_pytorch import CircuitConverter, build_slos_distribution_computegraph
from .utils.combinadics import Combinadics
from .utils.grouping import LexGrouping, ModGrouping

# Version and metadata
__version__ = "0.2.3"
__author__ = "Merlin Team"
__description__ = "Photonic Quantum Machine Learning Framework"

# Public API - what users see with `import merlin as ML`
__all__ = [
    # Core classes (most common usage)
    "QuantumLayer",
    "QuantumBridge",
    # Configuration enums
    "ComputationSpace",
    "MeasurementStrategy",
    "Combinadics",
    # Advanced components
    "ComputationProcess",
    "StatePattern",
    "generate_state",
    "OutputMapper",
    "Probabilities",
    "DetectorTransform",
    "resolve_detectors",
    "ModeExpectations",
    "MerlinProcessor",
    "Amplitudes",
    "LexGrouping",
    "ModGrouping",
    "CircuitConverter",
    "build_slos_distribution_computegraph",
    "NKernelAlignment",
    "FeatureMap",
    "FidelityKernel",
    "FeedForwardBlock",
    "FeedForwardBlockLegacy",
    "PoolingFeedForwardLegacy",
    "create_circuit",
    "define_layer_no_input",
    "define_layer_with_input",
    "CircuitBuilder",
    "AutoDiffProcess",
    "SamplingProcess",
]
