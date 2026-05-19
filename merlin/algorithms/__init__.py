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
Merlin algorithms package containing various quantum machine learning algorithms.
"""

from .feed_forward import FeedForwardBlock
from .feed_forward_legacy import (
    FeedForwardBlockLegacy,
    PoolingFeedForwardLegacy,
    create_circuit,
    define_layer_no_input,
    define_layer_with_input,
)
from .kernels import FeatureMap, FidelityKernel
from .layer import QuantumLayer
from .lora import QuantumAnsatz, QuantumLoRALayer, convert_to_quantum_lora
from .loss import NKernelAlignment

__all__ = [
    "FeatureMap",
    "FidelityKernel",
    "NKernelAlignment",
    "QuantumLayer",
    "QuantumLoRALayer",
    "QuantumAnsatz",
    "convert_to_quantum_lora",
    "FeedForwardBlock",
    "FeedForwardBlockLegacy",
    "PoolingFeedForwardLegacy",
    "create_circuit",
    "define_layer_no_input",
    "define_layer_with_input",
]
