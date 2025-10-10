"""
Merlin algorithms package containing various quantum machine learning algorithms.
"""

from .feed_forward import FeedForwardBlock
from .kernels import FeatureMap, FidelityKernel
from .layer import QuantumLayer
from .loss import NKernelAlignment

__all__ = [
    "FeatureMap",
    "FidelityKernel",
    "NKernelAlignment",
    "QuantumLayer",
    "FeedForwardBlock",
]
