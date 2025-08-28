"""Encoding module for quantum data encoding.

Provides flexible strategies for encoding classical data into quantum circuits.
"""

from .strategies import (
    EncodingStrategy,
    FourierEncoding,
    DataReuploadingEncoding,
    AmplitudeEncoding,
    HybridEncoding
)

from .input_states import (
    StatePattern,
    InputStateGenerator,
    AdaptiveStatePreparation
)

from .circuits import (
    EncodingCircuitBuilder,
    QuantumFeatureMap
)

__all__ = [
    # Strategies
    'EncodingStrategy',
    'FourierEncoding',
    'DataReuploadingEncoding',
    'AmplitudeEncoding',
    'HybridEncoding',

    # Input states
    'StatePattern',
    'InputStateGenerator',
    'AdaptiveStatePreparation',

    # Circuit builders
    'EncodingCircuitBuilder',
    'QuantumFeatureMap'
]
