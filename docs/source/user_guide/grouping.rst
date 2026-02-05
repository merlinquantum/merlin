:github_url: https://github.com/merlinquantum/merlin

==============
Grouping Guide
==============

Merlin now exposes quantum-to-classical conversion through two orthogonal concepts:

- :class:`~merlin.measurement.strategies.MeasurementStrategy` selects how results are extracted from the quantum simulation or hardware backend (see :doc:`./measurement_strategy`).
- :class:`~merlin.utils.grouping.LexGrouping` and :class:`~merlin.utils.grouping.ModGrouping` provide optional post-processing of outputs.

Grouping modules are often used to reshape a torch.tensor into smaller feature sets while preserving differentiability.

LexGrouping
-----------

Groups consecutive values into equally sized buckets. Padding with zeros ensures all buckets have the same width.

.. code-block:: python

    grouped = nn.Sequential(
        quantum_layer,
        ML.LexGrouping(input_size=quantum_layer.output_size, output_size=8),
    )

Useful when the order of Fock states carries meaning (e.g., lexicographic encoding). The module preserves probability mass and supports batched inputs.

Example (single vector)::

    >>> p = torch.tensor([0.1, 0.2, 0.4, 0.3])
    >>> mapper = ML.LexGrouping(input_size=4, output_size=2)
    >>> mapper(p)
    tensor([0.3000, 0.7000])

ModGrouping
-----------

Sums values sharing the same index modulo ``output_size``. When ``output_size`` exceeds ``input_size``, the layer pads with zeros.

.. code-block:: python

    grouped = nn.Sequential(
        quantum_layer,
        ML.ModGrouping(input_size=quantum_layer.output_size, output_size=8),
    )

This is effective for cyclic structures (e.g., periodic sensors) where indices wrapping around the distribution should be combined.

Example (single vector)::

    >>> p = torch.tensor([0.1, 0.2, 0.3, 0.4])
    >>> mapper = ML.ModGrouping(input_size=4, output_size=2)
    >>> mapper(p)
    tensor([0.4000, 0.6000])

Chaining Measurement Strategy and Grouping
------------------------------------------

.. code-block:: python

    import torch.nn as nn

    builder = CircuitBuilder(n_modes=4)
    # Add whatever to your builder

    quantum_layer = QuantumLayer(.
        input_size=2,
        builder=builder,
        n_photons=2,
        measurement_strategy=MeasurementStrategy.probs(),
    )

    quantum_pipeline = nn.Sequential(
        quantum_layer,
        ML.ModGrouping(input_size=quantum_layer.output_size, output_size=3),
        nn.Linear(3, 1),
    )