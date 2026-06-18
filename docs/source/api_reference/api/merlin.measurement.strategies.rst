merlin.measurement.strategies module
====================================

.. automodule:: merlin.measurement.strategies
   :no-members:

.. currentmodule:: merlin.measurement.strategies

Overview
--------

The measurement API is now centered on :class:`MeasurementStrategy`, an immutable object
that fully describes how quantum outputs are converted into classical outputs. The new
API uses explicit factory methods instead of enum-style access.

New API (recommended)
---------------------

Factory methods (preferred):

.. code-block:: python

   from merlin.core.computation_space import ComputationSpace
   from merlin.measurement.strategies import MeasurementStrategy

   # Full measurement probabilities
   strategy = MeasurementStrategy.probs(ComputationSpace.FOCK)

   # Per-mode expectations
   strategy = MeasurementStrategy.mode_expectations(ComputationSpace.UNBUNCHED)

   # Raw amplitudes (no detectors, no sampling)
   strategy = MeasurementStrategy.amplitudes(ComputationSpace.UNBUNCHED)

Partial measurement
-------------------

Partial measurement is explicit and validated. It returns a
:class:`~merlin.core.partial_measurement.PartialMeasurement`.

.. code-block:: python

   from merlin.core.computation_space import ComputationSpace
   from merlin.measurement.strategies import MeasurementStrategy

   # Measure only modes 0 and 2
   strategy = MeasurementStrategy.partial(
       modes=[0, 2],
       computation_space=ComputationSpace.FOCK,
   )

Deprecations and Migration Guide
--------------------------------

Enum-style access is deprecated and will be removed in v0.4:

- ``MeasurementStrategy.PROBABILITIES``
- ``MeasurementStrategy.MODE_EXPECTATIONS``
- ``MeasurementStrategy.AMPLITUDES``
- ``MeasurementStrategy.NONE`` (aliases amplitudes)

Passing ``measurement_strategy`` as a string is also deprecated (e.g. ``"PROBABILITIES"``).

Old to new mappings
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Deprecated
     - Recommended replacement
   * - ``MeasurementStrategy.PROBABILITIES``
     - ``MeasurementStrategy.probs(computation_space=...)``
   * - ``MeasurementStrategy.MODE_EXPECTATIONS``
     - ``MeasurementStrategy.mode_expectations(computation_space=...)``
   * - ``MeasurementStrategy.AMPLITUDES``
     - ``MeasurementStrategy.amplitudes(computation_space=...)``
   * - ``MeasurementStrategy.NONE``
     - ``MeasurementStrategy.amplitudes(computation_space=...)``
   * - ``"PROBABILITIES"`` (string)
     - ``MeasurementStrategy.probs(computation_space=...)``

Computation space now lives inside the strategy. If you already use the new factory
methods, do not also pass ``computation_space`` separately in constructors such as
``QuantumLayer``.

.. code-block:: python

   # Deprecated (legacy enum + separate computation_space)
   # QuantumLayer(..., measurement_strategy=MeasurementStrategy.PROBABILITIES,
   #             computation_space=ComputationSpace.FOCK)

   # Recommended
   QuantumLayer(..., measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK))

Reference
---------

.. autoclass:: BaseMeasurementStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DistributionStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ProbabilitiesStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ModeExpectationsStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AmplitudesStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PartialMeasurementStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MeasurementKind
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MeasurementStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autodata:: MeasurementStrategyLike

.. autofunction:: resolve_measurement_strategy
