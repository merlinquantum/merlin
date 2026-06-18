.. _user_guide_migration_guide:

Migration guide
===============

Migrating from ``no_bunching`` (deprecated)
-------------------------------------------

.. warning:: *Deprecated since version 0.3:*
   ``no_bunching`` is deprecated and is removed since version 0.3.0. Use
   ``computation_space`` instead inside the chosen ``measurement_strategy``. 
   See this migration section for the mapping.
   
The ``no_bunching`` flag is deprecated. 

If you are using a ``QuantumLayer`` and you need to control how Fock states are
truncated or encoded, define the ``computation_space`` inside the ``measurement_strategy``
instead. 

If you are using a Kernel, define the encoding or truncation of the Fock states in the
``computation_space`` parameter.


Map the legacy intent as follows:

- ``no_bunching=False`` → ``computation_space=ComputationSpace.FOCK`` (full Fock space)
- ``no_bunching=True`` → ``computation_space=ComputationSpace.UNBUNCHED`` (one photon per mode)
- Dual-rail encodings → ``computation_space=ComputationSpace.DUAL_RAIL``

This keeps measurement strategy selection orthogonal to simulation space configuration.

Migrating from legacy ``MeasurementStrategy``
---------------------------------------------

.. warning:: *Deprecated since version 0.3:*
   Enum-style and string access is deprecated and will be removed from `MeasurementStrategy` in v0.4.
   Use the new factory methods instead: ``MeasurementStrategy.probs(computation_space=...)``, ``MeasurementStrategy.mode_expectations(computation_space=...)``, ``MeasurementStrategy.amplitudes(computation_space=...)``.
   See this migration section for the mapping.

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
