merlin.core package
===================

.. automodule:: merlin.core
   :no-members:

Modules
-------

.. list-table::
   :widths: 35 65
   :header-rows: 0

   * - :doc:`merlin.core.base`
     - Abstract computation-process interfaces that define the low-level execution contract.
   * - :doc:`merlin.core.circuit`
     - The canonical :class:`~merlin.core.circuit.Circuit` container used across Merlin builders and processors.
   * - :doc:`merlin.core.components`
     - Public circuit-component dataclasses such as rotations, beam splitters, and interferometers.
   * - :doc:`merlin.core.computation_space`
     - Computation-space enums and coercion helpers for Fock and related bases.
   * - :doc:`merlin.core.execution`
     - Batch chunking and remote job execution units used by Merlin processors.
   * - :doc:`merlin.core.encoding_space`
     - Input encoding definitions and logical-to-Fock mapping helpers.
   * - :doc:`merlin.core.generators`
     - Legacy state and circuit generator helpers kept for compatibility.
   * - :doc:`merlin.core.partial_measurement`
     - Partial-measurement result objects, branches, and detector-output conversion aliases.
   * - :doc:`merlin.core.photonicbackend`
     - Photonic backend abstractions and backend-facing interfaces.
   * - :doc:`merlin.core.probability_distribution`
     - Probability-distribution objects, basis metadata, and conversion constructors.
   * - :doc:`merlin.core.process`
     - Computation processes and factory logic used to execute Merlin circuits.
   * - :doc:`merlin.core.merlin_processor`
     - The high-level :class:`~merlin.core.merlin_processor.MerlinProcessor` remote/local execution interface.
   * - :doc:`merlin.core.perceval_adapter`
     - Compatibility adapter and normalized state objects for Perceval backends.
   * - :doc:`merlin.core.sectored_distribution`
     - Data classes used for g2 simulation outputs.
   * - :doc:`merlin.core.state`
     - State-pattern helpers and input-state generation utilities.
   * - :doc:`merlin.core.state_vector`
     - The public :class:`~merlin.core.state_vector.StateVector` object and related conversions.

.. toctree::
   :hidden:

   merlin.core.base
   merlin.core.circuit
   merlin.core.components
   merlin.core.computation_space
   merlin.core.execution
   merlin.core.encoding_space
   merlin.core.generators
   merlin.core.partial_measurement
   merlin.core.photonicbackend
   merlin.core.probability_distribution
   merlin.core.process
   merlin.core.merlin_processor
   merlin.core.perceval_adapter
   merlin.core.sectored_distribution
   merlin.core.state
   merlin.core.state_vector
