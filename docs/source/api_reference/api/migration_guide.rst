Migrating from ``no_bunching`` (deprecated)
==========================================

.. deprecated:: 0.4
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