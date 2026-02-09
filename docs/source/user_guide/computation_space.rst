==================
Computation Spaces
==================

Photonics lets us choose *where* a computation takes place.  In Merlin this notion is
captured by the **computation space**, which determines the subset of the photonic Fock
space we simulate and therefore the type of detectors, encodings, and post-selection
strategies the model expects.  Selecting the right space is often the difference between
an experiment that is physically feasible and one that only works in theory.

Why computation space matters
-----------------------------

* **Detector requirements** – Full Fock space simulations assume photon-number-resolving
  (PNR) detectors.  Restricting the space to unbunched states tolerates threshold
  detectors that only count photons per mode.
* **Loss robustness** – Post-selecting on lossless states (e.g. unbunched or dual-rail
  configurations) narrows the state space in exchange for higher fidelity.
* **Sampling rate** – The narrower the computation space, the fewer simulated shots are
  discarded.  Logical encodings such as dual-rail sacrifice throughput, while unbunched
  spaces stay efficient when the number of modes is high compared to the number of
  photons.
* **Hybrid interoperability** – Matching a photonic computation space with a qubit-based
  model (for example from PennyLane) allows seamless hybrid training through
  :class:`~merlin.bridge.QuantumBridge`.

Core computation spaces
-----------------------

Merlin exposes three common working regimes; each is a subspace of the full Fock space.

.. list-table::
   :header-rows: 1
   :widths: 18 30 32 20

   * - Space
     - Description
     - Typical use case
     - Detector model
   * - Full Fock
     - All photon-number configurations over ``m`` modes.  Supports bunching and loss.
     - High-fidelity simulations and comparisons with analytic solutions.
     - PNR detectors
   * - Unbunched
     - Restricts to configurations with at most one photon per mode.
     - Circuits read with threshold detectors or when loss resilience is required.
     - Threshold detectors
   * - Dual-rail
     - Special case of the unbunched space: one photon shared across every pair of modes.
     - Logical qubit encodings, qubit ↔ photonic interfacing.
     - Threshold detectors

Configuring the computation space
---------------------------------

The :class:`~merlin.algorithms.layer.QuantumLayer` configures its computation space at
construction time. 

The ``measurement_strategy`` can define the computation space with its ``computation_space`` argument.
We can choose from 
  - ``merlin.ComputationSpace.UNBUNCHED``, do not allow multiple photons per modes.
  - ``merlin.ComputationSpace.FOCK``, allow multiple photons per modes (i.e. explore the full Fock space).
  - ``merlin.ComputationSpace.DUAL_RAIL``, use a dual rail encoding (two modes per photon).

Those computation spaces can also be assigned with the ``computation_space`` argument in the constructor but, it
is prefered to exploit the ``measurement_strategy`` since ``computation_space`` will be deprecated in the future.

It will be the only way to control the computation space as the ``no_bunching`` flag is deprecated.
.. deprecated:: 0.4
   The use of the ``no_bunching`` flag  is deprecated and will be removed in version 0.4.
   Use the ``computation_space`` flag inside ``measurement_strategy`` instead. See :doc:`/user_guide/migration_guide`.

Another parameter is also relevent.
``index_photons``
    Optional per-photon constraints on allowed modes.  This lets you carve out logical
    subspaces such as dual-rail groupings without rebuilding the circuit.

Internally the :class:`~merlin.core.process.ComputationProcessFactory` uses these 
to build the correct :mod:`perceval` simulation graph.  The same options propagate
through factory-created ansätze and algorithm builders.

Working with QLOQ encodings
---------------------------

Beyond dual-rail, Merlin supports **Qubit Logic on Qudits (QLOQ)** encodings where each
logical group of ``k_i`` qubits is mapped to a single photon living in a block of
``2^{k_i}`` modes:

.. math::

   \text{qubit_groups} = [k_1, k_2, \ldots, k_j], \qquad \sum_i k_i = n_\text{qubits}

This layout is especially useful when bridging matter-based qubits with flying photonic
qubits: you keep a qubit-based tensor on the ML side while the photonic circuit only
sees the corresponding one-photon-per-group states.

Example: three logical qubits encoded as ``[2, 1]`` means one photon is delocalised over
four modes (representing the two-qubit block) and another photon spans a two-mode
dual-rail pair.

Example setup
-------------

.. code-block:: python

   import perceval as pcvl
   import torch
   from merlin import MeasurementStrategy,ComputationSpace QuantumLayer

   # 3 logical qubits: first two qubits in a 4-mode block, third qubit dual-railed
   qubit_groups = [2, 1]
   n_photons = len(qubit_groups)
   n_modes = sum(2**k for k in qubit_groups)  # 6 modes

   circuit = pcvl.Circuit(n_modes)
   layer = QuantumLayer(
       input_size=0,
       circuit=circuit,
       n_photons=n_photons,
       measurement_strategy=MeasurementStrategy.probs(computation_space=ComputationSpace.UNBUNCHED), # stay inside the unbunched/QLOQ space
       dtype=torch.float32,
   )

   # Later you would feed a superposition state (or a QuantumBridge payload) into layer(...)

Bridging qubit logic and photonics
----------------------------------

The :class:`~merlin.bridge.QuantumBridge` is the companion to computation spaces.  It
turns a qubit statevector ``ψ ∈ ℂ^{2^n}`` into the appropriate photonic superposition by
grouping qubits according to ``qubit_groups``.  Its ``computation_space`` argument (``fock``,
``unbunched``, or ``dual_rail``) determines both the size of the emitted tensor and the
ordering, matching Merlin's internal combinadic conventions.  The resulting amplitudes
can be injected directly into :class:`~merlin.algorithms.layer.QuantumLayer`.

By pairing a PennyLane (or any gate-based) module, the QuantumBridge, and a configured
QuantumLayer in ``nn.Sequential``, you can train hybrid models end-to-end while staying
in the computation space that matches your detector and hardware constraints.

Takeaways
---------

* Pick the computation space that matches your hardware and loss profile.
* Use ``ComputationSpace.UNBUNCHED``/ ``index_photons`` to constrain Merlin's simulation graph without
  rewriting circuits.
* QLOQ encodings generalise dual-rail: group as many qubits as you want under a single
  photon.
* The QuantumBridge provides the “quantum bridge” between qubit logic encodings and the
  chosen photonic computation space, enabling plug-and-play hybrid networks.
