.. _feedforward_block:

Feedforward Circuits
====================

Feedforward is a key capability in photonic quantum circuits, where a *partial
measurement* determines the configuration of the downstream circuit.
This mechanism is comparable to *dynamic circuits* in the gate-based model of
quantum computing (see `IBM Dynamic Circuits <https://quantum.cloud.ibm.com/docs/en/guides/classical-feedforward-and-control-flow>`_).

The main difference is in the physical implementation:

- **Gate-based circuits:** gates are applied consecutively, and adapting the circuit
  requires performing a measurement and determining follow-up gates *within the
  coherence time of the qubits* (typically ms–s).
- **Photonic circuits:** feedforward involves measuring some modes while the remaining
  modes travel through a *delay line*. The delay must be short enough to avoid photon
  loss, while still allowing the photonic chip to be reconfigured. Measurement and
  reconfiguration must therefore happen on *sub-microsecond timescales*.

FeedForwardBlock in MerLin
--------------------------

Modern MerLin versions model feedforward circuits via the
:class:`~merlin.algorithms.feed_forward.FeedForwardBlock` class.  Instead of
describing the block procedurally, you simply provide a complete
:class:`pcvl.Experiment` containing:

1. The unitary layers between measurements.
2. Explicit detector declarations (PNR, threshold, ...).
3. One or more :class:`pcvl.FFCircuitProvider`
   instances that describe how the circuit is reconfigured after the detectors fire.

``FeedForwardBlock`` parses the experiment, creates the appropriate
:class:`~merlin.algorithms.layer.QuantumLayer` objects for every stage, and runs
them sequentially.  Classical inputs (``input_parameters``) are only consumed by
the first stage; once the first measurement happens the remaining branches are
propagated in amplitude-encoding mode.

.. note::

   The current implementation expects noise-free experiments (``NoiseModel()``
   or ``None``). Adding detectors and feed-forward configurators to a noisy
   experiment is rejected during construction.

**Measurement strategy**

``measurement_strategy`` controls the classical view exposed by
:meth:`~merlin.algorithms.feed_forward.FeedForwardBlock.forward`:

* ``merlin.MeasurementStrategy.probs()`` (default): returns a tensor of shape
  ``(batch_size, len(output_keys))``. Each column already corresponds to the
  fully specified Fock state listed in
  :py:attr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_keys`.
* ``merlin.MeasurementStrategy.mode_expectations()``: returns a tensor of shape
  ``(batch_size, num_modes)`` containing the per-mode photon expectations
  aggregated across **all** measurement keys. The
  :py:attr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_keys` list is
  retained for metadata while
  :py:attr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_state_sizes`
  stores ``num_modes`` for each entry.
* ``merlin.MeasurementStrategy.amplitudes()``: list of tuples
  ``(measurement_key, branch_probability, remaining_photons, amplitudes)``
  describing the mixed state produced after every partial measurement.

For tensor outputs the attribute
:py:attr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_keys` lists the
measurement tuple corresponding to each column. ``merlin.MeasurementStrategy.probs()`` therefore
directly aligns with the dictionary keys, whereas ``merlin.MeasurementStrategy.mode_expectations()``
retains the key ordering purely as metadata because the returned tensor is
already aggregated across all outcomes.

API Reference
-------------

.. autoclass:: merlin.algorithms.feed_forward.FeedForwardBlock
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Example
-------

.. code-block:: python

   import torch
   import perceval as pcvl
   from merlin.algorithms import FeedForwardBlock
   from merlin.measurement.strategies import MeasurementStrategy

   # Build an experiment with one detector stage and two branches
   exp = pcvl.Experiment()
   exp.add(0, pcvl.Circuit(3) // pcvl.BS())
   exp.add(0, pcvl.Detector.pnr())

   reflective = pcvl.Circuit(2) // pcvl.PERM([1, 0])
   transmissive = pcvl.Circuit(2) // pcvl.BS()
   provider = pcvl.FFCircuitProvider(1, 0, reflective)
   provider.add_configuration([1], transmissive)
   exp.add(0, provider)

   block = FeedForwardBlock(
       exp,
       input_state=[2, 0, 0],
       trainable_parameters=["theta"],   # optional Perceval prefixes
       input_parameters=["phi"],         # classical inputs for the first unitary
       measurement_strategy=MeasurementStrategy.probs(),
   )

   x = torch.zeros((1, 1))               # only the first stage consumes features
   outputs = block(x)                    # tensor (batch, num_keys, dim)
   for idx, key in enumerate(block.output_keys):
       distribution = outputs[:, idx]    # probabilities for this measurement

When the experiment does not expose classical inputs you may call ``block()``
without passing a tensor (an empty feature tensor is injected automatically).

.. note::

   ``FeedForwardBlock(input_state=...)`` accepts Fock occupation lists,
   ``pcvl.BasicState``, ``pcvl.StateVector``, or
   :class:`~merlin.core.state_vector.StateVector`. Raw ``torch.Tensor`` values
   are not accepted as ``input_state``; wrap amplitude tensors with
   :meth:`~merlin.core.state_vector.StateVector.from_tensor` first.

Known Limitation: Input Parameters Used Only Inside a Branch
--------------------------------------------------------------

.. note::

   This is a known limitation of ``FeedForwardBlock`` in MerLin 0.4, tracked in
   `issue #274 <https://github.com/merlinquantum/merlin/issues/274>`_. It is
   expected to be solved in **MerLin 0.5** with a new ``FeedForwardBlock`` backend.
   Until then, we propose the manual workaround below.

``FeedForwardBlock`` currently requires that every entry of ``input_parameters`` be
consumed by the **first** pre-measurement stage of the experiment. If a classical
parameter is only referenced inside one or more ``pcvl.FFCircuitProvider`` branch
configurations — i.e. it encodes a value *after* the measurement, in a specific
branch — construction fails immediately with:

.. code-block:: text

   ValueError: The first stage must use all of the input parameters. Create you own
   stages with variable input parameters with the partial measurement strategy instead

For example, the following experiment is rejected because ``x`` is only used inside the
branch circuits of the ``FFCircuitProvider``, not in the prefix unitary:

.. code-block:: python

   x = pcvl.P("x")
   provider = pcvl.FFCircuitProvider(1, 0, pcvl.Circuit(2))
   provider.add_configuration([0], pcvl.BS(x) // pcvl.BS(pcvl.P("A")))
   provider.add_configuration([1], pcvl.BS(x) // pcvl.BS(pcvl.P("B")))
   experiment.add(0, provider)

   FeedForwardBlock(
       experiment,
       trainable_parameters=["A", "B"],
       input_parameters=["x"],   # raises ValueError: "x" is not used in the first stage
   )

**Workaround: build the stages manually with partial measurement**

Until MerLin 0.5 ships, the same physical workflow can be reproduced by chaining
:class:`~merlin.algorithms.layer.QuantumLayer` instances yourself with the
:doc:`partial measurement strategy </quantum_expert_area/partial_measurement>`:

1. Run the prefix circuit with ``measurement_strategy=MeasurementStrategy.partial(...)``.
2. Inspect the returned :class:`~merlin.core.partial_measurement.PartialMeasurement`;
   it exposes one branch per possible detector outcome.
3. For each branch, build a downstream ``QuantumLayer`` using ``branch.amplitudes``
   (the conditional ``StateVector`` on the unmeasured modes) as ``input_state``, and
   declare whichever branch-local ``trainable_parameters`` / ``input_parameters`` that
   branch needs.
4. Feed the branch-local classical inputs to that downstream layer, and weight its
   output probabilities by ``branch.probability``.

.. code-block:: python

   import torch
   import perceval as pcvl
   from perceval import Circuit
   from merlin.algorithms.layer import QuantumLayer
   from merlin.core.computation_space import ComputationSpace
   from merlin.measurement.strategies import MeasurementStrategy

   input_state = [1, 1, 0]
   prefix = Circuit(3) // pcvl.Unitary.random(3)

   # Branch-specific circuits, each with its own local parameters.
   branch_circuits = {
       (0,): (pcvl.BS(pcvl.P("x")) // pcvl.BS(pcvl.P("A")), ["A"], ["x"]),
       (1,): (pcvl.BS(pcvl.P("x")) // pcvl.BS(pcvl.P("B")), ["B"], ["x"]),
       (2,): (Circuit(2), [], []),
   }

   # 1. Run the prefix stage and measure mode 0 only.
   partial_layer = QuantumLayer(
       circuit=prefix,
       input_state=input_state,
       measurement_strategy=MeasurementStrategy.partial(
           modes=[0],
           computation_space=ComputationSpace.FOCK,
       ),
   )
   partial_measurement = partial_layer()

   x = torch.tensor([[0.2]])  # branch-local classical input

   probabilities = {}
   for branch in partial_measurement.branches:
       circuit, trainable_parameters, input_parameters = branch_circuits[branch.outcome]

       # 2-3. Route the conditional state into the branch-specific layer.
       branch_layer = QuantumLayer(
           circuit=circuit,
           input_state=branch.amplitudes,          # conditional StateVector
           trainable_parameters=trainable_parameters,
           input_parameters=input_parameters,
           measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
       )

       # 4. Feed x only to branches that actually use it, then weight by branch.probability.
       branch_output = (
           branch_layer(x).squeeze(0) if input_parameters else branch_layer().squeeze(0)
       )
       for index, remaining_key in enumerate(branch_layer.output_keys):
           output_key = (branch.outcome[0], *remaining_key)
           probabilities[output_key] = branch.probability.squeeze(0) * branch_output[index]

This manual composition is exactly what ``FeedForwardBlock`` does internally for the
supported case (branch-local inputs aside), so it reproduces the same probabilities
while additionally allowing branch-local classical inputs.

Further Reading
---------------
- :doc:`/quantum_expert_area/internal_design`
- For circuit specific optimizations: :doc:`/quantum_expert_area/building_intuition`
- Output mapping strategies: :doc:`/user_guide/grouping`
