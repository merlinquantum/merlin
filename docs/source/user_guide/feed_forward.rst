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

   import perceval as pcvl
   from perceval import BasicState, Circuit
   from merlin.algorithms.feed_forward import FeedForwardBlock

   # Build a 4-mode experiment with a prefix unitary and measurement in mode 0
   experiment = pcvl.Experiment(4)
   experiment.add(0, Circuit(4) // pcvl.Unitary.random(4))  # prefix unitary
   experiment.add(0, pcvl.Detector.pnr())  # measure mode 0

   # Branch-local parameter "x" is only defined inside branch circuits
   provider = pcvl.FFCircuitProvider(1, 0, Circuit(3))
   provider.add_configuration([0], pcvl.BS(pcvl.P("x")) // pcvl.BS(pcvl.P("A")))
   provider.add_configuration([1], pcvl.BS(pcvl.P("x")) // pcvl.BS(pcvl.P("B")))
   experiment.add(0, provider)

   # FeedForwardBlock rejects this: "x" is not consumed by the first (prefix) stage
   FeedForwardBlock(
       experiment,
       input_state=BasicState([1, 1, 0, 0]),
       trainable_parameters=["A", "B"],
       input_parameters=["x"],   # raises ValueError: "x" is not used in the first stage
   )

**Workaround: build the stages manually with partial measurement**

Until MerLin 0.5 ships, the same physical workflow can be reproduced by chaining
:class:`~merlin.algorithms.layer.QuantumLayer` instances yourself with the
:doc:`partial measurement strategy </quantum_expert_area/partial_measurement>`.
Critically, to make the model trainable, you must construct layers **once** in
an ``nn.Module.__init__``, then call :meth:`~merlin.algorithms.layer.QuantumLayer.set_input_state`
at runtime to route conditional amplitudes through them. This is the only safe way
to preserve trainable parameters across training steps.

The pattern closely mirrors what ``FeedForwardBlock`` does internally for its
supported case (branch-local inputs aside). However, this workaround covers a
**single feedforward stage**; multi-stage experiments require nesting the same
pattern per stage.

Key implementation details:

- **Layer construction:** Build the prefix ``QuantumLayer`` and one branch layer
  per outcome in ``__init__``, stored in an ``nn.ModuleDict`` (or similar container).
- **Runtime routing:** In ``forward()``, call ``set_input_state(branch.amplitudes)``
  on each branch layer to inject the conditional measurement outcome.
- **Batch handling:** The prefix layer is called once with no features (produces
  batch size 1), while branch layers receive input with shape ``(batch_size, ...)``.
  Probabilities must be broadcast correctly: use ``unsqueeze(-1)`` or indexing to
  ensure branch and conditional probabilities are multiplied element-wise.

.. code-block:: python

   import torch
   import torch.nn as nn
   import perceval as pcvl
   from perceval import Circuit
   from merlin.algorithms.layer import QuantumLayer
   from merlin.core.computation_space import ComputationSpace
   from merlin.measurement.strategies import MeasurementStrategy

   class BranchFeedforward(nn.Module):
       """Feedforward model with branch-local parameters, trainable via gradient descent.

       **Note:** This example assumes a single measured mode (e.g., measured_modes=[0]).
       For arbitrary measured modes, adapt the key reconstruction accordingly.
       """

       def __init__(self, input_state, prefix_circuit):
           super().__init__()

           # Build the prefix stage (partial measurement after mode 0).
           self.partial_layer = QuantumLayer(
               circuit=prefix_circuit,
               input_state=input_state,
               measurement_strategy=MeasurementStrategy.partial(
                   modes=[0],
                   computation_space=ComputationSpace.FOCK,
               ),
           )

           # Pre-build branch layers for each outcome, keyed by measurement outcome.
           # Initialize with input_size (classical) and n_photons (quantum dimension).
           # A branch layer uses the photon count remaining after its measured
           # outcome, matching the routed conditional state dimension.
           self.branch_layers = nn.ModuleDict()

           # Build 2-mode branch circuits (remaining modes after measuring mode 0).
           c0 = Circuit(2)
           c0.add(0, pcvl.BS(pcvl.P("x")))
           c0.add(0, pcvl.BS(pcvl.P("A")))

           c1 = Circuit(2)
           c1.add(0, pcvl.BS(pcvl.P("x")))
           c1.add(0, pcvl.BS(pcvl.P("B")))

           c2 = Circuit(2)  # vacuum branch after measuring two photons

           branch_configs = {
               (0,): (c0, ["A"], ["x"]),
               (1,): (c1, ["B"], ["x"]),
               (2,): (c2, [], []),
           }

           for outcome, (circuit, trainable_params, input_params) in branch_configs.items():
               key = str(outcome)
               if sum(input_state) - sum(outcome) == 0:
                   continue
               self.branch_layers[key] = QuantumLayer(
                   circuit=circuit,
                   input_size=len(input_params),  # Classical input dimension
                   n_photons=sum(input_state) - sum(outcome),
                   trainable_parameters=trainable_params,
                   input_parameters=input_params,
                   measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
               )

       def forward(self, x=None):
           # Run the prefix stage to get partial measurement branches.
           # Note: called with no features; produces batch size 1.
           partial_measurement = self.partial_layer()

           probabilities = {}
           for branch in partial_measurement.branches:
               key = str(branch.outcome)
               if sum(branch.outcome) == sum(input_state):
                   probabilities[(branch.outcome[0], 0, 0)] = branch.probability
                   continue
               branch_layer = self.branch_layers[key]

               # Set the conditional amplitudes for this branch.
               # set_input_state() is the only way to route a StateVector through
               # a layer that was already constructed with trainable parameters.
               branch_layer.set_input_state(branch.amplitudes)

               # Execute the branch layer with its local classical inputs (if any).
               branch_config = {
                   (0,): ["x"],
                   (1,): ["x"],
               }
               input_params = branch_config[branch.outcome]
               if input_params:
                   branch_output = branch_layer(x)  # shape: (batch_size, n_keys)
               else:
                   branch_output = branch_layer()  # shape: (batch_size, n_keys)

               # Weight by branch probability with proper broadcasting.
               # branch.probability is (batch_size,), branch_output is (batch_size, n_keys).
               # unsqueeze(-1) broadcasts branch probability to (batch_size, 1).
               branch_prob_weighted = branch.probability.unsqueeze(-1)  # (batch_size, 1)

               for index, remaining_key in enumerate(branch_layer.output_keys):
                   # Construct full output key: (measured_outcome, *remaining_outcomes)
                   # For measured_modes=[0], this places the measurement outcome first.
                   output_key = (branch.outcome[0], *remaining_key)
                   branch_probs_for_key = branch_output[:, index]  # (batch_size,)
                   weighted_probs = branch_prob_weighted.squeeze(-1) * branch_probs_for_key
                   probabilities[output_key] = weighted_probs

           return probabilities

   # Usage
   input_state = [1, 1, 0]
   prefix = Circuit(3) // pcvl.Unitary.random(3)
   model = BranchFeedforward(input_state, prefix)

   # Single batch
   x = torch.tensor([[0.2]])  # (batch_size=1, input_dim=1)
   probs_single = model(x)

   # Multi-batch (batch_size=3)
   x = torch.tensor([[0.1], [0.2], [0.3]])  # (batch_size=3, input_dim=1)
   probs_multi = model(x)  # each probability has shape (3,)

   # Verify probabilities sum to 1 per batch element
   for i in range(x.shape[0]):
       batch_total = sum(p[i].item() for p in probs_multi.values())
       assert abs(batch_total - 1.0) < 1e-5

   # Now the model is trainable:
   optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
   # ... training loop ...

This manual composition reproduces the same probabilities as ``FeedForwardBlock``
(for single-stage experiments) while allowing branch-local classical inputs and
preserving trainable parameters that persist across calls.

.. note::

   The pattern is verified by the test suite in
   ``tests/algorithms/test_feedforward_manual_workaround.py``, which covers
   single-batch, multi-batch, trainability, and error cases. When generalizing
   to multiple measured modes (e.g., ``measured_modes=[0, 1]``), ensure key
   reconstruction uses the correct measured modes and remaining modes.

Further Reading
---------------
- :doc:`/quantum_expert_area/internal_design`
- For circuit specific optimizations: :doc:`/quantum_expert_area/building_intuition`
- Output mapping strategies: :doc:`/user_guide/grouping`
