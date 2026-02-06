.. _feedforward_block:

Feedforward Circuits
====================

Feedforward is a key capability in photonic quantum circuits, where a *partial
measurement* determines the configuration of the downstream circuit.
This mechanism is comparable to *dynamic circuits* in the gate-based model of
quantum computing (see `IBM Dynamic Circuits <https://research.ibm.com/blog/dynamic-circuits>`_).

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
:class:`perceval.Experiment` containing:

1. The unitary layers between measurements.
2. Explicit detector declarations (PNR, threshold, ...).
3. One or more :class:`perceval.components.feed_forward_configurator.FFCircuitProvider`
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
  :pyattr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_keys`.
* ``merlin.MeasurementStrategy.mode_expectations()``: returns a tensor of shape
  ``(batch_size, num_modes)`` containing the per-mode photon expectations
  aggregated across **all** measurement keys. The
  :pyattr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_keys` list is
  retained for metadata while
  :pyattr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_state_sizes`
  stores ``num_modes`` for each entry.
* ``merlin.MeasurementStrategy.amplitudes()``: list of tuples
  ``(measurement_key, branch_probability, remaining_photons, amplitudes)``
  describing the mixed state produced after every partial measurement.

For tensor outputs the attribute
:pyattr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_keys` lists the
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


Further Reading
---------------

- :ref:`internal_design`
- :ref:`circuit_specific_optimizations`
- :ref:`output_mappings`