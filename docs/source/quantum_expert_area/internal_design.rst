:github_url: https://github.com/merlinquantum/merlin

========================
Internal Design Overview
========================

This page provides an overview of key internal design elements that underpin
MerLin’s simulation pipeline and advanced workflows. It covers core pieces such
as the **partial** :class:`~merlin.measurement.detectors.DetectorTransform`, the
experimental :class:`~merlin.algorithms.feed_forward.FeedForwardBlock`, and the
SLOS TorchScript simulation helpers that execute photonic circuit evolution
efficiently in both angle- and amplitude-encoding regimes.

Partial DetectorTransform
=========================

MerLin’s detector transform normally converts a complete Fock probability
distribution into the classical detector outcomes dictated by the detector
model.  That path assumes *all* modes are detected simultaneously and it
operates on **real-valued probability tensors**.

`DetectorTransform` also supports a *partial measurement* mode, enabled by
passing ``partial_measurement=True``. In this configuration:

* You may pass ``None`` for the detectors attached to unmeasured modes.  Those
  modes remain quantum and their amplitudes are preserved.
* The forward pass now expects **complex-valued amplitude tensors** (instead of
  probabilities) and returns, for each measurement branch, the normalized
  amplitudes that correspond to the still-active modes.
* Internally the transform enumerates all measurement outcomes for the measured
  subset of modes, reweights the amplitudes by the corresponding detection
  probabilities, and yields:

  .. code-block:: python

      [
          {
              measurement_key: [
                  (probabilities, normalized_amplitudes),
                  ...
              ],
              ...
          },
          ...
      ]

  The outer list is indexed by the number of remaining photons. Each dictionary
  entry contains every measurement branch for that photon count. Each branch
  stores the accumulated probability weight and the normalized amplitudes for
  the unmeasured modes.

This partial interface is the backbone of feed-forward simulation where only a
subset of modes is observed at each stage and the remaining modes must be
propagated through additional circuits.

FeedForwardBlock
=================

``FeedForwardBlock`` is a new experimental block that consumes a full Perceval
experiment containing detectors and one or more
:class:`perceval.components.feed_forward_configurator.FFCircuitProvider`
instances. The block parses the experiment into a sequence of *stages*:

1. A unitary prefix (collapsed into a single :class:`~merlin.algorithms.layer.QuantumLayer`).
2. The detector set for the stage.
3. The matching feed-forward configurator that decides which circuit to insert
   based on the detector outcome.

The parser records these stages as ``FFStage`` instances. Each stage stores

* the collapsed unitary operating on the currently *active* optical modes,
* the tuple of global mode indices that remain active at the beginning of the stage,
* the subset of those modes that are measured, and
* the detector objects and ``FFCircuitProvider`` attached to the stage.

After a detector stage finishes, the measured modes are removed from the active
set so that every subsequent circuit is expressed solely in terms of the
remaining modes. This remapping guarantees that the :class:`QuantumLayer`
constructed for the stage matches the reduced dimensionality, while the
original mode identifiers are still available for bookkeeping.

For each ``FFStage`` the block builds a runtime bundle consisting of:

* A ``QuantumLayer`` configured in amplitude-encoding mode (for the pre-measurement unitary).
* A partial ``DetectorTransform`` tied to the stage’s measured modes.
* A dictionary of conditional ``QuantumLayer`` objects – one per feed-forward branch.

The current implementation expects noise-free experiments (a bare
``NoiseModel()`` or ``None``) and only the first stage is allowed to consume
classical inputs defined via ``input_parameters``. Once detectors fire, every
branch progresses in amplitude-encoding mode and additional classical tensors
are ignored.

During the forward pass, ``FeedForwardBlock`` iterates over the stages. Each
stage takes the incoming branch amplitudes, applies the unitary, runs the
partial detector transform, and spawns new branches for the next stage based on
the measured outcomes. Branch bookkeeping keeps track of:

* The amplitudes of the remaining (unmeasured) modes.
* The probability weight associated with that branch.
* The sequence of measurement results that led to the branch. These are exposed
  via dictionary keys in the *original* experiment mode ordering thanks to the
  stage-level remapping described above.

At the end of the execution the block can expose different classical views over
the surviving quantum modes. Much like :class:`~merlin.algorithms.layer.QuantumLayer`,
the ``measurement_strategy`` parameter switches between raw amplitudes, dense
probabilities, or per-mode expectations. ``measurement_strategy=merlin.MeasurementStrategy.amplitudes()``
returns a list of tuples ``(measurement_key, branch_probability,
remaining_photons, amplitudes)`` so callers can reason about the remaining
mixed state. ``merlin.MeasurementStrategy.probs()`` collapses every branch into a tensor of shape
``(batch_size, len(output_keys))`` where the columns already align with the
fully specified Fock states recorded in
:pyattr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_keys`.
``merlin.MeasurementStrategy.mode_expectations()`` produces a ``(batch_size, num_modes)`` tensor describing
the photon expectations per mode. The result is already aggregated across all
measurement keys, so :pyattr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_keys`
is retained solely for metadata while
:pyattr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_state_sizes`
equals ``num_modes`` for every key so consumers can reason about downstream
reshaping without additional bookkeeping.

This design allows every stage to be simulated with amplitude access, while
still exposing convenient classical views. The mixed-state format (list of
tuples) is particularly useful for downstream probabilistic reasoning or for
feeding the remaining amplitudes into additional differentiable modules.

SLOS Torch simulation helpers
=============================

MerLin provides TorchScript-optimized primitives in
:mod:`merlin.pcvl_pytorch.slos_torchscript` to simulate photonic circuits with
high throughput. These helpers separate graph construction from evaluation and
offer two primary execution paths matching the two common encoding schemes:

* :meth:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph.compute` –
  Simulates a single input Fock state across a batch of unitary matrices. This
  path is used by MerLin’s angle-encoding implementation, where the input state
  is fixed and the circuit parameters (unitary) vary across the batch.
* :meth:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph.compute_batch` –
  Simulates a collection of input Fock states (all with the same total photon
  number) against a single unitary. This path is used by MerLin’s
  amplitude-encoding implementation, where a superposition of inputs is
  propagated through a fixed circuit.

Both methods rely on a pre-built sparse computation graph created by
:class:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph`, which encodes
layer-by-layer transitions between intermediate Fock configurations. The graph
is parameterized by the computation space (e.g., ``FOCK``, ``UNBUNCHED``,
``DUAL_RAIL``), the number of modes and photons, and optional state mapping.

Creating a compute graph
------------------------

You can either build the graph explicitly for repeated reuse:

.. code-block:: python

  from merlin.pcvl_pytorch.slos_torchscript import build_slos_distribution_computegraph
  from merlin.core.computation_space import ComputationSpace
  import torch

  m = 6                     # number of modes
  n_photons = 2             # total photons
  graph = build_slos_distribution_computegraph(
     m,
     n_photons,
     computation_space=ComputationSpace.UNBUNCHED,
     keep_keys=True,
     dtype=torch.float,
  )

  # Prepare a batch of random unitaries (here 4 samples) with matching complex dtype
  complex_dtype = torch.cfloat  # inferred from chosen real dtype
  unitary_batch = torch.linalg.qr(torch.randn(4, m, m, dtype=complex_dtype))[0]
  input_state = [1, 1, 0, 0, 0, 0]
  keys, amplitudes = graph.compute(unitary_batch, input_state)

Or invoke the convenience function which builds a transient graph on the fly
based on the provided unitary (dtype and device inferred):

.. code-block:: python

  from merlin.pcvl_pytorch.slos_torchscript import compute_slos_distribution

  single_unitary = unitary_batch[0]
  keys, amplitudes = compute_slos_distribution(single_unitary, input_state)

If you need to sweep *inputs* for amplitude encoding, prepare a list of Fock
states (same photon count) and call ``compute_batch`` on the existing graph.

Contract
--------

* ``compute(unitary, input_state)``
  - Input: ``unitary`` with shape ``[B, m, m]`` (or ``[m, m]``), complex dtype
    matching the graph precision; ``input_state`` as a length-``m`` list of
    integers whose sum equals the photon count.
  - Output: ``(keys, amplitudes)`` where ``amplitudes`` has shape
    ``[B, S]`` and ``keys`` enumerates the output Fock states (or
    ``None`` if keys are not retained).
  - Usage: angle encoding (vary unitaries over the batch).

* ``compute_batch(unitary, input_states)``
  - Input: ``unitary`` with shape ``[m, m]`` (or ``[1, m, m]``) and
    ``input_states`` as a list of Fock states; all states must have the same
    total photon number.
  - Output: ``(keys, amplitudes)`` where ``amplitudes`` has shape
    ``[1, S, N]`` (or squeezed), with ``N`` the number of input states.
  - Usage: amplitude encoding (vary inputs while the unitary is fixed).

Integration in QuantumLayer
---------------------------

:class:`~merlin.algorithms.layer.QuantumLayer` selects the execution path based
on the encoding mode:

* Angle encoding: calls ``compute`` to evaluate a batch of circuit instances
  over a fixed input state.
* Amplitude encoding: calls ``compute_batch`` to propagate a collection of
  input states (superposition components) through a single unitary.

Refer to ``QuantumLayer`` for the exact wiring and measurement strategies
(``measurement_strategy=merlin.MeasurementStrategy.probs()``, 
``measurement_strategy=merlin.MeasurementStrategy.amplitudes()``, 
``measurement_strategy=merlin.MeasurementStrategy.mode_expectations()``) 
layered on top of the SLOS outputs.
