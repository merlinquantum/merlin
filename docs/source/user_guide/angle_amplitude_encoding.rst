.. _angle_and_amplitude_encoding:

=====================================
Angle Encoding and Amplitude Encoding
=====================================

This guide shows how to use **angle encoding** and **amplitude encoding** with
Merlin's :class:`~merlin.algorithms.layer.QuantumLayer`. You'll find when to use each,
how to build circuits with :class:`~merlin.builder.circuit_builder.CircuitBuilder` or native
Perceval, and complete, runnable snippets.

Prerequisites
-------------

- Python, PyTorch, and Merlin installed.
- Basic familiarity with Merlin's :class:`~merlin.algorithms.layer.QuantumLayer`.
- Optional: Perceval for custom circuits and experiments.

Conceptual overview
-------------------

- **Angle encoding** maps a *real feature vector* into *circuit parameters*
  (e.g., phase shifter angles). The circuit unitary depends on your data.
  Data is encoded at specific points in the circuit using phase shifters.
- **Amplitude encoding** feeds a *quantum state* directly to the layer as
  input. Instead of turning features into angles, you supply the input
  quantum state's amplitudes at the beginning of the circuit.
  The preferred way to do this is with a
  :class:`~merlin.core.state_vector.StateVector`.

Angle Encoding
--------------

When to use
^^^^^^^^^^^

Use angle encoding for classical-quantum pipelines: feature maps, kernels, or
hybrid neural networks where your inputs are real-valued tensors.

With CircuitBuilder
^^^^^^^^^^^^^^^^^^^

:class:`~merlin.builder.circuit_builder.CircuitBuilder` provides a declarative way to add an
angle-encoding stage into your photonic circuit.

1) Build a circuit with angle encoding:

.. code-block:: python

    import numpy as np
    from merlin.builder import CircuitBuilder

    # 1) Declare a circuit with 6 modes
    builder = CircuitBuilder(n_modes=6)

    # 2) Put trainable rotations (phase shifters) on every mode
    builder.add_rotations(modes=[0, 1, 2, 3, 4, 5], trainable=True)

    # 3) Add an angle-encoding layer; the 'name' will prefix the input parameters
    builder.add_angle_encoding(
        modes=[0, 1, 2, 3, 4, 5],
        name="input",
        scale=np.pi   # optional global scaling of features -> angles
    )

    # 4) Entangle some modes (e.g., MZI block between modes 0 and 5)
    builder.add_entangling_layer(modes=[0, 5], trainable=True, model="mzi")

    # 5) Add superposition/BS layers (increase expressivity)
    builder.add_superpositions(modes=[0, 1, 2, 3, 4, 5], trainable=True, depth=2)

2) Wrap it as a QuantumLayer and run a forward pass:

.. code-block:: python

    import torch
    from merlin.algorithms import QuantumLayer
    from merlin.core.state_vector import StateVector

    layer = QuantumLayer(
        input_size=6,                                    # number of classical features per sample
        builder=builder,                                 # the declarative circuit
        input_state=StateVector.from_basic_state([1, 0, 1, 0, 1, 0]),  # 3 photons in 6 modes
    )

    x = torch.rand((4, 6))       # batch of 4 samples
    probs = layer(x)              # default .probs() measurement

Parameter names and prefixes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~merlin.builder.circuit_builder.CircuitBuilder.add_angle_encoding` call registers
parameters prefixed by ``name`` (e.g., ``"input"``). Internally,
:class:`~merlin.algorithms.layer.QuantumLayer` will consume your real-valued input
tensor and map each feature to the corresponding prefixed angle(s).

Tips and constraints
^^^^^^^^^^^^^^^^^^^^

- **Modes vs. features**: By construction you typically shouldn't encode more
  independent features than available modes in the encoding step.
- **Scaling and combinations**: You can use ``scale=...`` to rescale inputs
  before turning them into angles. If you create multiple encoding stages with
  different names (prefixes), the layer can split the input tensor across them.
- **Kernels**: For quantum kernels, consider :class:`~merlin.algorithms.kernels.FeatureMap`
  and :class:`~merlin.algorithms.kernels.FidelityKernel` if you need a reusable feature
  map object.

Angle encoding using QuantumLayer.simple
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want a quick start without designing the circuit:

.. code-block:: python

    import torch
    from merlin.algorithms import QuantumLayer

    layer = QuantumLayer.simple(
        input_size=6,   # number of classical features
        output_size=10  # output dimensionality
    )

    x = torch.rand((2, 6))
    y = layer(x)  # probability vector of size `output_size`

Angle encoding with Perceval circuits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For full control, create a Perceval circuit, then expose input-parameter
prefix(es) that the layer will map features to:

.. code-block:: python

    import perceval as pcvl
    from merlin.algorithms import QuantumLayer
    from merlin.core.state_vector import StateVector

    # Build a 6-mode Perceval circuit
    circuit = pcvl.Circuit(6)

    # Example: add user-named input phase shifters (prefix 'input')
    for mode in range(6):
        circuit.add(mode, pcvl.PS(pcvl.P(f"input{mode}")))
        circuit.add(mode, pcvl.PS(pcvl.P(f"theta{mode}")))

    # (Add interferometers, MZIs, etc. as you like)
    # ...

    layer = QuantumLayer(
        input_size=6,
        circuit=circuit,
        input_state=StateVector.from_basic_state([1, 0, 1, 0, 1, 0]),
        input_parameters=["input"],    # map features -> parameters named 'input*'
        trainable_parameters=["theta"] # example trainable prefix used elsewhere in your circuit
    )

    import torch
    x = torch.rand((1, 6))
    probs = layer(x)

Amplitude Encoding
------------------

When to use
^^^^^^^^^^^

Choose amplitude encoding when you want to:

- **Map classical feature vectors** directly into Fock-space amplitudes. Your
  real-valued data becomes the quantum state itself and the circuit acts as a
  learned unitary transformation on it. Use
  :meth:`StateVector.from_tensor() <merlin.core.state_vector.StateVector.from_tensor>`
  to wrap the data.
- **Inject a prepared quantum state** into the circuit — for example, a state
  produced by an upstream simulator or another photonic block. In this case you
  can provide complex-valued data directly.

How it works
^^^^^^^^^^^^^

Amplitude encoding replaces the input quantum state of the circuit at runtime
with a state whose amplitudes come from your data. The circuit then acts as a
learned unitary transformation on that state.

The workflow is:

1. Start with a feature vector of length *d* (the Fock basis size). This can
   be real-valued classical data or complex-valued quantum-state amplitudes.
2. Wrap it via :meth:`StateVector.from_tensor() <merlin.core.state_vector.StateVector.from_tensor>`,
   which attaches the Fock metadata and auto-promotes real data to complex.
3. Pass the :class:`~merlin.core.state_vector.StateVector` to ``forward()`` —
   the layer detects the type and activates amplitude encoding automatically.

No special constructor flags are needed. The encoding mode is inferred purely
from the **type** of the first argument to ``forward()``:

.. list-table::
   :header-rows: 1
   :widths: 2 5 10

   * - #
     - Input type
     - Behaviour
   * - 1
     - | :class:`~merlin.core.state_vector.StateVector`
       | **(preferred)**
     - | Automatically activates amplitude encoding. The layer extracts the
       | active basis support, validates its dimension against the layer basis,
       | and propagates only active components through the circuit.
   * - 2
     - Complex ``torch.Tensor``
     - | A single complex-dtype tensor is treated identically to a :class:`~merlin.core.state_vector.StateVector`'s
       | underlying tensor. It uses the same active-support execution path and
       | is useful when you manage tensors directly without wrapping them.
   * - 3
     - | Real ``torch.Tensor``
       | + ``amplitude_encoding=True``
     - | **Legacy path — deprecated (will be removed in 0.4).**
       | The constructor flag forces amplitude interpretation on a real-valued
       | tensor. Migrate to path 1 or 2.

.. warning::
   *Deprecated since version 0.3:* The ``amplitude_encoding=True`` constructor parameter is deprecated and will
   be removed in **0.4**. Pass a :class:`~merlin.core.state_vector.StateVector`
   or a complex ``torch.Tensor`` to ``forward()`` instead.

Chunked execution tradeoff
^^^^^^^^^^^^^^^^^^^^^^^^^^

Amplitude inputs are accepted as ordinary dense tensors, because this is the
most convenient representation for users and for upstream PyTorch code. During
execution, :class:`~merlin.algorithms.layer.QuantumLayer` normalizes the input
and converts dense tensors, dense :class:`~merlin.core.state_vector.StateVector`
objects, sparse tensors, and sparse ``StateVector`` objects into the same
internal active-support representation:

.. code-block:: text

   active basis indices + compact coefficient tensor [batch_size, nnz]

Only those active Fock components are propagated through the simulator. Merlin
processes them in chunks, calls the batched simulator routine for each chunk,
weights the returned amplitudes by the corresponding input coefficients, and
accumulates directly into the final dense output tensor.

The benefit is peak temporary memory: it scales with approximately
``chunk_size * num_output_states`` instead of
``num_input_basis_states * num_output_states``. This matters when the Fock basis
is large but the input state has only a small active support.

The tradeoff is throughput. A smaller chunk size reduces peak memory but
requires more simulator calls; a larger chunk size can be faster when memory is
available. Control this with the ``simultaneous_processes`` argument:

.. code-block:: python

   output = layer(state_vector, simultaneous_processes=32)

If omitted, Merlin uses a conservative default. Changing the chunk size should
not change the numerical result; it only changes how much active support is
processed in each simulator call.

Setup
^^^^^

Just make sure:

- ``n_photons`` is set (so the layer knows the Hilbert space dimension).
- ``input_size`` and ``input_parameters`` are **not** set (they are for angle
  encoding only).

.. code-block:: python

    import perceval as pcvl
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy
    from merlin.core.computation_space import ComputationSpace

    circuit = pcvl.Circuit(4)
    # ... populate with beam splitters, phase shifters, etc. ...

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=2,
        measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
    )


Input dimensions
^^^^^^^^^^^^^^^^

The amplitude vector must have exactly *d* components, where *d* is the
Fock-space basis size:

.. math::

   d = \binom{n\_modes + n\_photons - 1}{n\_photons}

Check the expected dimension and basis ordering with:

.. code-block:: python

   print(len(layer.output_keys))   # 10 for (4 modes, 2 photons)
   print(layer.output_keys[:3])    # e.g. [(2,0,0,0), (1,1,0,0), (1,0,1,0)]


Encoding classical data with ``from_tensor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The primary amplitude encoding path wraps a classical feature vector as a
:class:`~merlin.core.state_vector.StateVector` via
:meth:`~merlin.core.state_vector.StateVector.from_tensor`. This method accepts
**real or complex** tensors — real data is automatically promoted to complex
internally — and validates that the last dimension matches the Fock basis size.

**Single sample:**

.. code-block:: python

    import torch
    from merlin.core.state_vector import StateVector

    # Classical feature vector of size d = 10  (for 4 modes, 2 photons)
    features = torch.randn(10)

    # Wrap as StateVector — real-to-complex promotion happens automatically
    sv = StateVector.from_tensor(features, n_modes=4, n_photons=2)

    # Pass to the layer — amplitude encoding is detected from the type
    output = layer(sv)

.. tip::

   :meth:`~merlin.core.state_vector.StateVector.from_tensor` handles real-to-complex promotion for you.
   The layer lazily normalizes amplitudes before computation, but explicitly
   normalizing upstream (e.g. via ``nn.functional.normalize``) can improve
   numerical stability during training.

Structured input encodings can infer their own dimensions. For example,
dual-rail infers ``n_photons`` from the logical tensor width and sets
``n_modes = 2 * n_photons``. QLOQ and other partitioned encodings infer
``n_modes`` and ``n_photons`` from their ``modes_per_photon`` contract. The
resolved values are available on the returned object:

.. code-block:: python

    import torch
    from merlin.core import EncodingSpace
    from merlin.core.state_vector import StateVector

    dual_rail = StateVector.from_tensor(
        torch.zeros(4),
        encoding=EncodingSpace.DUAL_RAIL,
    )
    assert dual_rail.n_modes == 4
    assert dual_rail.n_photons == 2

    qloq = EncodingSpace.qloq(qubit_groups=[2, 1])
    qloq_state = StateVector.from_tensor(torch.zeros(8), encoding=qloq)
    assert qloq_state.n_modes == 6
    assert qloq_state.n_photons == 2

If a circuit needs auxiliary vacuum modes, append them after constructing the
encoded state:

.. code-block:: python

    padded = dual_rail @ [0]
    assert padded.n_modes == 5
    assert padded.n_photons == 2


Using a complex tensor directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to manage raw tensors without the :class:`~merlin.core.state_vector.StateVector`: wrapper,
passing a **complex** ``torch.Tensor`` to ``forward()`` also triggers amplitude
encoding:

.. code-block:: python

    import torch

    amps = torch.randn(10, dtype=torch.complex64)
    output = layer(amps)   # complex dtype triggers amplitude encoding


Standardising other inputs as StateVector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constructors :meth:`~merlin.core.state_vector.StateVector.from_basic_state`,
:meth:`~merlin.core.state_vector.StateVector.from_perceval`, and the ``+`` operator are **not** forms
of amplitude encoding — they do not map classical data into amplitudes.
Their purpose is to give every quantum-state input a single, uniform type
(:class:`~merlin.core.state_vector.StateVector`) so the layer's dispatch
logic does not need special cases for lists, ``pcvl.BasicState``, or
``pcvl.StateVector``.

**From a known Fock state:**

.. code-block:: python

    from merlin.core.state_vector import StateVector

    sv = StateVector.from_basic_state([1, 0, 1, 0])
    output = layer(sv)

**From a superposition:**

.. code-block:: python

    sv = (
        StateVector.from_basic_state([1, 0, 1, 0])
        + StateVector.from_basic_state([0, 1, 0, 1])
    )
    output = layer(sv)   # (|1,0,1,0⟩ + |0,1,0,1⟩) / √2

**From a Perceval state:**

.. code-block:: python

    import perceval as pcvl
    from merlin.core.state_vector import StateVector

    pcvl_sv = (
        pcvl.StateVector(pcvl.BasicState([1, 0, 1, 0]))
        + pcvl.StateVector(pcvl.BasicState([0, 1, 0, 1]))
    )
    sv = StateVector.from_perceval(pcvl_sv)
    output = layer(sv)

In every case the layer processes the :class:`~merlin.core.state_vector.StateVector` through the same
code path — the only difference is where the amplitudes came from.


Restrictions
^^^^^^^^^^^^

- Only **one** :class:`~merlin.core.state_vector.StateVector` may be passed per
  ``forward()`` call.
- :class:`~merlin.core.state_vector.StateVector` and ``torch.Tensor`` inputs
  **cannot be mixed** in the same call.
- **Batched (2-D)** :class:`~merlin.core.state_vector.StateVector` inputs are
  supported. Pass a 2-D tensor of shape ``(batch_size, d)`` to
  :meth:`~merlin.core.state_vector.StateVector.from_tensor` and the layer processes the whole batch in a
  single ``forward()`` call.
- With ``MeasurementStrategy.amplitudes()`` the layer **bypasses** detectors
  and noise; ``shots`` must be unset or zero.
- If you need to combine a custom quantum input with classical angle-encoded
  features, set the quantum state via the ``input_state`` constructor parameter
  and pass the classical features as a real tensor to ``forward()``.


Returning typed objects
^^^^^^^^^^^^^^^^^^^^^^^

With ``return_object=True``, the measurement strategy determines the return
type. This applies equally to angle and amplitude encoding:

.. code-block:: python

    layer = QuantumLayer(
        builder=builder,
        n_photons=2,
        measurement_strategy=MeasurementStrategy.amplitudes(ComputationSpace.FOCK),
        return_object=True,
    )

    sv = StateVector.from_tensor(torch.randn(len(layer.output_keys)), n_modes=4, n_photons=2)
    sv_out = layer(sv)           # StateVector
    sv_out.n_modes               # 4
    sv_out[[1, 0, 1, 0]]        # complex amplitude for a specific Fock state

And with a probability strategy, a
:class:`~merlin.core.probability_distribution.ProbabilityDistribution`:

.. code-block:: python

    layer = QuantumLayer(
        builder=builder,
        n_photons=2,
        measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
        return_object=True,
    )

    sv = StateVector.from_tensor(torch.randn(len(layer.output_keys)), n_modes=4, n_photons=2)
    pd = layer(sv)               # ProbabilityDistribution
    pd.probabilities()           # dense probability tensor
    pd.filter(ComputationSpace.UNBUNCHED)  # post-select


Chaining quantum layers
^^^^^^^^^^^^^^^^^^^^^^^^

Because :class:`~merlin.algorithms.layer.QuantumLayer` can both consume and produce
:class:`~merlin.core.state_vector.StateVector` objects, you can chain layers
so that the output amplitudes of one feed into the next:

.. code-block:: python

    from merlin.core.state_vector import StateVector

    layer_1 = QuantumLayer(
        builder=builder_1,
        input_state=StateVector.from_basic_state([1, 0, 1, 0]),
        measurement_strategy=MeasurementStrategy.amplitudes(ComputationSpace.FOCK),
        return_object=True,
    )

    layer_2 = QuantumLayer(
        builder=builder_2,
        n_photons=2,
        measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
        return_object=True,
    )

    sv_mid = layer_1(x_input)    # StateVector
    pd_out = layer_2(sv_mid)     # ProbabilityDistribution

Gradients flow through both layers during backpropagation.


Encodings Key Differences
-------------------------

.. |br| raw:: html

   <br />

+---------------------------+----------------------------+------------------------------------------+
| Aspect                    | Angle Encoding             | Amplitude Encoding                       |
+===========================+============================+==========================================+
| Input to ``forward()``    | Real ``torch.Tensor``      | ``StateVector`` via ``from_tensor()``    |
|                           |                            | **(preferred)** or complex tensor        |
+---------------------------+----------------------------+------------------------------------------+
| Number of inputs          | User-defined               | Fixed by n_modes and n_photons           |
|                           | (``input_size``)           | (combinatorial formula)                  |
+---------------------------+----------------------------+------------------------------------------+
| Circuit dependence        | Features set parameters    | Data defines input quantum               |
|                           | (phases/angles)            | state; circuit is fixed unitary          |
+---------------------------+----------------------------+------------------------------------------+
| Setup (constructor)       | ``input_size``,            | ``n_photons``; no ``input_size`` or      |
|                           | ``add_angle_encoding(...)``| ``input_parameters``                     |
+---------------------------+----------------------------+------------------------------------------+
| Activation trigger        | Real tensor to             | ``StateVector.from_tensor(data, ...)``   |
|                           | ``forward()``              | or complex tensor to ``forward()``       |
+---------------------------+----------------------------+------------------------------------------+
| Typical use               | Feature maps, kernels,     | Classical data as amplitudes.            |
|                           | hybrid NN layers           | via ``from_tensor``; or injecting a |br| |
|                           |                            | prepared quantum state                   |
+---------------------------+----------------------------+------------------------------------------+
| Measurement options       | Probabilities, modes,      | Probabilities, modes,                    |
|                           | amplitudes (sim-only),     | amplitudes (sim-only), partial           |
|                           | partial (sim-only)         | (sim-only)                               |
+---------------------------+----------------------------+------------------------------------------+

Troubleshooting
---------------

- **Shape errors (angle encoding)**: Ensure ``input_size`` equals the number of
  features you feed into the layer and matches the encoding specification
  (number of input phase shifters and prefixes).
- **Too many features**: If you attempt to encode more features than modes in
  your encoding stage, reduce features using dimensionality reduction techniques
  such as PCA or UMAP, or expand the circuit's encoding modes.
- **Shape errors (amplitude encoding)**: The amplitude vector length must match
  the layer basis size: ``len(layer.output_keys)``. For batching, use
  ``(batch, len(output_keys))``.
- **Incompatible measurement strategy**: When
  ``MeasurementStrategy.amplitudes()`` is selected, do not set nonzero ``shots``
  or enable detectors/noise.
- **Unnormalized amplitudes**: Always normalize amplitude inputs to avoid
  unstable gradients and to ensure proper probability mass.
- **Mixing input types**: You cannot pass both a ``StateVector`` and a
  ``torch.Tensor`` in the same ``forward()`` call. Use either angle encoding
  (real tensors) or amplitude encoding (``StateVector`` / complex tensor), not
  both.
- **DeprecationWarning for** ``amplitude_encoding=True``: Migrate to passing a
  :class:`~merlin.core.state_vector.StateVector` or complex tensor to
  ``forward()`` instead of using the constructor flag.
- **Batched amplitude encoding**: Pass a 2-D tensor to
  :meth:`~merlin.core.state_vector.StateVector.from_tensor` and call ``forward()`` with the resulting
  :class:`~merlin.core.state_vector.StateVector`. The layer normalizes each sample independently and
  returns a ``(batch_size, output_size)`` tensor.

Complete Examples
-----------------

Angle encoding with builder and probabilities out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    import torch
    from merlin.builder import CircuitBuilder
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy
    from merlin.core.computation_space import ComputationSpace
    from merlin.core.state_vector import StateVector

    builder = CircuitBuilder(n_modes=6)
    builder.add_angle_encoding(modes=list(range(6)), name="input", scale=np.pi)
    builder.add_entangling_layer(trainable=True)
    builder.add_superpositions(modes=list(range(6)), trainable=True, depth=1)

    layer = QuantumLayer(
        input_size=6,
        builder=builder,
        input_state=StateVector.from_basic_state([1, 0, 1, 0, 1, 0]),
        measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
    )

    x = torch.rand((3, 6))
    probs = layer(x)  # shape: [3, layer.output_size]

Amplitude encoding with ``StateVector.from_tensor`` and amplitudes out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import perceval as pcvl
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy
    from merlin.core.computation_space import ComputationSpace
    from merlin.core.state_vector import StateVector

    # Simple unitary circuit placeholder; customize as needed
    circuit = pcvl.Circuit(4)
    # ... populate circuit ...

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=2,
        measurement_strategy=MeasurementStrategy.amplitudes(ComputationSpace.FOCK),
        return_object=True,
    )

    # Encode classical data as a StateVector (real data is auto-promoted to complex)
    d = len(layer.output_keys)              # basis size for (4 modes, 2 photons)
    features = torch.randn(d)               # real-valued classical features
    sv = StateVector.from_tensor(features, n_modes=4, n_photons=2)
    sv_out = layer(sv)                       # StateVector with output amplitudes

Amplitude encoding with complex tensor (no wrapper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy
    from merlin.core.computation_space import ComputationSpace

    layer = QuantumLayer(
        circuit=circuit,     # same circuit as above
        n_photons=2,
        measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
    )

    d = len(layer.output_keys)
    amps = torch.randn(d, dtype=torch.complex64)
    amps = amps / amps.abs().pow(2).sum().sqrt()

    probs = layer(amps)   # complex dtype triggers amplitude encoding

Amplitude encoding with probabilities out and post-selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy
    from merlin.core.computation_space import ComputationSpace
    from merlin.core.state_vector import StateVector

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=2,
        measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),
        return_object=True,
    )

    sv = StateVector.from_tensor(torch.randn(len(layer.output_keys)), n_modes=4, n_photons=2)
    pd = layer(sv)                                   # ProbabilityDistribution

    # Post-select to unbunched states
    pd_ub = pd.filter(ComputationSpace.UNBUNCHED)
    print(pd_ub.logical_performance)                  # fraction of mass kept
    print(pd_ub.probabilities())                      # renormalized probabilities

Measurement Strategies (Output Options)
---------------------------------------

Both angle and amplitude encoding support the following output measurement
strategies. For more details, see :doc:`measurement_strategy`.

- ``MeasurementStrategy.probs(computation_space)`` (default): returns a
  probability vector aligned with ``layer.output_keys``.
  With ``return_object=True``, returns a
  :class:`~merlin.core.probability_distribution.ProbabilityDistribution`.
- ``MeasurementStrategy.mode_expectations(computation_space)``: returns
  per-mode expected photon counts.
- ``MeasurementStrategy.amplitudes()``: returns complex amplitudes
  (simulation-only; bypasses detectors and noise).
  With ``return_object=True``, returns a
  :class:`~merlin.core.state_vector.StateVector`.
- ``MeasurementStrategy.partial(computation_space)``: returns a partial
  measurement result for selected modes.

References
----------

Tak Hur et al., "Quantum convolutional neural network for classical data
classification", 2022. https://arxiv.org/abs/2108.00661

.. seealso::

   - :doc:`Basic Concepts </quickstart/basic_concepts>` — overview of ``StateVector`` and
     ``ProbabilityDistribution``
   - :doc:`StateVector API </api_reference/api/merlin.core.state_vector>` — full API reference for ``StateVector`` including
     ``from_tensor``
