.. _angle_and_amplitude_encoding:

=====================================
Angle Encoding and Amplitude Encoding
=====================================

This guide shows how to use **angle encoding** and **amplitude encoding** with
Merlin's :class:`~merlin.algorithms.QuantumLayer`. You'll find when to use each,
how to build circuits with :class:`~merlin.builder.CircuitBuilder` or native Perceval, and complete, runnable snippets.

Prerequisites
-------------

- Python, PyTorch, and Merlin installed.
- Basic familiarity with Merlin's :class:`~merlin.algorithms.QuantumLayer`.
- Optional: Perceval for custom circuits and experiments.

Conceptual Overview
-------------------

- **Angle encoding** maps a *real feature vector*
  into *circuit parameters* (e.g., phase shifter angles). The circuit unitary
  depends on your data. Data is encoded at specific points in the circuit using phase shifters.
- **Amplitude encoding** feeds a *complex statevector* directly to the layer as input. Instead of
  turning features into angles, you supply the input quantum state's amplitudes at the beginning of the circuit.

Angle Encoding
--------------

When to use
^^^^^^^^^^^

Use angle encoding for classical-quantum pipelines: feature maps, kernels, or
hybrid neural networks where your inputs are real-valued tensors.

With CircuitBuilder
^^^^^^^^^^^^^^^^^^^

:class:`~merlin.builder.CircuitBuilder` provides a declarative way to add an
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

    layer = QuantumLayer(
        input_size=6,           # number of *classical* features per sample
        builder=builder,        # the declarative circuit
        input_state=[1, 0, 1, 0, 1, 0]   # 5 photons in 10-mode equivalent => here 6 modes, so 3 photons example
    )

    x = torch.rand((4, 6))      # batch of 4 samples
    probs = layer(x)            # default MeasurementStrategy.probs()

Parameter names and prefixes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~merlin.builder.CircuitBuilder.add_angle_encoding` call registers
parameters prefixed by ``name`` (e.g., ``"input"``). Internally,
:class:`~merlin.algorithms.QuantumLayer` will consume your real-valued input
tensor and map each feature to the corresponding prefixed angle(s).

Tips and constraints
^^^^^^^^^^^^^^^^^^^^

- **Modes vs. features**: By construction you typically shouldn't encode more
  independent features than available modes in the encoding step.
- **Scaling and combinations**: You can use ``scale=...`` to rescale inputs
  before turning them into angles. If you create multiple encoding stages with
  different names (prefixes), the layer can split the input tensor across them.
- **Kernels**: For quantum kernels, consider :class:`~merlin.kernels.FeatureMap`
  and :class:`~merlin.kernels.FidelityKernel` if you need a reusable feature
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
        input_state=[1, 0, 1, 0, 1, 0],
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

Choose amplitude encoding when you already have a prepared **quantum state** to
inject into the circuit (e.g., produced by an upstream simulator or another
photonic block). Here your input to ``forward`` is the **statevector**
amplitudes, not classical features. This is useful for providing a prepared quantum state as input to a photonic circuit.

How to use
^^^^^^^^^^

- To activate amplitude encoding, set ``amplitude_encoding=True`` on :class:`~merlin.algorithms.QuantumLayer`.
- Requires the parameter ``n_photons`` to define the computational subspace.
- Angle and amplitude encoding are mutually exclusive. Thus, ``input_size`` or ``input_parameters`` will not be used and may cause errors if provided.
- The **input tensor shape** must match the layer's state space size:
  ``len(layer.output_keys)`` (or ``[batch, len(output_keys)]``).


Key input/output dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The principal difference is that in amplitude encoding, the number of inputs is conditioned by the number of photons and modes.
The input dimension must equal the number of Fock states, given by ``num_states = len(layer.output_keys)``.
Mathematically, the number of Fock states is:

**(n_modes + n_photons - 1) choose n_photons**

This combinatorial formula gives the dimension of the Hilbert space for the photonic system.

Minimal example (amplitudes out)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy

    # Suppose you already have a circuit or builder; here we assume `circuit`
    # exists and is unitary with no post-selection.
    # For amplitude encoding, the optical layout defines the evolution,
    # but no classical input parameters are used.
    layer = QuantumLayer(
        circuit=circuit,              # or builder=..., or experiment=...
        n_photons=2,                  # required: defines the subspace
        amplitude_encoding=True,      # switch to amplitude input
        measurement_strategy=MeasurementStrategy.amplitudes()
    )

    # Build (or sample) an input statevector compatible with the layer basis
    num_states = len(layer.output_keys)     # basis size for 2 photons over the modes
    psi_in = torch.randn(num_states, dtype=torch.complex64)
    psi_in = psi_in / psi_in.norm()         # normalize - important to avoid exploding norms

    # Forward: returns complex amplitudes after the circuit
    psi_out = layer(psi_in)

Detectors, noise, and shots
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- With :data:`~merlin.measurement.MeasurementStrategy.amplitudes()`, only strong simulation is possible. Hence, the layer
  **bypasses** detectors and noise; ``shots`` must be unset or zero.
- With probability-like strategies, detector/noise models (if present in a
  :class:`perceval.Experiment`) are applied *after* converting amplitudes to
  probabilities; shot sampling is supported when compatible.

Quantum systems interoperability requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The amplitude vector must be **compatible with the basis** used by the layer.
  Check ``layer.output_keys`` to see state ordering.
- Always normalize your amplitude inputs to ensure proper probability mass and avoid unstable gradients.

Encodings Key Differences
-------------------------

+------------------------+----------------------------+----------------------------------+
| Aspect                 | Angle Encoding             | Amplitude Encoding               |
+========================+============================+==================================+
| Input to ``forward``   | Real features ``x``        | Complex statevector amplitudes   |
+------------------------+----------------------------+----------------------------------+
| Number of inputs       | User-defined               | Fixed by n_modes and n_photons   |
|                        | (``input_size``)           | (combinatorial formula)          |
+------------------------+----------------------------+----------------------------------+
| Circuit dependence     | Features set parameters    | State defines input quantum      |
|                        | (phases/angles)            | state to propagate               |
+------------------------+----------------------------+----------------------------------+
| Setup knobs            | ``add_angle_encoding(...)``| ``amplitude_encoding=True``,     |
|                        | scales, multiple prefixes  | ``n_photons``                    |
+------------------------+----------------------------+----------------------------------+
| Typical use            | Feature maps, kernels,     | Providing a prepared quantum     |
|                        | hybrid NN layers           | state as input to a photonic     |
|                        |                            | circuit                          |
+------------------------+----------------------------+----------------------------------+
| Measurement options    | Probabilities, modes,      | Probabilities, modes,            |
|                        | amplitudes (sim-only)      | amplitudes (sim-only)            |
+------------------------+----------------------------+----------------------------------+

Troubleshooting
---------------

- **Shape errors (angle encoding)**: Ensure ``input_size`` equals the number of
  features you feed into the layer and matches the encoding specification
  (number of input phase shifters and prefixes).
- **Too many features**: If you attempt to encode more features than modes in
  your encoding stage, reduce features using dimensionality reduction techniques such as PCA or UMAP, or expand the circuit's encoding modes.
- **Shape errors (amplitude encoding)**: The amplitude vector length must match
  the layer basis size: ``len(layer.output_keys)``. For batching, use
  ``[batch, len(output_keys)]``.
- **Incompatible measurement strategy**: When
  :data:`~merlin.measurement.MeasurementStrategy.amplitudes()` is selected, do not
  set nonzero ``shots`` or enable detectors/noise.
- **Unnormalized amplitudes**: Always normalize amplitude inputs to avoid
  unstable gradients and to ensure proper probability mass.

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

    builder = CircuitBuilder(n_modes=6)
    builder.add_angle_encoding(modes=list(range(6)), name="input", scale=np.pi)
    builder.add_entangling_layer(trainable=True)
    builder.add_superpositions(modes=list(range(6)), trainable=True, depth=1)

    layer = QuantumLayer(
        input_size=6,
        builder=builder,
        input_state=[1, 0, 1, 0, 1, 0],
        measurement_strategy=MeasurementStrategy.probs()
    )

    x = torch.rand((3, 6))
    probs = layer(x)  # shape: [3, layer.output_size]

Amplitude encoding with amplitudes out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import perceval as pcvl
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy

    # Simple unitary circuit placeholder; customize as needed
    circuit = pcvl.Circuit(4)
    # ... populate circuit ...

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=2,
        amplitude_encoding=True,
        measurement_strategy=MeasurementStrategy.amplitudes()
    )

    num_states = len(layer.output_keys)
    psi_in = torch.randn(num_states, dtype=torch.complex64)
    psi_in = psi_in / psi_in.norm()

    amps_out = layer(psi_in)  # complex amplitudes

Amplitude encoding with probabilities out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy

    layer = QuantumLayer(
        circuit=circuit,     # same circuit as above
        n_photons=2,
        amplitude_encoding=True,
        measurement_strategy=MeasurementStrategy.probs()
    )

    psi_in = torch.randn(len(layer.output_keys), dtype=torch.complex64)
    psi_in = psi_in / psi_in.norm()

    probs = layer(psi_in)   # classical probabilities

Measurement Strategies (Output Options)
---------------------------------------

Both angle and amplitude encoding support the following output measurement strategies. For more details, see :doc:`measurement_strategy`.

- :data:`~merlin.measurement.MeasurementStrategy.probs()` (default):
  returns a probability vector aligned with ``layer.output_keys``.
- :data:`~merlin.measurement.MeasurementStrategy.mode_expectations()`:
  returns per-mode expected photon counts.
- :data:`~merlin.measurement.MeasurementStrategy.amplitudes()`:
  returns complex amplitudes (simulation-only; bypasses detectors and noise).

References
----------

Tak Hur et al., "Quantum convolutional neural network for classical data classification", 2022. https://arxiv.org/abs/2108.00661
