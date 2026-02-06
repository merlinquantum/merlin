merlin.algorithms.layer module
==============================

.. automodule:: merlin.algorithms.layer
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   Quantum layers built from a :class:`perceval.Experiment` now apply the experiment's per-mode detector configuration before returning classical outputs. When no detectors are specified, ideal photon-number resolving detectors are used by default.

   If the experiment carries a :class:`perceval.NoiseModel` (via ``experiment.noise``), MerLin inserts a :class:`~merlin.measurement.photon_loss.PhotonLossTransform` ahead of any detector transform. The resulting ``output_keys`` and ``output_size`` therefore include every survival/loss configuration implied by the model, and amplitude read-out is disabled whenever custom detectors or photon loss are present.

Example: Quickstart QuantumLayer
--------------------------------

.. code-block:: python

    import torch.nn as nn
    from merlin import QuantumLayer

    simple_layer = QuantumLayer.simple(
        input_size=4,
    )

    model = nn.Sequential(
        simple_layer,
        nn.Linear(simple_layer.output_size, 3),
    )
    # Train and evaluate as a standard torch.nn.Module

.. note::
   :func:`QuantumLayer.simple` returns a thin ``SimpleSequential`` wrapper that behaves like a standard
   PyTorch module while exposing the inner quantum layer as ``.quantum_layer`` and any
   post-processing (:class:`~merlin.utils.grouping.ModGrouping` or :class:`~torch.nn.Identity`) as ``.post_processing``.
   The wrapper also forwards ``.circuit`` and ``.output_size`` so existing code that inspects these
   attributes continues to work.

.. image:: ../../_static/img/Circ_simple.png
   :alt: A Perceval Circuit built with QuantumLayer.simple
   :width: 600px
   :align: center

The simple quantum layer above implements a circuit of (input_size) modes and (input_size//2) photons. This circuit is made of:
- A fully trainable entangling layer acting on all modes;
- A full input encoding layer spanning all encoded features;
- A fully trainable entangling layer acting on all modes.

Example: Declarative builder API
--------------------------------

.. code-block:: python

    import torch.nn as nn
    from merlin import LexGrouping, MeasurementStrategy, QuantumLayer
    from merlin.builder import CircuitBuilder
    builder = CircuitBuilder(n_modes=6)
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(modes=list(range(4)), name="input")
    builder.add_rotations(trainable=True, name="theta")
    builder.add_superpositions(depth=1)

    builder_layer = QuantumLayer(
        input_size=4,
        builder=builder,
        n_photons=3,  # is equivalent to input_state=[1,1,1,0,0,0]
        measurement_strategy=MeasurementStrategy.probs(),
    )

    model = nn.Sequential(
        builder_layer,
        LexGrouping(builder_layer.output_size, 3),
    )
    # Train and evaluate as a standard torch.nn.Module

.. image:: ../../_static/img/Circ_builder.png
   :alt: A Perceval Circuit built with the CircuitBuilder
   :width: 600px
   :align: center

The circuit builder allows you to build your circuit layer by layer, with a high-level API. The example above implements a circuit of 6 modes and 3 photons.
This circuit is made of:
- A first entangling layer (trainable)
- Angle encoding on the first 4 modes (for 4 input parameters with the name "input")
- A trainable rotation layer to add more trainable parameters
- An entangling layer to add more expressivity

Other building blocks in the CircuitBuilder include:

- **add_rotations**: Add single or multiple phase shifters (rotations) to specific modes. Rotations can be fixed, trainable, or data-driven (input-encoded).
- **add_angle_encoding**: Encode classical data as quantum rotation angles, supporting higher-order feature combinations for expressive input encoding.
- **add_entangling_layer**: Insert a multi-mode entangling layer (implemented via a generic interferometer), optionally trainable, and tune its internal template with the ``model`` argument (``"mzi"`` or ``"bell"``) for different mixing behaviours.
- **add_superpositions**: Add one or more beam splitters (superposition layers) with configurable targets, depth, and trainability.

Example: Manual Perceval circuit (more control)
-----------------------------------------------

.. code-block:: python

    import torch.nn as nn
    import perceval as pcvl
    from merlin import LexGrouping, MeasurementStrategy, QuantumLayer
    modes = 6
    wl = pcvl.GenericInterferometer(
        modes,
        lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_li{i}")) //
        pcvl.BS() // pcvl.PS(pcvl.P(f"theta_lo{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    circuit = pcvl.Circuit(modes)
    circuit.add(0, wl)
    for mode in range(4):
        circuit.add(mode, pcvl.PS(pcvl.P(f"input{mode}")))
    wr = pcvl.GenericInterferometer(
        modes,
        lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ri{i}")) //
        pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ro{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    circuit.add(0, wr)

    manual_layer = QuantumLayer(
        input_size=4,  # matches the number of phase shifters named "input{mode}"
        circuit=circuit,
        input_state=[1, 0, 1, 0, 1, 0],
        trainable_parameters=["theta"],
        input_parameters=["input"],
        measurement_strategy=MeasurementStrategy.probs(),
    )

    model = nn.Sequential(
        manual_layer,
        LexGrouping(manual_layer.output_size, 3),
    )
    # Train and evaluate as a standard torch.nn.Module

.. image:: ../../_static/img/Circ_manual.png
   :alt: A Perceval Circuit built with the Perceval API
   :width: 600px
   :align: center

Here, the grouping can also be directly added to the ``MeasurementStrategy`` object used in the ``measurement_strategy`` parameter.

See the User guide and Notebooks for more advanced usage and training routines !

Input states and amplitude encoding
-----------------------------------

The *input state* of a photonic circuit specifies how the photons enter the device. Physically this can be a single
Fock state (a precise configuration of ``n_photons`` over ``m`` modes) or a superposed/entangled state within the same
computation space (for example Bell pairs or GHZ states). :class:`~merlin.algorithms.layer.QuantumLayer` accepts the
following representations:

* :class:`perceval.BasicState` – a single configuration such as ``pcvl.BasicState([1, 0, 1, 0])``;
* :class:`perceval.StateVector` – an arbitrary superposition of basic states with complex amplitudes;
* **Deprecated**: Python lists, e.g. ``[1, 0, 1, 0]``. Lists are still recognised for backward compatibility but are
  immediately converted to their Perceval counterparts—new code should build explicit ``BasicState`` objects.

When ``input_state`` is passed, the layer always injects that photonic state. In more elaborate pipelines you may want
to cascade circuits and let the output amplitudes of the previous layer become the input state of the next. Merlin
calls this *amplitude encoding*: the probability amplitudes themselves carry information and are passed to the next
layer as a tensor. Enabling this behaviour is done with ``amplitude_encoding=True``; in that mode the forward input of
``QuantumLayer`` is the complex photonic state.

The snippet below prepares a dual-rail Bell state as the initial condition and evaluates a batch of classical parameters:

.. code-block:: python

    import torch
    import perceval as pcvl
    from merlin.algorithms.layer import QuantumLayer
    from merlin.core import ComputationSpace
    from merlin.measurement.strategies import MeasurementStrategy
    from merlin.measurement.

    circuit = pcvl.Unitary(pcvl.Matrix.random_unitary(4))  # some haar-random 4-mode circuit

    bell = pcvl.StateVector()
    bell += pcvl.BasicState([1, 0, 1, 0])
    bell += pcvl.BasicState([0, 1, 0, 1])
    print(bell) # bell is a state vector of 2 photons in 4 modes

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=2,
        input_state=bell,
        measurement_strategy=MeasurementStrategy.probs(computation_space=ComputationSpace.DUAL_RAIL),
    )

    x = torch.rand(10, circuit.m)  # batch of classical parameters
    amplitudes = layer(x)
    assert amplitudes.shape == (10, 2**2)

For comparison, the ``amplitude_encoding`` variant supplies the photonic state during the forward pass:

.. code-block:: python

    import torch
    import perceval as pcvl
    from merlin.algorithms.layer import QuantumLayer
    from merlin.core import ComputationSpace

    circuit = pcvl.Circuit(3)

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=2,
        amplitude_encoding=True,
        computation_space=ComputationSpace.UNBUNCHED,
        dtype=torch.cdouble,
    )

    prepared_states = torch.tensor(
        [[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
         [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]],
        dtype=torch.cdouble,
    )

    out = layer(prepared_states)

In the first example the circuit always starts from ``bell``; in the second, each row of ``prepared_states`` represents a
different logical photonic state that flows through the layer. This separation allows you to mix classical angle
encoding with fully quantum, amplitude-based data pipelines.


Returning typed objects
-------------------------

When ``return_object`` is set to True, the output of a ``forward()`` call depends of the ``measurement_strategy``. By default,
it is set to False. See the following output matrix to size what to expect as the return of a forward call.

|   measurement_strategy   |  return_object=False   |  return_object=True       |
|   :-------------------   |  :------------------   |  :----------------:       |
|   AMPLTITUDES            |  torch.Tensor          |  StateVector              |
|   PROBABILITIES          |  torch.Tensor          |  ProbabilityDistribution  |
|   PARTIAL_MEASUREMENT    |  PartialMeasurement    |  PartialMeasurement       |
|   MODE_EXPECTATIONS      |  torch.Tensor          |  torch.Tensor             |

Most of the typed objects can give the ``torch.Tensor`` as an output with the ``.tensor`` parameter. Only the 
PartialMeasurement object is a little different. See its according documentation.

The snippet below prepares a basic quantum layer and returns a ``ProbabilityDistribution`` object:

.. code-block:: python

    import torch
    import perceval as pcvl
    from merlin.algorithms.layer import QuantumLayer
    from merlin.core import ComputationSpace
    from merlin.measurement.strategies import MeasurementStrategy
    from merlin import ProbabilityDistribution

    circuit = pcvl.Unitary(pcvl.Matrix.random_unitary(4))  # some haar-random 4-mode circuit

    bell = pcvl.StateVector()
    bell += pcvl.BasicState([1, 0, 1, 0])
    bell += pcvl.BasicState([0, 1, 0, 1])
    print(bell) # bell is a state vector of 2 photons in 4 modes

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=2,
        input_state=bell,
        measurement_strategy=MeasurementStrategy.probs(computation_space=ComputationSpace.DUAL_RAIL),
        return_object=True,
    )

    x = torch.rand(10, circuit.m)  # batch of classical parameters
    probs = layer(x)
    assert isinstance(probs,ProbabilityDistribution)
    assert isinstance(probs.tensor,torch.Tensor)

Deprecations
-------------------------
.. deprecated:: 0.4
   The use of the ``no_bunching`` flag  is deprecated and will be removed in version 0.4.
   Use the :func:`measurement_strategy` parameter instead. See :ref:`migration_guide`.