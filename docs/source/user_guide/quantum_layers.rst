:github_url: https://github.com/merlinquantum/merlin

.. _quantum_layers:

==============
Quantum Layers
==============

How to Build a QuantumLayer
===========================

Merlin's :class:`~merlin.algorithms.QuantumLayer` wraps a photonic circuit so it can be trained as a PyTorch module. You can instantiate it through three complementary entry points:

- :class:`~merlin.builder.CircuitBuilder` keeps the circuit declarative.
- :meth:`~merlin.algorithms.QuantumLayer.simple` assembles a ready-to-train 10-mode circuit with a configurable parameter budget.
- A custom :class:`perceval.Circuit` or :class:`perceval.Experiment` gives you complete control over the optical layout, detectors, and noise.


Using CircuitBuilder
--------------------

:class:`~merlin.builder.CircuitBuilder` is the most appropriate approach if you understand the basics of photonic QML without specifically knowing how to use `Perceval <https://perceval.quandela.net>`_. It has an intuitive API which allows you to build a photonic circuit that you can then send to a QuantumLayer.

1. Build the circuit.

.. code-block:: python

    import numpy as np
    from merlin.builder import CircuitBuilder

    # Instantiate the CircuitBuilder with the number of modes 
    builder = CircuitBuilder(n_modes=6)

    # Use add_rotations to add phase shifters
    builder.add_rotations(modes=[0, 1, 2, 3, 4, 5], trainable=True)

    # Use add_angle_encoding to add an angle-based input encoding to the circuit
    builder.add_angle_encoding(modes=[0, 1, 2, 3, 4, 5], name="input", scale=np.pi)

    # Use add_entangling_layer to add a layer that entangles the specified modes together
    builder.add_entangling_layer(modes=[0, 5], trainable=True, model="mzi")

    # Use add_superpositions to add beam splitters
    builder.add_superpositions(modes=[0, 1, 2, 3, 4, 5], trainable=True, depth=2)

2. Instantiate the layer directly from the builder.

.. code-block:: python

    from merlin.algorithms import QuantumLayer

    builder_layer = QuantumLayer(
        input_size=6,
        builder=builder,
        input_state = [1, 0, 1, 0, 1, 0]
    )

3. (Optional) Convert the declarative circuit into a Perceval circuit.

.. code-block:: python

   import perceval as pcvl

   circuit = builder.build()

   circuit_layer = QuantumLayer(
         input_size=6,
         circuit=circuit,
         input_state=[1, 0, 1, 0, 1, 0],
         input_parameters=list(builder.input_parameter_prefixes),
         trainable_parameters=list(builder.trainable_parameter_prefixes),
   )

Using QuantumLayer.simple
-------------------------

:meth:`~merlin.algorithms.QuantumLayer.simple` is appropriate for you if you do not have any photonic QML experience, but understand the basics of machine learning. It automatically builds a 10-modes photonic circuit based on the input size and desired number of parameters you provide. It is an approach that reduces the complexity for using a photonic circuit but also reduces the amount of control the user has on it.

.. code-block:: python

    import torch
    from merlin.algorithms import QuantumLayer

    simple_layer = QuantumLayer.simple(
        input_size=6,
        n_params=100,
        output_size=10
    )

Automatically, the input state to the circuit is chosen to be [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] (so 5 photons) and the minimum number of parameters is 90.

Using perceval.Circuit and perceval.Experiment
----------------------------------------------

This last approach is appropriate for `Perceval <https://perceval.quandela.net>`_ experts and users who want the highest level of control over the photonic circuit.

1. Build or load a :class:`perceval.Circuit`.

There are several ways to build a circuit with Perceval. One of them is to use a GenericIterferometer.

.. code-block:: python

    import perceval as pcvl
    from merlin.algorithms import QuantumLayer

    circuit = perceval.Circuit(n_modes=6)

    wl = pcvl.GenericInterferometer(
        6, 
        lambda i: pcvl.BS()
        // pcvl.PS(pcvl.P(f"theta_li{i}"))
        // pcvl.BS()
        // pcvl.PS(pcvl.P(f"theta_lo{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    circuit.add(0, wl)

    for mode in range(6):
        circuit.add(mode, pcvl.PS(pcvl.P(f"input{mode}")))

    wr = pcvl.GenericInterferometer(
        6,
        lambda i: pcvl.BS()
        // pcvl.PS(pcvl.P(f"theta_ri{i}"))
        // pcvl.BS()
        // pcvl.PS(pcvl.P(f"theta_ro{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    circuit.add(0, wr)

    # Display the circuit
    pcvl.pdisplay(circuit)

2. Include your circuit in a perceval.Experiment (optional).

This step is optional, as a QuantumLayer can be initialized without an experiment. However, doing it allows you to integrate a noise model and detectors into the QuantumLayer. For more information on those, see :doc:`../quantum_expert_area/index`.

.. code-block:: python

    experiment = perceval.Experiment(circuit)
    # Optionally add a noise model
    experiment.noise = perceval.NoiseModel(brightness=0.9, transmittance=1.0)
    # Optionally add detectors
    experiment.detectors[0] = perceval.Detector.threshold()

3. Build a QuantumLayer.

The QuantumLayer can be built from the circuit or the experiment, not both:

.. code-block:: python

    # With circuit
    circuit_layer = QuantumLayer(
        input_size=6,
        circuit=circuit,
        input_state=[1, 0, 1, 0, 1, 0],
        input_parameters=["input"],
        trainable_parameters=["theta"]
    )

    # Or with experiment
    experiment_layer = QuantumLayer(
        input_size=6,
        experiment=experiment,
        input_state=[1, 0, 1, 0, 1, 0],
        input_parameters=["input"],
        trainable_parameters=["theta"]
    )

The experiment must stay unitary and free from post-selection or heralding, because Merlin's differentiable backend currently supports only that regime.

Angle and Amplitude Encoding
============================

Measurement Strategy
====================

Select a :class:`~merlin.measurement.MeasurementStrategy` to control the classical output of the layer.

There are three measurement strategies:

- **MeasurementStrategy.PROBABILITIES** returns a probability vector with one entry per possible output Fock state. This is the **default** strategy.

.. code-block:: python
    
    from merlin.builder import CircuitBuilder
    import torch
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy
    
    # Create the builder
    builder = CircuitBuilder(n_modes=6)
    builder.add_angle_encoding(modes=list(range(6)), name="input")
    builder.add_entangling_layer(trainable=True)

    # Set the MeasurementStrategy
    layer = QuantumLayer(
        input_size=6,
        builder=builder,
        input_state = [1, 0, 1, 0, 1, 0],
        measurement_strategy=MeasurementStrategy.PROBABILITIES  # Optional since default value
    )

    x = torch.rand((1, 6))  # Generate a random datapoint
    output = layer(x)

    # The output size is accessible
    output_size = layer.output_size
    # The output state keys corresponding with the probabilities are also accessible
    output_keys = layer.output_keys

- **MeasurementStrategy.MODE_EXPECTATIONS** aggregates the per-mode photon expectations. The output width equals the number of modes in the underlying circuit.

.. code-block:: python

     import torch
     from merlin.algorithms import QuantumLayer
     from merlin.measurement import MeasurementStrategy

    # Set the MeasurementStrategy
    layer = QuantumLayer(
        input_size=6,
        builder=builder,
        input_state = [1, 0, 1, 0, 1, 0],
        measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS
    )

    x = torch.rand((1, 6))  # Generate a random datapoint
    output = layer(x)

    # The output size is the number of modes (so 6 in this situation)
    output_size = layer.output_size

- **MeasurementStrategy.AMPLITUDES** bypasses detectors and photon-loss transforms, returning the complex amplitudes for every possible Fock state. This strategy is available only in simulation: detectors, noise models, or positive ``shots`` are incompatible.

.. code-block:: python
    
    import torch
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy

    # Set the MeasurementStrategy
    layer = QuantumLayer(
        input_size=6,
        builder=builder,
        input_state = [1, 0, 1, 0, 1, 0],
        measurement_strategy=MeasurementStrategy.AMPLITUDES
    )

    x = torch.rand((1, 6))  # Generate a random datapoint
    # output will have dtype complex
    output = layer(x)

    # The output size is accessible
    output_size = layer.output_size
    # The output state keys corresponding with the amplitudes are also accessible
    output_keys = layer.output_keys
