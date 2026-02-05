:github_url: https://github.com/merlinquantum/merlin

===============
Basic Concepts
===============

This guide introduces the fundamental concepts behind Merlin's approach to quantum neural networks.

Merlin centres on three high-level tools you will see throughout the quickstart:

- **Photonic simulation** with fast classical solvers so you can prototype locally before targeting hardware.
- **CircuitBuilder** for declaratively authoring interferometers, encoding steps, and trainable components.
- **QuantumLayer** for dropping your circuit into any PyTorch model with automatic differentiation support.

Conceptual Overview
===================

Merlin bridges the gap between physical quantum circuits and high-level machine learning interfaces through a layered architecture. From lowest to highest level:

1. **Physical Quantum Circuits**: The actual photonic hardware (or fast simulation thereof)
2. **Photonic Backend**: Mathematical models of quantum circuits with configurable components
3. **CircuitBuilder** (:class:`~merlin.builder.circuit_builder.CircuitBuilder`): Declarative interface for assembling photonic circuits
4. **Encoding**: Strategies for mapping classical features to quantum parameters
5. **Measurement Strategy**: Strategies for converting quantum outputs to classical outputs
6. **QuantumLayer**: High-level PyTorch interface that combines all these concepts

Let's explore each level in detail.

1. Physical Foundation: Photonic Circuits
=========================================

At the foundation, Merlin uses **photonic quantum computing**, where information is encoded in photons (particles of light) traveling through optical circuits. These circuits consist of:

- **Modes**: Independent optical pathways (like waveguides) that can carry photons
- **Photons**: Quantum information carriers; more photons enable more complex quantum interference
- **Optical Components**: Beam splitters, phase shifters, and interferometers that manipulate photon paths

.. image:: ../_static/img/Interferometer_training.png
   :alt: 12-mode interferometer with 6 photons

On the image above, you can see a 12-mode interferometer with 6 photons entering. The photons interfere as they pass through the optical components, creating complex quantum states. We measure the output distribution of photons across the modes to extract information.
Here, we could write

.. code-block:: python
    # A simple photonic system
    n_modes = 12        # 4 optical pathways
    n_photons = 6     # 2 photons for quantum interference
    input_state = pcvl.BasicState("|1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0>")  # Alternating photon pattern

For a deeper understanding of photonic quantum computing fundamentals, see :doc:`../quantum_expert_area/architectures`.

First, we present the overview of the building of a QuantumLayer, brick by brick using the `CircuitBuilder`.  

.. image:: ../_static/img/diagrams.png
   :alt: Flow from data preprocessing to quantum layer and measurement


Overview of the Merlin hybrid workflow.

2. Backend : Mathematical Models
========================================

To run this `layer`, the **Backend** provides mathematical representations of quantum circuits, handling the complex quantum mechanics while exposing a clean interface for machine learning.

Key responsibilities:

- **State Evolution**: Computing how quantum states change through the circuit (see :doc:`../quantum_expert_area/SLOS`)
- **Simulation Modes**: Switching between sampling and deterministic simulation for rapid prototyping
- **Parameter Management**: Tracking which components are configurable vs. fixed
- **Measurement Simulation**: Converting quantum states to probability distributions

Merlin comes with high-performance classical simulators (SLOS and Clifford-based modes) so you can prototype and train without immediate access to hardware. Switching to hardware later only requires changing the backend configuration.

3. Encoding: Classical-to-Quantum Mapping
=========================================

**Encoding** defines how classical input features are mapped to quantum circuit parameters. This is crucial because quantum circuits operate on phases and amplitudes, not raw feature values.

**Key Steps**:

1. **Normalization**: Ensure inputs are in :math:`[0,1]` range
2. **Scaling**: Apply scaling for quantum parameter ranges
3. **Circuit Mapping**: Distribute to quantum parameters based on the configured circuit

Angle Encoding
^^^^^^^^^^^^^^

**Angle encoding** rotates programmable elements of the circuit by an angle proportional to each classical feature.

.. code-block:: python

    import merlin as ML
    import numpy as np

    builder = ML.CircuitBuilder(n_modes=4)
    builder.add_angle_encoding(scale=np.pi)    # Rotations proportional to input features

Angle encoding keeps circuit depth compact while still giving continuous control over the interferometer. Keep signals normalized (or pass them through a bounded activation such as ``torch.tanh``) so the mapped rotation angles remain in a sensible range.

Amplitude Encoding
^^^^^^^^^^^^^^^^^^

**Amplitude encoding** maps classical data values to the amplitudes of a quantum state.
Given a normalized vector :math:`x = (x_0, x_1, ..., x_{2^n-1})`, the encoding creates
a quantum state :math:`|\psi\gt = \sum_i x_i |i\gt` where :math:`|i\gt` represents the computational basis state.
This technique requires n qubits to encode :math:`2^n` data points, offering exponential
compression but requiring complex state preparation circuits, unless the state can be prepared at source.

Initial State Patterns
^^^^^^^^^^^^^^^^^^^^^^

The initial distribution of photons affects quantum behavior:

.. code-block:: python

    # Example state patterns
    ML.StatePattern.PERIODIC     # [1,0,1,0] - alternating photons
    ML.StatePattern.SPACED       # [1,0,0,1] - evenly spaced
    ML.StatePattern.SEQUENTIAL   # [1,1,0,0] - consecutive

Different patterns create different types of quantum interference and correlations.

For detailed encoding strategies and optimization techniques, see :doc:`../user_guide/angle_amplitude_encoding`.

4. Measurement Strategy: Quantum-to-Classical Conversion
==================================================

**Measurement Strategy** converts quantum measurement results (probability distributions or amplitudes) into classical outputs.

Quantum circuits produce probability distributions or amplitudes (in simulation) over possible photon configurations. Measurement strategy determines which formatting to use.

.. code-block:: python

    # Common measurement strategies
    ML.MeasurementStrategy.probs()  # Default: full Fock distribution
    ML.MeasurementStrategy.mode_expectations()   # Per-mode photon statistics
    ML.MeasurementStrategy.amplitudes()       # Complex amplitudes (simulation only)

To reduce the dimensionality of the Fock distribution after measurement, compose your layer with a grouping
:class:`~merlin.utils.grouping.LexGrouping` or :class:`~merlin.utils.grouping.ModGrouping`.

**Key Concept**: Measurement strategy bridges the gap between quantum measurements and classical outputs. The choice affects both the interpretability and expressivity of your quantum layer.

For detailed comparisons and selection guidelines, see :doc:`../user_guide/measurement_strategy` and :doc:`../user_guide/grouping`.

Grouping strategies
-------------------

Two simple, built-in grouping strategies are provided to reduce the high-dimensional
Fock outputs to a smaller set of classical features:

- ``LexGrouping``: assigns consecutive output indices to the same bucket (lexicographic
    ordering). This preserves locality in the Fock index ordering and is a good default
    when nearby basis states are expected to encode related features.
- ``ModGrouping``: maps output indices to buckets using a modulo operation, effectively
    interleaving states across groups. This is useful when you want to mix information
    from distant basis states or avoid clustering correlated states into the same bucket.

Example::

        # Collapse the Fock output into 3 features by grouping consecutive indices
        grouped_layer = ML.LexGrouping(quantum_layer.output_size, 3)

        # Alternatively, interleave states into 3 groups
        interleaved = ML.ModGrouping(quantum_layer.output_size, 3)

5. High-Level Interface: QuantumLayer
=====================================

The **QuantumLayer** combines all these concepts into a PyTorch-compatible interface that plays nicely with standard deep learning tooling. Build a circuit with the builder interface, then pass it to the layer alongside the parameters you want Merlin to manage:

.. code-block:: python

    import merlin as ML
    import numpy as np

    builder = ML.CircuitBuilder(n_modes=6)
    builder.add_angle_encoding(name="px", scale=np.pi)
    builder.add_entangling_layer(trainable=True)

    quantum_layer = ML.QuantumLayer(
        input_size=4,                                   # Classical feature dimension
        builder=builder,                                # CircuitBuilder instance
        n_photons=2,                                    # Number of photons in the register
        input_state=[1, 0, 1, 0, 1, 0],                 # Initial photon pattern
        input_parameters=["px"],                        # Prefix for angle-encoded features  
        measurement_strategy=ML.MeasurementStrategy.probs(computation_space=ML.ComputationSpace.FOCK), # Choose Fock space handling and output type
        return_object=False                             # Choose whether or not to return a typed object after a forward call depending on the measurement strategy.
    )

    # Optional: down-sample the Fock distribution to 3 features using a Linear Layer
    mapped_layer = nn.Sequential(
        quantum_layer,
        nn.Linear(quantum_layer.output_size, 3),
    )

Key parameters to tune when instantiating :class:`~merlin.algorithms.layer.QuantumLayer`:

- ``builder`` or ``circuit``: define the photonic circuit you want to simulate.
- ``n_photons`` and ``input_state``: set the quantum resources entering the interferometer.
- ``input_parameters``: prefixes generated by :meth:`~merlin.builder.circuit_builder.CircuitBuilder.add_angle_encoding`.
- ``measurement_strategy``: pick the classical readout (probabilities, mode expectations, amplitudes).
- ``computation_space``: control simulation modes and alternative encodings.
- ``return_object``: Choose to return a typed object as the forward output depending  on the ``measurement_strategy``. Take a look at :doc:`../api_reference/api/merlin.algorithms.layer` for more details about the return types.


Putting It All Together
=======================

Here's how all these concepts work together in practice:

.. code-block:: python

    import torch
    import torch.nn as nn
    import merlin as ML
    import numpy as np

    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()

            # Classical preprocessing
            self.classical_input = nn.Linear(8, 4, bias=False)

            # Quantum processing layer
            builder = ML.CircuitBuilder(n_modes=6)
            builder.add_angle_encoding(name="px", scale=np.pi)
            builder.add_entangling_layer(trainable=True)
            builder.add_superpositions(trainable=True)

            quantum_core = ML.QuantumLayer(
                input_size=4,
                builder=builder,
                n_photons=2,
                input_state=[1, 0, 1, 0, 1, 0],
                input_parameters=["px"],
                measurement_strategy=measurement_strategy=ML.MeasurementStrategy.probs(computation_space=ML.ComputationSpace.FOCK),
                return_object=False,
            )
            self.quantum = nn.Sequential(
                quantum_core,
                # Lexicographic grouping: collapse the high-dimensional Fock output into `6` buckets
                # by assigning consecutive output indices to the same group. This preserves local
                # structure in the Fock ordering and is useful when nearby basis states encode
                # similar features.
                ML.LexGrouping(quantum_core.output_size, 6),
            )

            # Classical output
            self.classifier = nn.Linear(6, 3)

        def forward(self, x):
            x = self.classical_input(x)
            x = torch.sigmoid(x)           # Normalize for quantum encoding
            x = self.quantum(x)            # Quantum transformation
            return self.classifier(x)

    # The quantum stack automatically handles:
    # - Photonic backend simulation
    # - Classical-to-quantum encoding
    # - Quantum computation
    # - Quantum-to-classical measurement (plus optional grouping)


Design Guidelines
=================

When choosing configurations, consider these general principles:

**Start Simple**: Begin with a small ``CircuitBuilder`` (4–6 modes), default ``PROBABILITIES`` measurement, and a lightweight classical head.

**Match Complexity to Problem**:

- Simple problems → few modes, shallow entangling layers
- Complex problems → more modes, combine entangling layers with superpositions

**Computational Constraints**:

- Limited resources → fewer photons, prefer ``ComputationSpace.UNBUNCHED`` when your circuit avoids photon bunching
- More resources available → increase photon count or depth for richer expressivity

**Experiment Systematically**: The quantum advantage often comes from the right combination of circuit design, encoding, measurement strategy, and optional grouping for your specific problem.

For detailed optimization strategies and advanced configurations, see the :doc:`../user_guide/index` section.

Next Steps
==========

Now that you understand the conceptual hierarchy:

1. **Start Simple**: Prototype with ``CircuitBuilder`` defaults and the built-in simulator
2. **Experiment**: Try different CircuitBuilder layouts, measurement strategies, and grouping modules for your use case
3. **Optimize**: Tune circuit size and encoding strategies based on performance
4. **Advanced Usage**: Explore custom circuit definitions when needed

For practical implementation, continue to :doc:`first_quantum_layer` to see these concepts in action.
