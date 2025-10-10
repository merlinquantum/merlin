:github_url: https://github.com/merlinquantum/merlin

===============
Basic Concepts
===============

This guide introduces the fundamental concepts behind Merlin's approach to quantum neural networks.

Conceptual Overview
===================

Merlin bridges the gap between physical quantum circuits and high-level machine learning interfaces through a layered architecture. From lowest to highest level:

1. **Physical Quantum Circuits**: The actual photonic hardware (or simulation thereof)
2. **Photonic Backend**: Mathematical models of quantum circuits with configurable components
3. **Ansatz** (:mod:`merlin.builder.ansatz`): Logical circuit templates that define shape of quantum circuits to be implemented on the backend
4. **Encoding**: Strategies for mapping classical features to quantum parameters
5. **Output Mapping**: Methods for converting quantum states to classical outputs
6. **QuantumLayer**: High-level PyTorch interface that combines all these concepts

Let's explore each level in detail.

1. Physical Foundation: Photonic Circuits
=========================================

At the foundation, Merlin uses **photonic quantum computing**, where information is encoded in photons (particles of light) traveling through optical circuits. These circuits consist of:

- **Modes**: Independent optical pathways (like waveguides) that can carry photons
- **Photons**: Quantum information carriers; more photons enable more complex quantum interference
- **Optical Components**: Beam splitters, phase shifters, and interferometers that manipulate photon paths

.. code-block:: python

    # A simple photonic system
    n_modes = 4        # 4 optical pathways
    n_photons = 2      # 2 photons for quantum interference
    # Initial state: [1, 0, 1, 0] = photons in modes 0 and 2

For a deeper understanding of photonic quantum computing fundamentals, see :doc:`../research/architectures`.

2. Backend : Mathematical Models
========================================

The **Backend** provides mathematical representations of quantum circuits, handling the complex quantum mechanics while exposing a clean interface for machine learning.

Key responsibilities:

- **State Evolution**: Computing how quantum states change through the circuit
- **Parameter Management**: Tracking which components are configurable vs. fixed
- **Measurement Simulation**: Converting quantum states to probability distributions



3. Ansatz (:mod:`merlin.builder.ansatz`): Logical Circuit Templates
==================================================================

.. note::

   The ansatz abstractions moved from ``merlin.core.ansatz`` to
   :mod:`merlin.builder.ansatz` to align with the builder utilities.

An **Ansatz** is a logical template that defines the structure of your quantum circuit, specifying:

- The arrangement of optical components
- Which parameters can be trained (learnable weights)
- Which parameters encode input features

Merlin provides different ansatz types that determine how features are mapped to quantum parameters and how complex the resulting transformations can be:

.. code-block:: python

    # Different ansatz types offer different complexity/efficiency tradeoffs
    circuit_type=ML.CircuitType.PARALLEL        # Simple, efficient
    circuit_type=ML.CircuitType.SERIES          # Balanced, good default
    circuit_type=ML.CircuitType.PARALLEL_COLUMNS # Complex, expressive

**Key Concept**: Ansatz types represent different strategies for organizing quantum circuits:

For instance, the `PARALLEL` ansatz is corresponding to the following circuit structure:

CIRCUIT STRUCTURE HERE AND DESCRIPTION.


The choice depends on your problem complexity and computational constraints. For detailed comparisons and guidance on choosing ansatz types, see the :doc:`../user_guide/circuit_types` section.

4. Encoding: Classical-to-Quantum Mapping
=========================================

**Encoding** defines how classical input features are mapped to quantum circuit parameters. This is crucial because quantum circuits operate on phases and amplitudes, not raw feature values.

Basic Encoding Process
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Classical features (must be normalized to [0,1])
    x = [0.3, 0.7, 0.9]

    # Quantum encoding (automatic in Merlin)
    quantum_parameters = np.pi * x * bandwidth_coefficients

**Key Steps**:

1. **Normalization**: Ensure inputs are in :math:`[0,1]` range
2. **Scaling**: Apply scaling for quantum parameter ranges
3. **Circuit Mapping**: Distribute to quantum parameters based on ansatz

Amplitude encoding Process
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Amplitude encoding** maps classical data values to the amplitudes of a quantum state.
Given a normalized vector :math:`x = (x_0, x_1, ..., x_{2^n-1})`, the encoding creates
a quantum state :math:`|\psi\gt = \sum_i x_i |i\gt` where :math:`|i\gt` represents the computational basis state.
This technique requires n qubits to encode :math:`2^n` data points, offering exponential
compression but requiring complex state preparation circuits, unless the state can be prepared at source.


**Key Steps**:

1. **Normalization**: Ensure inputs are in :math:`[0,1]` range
2. **Scaling**: Apply scaling for quantum parameter ranges
3. **Circuit Mapping**: Distribute to quantum parameters based on ansatz

Initial State Patterns
^^^^^^^^^^^^^^^^^^^^^^

The initial distribution of photons affects quantum behavior:

.. code-block:: python

    # Example state patterns
    ML.StatePattern.PERIODIC     # [1,0,1,0] - alternating photons
    ML.StatePattern.SPACED       # [1,0,0,1] - evenly spaced
    ML.StatePattern.SEQUENTIAL   # [1,1,0,0] - consecutive

Different patterns create different types of quantum interference and correlations.

For detailed encoding strategies and optimization techniques, see :doc:`../user_guide/encoding`.

5. Output Mapping: Quantum-to-Classical Conversion
==================================================

**Output Mapping** converts quantum measurement results (probability distributions) into classical neural network activations.

Quantum circuits produce probability distributions over possible photon configurations. Output mapping strategies determine how these probabilities become the classical outputs your PyTorch model sees.

.. code-block:: python

    # Common output mapping strategies
    ML.OutputMappingStrategy.LINEAR      # Learnable linear combination (most flexible)
    ML.OutputMappingStrategy.LEXGROUPING # Groups probabilities by quantum structure
    ML.OutputMappingStrategy.NONE        # Direct quantum probabilities

**Key Concept**: Output mapping bridges the gap between quantum measurements and classical neural network expectations. The choice affects both the interpretability and expressivity of your quantum layer.

For detailed comparisons and selection guidelines, see :doc:`../user_guide/output_mappings`.

6. High-Level Interface: QuantumLayer
=====================================

The **QuantumLayer** combines all these concepts into a PyTorch-compatible interface:

.. code-block:: python

    # High-level interface combining all concepts
    quantum_layer = ML.QuantumLayer(
        input_size=4,                                              # Classical input dimension
        output_size=3,                                             # Desired output dimension
        circuit=circuit,                                           # Photonic backend + ansatz
        trainable_parameters=["theta"],                            # Which parameters to train
        input_parameters=["px"],                                   # Encoding parameters
        input_state=[1, 0, 1, 0, 1, 0],                            # Initial photon state
        output_mapping_strategy=ML.OutputMappingStrategy.LINEAR    # Output mapping choice
    )

Using the Experiment Interface
==============================

For most users, Merlin provides a simplified interface that handles these complexities automatically:

.. code-block:: python

    # Simple experiment configuration
    experiment = ML.Experiment(
        circuit_type=ML.CircuitType.SERIES,                    # Ansatz choice
        n_modes=4,                                             # Circuit size
        n_photons=2,                                           # Quantum resource
        state_pattern=ML.StatePattern.PERIODIC,                # Encoding strategy
        use_bandwidth_tuning=True,                             # Learnable encoding
        reservoir_mode=False                                   # Full training vs reservoir
    )

    # Creates quantum layer automatically
    quantum_layer = experiment.create_layer(
        input_size=4,
        output_size=3,
        output_mapping_strategy=ML.OutputMappingStrategy.LINEAR
    )

Putting It All Together
=======================

Here's how all these concepts work together in practice:

.. code-block:: python

    import torch
    import torch.nn as nn
    import merlin as ML

    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()

            # Classical preprocessing
            self.classical_input = nn.Linear(8, 4)

            # Quantum processing layer
            experiment = ML.Experiment(
                circuit_type=ML.CircuitType.SERIES,        # Ansatz: balanced complexity
                n_modes=6,                                 # Photonic backend: 6 modes
                n_photons=2,                               # 2 photons for interference
                state_pattern=ML.StatePattern.PERIODIC,    # Encoding: alternating photons
                use_bandwidth_tuning=True                  # Learnable encoding scaling
            )

            self.quantum_layer = experiment.create_layer(
                input_size=4,
                output_size=6,
                output_mapping_strategy=ML.OutputMappingStrategy.LINEAR  # Flexible output mapping
            )

            # Classical output
            self.classifier = nn.Linear(6, 3)

        def forward(self, x):
            x = self.classical_input(x)
            x = torch.sigmoid(x)           # Normalize for quantum encoding
            x = self.quantum_layer(x)      # Quantum transformation
            return self.classifier(x)

    # The quantum layer automatically handles:
    # - Photonic backend simulation
    # - Classical-to-quantum encoding
    # - Quantum computation
    # - Quantum-to-classical output mapping

Design Guidelines
=================

When choosing configurations, consider these general principles:

**Start Simple**: Begin with default settings (SERIES ansatz, LINEAR output mapping) and adjust based on performance.

**Match Complexity to Problem**:

- Simple problems → PARALLEL ansatz, smaller circuits
- Complex problems → SERIES or PARALLEL_COLUMNS ansatz, larger circuits

**Computational Constraints**:

- Limited resources → smaller circuits, PARALLEL ansatz
- More resources available → larger circuits, more expressive ansatz

**Experiment Systematically**: The quantum advantage often comes from the right combination of ansatz, encoding, and output mapping for your specific problem.

For detailed optimization strategies and advanced configurations, see the :doc:`../user_guide/index` section.

Next Steps
==========

Now that you understand the conceptual hierarchy:

1. **Start Simple**: Begin with the Experiment interface and default settings
2. **Experiment**: Try different ansatz types and output mappings for your use case
3. **Optimize**: Tune circuit size and encoding strategies based on performance
4. **Advanced Usage**: Explore custom circuit definitions when needed

For practical implementation, continue to :doc:`first_quantum_layer` to see these concepts in action.
