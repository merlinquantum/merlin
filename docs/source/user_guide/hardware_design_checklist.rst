Hardware-Aware QML Guidelines with MerLin
=========================================

This guide outlines the best practices, constraints, and recommended workflows for designing Quantum Machine Learning (QML) models compatible with physical photonic Quantum Processing Units (QPUs).

Computation Spaces
------------------

When designing models for physical hardware execution, selecting the appropriate computation space is critical.

* **Recommended:** Use ``ComputationSpace.unbunched`` or ``ComputationSpace.DUAL_RAIL``.
* **Avoid:** ``ComputationSpace.bunched`` and ``ComputationSpace.FOCK``. Physical photonic QPUs do not natively support bunched states or arbitrary Fock spaces due to hardware limitations.

State Encoding
--------------

Photonic QPUs do not natively perform amplitude encoding. 

* **Hardware Execution:** Use **angle/phase encoding** as it directly maps to physical phase shifters.
* **Simulation:** Amplitude encoding remains acceptable for pure simulation workflows but should be avoided for hardware-targeted models.

Supported Output Formats
------------------------

Due to the destructive nature of quantum measurements, hardware execution restricts the available output types:

* **Supported:**
  
  * ``probabilities`` (Full probability distribution)
  * ``mode_expectations`` (Average photon counts per mode)
  * ``sampled counts/probabilities`` (Shot-based measurements)

* **Unsupported:** Raw amplitudes or ``StateVector`` outputs cannot be directly retrieved from physical hardware.

Input-State Constraints
-----------------------

Photonic QPUs natively process ``BasicState`` inputs (e.g., specific photon configurations). 

.. note::
   If your pipeline relies on a ``StateVector``, you must implement a state-preparation circuit prior to the main ansatz to convert the state natively on the hardware.

Dual-Rail Constraints
^^^^^^^^^^^^^^^^^^^^^

When using Dual-Rail encoding, you **must** initialize the ``BasicState`` with exactly **one photon per pair of two modes**. Failing to respect this constraint triggers post-selection mechanisms that discard invalid states, resulting in severe information loss and degraded performance.

Circuit Design Recommendations
------------------------------

To ensure efficient execution and high fidelity, adhere to the following hardware-aware design principles:

* **Component Selection:** Favor native photonic components such as Mach-Zehnder Interferometers (MZI), beam splitters, and phase shifters, as they map directly to the QPU's physical architecture.
* **Circuit Depth:** Avoid unnecessarily deep circuits. Large simulated circuits may scale poorly during hardware compilation (transpilation), leading to suboptimal or inefficient physical implementations on a constrained QPU.

Recommended Architecture Pattern
--------------------------------

The standard workflow follows a hybrid classical-quantum-classical pipeline:

1. **Classical Preprocessing:** Normalize and prepare features.
2. **Quantum Layer:** Inject features using angle encoding into a hardware-optimized ansatz.
3. **Classical Post-processing:** Bind the output tensor to a classical neural network for final task mapping.

Implementation Example
----------------------

Below is a standard hardware-compatible circuit definition using ``merlin`` and ``perceval``:

.. code-block:: python

   import perceval as pcvl
   import torch.nn as nn
   import merlin as ML
   from ..builder import CircuitBuilder

   # 1. Define the parameterized photonic circuit
   builder = CircuitBuilder(n_modes=6)
   builder.add_entangling_layer(trainable=True, name="U1")
   builder.add_angle_encoding(modes=[0, 1, 2, 3], name="input")   # Maps 4 features -> 4 modes
   builder.add_rotations(trainable=True, name="theta")             # Extra expressivity
   builder.add_superpositions(depth=1)                            # Fixed mixing layer

   # 2. Instantiate the hardware-aware Quantum Layer
   core = QuantumLayer(
       input_size=4,                                      # Number of classical features
       Experiment=ComputationSpace.UNBUNCHED,             # UNBUNCHED or DUAL_RAIL for hardware
       builder=builder,
       n_photons=3,                                       # Equivalent to input_state = [1, 1, 1, 0, 0, 0]
       dtype=torch.float32,
       measurement_strategy=MeasurementStrategy.probs(),  # Hardware-compatible strategy
   )

   # 3. Connect the quantum layer to a classical PyTorch network
   model = nn.Sequential(
       core,
       LexGrouping(core.output_size, 3),                  # Transforms output to a tensor of shape (B, 3)
   )

Deployment Workflow
-------------------

.. important::
   Always **train the model on a GPU** using quantum simulation to compute gradients efficiently, then **deploy and infer on the QPU** for real-world hardware validation.