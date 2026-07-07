Hardware-Aware QML Guidelines with MerLin
=========================================

This guide outlines the best practices, constraints, and recommended workflows for designing Quantum Machine Learning (QML) models compatible with physical photonic Quantum Processing Units (QPUs).

Design Recommendations
----------------------

To simplify your workflow, this guide classifies QML design components into three distinct categories based on physical hardware readiness:

1. **Hardware-Oriented Designs (Go)**
   These features map directly to the physical constraints of the QPU and ensure optimal fidelity.
   
   * **Computation Space:** ``ComputationSpace.UNBUNCHED`` or ``ComputationSpace.DUAL_RAIL``.
   * **State Encoding:** Angle/phase encoding (directly maps to physical phase shifters).
   * **Input State:** ``BasicState`` initialization (with exactly 1 photon per pair of modes for Dual-Rail).
   * **Output Strategy:** ``MeasurementStrategy.probs()`` or ``mode_expectations()``.
   * **Components:** Native Mach-Zehnder Interferometers (MZI), beam splitters, and shallow circuit depths.
   * **Superposed states:** But only available with a state-preparation circuit.

2. **Simulation-Only Designs (No-Go on Hardware)**
   These features work perfectly in software simulation but are physically impossible to execute on current QPUs.
   
   * **Output Formats:** Raw amplitudes or ``StateVector`` outputs (due to the destructive nature of physical measurements).
   * **Computation Space:** ``ComputationSpace.FOCK`` (since physical QPUs do not have photon-number-resolving detectors yet).

3. **Discouraged-but-Possible Designs (Proceed with Caution)**
   These features can theoretically be sent to hardware but require complex workarounds, scale poorly, or drastically degrade performance.
   
   * **State Encoding:** Amplitude encoding (the required arbitrary dense state preparation is not hardware-realistic and creates oversized circuits).
   * **Input States:** Arbitrary superpositions without a native, hardware-validated state-preparation circuit.
   * **Circuit Depth:** Very deep simulated architectures (they scale poorly during transpilation and suffer from hardware decoherence).

Computation Spaces
------------------

When designing models for physical hardware execution, selecting the appropriate computation space is critical.

Voici une version corrigée et enrichie de tes explications. J'ai affiné le vocabulaire technique pour qu'il soit parfaitement aligné avec la documentation officielle de MerLin et les réalités matérielles de la photonique quantique.

* **Recommended:** Use `ComputationSpace.UNBUNCHED` or `ComputationSpace.DUAL_RAIL`.
* **UNBUNCHED:** This space restricts the simulated configurations to at most one photon per mode.
   It aligns perfectly with the physical reality of current threshold detectors, which can only detect the presence or absence of a photon (0 or 1), but cannot count them.
   It is the default computation space in MerLin.
* **DUAL-RAIL:** This is a special case of the unbunched space, where exactly one photon is shared across every pair of modes.
   By restricting the allowed states, this logical encoding provides robustness against photon loss and ensures higher fidelity on physical hardware.


* **Avoid:** `ComputationSpace.FOCK`.
* **FOCK:** This space simulates the full Fock space, allowing all photon-number configurations, including the accumulation of multiple photons in a single mode (bunching). 
   You must avoid this on current physical QPUs because it assumes the use of photon-number-resolving (PNR) detectors, which are not yet available on standard hardware. 
   It should be reserved exclusively for high-fidelity software simulations.

Physical photonic QPUs do not have photon-number-resolving detectors yet.

State Encoding
--------------

Since arbitrary dense state preparation is not hardware realistic, algorithms used for encoding take too many resources. We strongly recommend not using it for hardware implementations.

* **Hardware Execution:** Use **angle/phase encoding** as it directly maps to physical phase shifters.
* **Simulation:** Amplitude encoding remains acceptable for pure simulation workflows but should be avoided for hardware-targeted models.

Supported Output Formats
------------------------

Due to the destructive nature of quantum measurements, hardware execution restricts the available output types:

* **Supported:**
  
  * ``probabilities`` (Full probability distribution derived from output samples)
  * ``mode_expectations`` (Average photon counts per mode)
  * ``sampled counts/probabilities`` (Shot-based measurements)

* **Unsupported:** Raw amplitudes or ``StateVector`` outputs cannot be directly retrieved from physical hardware.
   We can only infer state probabilities at the end of the circuit, using shots and samples.

Understanding Shot-Based Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Physical QPUs cannot output raw amplitudes or a ``StateVector``. Due to the destructive nature of quantum measurement, a single execution of a circuit yields only a single classical outcome (a sample). 

To compute expectations or output probabilities, the hardware must physically prepare, execute, and measure the exact same quantum circuit repeatedly. Each independent repetition is called a **shot**. By accumulating a large number of shots, we can reconstruct the underlying probability distribution of the possible output states.

**Hardware Realities to Consider:**

* **Photon Loss:** Physical components are imperfect. During execution, photons can be lost before reaching the detectors, resulting in invalid or degraded states.
* **Runtime vs. Accuracy Trade-off:** While a higher number of shots yields a more accurate statistical estimation (reducing shot noise), it linearly increases the physical runtime and execution cost on the QPU. You must find the optimal balance between statistical fidelity and execution time for your specific model.

Photonic QPUs natively process ``BasicState`` inputs (e.g., specific photon configurations).

.. note::
   If your pipeline relies on a ``StateVector``, you must implement a state-preparation circuit prior to the main ansatz to convert the state natively on the hardware. Avoid arbitrary superposition unless there is a real state preparation path.

**Shot efficiency guidance**

Note that the larger the state space, the more samples are required. If the number of shots is too low, the sampled distribution will not be representative, making the resulting output inaccurate. 
Conversely, requesting a massive number of shots does not cause any issues in software simulation, as the Law of Large Numbers ensures the distribution converges perfectly. 
However, on a physical QPU, photons can be lost during execution. 
Because of this photon loss, many shots are discarded as invalid, meaning you must run a much higher number of raw shots to gather enough valid samples.

A good strategy to ensure using shot efficiently can be setting an arbitrary high number of shots (e.g., a default of 1024 or 10000) for every execution is highly inefficient. 
To optimize hardware usage and mitigate execution costs, we strongly recommend implementing a **Diminishing Returns Strategy** (Incremental Execution).

* **Incremental Execution:** Instead of requesting a massive block of shots upfront, execute the circuit in smaller, successive batches (e.g., chunks of 100 shots).
* **Statistical Stopping Criterion:** Continuously monitor and compute the empirical output distribution after each batch. 
* **The Diminishing Returns Point:** Halt the execution precisely when the system identifies that adding new shots no longer significantly alters the estimated probability distribution. 

This dynamic approach ensures you only spend QPU time when it actively contributes to resolving the statistical noise, cutting down unnecessary hardware usage without sacrificing the accuracy of your gradients or final inferences.

Input-State Constraints


Dual-Rail Constraints
^^^^^^^^^^^^^^^^^^^^^

When using Dual-Rail encoding, you **must** initialize the ``BasicState`` with exactly **one photon per pair of two modes**. 
Failing to respect this constraint triggers post-selection mechanisms that discard states that falls outside the computation space,
resulting in severe information loss and degraded performance.

Circuit Design Recommendations
------------------------------

To ensure efficient execution and high fidelity, adhere to the following hardware-aware design principles:

* **Component Selection:** Favor native photonic components such as Mach-Zehnder Interferometers (MZI), beam splitters, and phase shifters, as they map directly to the QPU's physical architecture.
* **Circuit Depth:** Avoid unnecessarily deep circuits. Large simulated circuits may scale poorly during hardware compilation (transpilation), leading to suboptimal or inefficient physical implementations on a constrained QPU.
* **permutations:** They are natively present on the circuit. Avoid design that create excessive bunching

Recommended Pipeline
--------------------

The standard workflow follows a hybrid classical-quantum-classical pipeline:

1. **Classical Preprocessing:** Normalize and prepare features.
2. **Angle Encoding:** Inject features using angle encoding into a hardware-optimized ansatz.
3. **Output Measurement Strategy:** ``MeasurementStrategy.probs(...)`` or ``mode_expectations(...)`` to return output probabilities for each state.
4. **Classical Post-processing:** Bind the output tensor to a classical neural network for final task mapping.
5. **Classical ML Algorithm:** Read the output of the ``QuantumLayer`` and process it.
6. **Train/Inference Separation:** Train the model on a GPU and infer it on a QPU.

Implementation Example
----------------------

Below is a standard hardware-compatible circuit definition using ``merlin`` and ``perceval``:

.. code-block:: python

   import torch
   import torch.nn as nn
   import perceval as pcvl
   from merlin import QuantumLayer, LexGrouping
   from merlin.builder import CircuitBuilder
   from merlin import ComputationSpace, MeasurementStrategy

   # 1. Define the parameterized photonic circuit
   builder = CircuitBuilder(n_modes=6)
   builder.add_entangling_layer(trainable=True, name="U1")
   builder.add_angle_encoding(modes=[0, 1, 2, 3], name="input")   # Maps 4 features -> 4 modes
   builder.add_rotations(trainable=True, name="theta")             # Extra expressivity
   builder.add_superpositions(depth=1)                            # Fixed mixing layer

   # 2. Instantiate the hardware-aware Quantum Layer (Fixed for MerLin v0.4)
   core = QuantumLayer(
      input_size=6,                                      # Number of classical features
      builder=builder,
      n_photons=3,                                       # Equivalent to input_state = [1, 0, 1, 0, 1, 0]
      dtype=torch.float32,
      measurement_strategy=MeasurementStrategy.probs(computation_space=ComputationSpace.UNBUNCHED),
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

Summary of Unsupported or Discouraged Features
----------------------------------------------

For your first hardware-aware implementation, ensure your pipeline **does not** use any of the following features:

.. list-table:: Hardware Compatibility Summary
   :widths: 25 35 40
   :header-rows: 1

   * - Feature
     - Status
     - Alternative / Reason
   * - ``FOCK``
     - **Unsupported**
     - QPUs currently lack photon-number-resolving detectors. Use ``UNBUNCHED`` or ``DUAL_RAIL``.
   * - ``StateVector`` / Raw Amplitudes
     - **Unsupported**
     - Physical hardware only yields destructive, shot-based measurements. Use ``MeasurementStrategy.probs()``.
   * - Amplitude Encoding
     - **Discouraged**
     - Arbitrary dense state preparation requires exponential circuit depth, which is not hardware-realistic. Use **angle/phase encoding**.
   * - Arbitrary Superposition (Input)
     - **Discouraged**
     - Hard to prepare natively without a dedicated state-preparation circuit. Stick to ``BasicState`` initialization.
   * - Deep Simulated Circuits
     - **Discouraged**
     - Leads to high error rates and compilation (transpilation) failures due to QPU coherence time limits. Keep circuits shallow.