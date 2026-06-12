.. _quantum_layer:

=======================
QuantumLayer Essentials
=======================

The :class:`~merlin.algorithms.layer.QuantumLayer` is MerLin’s core building block for integrating quantum computation, as a single module, in a machine learning pipeline. 
It combines a Perceval photonic circuit (or experiment), optional classical parameters, and detector logic into a single differentiable module.

---------------
Overview
---------------

- **Autograd ready** – QuantumLayer exposes a PyTorch ``Module`` interface,
  supports batching and differentiable forward passes, and plays nicely with
  optimisers or higher-level architectures.
- **Input encoding strategies** - Pick a data encoding method: angle or amplitude encoding.. See :doc:`./angle_amplitude_encoding` for more information.
- **Output measurement strategies** – Select between probabilities, per-mode expectations,
  or raw amplitudes through :class:`~merlin.measurement.strategies.MeasurementStrategy`.
  The layer validates incompatible combinations (e.g. detectors with amplitude read-out). For more information on this and all possible output configurations, visit :doc:`./measurement_strategy`.
  Grouping can be configured directly in the measurement strategy parameter. See :doc:`./grouping` for more information.
- **Multiple construction paths** – Build layers from
  the convenience :meth:`~merlin.algorithms.layer.QuantumLayer.simple` factory,
  a :class:`~merlin.builder.circuit_builder.CircuitBuilder`, a custom
  :class:`pcvl.Circuit` or a fully specified :class:`pcvl.Experiment`.
- **Detector awareness** – Layers automatically derive detector transforms from
  the experiment, enabling threshold, PNR, or hybrid detection schemes.
- **Photon-loss aware** – Experiments carrying a :class:`pcvl.NoiseModel`
  trigger an automatic photon-loss transform so survival and loss outcomes share
  a single, normalised output distribution.

-----------------------
Initialisation recipes
-----------------------

``QuantumLayer.simple()``
~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~merlin.algorithms.layer.QuantumLayer.simple` helper generates a
trainable interferometer with angle encoding that has input size + 1 modes. 
It is convenient for quick experiments, baselines or for machine
learning experts without any prior knowledge in quantum machine learning.

.. code-block:: python

   import merlin as ML

   layer = ML.QuantumLayer.simple(
       input_size=4,
       measurement_strategy=ML.MeasurementStrategy.probs(),
   )

   x = torch.rand(16, 4)
   probs = layer(x)

CircuitBuilder
~~~~~~~~~~~~~~

Use MerLin’s :class:`~merlin.builder.circuit_builder.CircuitBuilder` utilities to describe a circuit at a higher level. The builder maintains a record of the trainable parameters and the parameters used as layer inputs. A prefix-based naming scheme separates trainable parameters from those used as layer inputs. This is an ideal tool for quantum machine learning experts who do not have any experience with Perceval. More information in the CircuitBuilder API reference: :class:`~merlin.builder.circuit_builder.CircuitBuilder`

.. code-block:: python

   import torch
   import merlin as ML

   builder = ML.CircuitBuilder(n_modes=4)
   builder.add_superpositions(depth=1)
   builder.add_angle_encoding(modes=[0, 1], name="x")
   builder.add_rotations(trainable=True, name="theta")

   layer = ML.QuantumLayer(
       input_size=2,
       builder=builder,
       measurement_strategy=ML.MeasurementStrategy.probs(computation_space=ML.ComputationSpace.UNBUNCHED),
   )

   x = torch.rand(4, 2)
   probs = layer(x)

Custom circuit
~~~~~~~~~~~~~~

When you already have a :class:`pcvl.Circuit`, provide the classical input
layout and the trainable parameter prefixes explicitly. This initialization requires
a good understanding of Perceval.

.. code-block:: python

   import perceval as pcvl
   import torch
   import merlin as ML

   circuit = pcvl.Circuit(3)
   circuit.add((0, 1), pcvl.BS())
   circuit.add(0, pcvl.PS(pcvl.P("phi")))

   layer = ML.QuantumLayer(
       input_size=1,
       circuit=circuit,
       input_parameters=["phi"],
       trainable_parameters=["theta"],
       input_state=[1, 0, 0],
       measurement_strategy=ML.MeasurementStrategy.probs(),
   )

   x = torch.linspace(0.0, 1.0, steps=8).unsqueeze(1)
   probs = layer(x)

.. note::

  ``input_state=[...]`` is accepted as a convenience input, but the layer stores it as a Perceval
  `pcvl.BasicState <https://perceval.quandela.net/docs/v1.2/reference/utils/states.html>`_ (access the occupation vector via ``list(layer.input_state)``).

Experiment-driven
~~~~~~~~~~~~~~~~~

If you want to simulate a noise model or specify detectors characteristics, configure a :class:`pcvl.Experiment` and pass it directly. The QuantumLayer inherits the circuit, detectors, and any photon-loss noise model you attached. This scheme is the one that gives the user the most options when utilizing a QuantumLayer.

.. code-block:: python

   import perceval as pcvl
   import torch
   import merlin as ML

   circuit = pcvl.Circuit(2)
   circuit.add((0, 1), pcvl.BS())

   experiment = pcvl.Experiment(circuit)
   experiment.detectors[0] = pcvl.Detector.threshold()
   experiment.detectors[1] = pcvl.Detector.pnr()
   experiment.noise = pcvl.NoiseModel(brightness=0.95, transmittance=0.9)

   layer = ML.QuantumLayer(
       input_size=0,
       experiment=experiment,
       input_state=[1, 1],
       measurement_strategy=ML.MeasurementStrategy.probs(),
   )

   probs = layer()
   detector_keys = layer.output_keys

---------------------------------
Photon loss and detectors
---------------------------------

- Without an experiment, the layer defaults to ideal PNR detection on every
  mode, mirroring Perceval’s default behaviour.
- ``experiment.noise = pcvl.NoiseModel(...)`` adds photon-loss sampling ahead of
  detector transforms. The resulting ``output_keys`` and ``output_size`` cover
  every survival/loss configuration implied by the noise model.
- ``MeasurementStrategy.amplitudes()`` requires access to raw complex amplitudes
  and is therefore incompatible with custom detectors **or** photon-loss noise
  models. Attempting this combination raises a ``RuntimeError``. To emulate a
  detector pipeline while still inspecting amplitudes, run the layer without
  detectors and apply :class:`~merlin.measurement.detectors.DetectorTransform`
  manually to the resulting amplitudes.
- Call :meth:`~merlin.algorithms.layer.QuantumLayer.output_keys` to inspect
  the classical outcomes produced by the detector transform.

-----------
Notes
-----------

- ``input_state`` must match the number of circuit modes. When unspecified, the photons (denoted by ``n_photons``) are evenly distributed across the modes (for instance, for dual-rail it defaults to ``[1,0,1,0,...]``).
- Both strong simulation (SLOS, which computes exact probabilities) and weak simulation (sampling) are supported. Sampling can be enabled using the ``shots`` and ``sampling_method`` parameters. See the :doc:`../quantum_expert_area/SLOS` for more information about strong and weak simulations.
- The ``layer.parameters()`` method provides access to the trainable parameters (if any), just like any standard PyTorch layer.
- Inspect ``layer.has_custom_noise_model`` and ``layer.output_keys`` to confirm
  whether photon loss is active and how it alters the output distribution.

.. warning::
   *Deprecated since version 0.3:* The use of the ``no_bunching`` flag  is deprecated and is removed since version 0.3.0.
   Use the ``computation_space`` flag inside ``measurement_strategy`` instead. See :doc:`/user_guide/migration_guide`.

----------------------
API Reference
----------------------

.. autoclass:: merlin.algorithms.layer.QuantumLayer
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
