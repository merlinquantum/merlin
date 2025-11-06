.. _experiment_guide:

Experiments
===========

Photonic experiments in `Perceval <https://perceval.quandela.net/>`_ bundle a
unitary circuit together with classical *detectors* that interpret the raw Fock
state probabilities produced by the simulator. MerLin reuses this abstraction:
passing a :class:`perceval.Experiment` to :class:`~merlin.algorithms.layer.QuantumLayer` or to :class:`~merlin.algorithms.kernels.FeatureMap`
lets you specify how each optical mode should be measured without rewriting the
quantum layer or quantum feature map itself.

---------------
Why use an Experiment?
---------------

- **Single source of truth** – The circuit and every detector live in one object
  that can be shared across QuantumLayers, kernels, or feed-forward blocks.
- **Detector customization** – You can mix photon-number-resolving (PNR),
  threshold, or partially projected detectors mode-by-mode while keeping the
  rest of the configuration identical.
- **Photon-loss modelling** – Attach a :class:`perceval.NoiseModel` and MerLin
  will propagate its brightness/transmittance parameters before any detector
  logic, exposing photon loss events in the returned classical outcomes.
- **Consistent defaults** – Any unspecified mode automatically falls back to an
  ideal PNR detector, matching Perceval's behaviour.

-----------
Detector overview
-----------

Detectors are used to detect the number of photons on each mode. Indeed, every detector detects for one mode. Perceval exposes several detector families; the most common ones in MerLin are:

``pcvl.Detector.pnr()``
    Ideal photon-number-resolving detector. This detector detects any number of photons present. Leaves the Fock basis unchanged.

``pcvl.Detector.threshold()``
    Binary detector that only distinguishes between “zero” and “non-zero”
    photons on a mode.

``pcvl.Detector.ppnr(n_wires, max_detections=None)``
    Partially projected detector that groups several optical modes (``n_wires``)
    into a logical output with optional truncation through ``max_detections``. This detector can detect up to ``max_detections`` photons and ``max_detections`` is bounded above by ``n_wires``.

Detectors can return multiple classical outcomes for a single quantum state. The
layer converts the raw Fock probabilities into detector keys through
``DetectorTransform`` (see :mod:`merlin.sampling.detectors`), so probabilities
remain properly normalised regardless of the detection model.

-----------
Photon loss with NoiseModel
-----------

Perceval’s :class:`~perceval.NoiseModel` stores photon-loss parameters that
MerLin reads automatically. Assign it to the ``experiment.noise`` attribute with 
*brightness* and *transmittance*. The quantum layer combines
them into a survival probability and inserts a
:class:`~merlin.measurement.photon_loss.PhotonLossTransform` ahead of any
detector mapping. As a result, ``output_keys`` include configurations where
photons disappear before detection and probability mass still sums to one.

.. code-block:: python

   experiment = pcvl.Experiment(circuit)
   # Model 10% loss from the source (brightness) and 15% from propagation loss
   experiment.noise = pcvl.NoiseModel(brightness=0.9, transmittance=0.85)

   layer = ML.QuantumLayer(
       input_size=0,
       experiment=experiment,
       input_state=[1, 1],
   )

   probs = layer()
   layer.output_keys  # includes both survival and loss outcomes

If only one of the parameters is set, MerLin assumes the other equals 1.0.
Detectors and photon loss stack cleanly: the loss transform expands the Fock
basis and the detector transform then maps it to classical outcomes (threshold,
PNR, etc.).

--------
Usage example
--------

.. code-block:: python

   import perceval as pcvl
   import torch
   import merlin as ML

   # 1. Build the base circuit
   circuit = pcvl.Circuit(3)
   circuit.add((0, 1), pcvl.BS())
   circuit.add(2, pcvl.PS(pcvl.P("px")))
   circuit.add((1, 2), pcvl.BS())

   # 2. Wrap it in a Perceval Experiment, configure noise model and detectors
   experiment = pcvl.Experiment(circuit)
   experiment.noise = pcvl.NoiseModel(brightness=0.9, transmittance=0.85)
   experiment.detectors[0] = pcvl.Detector.threshold()
   experiment.detectors[1] = pcvl.Detector.pnr()
   experiment.detectors[2] = pcvl.Detector.ppnr(n_wires=2)

   # 3.1 Feed the experiment into a QuantumLayer with an input Fock state
   layer = ML.QuantumLayer(
       input_size=0,
       experiment=experiment,
       input_state=[1, 1, 1],
       input_parameters=["px"],
   )
   x = torch.rand(4, 1)  # Generate data
   probs = layer(x)  # NoiseModel-aware & Detector-aware probability tensor
   keys = layer.output_keys  # Classical outcomes produced by the detectors

   # 3.2 Feed the experiment into a quantum kernel FeatureMap to build a FidelityKernel
   feature_map = ML.FeatureMap(
       input_size=1,
       experiment=experiment,
       input_parameters=["px"],
   )
   kernel = ML.FidelityKernel(
       feature_map=feature_map,
       input_state=[1, 1, 1],
   )

   X_train = torch.rand(4, 1)
   X_test = torch.rand(2, 1)
   # Construct the training & test kernel matrices
   K_train = kernel(X_train)
   K_test = kernel(X_test, X_train)

-----------
Practical notes
-----------

- Experiments used with QuantumLayer **must be unitary** and cannot carry
  Perceval heralding detectors.
- If at least one detector is defined, the quantum layer needs to have ``computation_space="fock"`` (default value). Photon filtering and detector post-processing are incompatible.
- Photon-loss noise models extend the classical basis returned by the layer. The
  **amplitude measurement strategy is therefore unavailable when detectors or a
  noise model are attached** to the experiment.
- Provide either brightness, transmittance, or both. Any missing parameter is
  treated as 1.0 so you can model source-only or circuit-only loss independently.
- Detector assignments use standard Python indexing or the Perceval
  ``.detectors`` mapping interface. Out-of-range indices raise the original
  Perceval error.
- Experiments are reusable: the same object can be passed to
  :class:`~merlin.algorithms.kernels.FeatureMap` or multiple QuantumLayers to
  guarantee consistent measurement semantics.

-----------
Related API
-----------

- :class:`perceval.Experiment`
- :class:`perceval.Detector`
- :class:`~merlin.algorithms.layer.QuantumLayer`
- :func:`merlin.sampling.detectors.resolve_detectors`
