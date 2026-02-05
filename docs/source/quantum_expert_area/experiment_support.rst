:github_url: https://github.com/merlinquantum/merlin

==================
Experiment Support
==================

Photonic experiments in `Perceval <https://perceval.quandela.net/>`_ bundle the elements of an optical circuit and its post-processing rules. MerLin uses this abstraction:
passing a :class:`perceval.Experiment` to :class:`~merlin.algorithms.layer.QuantumLayer` or to :class:`~merlin.algorithms.kernels.FeatureMap` lets you specify how each optical mode should be measured.

Why use an Experiment?
---------------

- **Single source of truth** – The circuit and every detector live in one object
  that can be shared across QuantumLayers or kernels.
- **Detector customization** – You can mix photon-number-resolving (PNR),
  threshold, or partially projected detectors mode-by-mode while keeping the
  rest of the configuration identical.
- **Photon-loss modelling** – Attach a :class:`perceval.NoiseModel` and MerLin
  will propagate its brightness/transmittance parameters before any detector
  logic, exposing photon loss events in the returned classical outcomes.

Noisy Simulations
-----------------

Perceval’s `NoiseModel <https://perceval.quandela.net/docs/v1.0/reference/utils/noise_model.html>`_ stores photon-loss parameters that
MerLin reads automatically. Assign it to the ``experiment.noise`` attribute with 
*brightness* and *transmittance*. The quantum layer combines
them into a survival probability and inserts a
:class:`~merlin.measurement.photon_loss.PhotonLossTransform`. As a result, ``output_keys`` include configurations where
photons disappear before detection and probability mass still sums to one.

For now, MerLin supports a global photon survival probability which applies to every mode. It is calculated as such:

.. math::

   survival\_probability = brightness \times transmittance

--------
Usage
--------

.. code-block:: python

    import perceval as pcvl
    import torch
    import merlin as ML

    circuit = pcvl.Circuit(3)
    circuit.add(0, pcvl.PS(pcvl.P("px")))
    circuit.add((0, 1), pcvl.BS())
    circuit.add((1, 2), pcvl.BS())

    experiment = pcvl.Experiment(circuit)
    # Model 10% loss from the source (brightness) and 15% from propagation loss (transmittance)
    experiment.noise = pcvl.NoiseModel(brightness=0.9, transmittance=0.85)

    # Noisy quantum layer
    layer = ML.QuantumLayer(
        input_size=1,
        experiment=experiment,
        input_parameters=["px"],
        input_state=[1, 1, 1],
    )

    x = torch.rand(3, 1)  # Generated data
    probs = layer(x)
    states = layer.output_keys  # includes both photon survival and photon loss outcomes

The default value for brightness and transmittance during noise model initialization is 1.0.

Detector Support
----------------

Perceval's `Dectectors <https://perceval.quandela.net/docs/v1.0/reference/components/detector.html>`_ are used to detect the number of photons on each mode. Indeed, every detector detects for one mode. Perceval exposes several detector families:

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
:class:`~merlin.measurement.detectors.DetectorTransform`, so probabilities
remain properly normalised regardless of the detection model. Finally, this detector transform is applied after the photon loss transform so they can both be used simultaneously without problem.

On a sidenote, detectors will be ignored if ``ComputationSpace`` is not ``FOCK`` because using a different computation space already translates to using special detectors.

--------
Usage
--------

.. code-block:: python

    import perceval as pcvl
    import torch
    import merlin as ML

    circuit = pcvl.Circuit(3)
    circuit.add(0, pcvl.PS(pcvl.P("px")))
    circuit.add((0, 1), pcvl.BS())
    circuit.add((1, 2), pcvl.BS())

    experiment = pcvl.Experiment(circuit)
    experiment.detectors[0] = pcvl.Detector.threshold()
    experiment.detectors[1] = pcvl.Detector.pnr()
    experiment.detectors[2] = pcvl.Detector.ppnr(n_wires=2)

    # Quantum layer with Detectors
    layer = ML.QuantumLayer(
        input_size=1,
        experiment=experiment,
        input_parameters=["px"],
        input_state=[1, 1, 1],
        computation_space=ML.ComputationSpace.FOCK  # Optional since this is the default value
    )

    x = torch.rand(3, 1)  # Generated data
    probs = layer(x)

    # Quantum kernel with Detectors
    feature_map = ML.FeatureMap(
        input_size=1,
        experiment=experiment,
        input_parameters=["px"]
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

Practical notes
===============

- Experiments used with QuantumLayer **must be unitary** and cannot carry
  Perceval heralding detectors.
- If at least one detector is defined, the quantum layer needs to have ``ComputationSpace.FOCK`` (default value). Photon filtering and detector post-processing are incompatible.
- Adding photon loss or detectors corresponds to performing a measurement of the quantum state. This collapses the quantum state and is therefore **incompatible with amplitude retrieval** (MeasurementStrategy.amplitudes()).
- Provide either brightness, transmittance, or both. Any missing parameter is
  treated as 1.0 so you can model source-only or circuit-only loss independently.
- Detector assignments use standard Python indexing or the Perceval
  ``.detectors`` mapping interface. Out-of-range indices raise the original
  Perceval error.
- Experiments are reusable: the same object can be passed to
  :class:`~merlin.algorithms.kernels.FeatureMap` or multiple QuantumLayers to
  guarantee consistent measurement semantics.