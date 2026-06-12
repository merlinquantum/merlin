=============================
Photonic Kernel Methods
=============================

Introduction
------------

Quantum kernels leverage quantum circuits to compute similarity measures between data points in ways that classical kernels cannot. Recent experimental research has demonstrated that photonic quantum kernels can outperform state-of-the-art classical methods including Gaussian and neural tangent kernels for certain classification tasks [1]_, exploiting quantum interference effects that are computationally intractable for classical computers to simulate.

In photonic quantum computing, these kernels can be implemented using linear optical circuits operating at room temperature, making them particularly attractive for near-term quantum machine learning applications. This guide explains how MerLin implements photonic quantum kernels for machine learning tasks like classification and regression, making these capabilities accessible through familiar PyTorch and scikit-learn interfaces.

The theoretical foundation for quantum kernel methods builds on the observation that quantum computing and kernel methods share a common principle: efficiently performing computations in exponentially large Hilbert or Fock spaces [2]_. By encoding classical data into quantum states, we can access feature spaces that are difficult or impossible for classical computers to work with efficiently [3]_.

What is a quantum kernel?
-------------------------

A quantum kernel measures similarity between data points by comparing their quantum states after encoding through a photonic circuit.

**Mathematical formulation**

Given a photonic feature map that embeds a classical datapoint :math:`x \in \mathbb{R}^d` into a unitary :math:`U(x)`, the fidelity kernel between two inputs :math:`x_1, x_2` and a chosen input Fock state :math:`|s\rangle` is

.. math::

   k(x_1, x_2) \;=\; \big|\langle s\,|\, U^{\dagger}(x_2)\, U(x_1) \,|\, s\rangle\big|^2 \,.

**Physical interpretation**

- :math:`U(x)` encodes your classical data into a quantum circuit transformation
- The overlapping application :math:`U^{\dagger}(x_2)U(x_1)` compares how the two encodings relate
- The squared amplitude gives a real-valued similarity measure in :math:`[0,1]`
- When :math:`x_1 = x_2`, the kernel returns 1 (perfect similarity)


In MerLin, :class:`~merlin.algorithms.kernels.FidelityKernel` evaluates this kernel efficiently with the SLOS simulator, optionally including photon-loss and detector models from a :class:`pcvl.Experiment`.

Core building blocks
--------------------

MerLin exposes three cooperating components:

- :class:`~merlin.algorithms.kernels.FeatureMap`
  Encodes classical inputs into a photonic circuit and produces the corresponding unitary matrix. You can pass a pre-built :class:`pcvl.Circuit`, a declarative :class:`~merlin.builder.circuit_builder.CircuitBuilder`, or a full :class:`pcvl.Experiment`.

- :class:`~merlin.algorithms.kernels.FidelityKernel`
  Given a feature map, computes Gram matrices (train/test) by simulating transition probabilities through SLOS. The input Fock state is inferred by default and can be overridden when a specific photon pattern is required. Supports optional sampling, photon loss, and detector transforms.

- :class:`~merlin.builder.circuit_builder.CircuitBuilder`
  Declaratively builds circuits with angle-encoding metadata. This is the preferred path when the feature map circuit is created in MerLin.

Quick Start Decision Guide
--------------------------

**"I want to quickly try quantum kernels on my data"**
    → Build a feature map with :meth:`~merlin.algorithms.kernels.FeatureMap.simple` and pass it to :class:`~merlin.algorithms.kernels.FidelityKernel`

**"I need to customize the circuit architecture"**
    → Use :class:`~merlin.builder.circuit_builder.CircuitBuilder`, then wrap it in :class:`~merlin.algorithms.kernels.FeatureMap`

**"I have an existing Perceval circuit/experiment"**
    → Create a :class:`~merlin.algorithms.kernels.FeatureMap` from your circuit or experiment, then wrap it in :class:`~merlin.algorithms.kernels.FidelityKernel`

**"I need to model realistic hardware effects"**
    → Create a :class:`pcvl.Experiment` with :class:`pcvl.NoiseModel` and detectors

**"I want to compare classical vs quantum performance"**
    → Compute both kernel matrices and use with scikit-learn ``SVC(kernel="precomputed")``

How feature maps encode data
----------------------------

For kernel computation, :class:`~merlin.algorithms.kernels.FidelityKernel`
treats :class:`~merlin.algorithms.kernels.FeatureMap` as a descriptor and
delegates encoding to its internal ``_CCInvQuantumLayer`` backend. The supported
encoding contract is:

1. For feature maps created from a
   :class:`~merlin.builder.circuit_builder.CircuitBuilder`, builder-provided
   angle-encoding metadata defines how raw input features are converted into
   circuit parameters.
2. For feature maps created directly from a :class:`pcvl.Circuit` or
   :class:`pcvl.Experiment`, ``input_size`` must match the number of circuit
   input parameters selected by ``input_parameters``. Inputs are passed with
   that parameter dimension.

Detectors, photon loss and experiments
--------------------------------------

If the feature map exposes a :class:`pcvl.Experiment`, the kernel composes a photon‑loss transform derived from the experiment's :class:`pcvl.NoiseModel` and then applies detector transforms (threshold or PNR) before reading probabilities. This means kernel values naturally reflect survival probabilities and detector post‑processing.

If no experiment is provided, the kernel constructs one from the circuit (unitary, no detectors, no noise).

Parameters and behaviour
------------------------

Below is a summary of key constructor arguments and their effects. See the API reference for full signatures.

FeatureMap Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``circuit``
     - Circuit | None
     - None
     - Perceval circuit defining the photonic transformation
   * - ``builder``
     - CircuitBuilder | None
     - None
     - Alternative: build circuit declaratively
   * - ``experiment``
     - Experiment | None
     - None
     - Full experiment including noise/detectors
   * - ``input_size``
     - int
     - *required*
     - Dimensionality of input feature vectors
   * - ``input_parameters``
     - str | List[str]
     - ``"input"``
     - Parameter prefix(es) for feature encoding
   * - ``trainable_parameters``
     - List[str] | None
     - None
     - Additional parameters to expose for gradient training
   * - ``dtype``
     - torch.dtype
     - torch.float32
     - Numerical precision
   * - ``device``
     - torch.device
     - cpu
     - Computation device

FidelityKernel Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``feature_map``
     - FeatureMap
     - *required*
     - The feature map instance to use
   * - ``input_state``
     - List[int] | None
     - None
     - Fock state :math:`|s\rangle`; omit it to infer the state from the circuit size and ``n_photons``
   * - ``n_photons``
     - int | None
     - None
     - Number of photons used to infer ``input_state`` when ``input_state`` is omitted
   * - ``shots``
     - int | None
     - None
     - If positive, use pseudo-sampling; ``None`` or ``0`` means exact probabilities
   * - ``sampling_method``
     - str
     - ``"multinomial"``
     - Sampling strategy: multinomial/binomial/gaussian
   * - ``computation_space``
     - ComputationSpace | str | None = None
     - None
     - Logical state space: ``FOCK``, ``UNBUNCHED``, or ``DUAL_RAIL``
   * - ``force_psd``
     - bool
     - True
     - Project Gram matrix to positive semi-definite
   * - ``dtype``
     - torch.dtype
     - *from feature_map*
     - Simulation precision
   * - ``device``
     - torch.device
     - *from feature_map*
     - Simulation device

Implementation highlights
-------------------------

Internally, :class:`~merlin.algorithms.kernels.FidelityKernel` delegates
pairwise circuit construction and SLOS evaluation to its ``_CCInvQuantumLayer``
backend. The backend builds the pairwise circuits
:math:`U^{\dagger}(x_2) U(x_1)` in a vectorised way and asks the SLOS graph to
compute detection probabilities for the resolved input state. If photon loss
and/or detectors are defined, the raw probabilities are transformed accordingly
before the scalar kernel is read.

When constructing a training Gram matrix (``x2 is None``), only the upper triangle is simulated and mirrored to the lower triangle, then the diagonal is set to 1. With ``force_psd=True``, the matrix is symmetrised and projected to PSD by zeroing negative eigenvalues in an eigendecomposition.

Input state inference
---------------------

The input state does not need to be provided for standard fidelity kernels.
When ``input_state`` is omitted, :class:`~merlin.algorithms.kernels.FidelityKernel`
uses the number of modes in ``feature_map.circuit`` to build an alternating
single-photon state, for example ``[1, 0, 1, 0, 1]`` for five modes. If
``n_photons`` is provided, the kernel places that many photons into the inferred
state, filling alternating positions first. Pass ``input_state`` only when the
experiment requires an explicit occupation pattern.

Quickstarts and recipes
-----------------------

Minimal example (factory)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from merlin import ComputationSpace
   from merlin.algorithms.kernels import FeatureMap, FidelityKernel

   # Build a feature map with default circuit topology (n_modes = input_size + 1 = 3)
   feature_map = FeatureMap.simple(input_size=2)

   # Wrap it in a fidelity kernel
   kernel = FidelityKernel(
       feature_map=feature_map,
       computation_space=ComputationSpace.FOCK,
       dtype=torch.float32,
       device=torch.device("cpu"),
   )

   X_train = torch.rand(10, 2)
   X_test = torch.rand(5, 2)
   K_train = kernel(X_train)  # (10, 10)
   K_test = kernel(X_test, X_train)  # (5, 10)

Custom experiment with detectors and loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import perceval as pcvl
   from merlin import ComputationSpace
   from merlin.algorithms.kernels import FeatureMap, FidelityKernel

   circuit = pcvl.Circuit(6)
   circuit.add(0, pcvl.PS(pcvl.P("px0")))
   circuit.add(2, pcvl.PS(pcvl.P("px1")))
   circuit.add(4, pcvl.PS(pcvl.P("px2")))

   experiment = pcvl.Experiment(circuit)
   experiment.noise = pcvl.NoiseModel(brightness=0.9, transmittance=0.85)
   experiment.detectors[0] = pcvl.Detector.threshold()
   experiment.detectors[2] = pcvl.Detector.threshold()
   experiment.detectors[4] = pcvl.Detector.threshold()

   fmap = FeatureMap(
       input_size=3,
       input_parameters="px",
       experiment=experiment,
   )

   kernel = FidelityKernel(
       feature_map=fmap,
       shots=0,
       computation_space=ComputationSpace.FOCK,
   )

   X = torch.rand(8, 3)
   K = kernel(X)  # (8, 8)

Declarative builder + kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from merlin.algorithms.kernels import FeatureMap, FidelityKernel
   from merlin.builder import CircuitBuilder

   builder = CircuitBuilder(n_modes=6)
   builder.add_superpositions(depth=1)
   builder.add_angle_encoding(
       modes=[0, 1, 2, 3],
       name="input",
       scale=float(torch.pi),
   )
   builder.add_rotations(trainable=True, name="phi")
   builder.add_superpositions(depth=1)

   feature_map = FeatureMap(
       builder=builder,
       input_size=4,
       input_parameters=None,
   )
   kernel = FidelityKernel(
       feature_map=feature_map,
       n_photons=2,
       shots=0,
   )

   X = torch.rand(32, 4)
   K = kernel(X)

.. note::

  To force a specific photon pattern, pass ``input_state=[...]``. The list is
  converted to a Perceval
  `pcvl.BasicState <https://perceval.quandela.net/docs/v1.2/reference/utils/states.html>`_ internally.

Using with scikit‑learn (precomputed kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.svm import SVC

   K_train = kernel(X_train)
   K_test = kernel(X_test, X_train)
   clf = SVC(kernel="precomputed").fit(K_train, y_train)
   y_pred = clf.predict(K_test)

Comparing quantum vs classical kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sklearn.svm import SVC
    from sklearn.metrics.pairwise import rbf_kernel
    import torch
    from merlin.algorithms.kernels import FeatureMap, FidelityKernel

    # Prepare data
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Quantum kernel
    feature_map = FeatureMap.simple(input_size=4)  # n_modes = input_size + 1 = 5
    qkernel = FidelityKernel(feature_map=feature_map)
    K_train_q = qkernel(X_train_t).detach().numpy()
    K_test_q = qkernel(X_test_t, X_train_t).detach().numpy()

    clf_q = SVC(kernel="precomputed")
    clf_q.fit(K_train_q, y_train)
    acc_quantum = clf_q.score(K_test_q, y_test)

    # Classical RBF kernel
    gamma = 1.0 / X_train_t.shape[1]
    K_train_rbf = rbf_kernel(X_train, gamma=gamma)
    K_test_rbf = rbf_kernel(X_test, X_train, gamma=gamma)

    clf_rbf = SVC(kernel="precomputed")
    clf_rbf.fit(K_train_rbf, y_train)
    acc_classical = clf_rbf.score(K_test_rbf, y_test)

    print(f"Quantum kernel accuracy: {acc_quantum:.3f}")
    print(f"Classical RBF accuracy: {acc_classical:.3f}")

Performance and batching tips
-----------------------------

- Build feature maps once and reuse them; the converter caches parameter specs.
- Prefer contiguous tensors on the same device/dtype for inputs to minimise transfers.
- When memory is constrained, reduce the number of modes/photons or change ``ComputationSpace.FOCK`` to ``ComputationSpace.UNBUNCHED`` where physically appropriate.

Limitations and caveats
-----------------------

- The feature map encodes classical features via angle encoding; amplitude encoding of state vectors is not part of the kernel API.
- ``ComputationSpace.UNBUNCHED`` cannot be used together with detectors defined in the experiment.
- Consider GPU acceleration via ``device=torch.device("cuda")`` for large datasets

API reference
-------------

See :mod:`merlin.algorithms.kernels` for complete class and method signatures and additional usage notes.

References
----------

.. [1] Z. Yin et al., "Experimental quantum-enhanced kernel-based machine learning on a photonic processor," Nature Photonics (2025). https://www.nature.com/articles/s41566-025-01682-5
.. [2] M. Schuld and N. Killoran, "Quantum machine learning in feature Hilbert spaces." https://arxiv.org/abs/1803.07128
.. [3] V. Havlicek et al., "Supervised learning with quantum-enhanced feature spaces," Nature (2019). https://www.nature.com/articles/s41586-019-0980-2
