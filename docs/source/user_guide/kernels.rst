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
  Given a feature map and an input Fock state, computes Gram matrices (train/test) by simulating transition probabilities through SLOS. Supports optional sampling, photon loss and detector transforms.

- :class:`~merlin.algorithms.kernels.KernelCircuitBuilder`
  A convenience helper to create a :class:`~merlin.algorithms.kernels.FeatureMap` and then a :class:`~merlin.algorithms.kernels.FidelityKernel` with minimal boilerplate.

Quick Start Decision Guide
--------------------------

**"I want to quickly try quantum kernels on my data"**
    → Use ``FidelityKernel.simple()`` with default parameters

**"I need to customize the circuit architecture"**
    → Use ``KernelCircuitBuilder`` for declarative circuit construction

**"I have an existing Perceval circuit/experiment"**
    → Create a ``FeatureMap`` from your circuit, then wrap in ``FidelityKernel``

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

The callable ``encoder`` accepted by :class:`~merlin.algorithms.kernels.FeatureMap`
is legacy-only. It is primarily used by deprecated
:meth:`~merlin.algorithms.kernels.FeatureMap.compute_unitary`. ``FidelityKernel``
still supports it for compatibility when a direct circuit or experiment needs
to map raw inputs to a larger circuit parameter vector, but this emits a
``DeprecationWarning``. New code should put encoding logic in
:class:`~merlin.builder.circuit_builder.CircuitBuilder` or pass inputs with the
circuit parameter dimension.

Direct circuit and experiment feature maps whose ``input_size`` differs from
the number of circuit input parameters are also supported only for
compatibility. Without ``encoder``, the kernel uses the legacy subset-sum
expansion or truncation behavior and emits a ``DeprecationWarning``.

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
   * - ``encoder``
     - Callable | None
     - None
     - Deprecated compatibility encoding; prefer ``CircuitBuilder.add_angle_encoding``
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
     - List[int]
     - *required*
     - Fock state :math:`|s\rangle`; must match circuit modes
   * - ``shots``
     - int
     - 0
     - If > 0, use sampling; 0 means exact probabilities
   * - ``sampling_method``
     - str
     - ``"multinomial"``
     - Sampling strategy: multinomial/binomial/gaussian
   * - ``computation_space``
     - ComputationSpace | str | None = None
     - None
     - Chose the computation state between UNBUNCHED (maximum one photon per mode), FOCK (multiple phorons per mode allowed) and DUAL_RAIL
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

.. warning:: *Deprecated since version 0.3:*
   The use of the ``no_bunching`` flag  is deprecated and is removed since version 0.3.0.
   Use the ``computation_space`` flag instead. See :doc:`/user_guide/migration_guide`.

Implementation highlights
-------------------------

Internally, :class:`~merlin.algorithms.kernels.FidelityKernel` delegates
pairwise circuit construction and SLOS evaluation to its ``_CCInvQuantumLayer``
backend. The backend builds the pairwise circuits
:math:`U^{\dagger}(x_2) U(x_1)` in a vectorised way and asks the SLOS graph to
compute detection probabilities for the chosen input state. If photon loss
and/or detectors are defined, the raw probabilities are transformed accordingly
before the scalar kernel is read.

When constructing a training Gram matrix (``x2 is None``), only the upper triangle is simulated and mirrored to the lower triangle, then the diagonal is set to 1. With ``force_psd=True``, the matrix is symmetrised and projected to PSD by zeroing negative eigenvalues in an eigendecomposition.

Quickstarts and recipes
-----------------------

Minimal example (factory)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from merlin import ComputationSpace
   from merlin.algorithms.kernels import FidelityKernel

   # Build a kernel where inputs of size 2 are encoded in a 4-mode circuit
   kernel = FidelityKernel.simple(
       input_size=2,
       n_modes=4,  # Optional; defaults to input_size + 1
       computation_space=ComputationSpace.FOCK,  # Allow bunched outcomes if needed
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

   # Circuit and experiment
   circuit = pcvl.Circuit(6)
   experiment = pcvl.Experiment(circuit)
   experiment.noise = pcvl.NoiseModel(brightness=0.9, transmittance=0.85)

   # Feature map from the experiment
   fmap = FeatureMap(
       input_size=3,
       input_parameters=["px"],
       experiment=experiment,
   )

   # Fidelity kernel using a spaced input pattern
   kernel = FidelityKernel(
       feature_map=fmap,
       input_state=[1, 0, 1, 0, 1, 0],
       shots=0,
       computation_space=ComputationSpace.FOCK,
   )

   X = torch.rand(8, 3)
   K = kernel(X)  # (8, 8)

Declarative builder + kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from merlin.algorithms.kernels import KernelCircuitBuilder

   builder = (
       KernelCircuitBuilder()
       .input_size(4)
       .n_modes(6)
       .angle_encoding(scale=torch.pi)
       .trainable(enabled=True, prefix="phi")
   )
   kernel = builder.build_fidelity_kernel(input_state=[1, 1, 0, 0, 0, 0], shots=0)

   X = torch.rand(32, 4)
   K = kernel(X)

.. note::

  ``input_state=[...]`` is accepted for convenience and is converted to a Perceval
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
    import numpy as np

    # Prepare data
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Quantum kernel
    qkernel = FidelityKernel.simple(input_size=4, n_modes=6)     # Here the number of modes is optional, if n_modes is not given, n_modes=input_size+1
    K_train_q = qkernel(X_train_t).numpy()
    K_test_q = qkernel(X_test_t, X_train_t).numpy()

    clf_q = SVC(kernel="precomputed")
    clf_q.fit(K_train_q, y_train)
    acc_quantum = clf_q.score(K_test_q, y_test)

    # Classical RBF kernel
    K_train_rbf = rbf_kernel(X_train, gamma='scale')
    K_test_rbf = rbf_kernel(X_test, X_train, gamma='scale')

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
