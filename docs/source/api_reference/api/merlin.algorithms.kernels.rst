merlin.algorithms.kernels module
================================

.. automodule:: merlin.algorithms.kernels
   :no-members:

.. currentmodule:: merlin.algorithms.kernels

.. autoclass:: FeatureMap
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FidelityKernel
   :members:
   :undoc-members:
   :show-inheritance:

Deprecations
------------

.. warning:: *Deprecated since version 0.4:*
   :class:`KernelCircuitBuilder` is deprecated and will be removed in release
   0.5. Use :class:`~merlin.builder.circuit_builder.CircuitBuilder` with
   :class:`FeatureMap` and :class:`FidelityKernel` directly instead.

.. autoclass:: KernelCircuitBuilder
   :members:
   :undoc-members:
   :show-inheritance:

.. warning:: *Deprecated since version 0.3:*
   The ``no_bunching`` flag accepted by legacy kernel constructors is removed
   since version 0.3.0. Use the ``computation_space`` parameter instead.
   See :doc:`/user_guide/migration_guide`.

.. warning:: *Deprecated since version 0.4:*
   Direct unitary construction through :meth:`FeatureMap.compute_unitary` is a
   legacy path. It still owns compiler state for backwards compatibility, but
   :class:`FidelityKernel` no longer uses it. Use :class:`FidelityKernel` for
   kernel computations.

.. note::

   :class:`~merlin.algorithms.kernels.FeatureMap` is the descriptor used by
   :class:`~merlin.algorithms.kernels.FidelityKernel`: it stores the circuit or
   experiment, input size, parameter prefixes, dtype, and device.
   :class:`~merlin.algorithms.kernels.FidelityKernel` uses the internal
   ``CCInvQuantumLayer`` adapter, and ``CCInvQuantumLayer`` uses the
   :class:`~merlin.algorithms.layer.QuantumLayer` backend.

.. note::

   When the wrapped :class:`~merlin.algorithms.kernels.FeatureMap` exposes a
   :class:`pcvl.Experiment`, fidelity kernels compose the attached
   :class:`pcvl.NoiseModel` (photon loss) before applying any detector
   transforms. The resulting kernel values therefore reflect both survival
   probabilities and detector post-processing.


Examples
--------

Quickstart: Fidelity kernel in a few lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from merlin import ComputationSpace
    from merlin.algorithms.kernels import FeatureMap, FidelityKernel

    # Build a kernel where inputs of size 2 are encoded in a 3-mode circuit
    feature_map = FeatureMap.simple(
        input_size=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    kernel = FidelityKernel(
        feature_map=feature_map,
        shots=0,                 # exact probabilities (no sampling)
        computation_space=ComputationSpace.FOCK,       # allow bunched outcomes if needed
    )

    # X_train: (N, 2), X_test: (M, 2)
    X_train = torch.rand(10, 2)
    X_test = torch.rand(5, 2)

    K_train = kernel(X_train)           # (N, N)
    K_test = kernel(X_test, X_train)    # (M, N)

Custom experiment with FeatureMap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import perceval as pcvl
    from merlin.algorithms.kernels import FeatureMap, FidelityKernel

    # Define a photonic circuit with two input parameters
    circuit = pcvl.Circuit(6)
    circuit.add(0, pcvl.PS(pcvl.P("x0")))
    circuit.add(1, pcvl.PS(pcvl.P("x1")))

    # Define the Experiment
    experiment = pcvl.Experiment(circuit)
    # Add noise models, detectors, etc...
    experiment.noise = pcvl.NoiseModel(brightness=0.9)
    experiment.detectors[0] = pcvl.Detector.threshold()
    experiment.detectors[5] = pcvl.Detector.ppnr(n_wires=3)

    # Use the experiment to create a FeatureMap
    feature_map = FeatureMap(
        experiment=experiment,
        input_size=2,
        input_parameters="x",
    )

    # Build the kernel with a specific input state
    kernel = FidelityKernel(
        feature_map=feature_map,
        input_state=[2, 0, 2, 0, 2, 0],
        computation_space=ComputationSpace.FOCK,
    )

    X = torch.rand(8, 2)
    K = kernel(X)  # (8, 8)

Use with scikit-learn (precomputed kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from sklearn.svm import SVC
    from merlin.algorithms.kernels import FeatureMap, FidelityKernel
    from merlin.builder import CircuitBuilder

    # Build a kernel with 4 input features in 5 modes
    builder = CircuitBuilder(n_modes=5)
    builder.add_superpositions(depth=1)
    builder.add_angle_encoding(modes=[0, 1, 2, 3], name="input")
    builder.add_superpositions(depth=1)

    feature_map = FeatureMap(builder=builder, input_size=4, input_parameters=None)
    kernel = FidelityKernel(feature_map=feature_map, input_state=[1, 0, 1, 0, 1])

    K_train = kernel(X_train)
    K_test = kernel(X_test, X_train)

    # Train a precomputed-kernel SVM
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
