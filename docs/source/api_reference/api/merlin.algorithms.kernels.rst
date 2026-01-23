merlin.algorithms.kernels module
================================

.. automodule:: merlin.algorithms.kernels
   :members: FeatureMap, FidelityKernel, KernelCircuitBuilder
   :undoc-members:
   :show-inheritance:

.. note::

   When the wrapped :class:`~merlin.algorithms.kernels.FeatureMap` exposes a
   :class:`perceval.Experiment`, fidelity kernels compose the attached
   :class:`perceval.NoiseModel` (photon loss) before applying any detector
   transforms. The resulting kernel values therefore reflect both survival
   probabilities and detector post-processing.


Examples
--------

Quickstart: Fidelity kernel in a few lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from merlin.algorithms.kernels import FidelityKernel

    # Build a kernel where inputs of size 2 are encoded in a 4-mode circuit
    kernel = FidelityKernel.simple(
        input_size=2,
        n_modes=4,               # Here the number of modes is optional, if n_modes is not given, n_modes=input_size
        shots=0,                 # exact probabilities (no sampling)
        no_bunching=False,       # allow bunched outcomes if needed
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # X_train: (N, 2), X_test: (M, 2)
    X_train = torch.rand(10, 2)
    X_test = torch.rand(5, 2)

    K_train = kernel(X_train)               # (N, N)
    K_test = kernel(X_test, X_train)        # (M, N)

Custom experiment with FeatureMap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import perceval as pcvl
    from merlin.algorithms.kernels import FeatureMap, FidelityKernel

    # Define a photonic circuit
    circuit = pcvl.Circuit(6)
    # Add whatever to the circuit...

    # Define the Experiment
    experiment = pcvl.Experiment(circuit)
    # Add noise models, detectors, etc...
    experiment.noise = pcvl.NoiseModel(brightness=0.9)
    experiment.detectors[0] = pcvl.Detector.threshold()
    experiment.detectors[5] = pcvl.Detector.ppnr(n_wires=3)

    # Use the experiment to create a FeatureMap automatically
    feature_map = FeatureMap.from_photonic_backend(
        input_size=0,
        experiment=experiment,
    )

    # Build the kernel with a specific input state
    kernel = FidelityKernel(
        feature_map=feature_map,
        input_state=[2, 0, 2, 0, 2, 0],
        no_bunching=False,
    )

    X = torch.rand(8, 3)
    K = kernel(X)  # (8, 8)

Use with scikit-learn (precomputed kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from sklearn.svm import SVC
    from merlin.algorithms.kernels import FidelityKernel

    # Build kernel and compute Gram matrices
    kernel = FidelityKernel.simple(input_size=4, n_modes=6)
    K_train = kernel(X_train)
    K_test = kernel(X_test, X_train)

    # Train a precomputed-kernel SVM
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
