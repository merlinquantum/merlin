:github_url: https://github.com/merlinquantum/merlin

======================
Algorithms with MerLin
======================

The Quantum Kernel
==================


What is a fidelity quantum kernel?
----------------------------------

Definition
----------

A fidelity quantum kernel measures the similarity of two inputs via the squared overlap of the quantum states prepared by a data-encoding circuit:
:math:`K(x, z) = |\langle \psi(x) \mid \psi(z) \rangle|^2`, where :math:`\lvert \psi(x) \rangle = U(x)\lvert \psi_0 \rangle`.
Equivalently, with density matrices :math:`\rho(x) = \lvert \psi(x) \rangle \langle \psi(x) \rvert`, one has
:math:`K(x, z) = \mathrm{Tr}[\rho(x)\rho(z)]` (the Hilbert–Schmidt inner product for pure states).

Key properties
--------------

- Bounded: :math:`0 \le K(x, z) \le 1`, and :math:`K(x, x)=1`.
- PSD in theory: Gram matrices built from :math:`K` are positive semi-definite in exact arithmetic; small negative eigenvalues can arise numerically. Use ``force_psd=True`` if needed.
- Phase-invariant: Insensitive to global phases of the state.
- Nonlinear features: Nonlinearity arises from the unitary embedding :math:`U(x)` of classical data into quantum states.

Practical notes
---------------

- In MerLin, a FeatureMap encodes inputs into a photonic circuit; FidelityKernel evaluates fidelities between the resulting states.
- ``kernel(X, Z)`` returns an :math:`N \times M` cross-kernel for datasets of sizes :math:`N` and :math:`M`; ``kernel(X)`` returns the :math:`N \times N` Gram matrix.
- Larger/deeper circuits can increase expressivity but also numerical error; prefer double precision and PSD projection for stability when needed.


Overview
--------

MerLin provides a quantum kernel implementation that measures similarity between data points via the fidelity of quantum states prepared by a photonic circuit. It is built around two core concepts:

- FeatureMap: encodes classical inputs into a parameterized quantum circuit.
- FidelityKernel: computes kernel matrices K where K[i, j] ≈ |⟨φ(x_i)|φ(x_j)⟩|^2.

You can construct kernels in multiple ways:

- Quickstart factory: FidelityKernel.simple(...)
- Direct pcvl circuit: build a FeatureMap from a perceval Circuit and wrap it in FidelityKernel
- Fluent builder: KernelCircuitBuilder(...).build_fidelity_kernel()


Quickstart: Iris Classification
-------------------------------

This example mirrors the tests using scikit-learn’s Iris dataset. We build a quantum kernel that matches the 4 Iris features, compute precomputed kernel matrices, and train a classical SVM.

.. code-block:: python

    import torch
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    from merlin.algorithms.kernels import FidelityKernel

    # Load data and split
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to tensors (Iris has 4 features)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Create a simple quantum kernel
    kernel = FidelityKernel.simple(
        input_size=4,      # number of input features
        n_modes=6,         # photonic modes in the circuit
        n_photons=2,       # number of photons
        trainable=True     # set False for a fixed, non-trainable kernel
    )

    # Compute precomputed kernel matrices
    K_train = kernel(X_train_t).detach().numpy()
    K_test = kernel(X_test_t, X_train_t).detach().numpy()

    # Train an SVM with precomputed kernel
    clf = SVC(kernel="precomputed", random_state=42)
    clf.fit(K_train, y_train)
    accuracy = clf.score(K_test, y_test)
    print(f"Iris accuracy: {accuracy:.3f}")

Notes
-----

- K_train is symmetric and approximately positive semi-definite; diagonal entries are ≈ 1. Use force_psd=True if you want an explicit PSD projection.
- Values typically lie in [0, 1] (allowing for small numerical tolerances).
- Set ``trainable=False`` for a fixed, non-trainable kernel; keep it ``True`` (default) to optimise phase parameters with learning algorithms such as ``NKernelAlignment``.


Advanced: Custom Circuit
--------------------------------------

You can also build a custom Perceval circuit, then wrapping it in a FeatureMap and FidelityKernel. Below is a compact version of the pattern used in the test suite.

.. code-block:: python

    import torch
    import perceval as pcvl
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    from merlin.algorithms.kernels import FeatureMap, FidelityKernel

    def create_quantum_circuit(m, size=400):
        # Two interferometers (wl, wr) with a middle data-encoding section (c_var)
        wl = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"phase_1_{i}")) // pcvl.BS() // pcvl.PS(pcvl.P(f"phase_2_{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE,
        )
        c = pcvl.Circuit(m)
        c.add(0, wl, merge=True)

        c_var = pcvl.Circuit(m)
        for i in range(size):
            px = pcvl.P(f"px-{i + 1}")
            c_var.add(i % m, pcvl.PS(px))
        c.add(0, c_var, merge=True)

        wr = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"phase_3_{i}")) // pcvl.BS() // pcvl.PS(pcvl.P(f"phase_4_{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE,
        )
        c.add(0, wr, merge=True)
        return c

    def get_quantum_kernel(modes=10, input_size=4, photons=4, no_bunching=False):
        circuit = create_quantum_circuit(m=modes, size=input_size)
        feature_map = FeatureMap(
            circuit=circuit,
            input_size=input_size,
            input_parameters=["px"],
            trainable_parameters=["phase"],
            dtype=torch.float64,
        )
        input_state = [0] * modes
        for p in range(min(photons, modes // 2)):
            input_state[2 * p] = 1
        return FidelityKernel(
            feature_map=feature_map,
            input_state=input_state,
            no_bunching=no_bunching,
        )

    # Iris workflow with the custom kernel
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    kernel = get_quantum_kernel(input_size=4, modes=10, photons=4)
    K_train = kernel(X_train_t).detach().numpy()
    K_test = kernel(X_test_t, X_train_t).detach().numpy()

    clf = SVC(kernel="precomputed", random_state=42)
    clf.fit(K_train, y_train)
    print("Accuracy:", clf.score(K_test, y_test))

Tip: This circuit is deep and can accumulate numerical errors!


Training the Kernel (NKA loss)
------------------------------

FidelityKernel supports training of selected parameters via the NKernelAlignment loss. The tests include a minimal training loop on a binary Iris subset (classes 0 vs 1):

.. code-block:: python

    import torch
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    from merlin.algorithms.kernels import FidelityKernel
    from merlin.algorithms.loss import NKernelAlignment

    # Prepare a binary problem (setosa vs versicolor)
    iris = load_iris()
    X, y = iris.data, iris.target
    mask = y < 2
    X, y = X[mask], y[mask]
    y = 2 * y - 1  # {-1, +1}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train_t = torch.tensor(X_train, dtype=torch.float64)
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    # Trainable kernel
    kernel = FidelityKernel.simple(input_size=4, n_modes=6, n_photons=2)

    optimizer = torch.optim.Adam(kernel.parameters(), lr=1e-2)
    loss_fn = NKernelAlignment()
    for _ in range(3):
        optimizer.zero_grad()
        K = kernel(X_train_t)
        loss = loss_fn(K, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluate with a classical SVM
    K_train = kernel(X_train_t).detach().numpy()
    K_test = kernel(X_test_t, X_train_t).detach().numpy()
    clf = SVC(kernel="precomputed", random_state=42)
    clf.fit(K_train, ((y_train + 1) // 2))
    print("Binary accuracy:", clf.score(K_test, ((y_test + 1) // 2)))


Other Construction Methods
--------------------------

For more control, you can create kernels from a custom perceval circuit or use the fluent builder:

.. code-block:: python

    import perceval as pcvl
    from merlin.core.generators import CircuitType
    from merlin.algorithms.kernels import FeatureMap, FidelityKernel, KernelCircuitBuilder

    # From a custom pcvl circuit
    params = [pcvl.P(f"x{i+1}") for i in range(4)]
    circuit = pcvl.Circuit(4)
    for mode, param in enumerate(params):
        circuit.add(mode, pcvl.PS(param))
    circuit.add(0, pcvl.BS())
    circuit.add(2, pcvl.BS())

    feature_map = FeatureMap(
        circuit=circuit,
        input_size=4,
        input_parameters="x",
    )
    kernel_manual = FidelityKernel(
        feature_map=feature_map,
        input_state=[1, 1, 0, 0],
    )

    # Builder pattern
    builder = KernelCircuitBuilder()
    kernel_builder = (
        builder
        .input_size(4)
        .n_modes(4)
        .n_photons(2)
        .circuit_type(CircuitType.SERIES)
        .reservoir_mode(True)
        .build_fidelity_kernel()
    )


API Pointers
------------

- merlin.algorithms.kernels.FeatureMap: create encoders for datapoints.
- merlin.algorithms.kernels.FidelityKernel: compute and train quantum kernels.
- merlin.algorithms.kernels.KernelCircuitBuilder: fluent construction of feature maps and kernels.
- perceval.Circuit / perceval components: build fully custom photonic circuits.
