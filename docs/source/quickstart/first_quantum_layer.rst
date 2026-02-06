:github_url: https://github.com/merlinquantum/merlin

=========================
Your First Quantum Layer
=========================

This walkthrough mirrors the ``FirstQuantumLayers.ipynb`` notebook: we build and train
photonic layers that classify the Iris dataset. Along the way, we focus on three core
concepts you will reuse in every project:

- **Angle encoding** via :class:`~merlin.builder.circuit_builder.CircuitBuilder`. See :doc:`../user_guide/angle_amplitude_encoding` for more details on input encoding with MerLin. 
- **Output measurement strategies** that turn photonic outcomes into classical tensors. See :doc:`../user_guide/measurement_strategy`.
- **Computation spaces** controlling how the simulator truncates Fock states. See :doc:`../user_guide/computation_space`.

Once the foundations are in place, we show how to reuse the same circuit through a
:class:`perceval.Experiment` for detector-aware execution.


Set up the dataset
==================

.. code-block:: python

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    from merlin import LexGrouping, MeasurementStrategy, QuantumLayer
    from merlin.builder import CircuitBuilder

    torch.manual_seed(0)
    np.random.seed(0)

    iris = load_iris()
    X = iris.data.astype("float32")
    y = iris.target.astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Normalise features before encoding them as phases
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

Shared training utilities
=========================

.. code-block:: python

    def run_experiment(model: nn.Module, epochs: int = 60, lr: float = 0.05):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train)
            loss = F.cross_entropy(logits, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_preds = model(X_train).argmax(dim=1)
            test_preds = model(X_test).argmax(dim=1)
            train_acc = (train_preds == y_train).float().mean().item()
            test_acc = (test_preds == y_test).float().mean().item()
        return train_acc, test_acc

CircuitBuilder walkthrough
==========================

:class:`CircuitBuilder` is the recommended way to author circuits for training. We
stack entangling layers, angle encoding, and additional rotations before handing the
result to :class:`~merlin.algorithms.layer.QuantumLayer`.

.. code-block:: python

    builder = CircuitBuilder(n_modes=6)
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(
        modes=list(range(X_train.shape[1])),  # one mode per Iris feature
        name="input",
        scale=np.pi,
    )
    builder.add_rotations(trainable=True, name="theta")
    builder.add_superpositions(depth=1, trainable=True)

    quantum_core = QuantumLayer(
        input_size=X_train.shape[1],
        builder=builder,
        n_photons=3,                             # 3 photons evenly distributed on 6 modes
        measurement_strategy=MeasurementStrategy.probs(),
    )

    model = nn.Sequential(
        quantum_core,
        LexGrouping(quantum_core.output_size, 3),
        nn.Linear(3, 3),
    )

    train_acc, test_acc = run_experiment(model, epochs=80, lr=0.05)
    print(f"Train accuracy: {train_acc:.3f} – Test accuracy: {test_acc:.3f}")

Angle encoding highlights
-------------------------

- ``add_angle_encoding`` generates input-driven phase shifters whose prefixes (``name``)
  must appear inside ``input_parameters`` when you instantiate the layer manually.
- Scaling by ``np.pi`` keeps rotations in a range compatible with Perceval’s phase
  conventions.
- Normalise your features to :math:`[-1, 1]` or :math:`[0, 1]` before feeding them into
  the layer so the implied rotations stay stable during training.
- In this tutorial, we focus on angle encoding. Amplitude encoding is also available. More information can be found in the :doc:`../user_guide/angle_amplitude_encoding` documentation.

Exploring output measurement strategies
=======================================

MerLin exposes three strategies: probabilities (default), per-mode expectations, and
complex amplitudes (simulation only). Swap the strategy to pick the classical output
that best matches the rest of your model.

.. code-block:: python

    strategies = {
        "probabilities": MeasurementStrategy.probs(),
        "mode_expectations": MeasurementStrategy.mode_expectations(),
    }

    for label, strategy in strategies.items():
        layer = QuantumLayer(
            input_size=X_train.shape[1],
            builder=builder,
            n_photons=3,
            measurement_strategy=strategy,
        )
        head = nn.Sequential(
            layer,
            nn.Linear(layer.output_size, 3),
        )
        train_acc, test_acc = run_experiment(head, epochs=60, lr=0.05)
        print(f"{label}: train {train_acc:.3f} – test {test_acc:.3f}")

    # Amplitudes provide complex tensors — convert them before handing off to nn.Linear
    amp_layer = QuantumLayer(
        input_size=X_train.shape[1],
        builder=builder,
        n_photons=3,
        measurement_strategy=MeasurementStrategy.amplitudes(),
    )
    class ComplexToReal(nn.Module):
        def forward(self, x):
            # view_as_real -> (..., 2) with last dim holding [real, imag]
            parts = torch.view_as_real(x)
            return parts.flatten(start_dim=1)

    amp_head = nn.Sequential(
        amp_layer,
        ComplexToReal(),
        nn.Linear(2 * amp_layer.output_size, 3),
    )
    train_acc, test_acc = run_experiment(amp_head, epochs=60, lr=0.05)
    print(f"amplitudes (with real/imag flattening): train {train_acc:.3f} – test {test_acc:.3f}")

Measurement strategy tips
-------------------------

- ``MeasurementStrategy.probs()`` returns the Fock state probability distribution – Ideal for attaching dense classical heads, simple linear probings or grouping strategies.
- ``MeasurementStrategy.mode_expectations()`` compresses the outputs to one value per mode, reducing the
  number of classical weights you need downstream.
- ``MeasurementStrategy.amplitudes()`` yields tensors with complex values and is restricted to noiseless simulations without detectors. Convert them with ``torch.view_as_real`` or flatten real/imaginary parts before feeding the data to classical layers.

More informations on measurement strategies can be found here: :doc:`../user_guide/measurement_strategy`.

Choosing a computation space
============================

The ``computation_space`` parameter is not recommended, it will be deprecated in the future. Instead, define the computation_space,
how Perceval truncates the Fock space, in the MeasurementStrategy chosen. By default the computation space is the ``ComputationSpace.UNBUNCHED``.
Override the default when you need explicit control:

.. code-block:: python

    from merlin import ComputationSpace

    fock_layer = QuantumLayer(
        input_size=X_train.shape[1],
        builder=builder,
        n_photons=3,
        measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK),       # Full Fock basis
    )

    unbunched_layer = QuantumLayer(
        input_size=X_train.shape[1],
        builder=builder,
        n_photons=3,
        measurement_strategy=MeasurementStrategy.probs(ComputationSpace.UNBUNCHED),  # Forbid multiple photons per mode
    )

    dual_rail_layer = QuantumLayer(
        input_size=X_train.shape[1],
        builder=builder,
        n_photons=3,
        measurement_strategy=MeasurementStrategy.probs(ComputationSpace.DUAL_RAIL),  # Pair modes to encode qubits
    )

- ``FOCK`` keeps the entire combinatorial space of the declared photons.
- ``UNBUNCHED`` assumes at most one photon per mode, reducing the state count when the
  circuit satisfies that constraint.
- ``DUAL_RAIL`` models qubits as photon pairs, which is useful when interfacing with
  dual-rail encodings or threshold detectors.

Detector-aware execution with Experiments
=========================================

Wrapping the circuit in a :class:`perceval.Experiment` lets you attach detectors and
noise models without re-authoring the layer. The experiment becomes the single source
of truth for measurement semantics.

.. code-block:: python

    import perceval as pcvl

    # Reuse the circuit produced by the builder
    circuit = builder.circuit
    experiment = pcvl.Experiment(circuit)
    experiment.noise = pcvl.NoiseModel(brightness=0.95, transmittance=0.9)
    experiment.detectors[0] = pcvl.Detector.threshold()
    experiment.detectors[1] = pcvl.Detector.pnr()

    experiment_layer = QuantumLayer(
        input_size=X_train.shape[1],
        experiment=experiment,
        input_state=[1, 1, 1, 0, 0, 0],
        input_parameters=["input"],
        measurement_strategy=MeasurementStrategy.probs(),
    )

    model_with_noise = nn.Sequential(
        experiment_layer,
        LexGrouping(experiment_layer.output_size, 3),
        nn.Linear(3, 3),
    )
    train_acc, test_acc = run_experiment(model_with_noise, epochs=80, lr=0.05)
    print(f"Experiment-backed layer – test accuracy: {test_acc:.3f}")

Experiment notes
----------------

- Attaching detectors or photon-loss models disables ``MeasurementStrategy.amplitudes()``
  because amplitudes are no longer observable.
- ``input_parameters`` must match the prefixes emitted by ``add_angle_encoding`` (``"input"`` in this example).
- You can reuse the same experiment across multiple layers or kernel feature maps to
  keep detector settings consistent.

Next steps
==========

- Swap out ``builder.add_superpositions`` or introduce additional entangling layers to
  explore deeper circuits.
- Set ``return_object`` to True to get a more detailed object as a result of a forward call. Take a look at :doc:`../api_reference/api/merlin.algorithms.layer` for more details about returned typed objects.
- Combine :class:`~merlin.utils.grouping.LexGrouping` or :class:`~merlin.utils.grouping.ModGrouping` modules to tailor the classical feature
  count to your downstream model.
- Re-run the experiments with alternative computation spaces to benchmark accuracy vs.
  runtime trade-offs.
- Take a look at the :doc:`../user_guide/index` for more detailed explanations.
