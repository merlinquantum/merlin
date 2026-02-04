from __future__ import annotations

import perceval as pcvl
import pytest
import torch
import torch.nn.functional as F

from merlin import MeasurementStrategy, QuantumLayer
from merlin.builder import CircuitBuilder
from merlin.datasets import iris as iris_dataset


@pytest.fixture
def iris_batch():
    features, labels, _ = iris_dataset.get_data_train()
    x = torch.tensor(features[:16], dtype=torch.float32)
    y = torch.tensor(labels[:16], dtype=torch.long)
    return x, y


def _check_training_step(
    layer: QuantumLayer | torch.nn.Sequential,
    inputs: torch.Tensor,
    targets: torch.Tensor,
):
    layer.train()
    layer.zero_grad()
    logits = layer(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert logits.shape == (inputs.shape[0], 3)
    assert torch.isfinite(loss)
    assert any(g is not None for g in grads)
    assert all(torch.all(torch.isfinite(g)) for g in grads if g is not None)


def _train_for_classification(
    layer: QuantumLayer | torch.nn.Sequential,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    min_relative_improvement: float = 0.05,
    min_accuracy: float = 0.6,
) -> tuple[float, float]:
    layer.train()
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.05)

    with torch.no_grad():
        initial_loss = F.cross_entropy(layer(inputs), targets).item()

    for _ in range(60):
        optimizer.zero_grad()
        logits = layer(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    layer.eval()
    with torch.no_grad():
        final_logits = layer(inputs)
        final_loss = F.cross_entropy(final_logits, targets).item()
        predictions = final_logits.argmax(dim=1)
        accuracy = (predictions == targets).float().mean().item()
    if min_relative_improvement > 0:
        print(f"Loss improved from {initial_loss:.4f} to {final_loss:.4f} ")
        assert final_loss <= initial_loss
    assert accuracy >= min_accuracy
    return initial_loss, final_loss


def test_builder_api_pipeline_on_iris(iris_batch):
    features, labels = iris_batch

    builder = CircuitBuilder(n_modes=10)
    builder.add_superpositions(depth=1)
    builder.add_angle_encoding(modes=list(range(features.shape[1])), name="input")
    builder.add_rotations(trainable=True, name="theta")
    builder.add_superpositions(depth=1)

    layer = QuantumLayer(
        input_size=features.shape[1],
        builder=builder,
        n_photons=5,
        measurement_strategy=MeasurementStrategy.probs(),
        dtype=features.dtype,
    )
    pcvl.pdisplay(layer.computation_process.circuit, output_format=pcvl.Format.TEXT)
    model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))
    _check_training_step(model, features, labels)
    _train_for_classification(model, features, labels)


def test_simple_api_pipeline_on_iris(iris_batch):
    features, labels = iris_batch

    layer = QuantumLayer.simple(
        input_size=features.shape[1],
        dtype=features.dtype,
    )
    base = layer.quantum_layer if hasattr(layer, "quantum_layer") else layer
    pcvl.pdisplay(base.computation_process.circuit, output_format=pcvl.Format.TEXT)
    print(
        f"Nb of parameters = {sum(p.numel() for p in layer.parameters() if p.requires_grad)}"
    )
    model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))
    _check_training_step(model, features, labels)
    _train_for_classification(model, features, labels)


def test_manual_pcvl_circuit_pipeline_on_iris(iris_batch):
    features, labels = iris_batch

    wl = pcvl.GenericInterferometer(
        4,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_li{i}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_lo{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    circuit = pcvl.Circuit(4)
    circuit.add(0, wl)
    for mode in range(4):
        circuit.add(mode, pcvl.PS(pcvl.P(f"input{mode}")))

    wr = pcvl.GenericInterferometer(
        4,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_ri{i}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"theta_ro{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    circuit.add(0, wr)
    layer = QuantumLayer(
        input_size=features.shape[1],
        circuit=circuit,
        n_photons=1,
        trainable_parameters=["theta"],
        input_parameters=["input"],
        measurement_strategy=MeasurementStrategy.probs(),
        dtype=features.dtype,
    )
    model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))
    _check_training_step(model, features, labels)
    _train_for_classification(
        model,
        features,
        labels,
        min_relative_improvement=0.0,
        min_accuracy=0.4,
    )
