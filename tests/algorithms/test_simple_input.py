from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from merlin import Probabilities, QuantumLayer
from merlin.utils.grouping import ModGrouping


@pytest.fixture
def quantum_layer_api():
    return QuantumLayer


def _unwrap(layer: nn.Module) -> QuantumLayer:
    if hasattr(layer, "quantum_layer"):
        return layer.quantum_layer  # type: ignore[attr-defined]
    if isinstance(layer, nn.Sequential) and len(layer) > 0:
        return _unwrap(layer[0])
    return layer  # type: ignore[return-value]


def test_none_strategy_without_output_size(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=3,
        dtype=torch.float32,
    )
    base = _unwrap(layer)

    x = torch.rand(4, 3)
    output = layer(x)

    assert output.shape == (4, base.output_size)
    assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-5)


def test_none_strategy_with_matching_output_size(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    reference_layer = QuantumLayer.simple(
        input_size=3,
    )
    dist_size = _unwrap(reference_layer).output_size

    layer = QuantumLayer.simple(
        input_size=3,
        output_size=dist_size,
    )

    base = _unwrap(layer)
    x = torch.rand(2, 3)
    output = layer(x)
    assert output.shape == (2, dist_size)
    assert base.output_size == dist_size


def test_simple_groups_output_when_requested(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    target_size = 16
    layer = QuantumLayer.simple(
        input_size=3,
        output_size=target_size,
    )

    assert hasattr(layer, "post_processing")
    assert isinstance(layer.post_processing, ModGrouping)
    base = _unwrap(layer)
    x = torch.rand(4, 3)
    output = layer(x)
    assert base.output_size != target_size
    assert output.shape == (4, target_size)
    assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-5)


def test_linear_strategy_creates_linear_mapping(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=3,
    )
    model = nn.Sequential(layer, nn.Linear(layer.output_size, 5))

    base = _unwrap(layer)
    assert isinstance(base.measurement_mapping, Probabilities)
    x = torch.rand(6, 3)
    output = model(x)
    assert output.shape == (6, 5)


def test_default_strategy_is_none(quantum_layer_api):
    QuantumLayer = quantum_layer_api
    sig = inspect.signature(QuantumLayer.simple)
    assert "measurement_strategy" not in sig.parameters


def test_simple_signature_does_not_include_reservoir_mode(quantum_layer_api):
    QuantumLayer = quantum_layer_api
    sig = inspect.signature(QuantumLayer.simple)
    assert "reservoir_mode" not in sig.parameters


def test_gradient_flow_for_strategies(quantum_layer_api):
    QuantumLayer = quantum_layer_api
    # nb_params = 90

    layer = QuantumLayer.simple(
        input_size=3,
    )
    model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 4))

    x = torch.rand(8, 3, requires_grad=True)
    loss = model(x).sum()
    loss.backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()
    )

    layer_none = QuantumLayer.simple(
        input_size=3,
    )

    x = torch.rand(8, 3, requires_grad=True)
    loss = layer_none(x).sum()
    loss.backward()
    base_none = _unwrap(layer_none)
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in base_none.parameters()
    )
    # mzi_param_count = sum(
    #     param.numel()
    #     for name, param in base_none.named_parameters()
    #     if name.startswith("mzi_extra")
    # )
    # interferometer_param_count = sum(
    #     param.numel()
    #     for name, param in base_none.named_parameters()
    #     if name.startswith("gi_simple")
    # )

    # total_trainable = mzi_param_count + interferometer_param_count
    # expected_total = max(nb_params, interferometer_param_count)
    # expected_mzi = max(nb_params - interferometer_param_count, 0)

    # assert total_trainable == expected_total
    # assert mzi_param_count == expected_mzi


def test_quantum_layer_simple_raises_when_input_exceeds_modes(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    with pytest.raises(
        ValueError,
    ):
        QuantumLayer.simple(
            input_size=22,
        )


def test_batch_shapes_and_probabilities(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=4,
    )

    for batch_size in [1, 5, 16]:
        x = torch.rand(batch_size, 4)
        output = layer(x)
        assert output.shape == (batch_size, layer.output_size)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.all(output >= 0)


def test_dtype_propagation(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    for dtype in (torch.float32, torch.float64):
        layer = QuantumLayer.simple(
            input_size=3,
            dtype=dtype,
        )

        base = _unwrap(layer)
        for param in base.parameters():
            assert param.dtype == dtype

        x = torch.rand(2, 3, dtype=dtype)
        output = layer(x)
        assert output.dtype == dtype


def test_circuit_and_output_size_access(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    simple_layer = QuantumLayer.simple(
        input_size=3,
    )
    circuit = simple_layer.circuit
    output_size = simple_layer.output_size

    assert circuit == simple_layer.quantum_layer.circuit
    assert output_size == simple_layer.quantum_layer.output_size

    simple_layer = QuantumLayer.simple(input_size=3, output_size=4)
    output_size = simple_layer.output_size
    assert output_size == 4
    assert output_size != simple_layer.quantum_layer.output_size
    assert output_size == simple_layer.post_processing.output_size
