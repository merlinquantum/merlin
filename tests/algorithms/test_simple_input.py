from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from merlin import OutputMappingStrategy, QuantumLayer

_PCVL_HOME = Path(__file__).resolve().parents[2] / ".pcvl_home"
(
    _PCVL_HOME / "Library" / "Application Support" / "perceval-quandela" / "job_group"
).mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(_PCVL_HOME)


@pytest.fixture(autouse=True)
def perceval_home(monkeypatch):
    monkeypatch.setenv("HOME", str(_PCVL_HOME))


@pytest.fixture
def quantum_layer_api():
    return QuantumLayer, OutputMappingStrategy


def test_none_strategy_without_output_size(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        output_mapping_strategy=OutputMappingStrategy.NONE,
        dtype=torch.float32,
    )

    x = torch.rand(4, 3)
    output = layer(x)

    assert output.shape == (4, layer.output_size)
    assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-5)


def test_none_strategy_with_matching_output_size(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    reference_layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    )
    dist_size = reference_layer.output_size

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        output_size=dist_size,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    )

    x = torch.rand(2, 3)
    output = layer(x)
    assert output.shape == (2, dist_size)


def test_none_strategy_with_mismatched_output_size(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    with pytest.raises(ValueError):
        QuantumLayer.simple(
            input_size=3,
            n_params=60,
            output_size=10,
            output_mapping_strategy=OutputMappingStrategy.NONE,
        )


def test_linear_strategy_requires_output_size(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    with pytest.raises(ValueError):
        QuantumLayer.simple(
            input_size=3,
            n_params=60,
            output_mapping_strategy=OutputMappingStrategy.LINEAR,
        )


def test_linear_strategy_creates_linear_mapping(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        output_size=5,
        output_mapping_strategy=OutputMappingStrategy.LINEAR,
    )

    assert isinstance(layer.output_mapping, nn.Linear)
    x = torch.rand(6, 3)
    output = layer(x)
    assert output.shape == (6, 5)


def test_default_strategy_is_none(quantum_layer_api):
    QuantumLayer, _ = quantum_layer_api
    sig = inspect.signature(QuantumLayer.simple)
    assert (
        sig.parameters["output_mapping_strategy"].default == OutputMappingStrategy.NONE
    )


def test_trainable_parameter_budget_matches_request(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    requested_params = 37
    with pytest.warns(RuntimeWarning):
        layer = QuantumLayer.simple(
            input_size=3,
            n_params=requested_params,
            output_mapping_strategy=OutputMappingStrategy.NONE,
        )

    theta_param_count = sum(
        param.numel()
        for name, param in layer.named_parameters()
        if name.startswith("theta")
    )
    interferometer_param_count = sum(
        param.numel()
        for name, param in layer.named_parameters()
        if name.startswith("gi_simple")
    )

    total_trainable = theta_param_count + interferometer_param_count
    expected_total = max(requested_params, interferometer_param_count)
    expected_theta = max(requested_params - interferometer_param_count, 0)

    assert total_trainable == expected_total
    assert theta_param_count == expected_theta


def test_gradient_flow_for_strategies(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api
    nb_params = 40

    layer_linear = QuantumLayer.simple(
        input_size=3,
        n_params=nb_params,
        output_size=4,
        output_mapping_strategy=OutputMappingStrategy.LINEAR,
    )

    x = torch.rand(8, 3, requires_grad=True)
    loss = layer_linear(x).sum()
    loss.backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in layer_linear.parameters()
    )

    layer_none = QuantumLayer.simple(
        input_size=3,
        n_params=nb_params,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    )

    x = torch.rand(8, 3, requires_grad=True)
    loss = layer_none(x).sum()
    loss.backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in layer_none.parameters()
    )
    theta_param_count = sum(
        param.numel()
        for name, param in layer_none.named_parameters()
        if name.startswith("theta")
    )
    interferometer_param_count = sum(
        param.numel()
        for name, param in layer_none.named_parameters()
        if name.startswith("gi_simple")
    )

    total_trainable = theta_param_count + interferometer_param_count
    expected_total = max(nb_params, interferometer_param_count)
    expected_theta = max(nb_params - interferometer_param_count, 0)

    assert total_trainable == expected_total
    assert theta_param_count == expected_theta


def test_quantum_layer_simple_raises_when_input_exceeds_modes(quantum_layer_api):
    QuantumLayer, _ = quantum_layer_api

    with pytest.raises(
        ValueError, match="You cannot encore more features than mode with Builder"
    ):
        QuantumLayer.simple(
            input_size=12,
            n_params=30,
            output_mapping_strategy=OutputMappingStrategy.NONE,
        )


def test_batch_shapes_and_probabilities(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=4,
        n_params=80,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    )

    for batch_size in [1, 5, 16]:
        x = torch.rand(batch_size, 4)
        output = layer(x)
        assert output.shape == (batch_size, layer.output_size)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.all(output >= 0)


def test_dtype_propagation(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    for dtype in (torch.float32, torch.float64):
        layer = QuantumLayer.simple(
            input_size=3,
            n_params=60,
            dtype=dtype,
            output_mapping_strategy=OutputMappingStrategy.NONE,
        )

        for param in layer.parameters():
            assert param.dtype == dtype

        x = torch.rand(2, 3, dtype=dtype)
        output = layer(x)
        assert output.dtype == dtype
