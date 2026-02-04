from __future__ import annotations

import os
from pathlib import Path

import perceval as pcvl
import pytest
import torch

from merlin import MeasurementStrategy, QuantumLayer
from merlin.builder import CircuitBuilder
from merlin.core.components import (
    BeamSplitter,
    GenericInterferometer,
    ParameterRole,
    Rotation,
)
from merlin.pcvl_pytorch.locirc_to_tensor import CircuitConverter

_PS_TYPE = type(pcvl.PS(0.0))
_BS_TYPE = type(pcvl.BS())


_PCVL_HOME = Path(__file__).resolve().parents[2] / ".pcvl_home"
(
    _PCVL_HOME / "Library" / "Application Support" / "perceval-quandela" / "job_group"
).mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(_PCVL_HOME)


@pytest.fixture(autouse=True)
def perceval_home(monkeypatch):
    monkeypatch.setenv("HOME", str(_PCVL_HOME))


def test_add_rotations_assigns_trainable_names_per_mode():
    builder = CircuitBuilder(n_modes=3)
    builder.add_rotations(trainable=True)

    rotations = builder.circuit.components
    assert len(rotations) == 3

    for idx, rotation in enumerate(rotations):
        assert isinstance(rotation, Rotation)
        assert rotation.role == ParameterRole.TRAINABLE
        assert rotation.target == idx
        assert rotation.custom_name == f"theta_{idx}_{idx}"
        assert rotation.axis == "z"


def test_add_rotations_input_custom_prefix_uses_global_counter():
    builder = CircuitBuilder(n_modes=4)
    builder.add_rotations(modes=[1, 3], role=ParameterRole.INPUT, name="feature")

    rotations = builder.circuit.components
    assert [rotation.target for rotation in rotations] == [1, 3]
    assert [rotation.custom_name for rotation in rotations] == ["feature1", "feature2"]
    assert all(rotation.role == ParameterRole.INPUT for rotation in rotations)


def test_complex_builder_pipeline_exports_pcvl_circuit():
    builder = CircuitBuilder(n_modes=3)
    builder.add_angle_encoding(name="input")
    builder.add_superpositions(depth=1, name="ent")
    builder.add_rotations(modes=1, angle=0.25)

    pcvl_circuit = builder.to_pcvl_circuit(pcvl)
    assert isinstance(pcvl_circuit, pcvl.Circuit)
    assert pcvl_circuit.m == 3

    ps_ops = [
        gate for _, gate in pcvl_circuit._components if isinstance(gate, _PS_TYPE)
    ]
    assert ps_ops, "Input encoding should add phase shifters"
    assert any(
        param.name.startswith("input")
        for gate in ps_ops
        for param in gate.get_parameters()
    )

    entangling_ops = [
        gate for _, gate in pcvl_circuit._components if isinstance(gate, _BS_TYPE)
    ]
    assert entangling_ops, "Entangling layer should contribute beam splitters"


def test_trainable_entangling_layer_generates_parameterised_mixers():
    builder = CircuitBuilder(n_modes=4)
    builder.add_superpositions(depth=1, trainable=True, name="mix")

    circuit = builder.to_pcvl_circuit(pcvl)
    param_names = {param.name for param in circuit.get_parameters()}

    assert any(name.startswith("mix_theta_") for name in param_names)
    assert any(name.startswith("mix_phi_") for name in param_names)
    assert "mix" in builder.trainable_parameter_prefixes


def test_pcvl_export_keeps_theta_and_phi_fallback_names_distinct():
    builder = CircuitBuilder(n_modes=2)
    builder.circuit.add(
        BeamSplitter(
            targets=(0, 1),
            theta_role=ParameterRole.TRAINABLE,
            phi_role=ParameterRole.TRAINABLE,
        )
    )

    circuit = builder.to_pcvl_circuit(pcvl)
    param_names = {param.name for param in circuit.get_parameters()}

    assert any(name.startswith("theta_bs_") for name in param_names)
    assert any(name.startswith("phi_bs_") for name in param_names)
    assert len(param_names) == 2


def test_to_pcvl_circuit_supports_gradient_backpropagation():
    builder = CircuitBuilder(n_modes=2)
    builder.add_rotations(trainable=True, name="theta")
    builder.add_superpositions(
        targets=(0, 1),
        trainable_theta=True,
        trainable_phi=True,
    )
    builder.add_superpositions(depth=1)

    pcvl_circuit = builder.to_pcvl_circuit(pcvl)

    converter = CircuitConverter(pcvl_circuit, input_specs=["theta", "phi"])

    total_params = sum(len(params) for params in converter.spec_mappings.values())
    assert total_params == 4

    theta_params = torch.tensor(
        [0.1, -0.2, 0.3], dtype=torch.float32, requires_grad=True
    )
    phi_params = torch.tensor([0.4], dtype=torch.float32, requires_grad=True)

    unitary = converter.to_tensor(theta_params, phi_params)
    loss = (unitary.real.pow(2) + unitary.imag.pow(2)).sum()
    loss.backward()

    assert theta_params.grad is not None
    assert phi_params.grad is not None
    assert torch.all(torch.isfinite(theta_params.grad))
    assert torch.all(torch.isfinite(phi_params.grad))
    assert theta_params.grad.norm() > 0
    assert phi_params.grad.norm() > 0


def test_builder_integrates_directly_with_quantum_layer():
    builder = CircuitBuilder(n_modes=3)
    builder.add_angle_encoding(name="input")
    builder.add_rotations(trainable=True, name="theta")
    builder.add_superpositions(depth=1)

    layer = QuantumLayer(
        input_size=3,
        builder=builder,
        n_photons=1,
        measurement_strategy=MeasurementStrategy.probs(),
        dtype=torch.float32,
    )
    model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

    assert isinstance(layer.computation_process.circuit, pcvl.Circuit)

    x = torch.rand(4, 3)
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (4, 3)
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_angle_encoding_metadata_and_scaling():
    builder = CircuitBuilder(n_modes=4)
    builder.add_angle_encoding(
        modes=[0, 1, 2],
        name="input",
        scale=0.5,
    )

    specs = builder.angle_encoding_specs
    assert "input" in specs
    combos = specs["input"]["combinations"]
    scales = specs["input"]["scales"]

    assert combos == [(0,), (1,), (2,)]

    assert scales[0] == 0.5
    assert scales[1] == 0.5
    assert scales[2] == 0.5

    rotations = [
        component
        for component in builder.circuit.components
        if isinstance(component, Rotation) and component.role == ParameterRole.INPUT
    ]
    assert len(rotations) == len(combos)


def test_angle_encoding_subset_combinations_extend_metadata():
    builder = CircuitBuilder(n_modes=6)
    builder.add_angle_encoding(
        modes=[0, 1, 2],
        name="input",
        subset_combinations=True,
        max_order=2,
    )

    spec = builder.angle_encoding_specs["input"]
    combos = spec["combinations"]

    # First entries keep singleton order, higher-order combos are appended
    assert combos[:3] == [(0,), (1,), (2,)]
    assert (0, 1) in combos
    assert (1, 2) in combos

    rotations = [
        comp
        for comp in builder.circuit.components
        if isinstance(comp, Rotation) and comp.role == ParameterRole.INPUT
    ]
    assert len(rotations) == len(combos)


def test_angle_encoding_applies_scaling_in_quantum_layer():
    builder = CircuitBuilder(n_modes=4)
    builder.add_angle_encoding(
        modes=[0, 1, 2],
        name="input",
        scale=0.5,
    )

    layer = QuantumLayer(
        input_size=3,
        builder=builder,
        n_photons=1,
        measurement_strategy=MeasurementStrategy.probs(),
        dtype=torch.float32,
    )
    torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

    x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    params = layer.prepare_parameters([x])
    encoded = params[-1]

    assert encoded.shape == (1, 3)

    singles = encoded[0, :3].detach()
    expected_singles = torch.tensor([0.05, 0.1, 0.15], dtype=torch.float32)
    assert torch.allclose(singles, expected_singles, atol=1e-6)


def test_angle_encoding_subset_combinations_in_quantum_layer():
    builder = CircuitBuilder(n_modes=8)
    builder.add_angle_encoding(
        modes=[0, 1, 2],
        name="input",
        subset_combinations=True,
        max_order=2,
    )

    layer = QuantumLayer(
        input_size=3,
        builder=builder,
        n_photons=1,
        measurement_strategy=MeasurementStrategy.probs(),
        dtype=torch.float32,
    )

    x = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float32)
    encoded = layer.prepare_parameters([x])[-1]

    combos = builder.angle_encoding_specs["input"]["combinations"]
    assert encoded.shape[1] == len(combos)

    assert torch.allclose(encoded[0, 0], x[0, 0])
    assert torch.allclose(encoded[0, 1], x[0, 1])
    assert torch.allclose(encoded[0, 2], x[0, 2])

    pair_idx = combos.index((0, 1))
    assert torch.allclose(encoded[0, pair_idx], x[0, 0] + x[0, 1])


def test_angle_encoding_raises_when_modes_exceeded():
    builder = CircuitBuilder(n_modes=3)

    with pytest.raises(
        ValueError, match="You cannot encore more features than mode with Builder"
    ):
        builder.add_angle_encoding(modes=[0, 1, 2, 3])


def test_angle_encoding_tracks_logical_indices_for_sparse_modes():
    builder = CircuitBuilder(n_modes=6)

    builder.add_angle_encoding(modes=[0, 2, 4], name="input")

    spec = builder.angle_encoding_specs["input"]
    # Logical indices should stay compact regardless of which physical modes were used
    assert spec["combinations"] == [(0,), (1,), (2,)]
    assert spec["scales"] == {0: 1.0, 1: 1.0, 2: 1.0}


def test_trainable_name_deduplication_for_rotation_layer():
    builder = CircuitBuilder(n_modes=2)

    builder.add_rotations(modes=[0, 1], trainable=True, name="theta")
    builder.add_rotations(modes=[0, 1], trainable=True, name="theta")
    pcvl.pdisplay(builder.to_pcvl_circuit(pcvl), output_format=pcvl.Format.TEXT)
    rotations = [
        comp for comp in builder.circuit.components if isinstance(comp, Rotation)
    ]
    assert [rot.custom_name for rot in rotations] == [
        "theta_0",
        "theta_1",
        "theta_0_1",
        "theta_1_1",
    ]
    # Prefix list should still expose the user-provided stem
    assert builder.trainable_parameter_prefixes == ["theta"]


def test_trainable_name_deduplication_for_single_rotation():
    builder = CircuitBuilder(n_modes=2)

    builder.add_rotations(modes=0, trainable=True, name="phi")
    builder.add_rotations(modes=1, trainable=True, name="phi")

    rotations = [
        comp for comp in builder.circuit.components if isinstance(comp, Rotation)
    ]
    assert [rot.custom_name for rot in rotations] == ["phi", "phi_1"]
    assert builder.trainable_parameter_prefixes == ["phi"]


def test_entangling_layer_defaults():
    builder = CircuitBuilder(n_modes=4)
    builder.add_entangling_layer()

    component = builder.circuit.components[-1]
    assert isinstance(component, GenericInterferometer)
    assert component.start_mode == 0
    assert component.span == 4
    assert component.trainable is True
    assert component.model == "mzi"
    assert any(
        prefix.startswith("el") for prefix in builder.trainable_parameter_prefixes
    )


def test_entangling_layer_mode_range_and_non_trainable():
    builder = CircuitBuilder(n_modes=5)
    builder.add_entangling_layer(modes=[2], trainable=False)

    component = builder.circuit.components[-1]
    assert isinstance(component, GenericInterferometer)
    assert component.start_mode == 2
    assert component.span == 3
    assert component.trainable is False
    assert builder.trainable_parameter_prefixes == []

    builder.add_entangling_layer(modes=[1, 3], trainable=True, name="block")
    last = builder.circuit.components[-1]
    assert last.start_mode == 1 and last.span == 3
    assert last.model == "mzi"
    assert "block" in builder.trainable_parameter_prefixes


def test_entangling_layer_invalid_modes():
    builder = CircuitBuilder(n_modes=4)

    with pytest.raises(ValueError):
        builder.add_entangling_layer(modes=[5])

    with pytest.raises(ValueError):
        builder.add_entangling_layer(modes=[1, 1])

    with pytest.raises(ValueError):
        builder.add_entangling_layer(modes=[0, 1, 2])


def test_entangling_layer_invalid_model():
    builder = CircuitBuilder(n_modes=3)
    with pytest.raises(ValueError, match="model must be either 'mzi' or 'bell'"):
        builder.add_entangling_layer(model="xyz")


@pytest.mark.parametrize("model", ["mzi", "bell"])
def test_entangling_layer_model_selection_to_pcvl(model):
    builder = CircuitBuilder(n_modes=4)
    builder.add_entangling_layer(model=model, name="custom")

    component = builder.circuit.components[-1]
    assert component.model == model

    pcvl_circuit = builder.to_pcvl_circuit(pcvl)
    params = pcvl_circuit.get_parameters()
    assert any(p.name.startswith("custom_li") for p in params)
    assert any(p.name.startswith("custom_lo") for p in params)


@pytest.mark.parametrize("model", ["mzi", "bell"])
def test_entangling_layer_models_forward_backward(model):
    builder = CircuitBuilder(n_modes=4)
    builder.add_angle_encoding(modes=[0, 1, 2, 3], name="input")
    builder.add_entangling_layer(model=model, trainable=True, name=f"{model}_ent")
    builder.add_rotations(trainable=True, name="theta")
    builder.add_superpositions(depth=1)

    layer = QuantumLayer(
        input_size=4,
        builder=builder,
        n_photons=1,
        measurement_strategy=MeasurementStrategy.probs(),
        dtype=torch.float32,
    )

    model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 4))
    x = torch.rand(2, 4)
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (2, 4)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_entangling_layer_to_pcvl_registers_parameters():
    builder = CircuitBuilder(n_modes=4)
    builder.add_entangling_layer(name="bridge")

    pcvl_circuit = builder.to_pcvl_circuit(pcvl)
    params = pcvl_circuit.get_parameters()
    assert any(p.name.startswith("bridge_li") for p in params)
    assert any(p.name.startswith("bridge_lo") for p in params)


def test_entangling_layer_layer_trains():
    builder = CircuitBuilder(n_modes=4)
    builder.add_angle_encoding(modes=[0, 1, 2, 3], name="input")
    builder.add_entangling_layer(trainable=True, name="gi")

    layer = QuantumLayer(
        input_size=4,
        builder=builder,
        n_photons=1,
        measurement_strategy=MeasurementStrategy.probs(),
        dtype=torch.float32,
    )
    model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 4))

    x = torch.rand(5, 4)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    pcvl.pdisplay(layer.computation_process.circuit, output_format=pcvl.Format.TEXT)
    assert logits.shape == (5, 4)
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()
    )


def test_entangling_layer_with_additional_components_trains():
    builder = CircuitBuilder(n_modes=5)
    builder.add_angle_encoding(modes=[0, 1, 2, 3, 4], name="input")
    builder.add_entangling_layer(trainable=True, name="core", modes=[2])
    builder.add_rotations(trainable=True, name="theta")
    builder.add_superpositions(depth=1)

    layer = QuantumLayer(
        input_size=5,
        builder=builder,
        n_photons=1,
        measurement_strategy=MeasurementStrategy.probs(),
        dtype=torch.float32,
    )
    pcvl.pdisplay(layer.computation_process.circuit, output_format=pcvl.Format.TEXT)
    model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 5))

    x = torch.rand(3, 5)
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (3, 5)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


@pytest.mark.parametrize("model", ["mzi", "bell"])
def test_entangling_layer_models_on_gpu(model):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU entangling-layer test.")

    device = torch.device("cuda")
    builder = CircuitBuilder(n_modes=4)
    builder.add_angle_encoding(modes=[0, 1, 2, 3], name="input")
    builder.add_entangling_layer(model=model, trainable=True, name=f"{model}_ent")
    builder.add_rotations(trainable=True, name="theta")
    builder.add_superpositions(depth=1)

    layer = QuantumLayer(
        input_size=4,
        builder=builder,
        n_photons=1,
        measurement_strategy=MeasurementStrategy.probs(),
        dtype=torch.float32,
    ).to(device)

    x = torch.rand(2, 4, device=device)
    logits = layer(x)
    loss = logits.sum()
    loss.backward()

    assert logits.device.type == device.type
    assert logits.shape == (2, 4)
    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads if g is not None)
