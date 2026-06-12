# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import perceval as pcvl
import pytest
import torch

import merlin as ML
from merlin.algorithms.layer_utils import (
    _build_simple_circuit,
    apply_angle_encoding,
    compute_new_memristive_ps_angles,
    feature_count_for_prefix,
    normalize_output_key,
    prepare_input_encoding,
    prepare_input_state,
    resolve_circuit,
    setup_noise_and_detectors,
    split_inputs_by_prefix,
    validate_and_resolve_circuit_source,
    validate_encoding_mode,
    vet_experiment,
)
from merlin.core.computation_space import ComputationSpace
from merlin.measurement.strategies import MeasurementStrategy


def test_validate_encoding_mode_constraints():
    with pytest.raises(ValueError, match="input_size"):
        validate_encoding_mode(True, 2, 1, None)

    with pytest.raises(ValueError, match="n_photons"):
        validate_encoding_mode(True, None, None, None)

    with pytest.raises(ValueError, match="input parameters"):
        validate_encoding_mode(True, None, 1, ["x"])

    config = validate_encoding_mode(False, 3, None, ["x"])
    assert config.input_size == 3
    assert config.input_parameters == ["x"]


def test_prepare_input_state_basic_state():
    state, resolved = prepare_input_state(
        pcvl.BasicState([1, 0, 1]),
        None,
        ComputationSpace.UNBUNCHED,
        None,
        torch.complex64,
    )
    assert state == pcvl.BasicState([1, 0, 1])
    assert resolved is None


def test_prepare_input_state_statevector():
    sv = pcvl.StateVector()
    sv += pcvl.StateVector(pcvl.BasicState([1, 0])) * 1.0

    state, resolved = prepare_input_state(
        sv,
        None,
        ComputationSpace.UNBUNCHED,
        None,
        torch.complex64,
    )
    assert isinstance(state, torch.Tensor)
    assert resolved == 1


def test_prepare_input_state_empty_statevector_rejected():
    empty_sv = pcvl.StateVector()
    with pytest.raises(ValueError, match="StateVector cannot be empty"):
        prepare_input_state(
            empty_sv,
            None,
            ComputationSpace.UNBUNCHED,
            None,
            torch.complex64,
        )


def test_prepare_input_state_experiment_override_warns():
    experiment = pcvl.Experiment(pcvl.Circuit(2))
    experiment.with_input(pcvl.BasicState([1, 0]))

    with pytest.warns(UserWarning, match="experiment.input_state"):
        state, _ = prepare_input_state(
            [0, 1],
            None,
            ComputationSpace.UNBUNCHED,
            None,
            torch.complex64,
            experiment=experiment,
        )
    assert state == pcvl.BasicState([1, 0])


def test_prepare_input_state_default_generation():
    state, _ = prepare_input_state(
        None,
        2,
        ComputationSpace.UNBUNCHED,
        None,
        torch.complex64,
        circuit_m=4,
        amplitude_encoding=False,
    )
    assert state == ML.generate_state(4, 2, ML.StatePattern.SPACED)


def test_validate_and_resolve_circuit_source_builder_conflict():
    builder = ML.CircuitBuilder(n_modes=2)
    with pytest.raises(ValueError, match="do not also specify"):
        validate_and_resolve_circuit_source(
            builder,
            None,
            None,
            trainable_parameters=["theta"],
            input_parameters=None,
        )


def test_validate_and_resolve_circuit_source_multiple_sources():
    with pytest.raises(ValueError, match="exactly one"):
        validate_and_resolve_circuit_source(
            None,
            pcvl.Circuit(1),
            pcvl.Experiment(pcvl.Circuit(1)),
            None,
            None,
        )


def test_validate_and_resolve_circuit_source_builder_prefixes():
    builder = ML.CircuitBuilder(n_modes=2)
    builder.add_angle_encoding(modes=[0], name="x")
    source = validate_and_resolve_circuit_source(builder, None, None, None, None)
    assert source.source_type == "builder"
    assert source.input_parameters == ["x"]


def test_vet_experiment_rejects_post_select():
    experiment = pcvl.Experiment(pcvl.Circuit(1))
    experiment.set_postselection(pcvl.PostSelect("[0]==1"))
    with pytest.raises(ValueError, match="post-selection"):
        vet_experiment(experiment)


def test_vet_experiment_rejects_time_dependent():
    experiment = pcvl.Experiment(pcvl.Circuit(1))
    experiment.add(0, pcvl.TD(1))
    with pytest.raises(ValueError, match="unitary"):
        vet_experiment(experiment)


def test_resolve_circuit_experiment_path():
    circuit = pcvl.Circuit(2)
    experiment = pcvl.Experiment(circuit)
    source = validate_and_resolve_circuit_source(None, None, experiment, None, None)
    resolved = resolve_circuit(source, pcvl)
    assert resolved.experiment is experiment
    assert resolved.circuit.m == 2


def test_setup_noise_and_detectors_amplitudes_rejects_detectors():
    experiment = pcvl.Experiment(pcvl.Circuit(2))
    experiment._add_detector(mode=0, detector=pcvl.Detector.threshold())
    result = validate_and_resolve_circuit_source(None, None, experiment, None, None)
    resolved = resolve_circuit(result, pcvl)

    with pytest.raises(
        RuntimeError, match="does not support experiments with detectors"
    ):
        setup_noise_and_detectors(
            resolved.experiment,
            resolved.circuit,
            ComputationSpace.FOCK,
            MeasurementStrategy.amplitudes(computation_space=ComputationSpace.FOCK),
        )


def test_setup_noise_and_detectors_computation_space_overrides():
    experiment = pcvl.Experiment(pcvl.Circuit(2))
    experiment._add_detector(mode=0, detector=pcvl.Detector.threshold())
    result = validate_and_resolve_circuit_source(None, None, experiment, None, None)
    resolved = resolve_circuit(result, pcvl)

    config = setup_noise_and_detectors(
        resolved.experiment,
        resolved.circuit,
        ComputationSpace.UNBUNCHED,
        MeasurementStrategy.probs(computation_space=ComputationSpace.UNBUNCHED),
    )
    assert config.has_custom_detectors is True
    assert len(config.detectors) == 2
    assert config.detector_warnings


def test_apply_angle_encoding_basic():
    spec = {"combinations": [(0, 1)], "scales": {0: 1.0, 1: 2.0}}
    x = torch.tensor([1.0, 2.0])
    encoded = apply_angle_encoding(x, spec)
    assert encoded.shape == (1,)
    assert torch.allclose(encoded, torch.tensor([5.0]))


def test_prepare_input_encoding_passthrough():
    x = torch.tensor([1.0, 2.0])
    assert torch.allclose(prepare_input_encoding(x), x)


def test_split_inputs_by_prefix_uses_specs():
    specs = {"x": {"combinations": [(0,), (1,)]}, "y": {"combinations": [(0,)]}}
    tensor = torch.tensor([1.0, 2.0, 3.0])
    splits = split_inputs_by_prefix(["x", "y"], tensor, specs)
    assert splits is not None
    assert [t.numel() for t in splits] == [2, 1]


def test_feature_count_for_prefix_spec_mappings():
    specs: dict[str, dict[str, object]] = {}
    spec_mappings = {"x": ["x0", "x1", "x2"]}
    assert feature_count_for_prefix("x", specs, spec_mappings) == 3


def test_normalize_output_key_tensor():
    key = normalize_output_key(torch.tensor([1, 0, 2]))
    assert key == (1, 0, 2)


def test_compute_new_memristive_ps_angles():

    def exponential_decay(state: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        tau = 5.0
        target = torch.full([state.size(0)], output[2])
        return state + (target - state) / tau

    def sum_outputs(state: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return state + (output.mean().repeat(2))

    memristive_metadata = [
        {
            "target_mode": 0,
            "name": "mem1",
            "update_rule": exponential_decay,
            "initial_state": 1,
        },
        {
            "target_mode": 1,
            "name": "mem2",
            "update_rule": sum_outputs,
            "initial_state": 1000,
        },
    ]

    memristive_state = [torch.Tensor([1, 2]), torch.Tensor([10, 1000])]
    output = torch.Tensor([1, 2, 3, 4])

    res = compute_new_memristive_ps_angles(
        memristive_metadata=memristive_metadata,
        memristive_state=memristive_state,
        output=output,
    )
    expected = [torch.Tensor([1.4, 2.2]), torch.Tensor([12.5, 1002.5])]

    for x, y in zip(res, expected, strict=True):
        assert torch.allclose(x, y)


class TestBuildSimpleCircuit:
    """Tests for :func:`~merlin.algorithms.layer_utils._build_simple_circuit`."""

    def test_returns_circuit_builder(self):
        from merlin.builder.circuit_builder import CircuitBuilder

        builder = _build_simple_circuit(input_size=3, n_modes=4)
        assert isinstance(builder, CircuitBuilder)

    def test_n_modes_propagated(self):
        builder = _build_simple_circuit(input_size=4, n_modes=5)
        circuit = builder.to_pcvl_circuit(__import__("perceval"))
        assert circuit.m == 5

    def test_default_n_modes_is_input_size_plus_one(self):
        for input_size in (1, 3, 5):
            builder = _build_simple_circuit(input_size=input_size)
            circuit = builder.to_pcvl_circuit(__import__("perceval"))
            assert circuit.m == input_size + 1

    def test_default_scale_is_one(self):
        builder_default = _build_simple_circuit(input_size=3)
        builder_explicit = _build_simple_circuit(input_size=3, angle_encoding_scale=1.0)
        # Both builders should produce the same angle-encoding specs
        assert (
            builder_default.angle_encoding_specs
            == builder_explicit.angle_encoding_specs
        )

    def test_trainable_prefixes(self):
        builder = _build_simple_circuit(n_modes=4, input_size=3)
        assert "LI_simple" in builder.trainable_parameter_prefixes
        assert "RI_simple" in builder.trainable_parameter_prefixes

    def test_input_prefix(self):
        builder = _build_simple_circuit(n_modes=4, input_size=3)
        assert "input" in builder.input_parameter_prefixes

    def test_angle_encoding_scale_stored(self):
        scale = 2.5
        builder = _build_simple_circuit(n_modes=4, input_size=3, angle_encoding_scale=scale)
        specs = builder.angle_encoding_specs
        # All feature scales in the "input" spec should equal the provided scale
        input_spec = specs.get("input", {})
        scales = input_spec.get("scales", {})
        assert scales, "Expected non-empty scales dict"
        for v in scales.values():
            assert v == scale

    def test_produces_identical_circuit_for_layer_and_feature_map(self):
        """QuantumLayer.simple and FeatureMap.simple must share the same circuit topology."""
        import merlin as ML
        from merlin.algorithms.kernels import FeatureMap

        ql = ML.QuantumLayer.simple(input_size=3)
        fm = FeatureMap.simple(input_size=3)

        ql_circuit = ql.quantum_layer.circuit
        fm_circuit = fm.circuit

        assert ql_circuit.m == fm_circuit.m
        # Both circuits encode the same number of modes
        assert ql_circuit.m == 4  # input_size + 1

    def test_quantum_layer_and_feature_map_share_parameter_names(self):
        """Both simple factories must expose the same trainable parameter names."""
        import merlin as ML
        from merlin.algorithms.kernels import FeatureMap

        ql_params = [k for k, _ in ML.QuantumLayer.simple(input_size=3).quantum_layer.named_parameters()]
        fm_params = FeatureMap.simple(input_size=3).trainable_parameters

        assert "LI_simple" in ql_params
        assert "RI_simple" in ql_params
        assert "LI_simple" in fm_params
        assert "RI_simple" in fm_params
