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
    apply_angle_encoding,
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
    assert state == [1, 0, 1]
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
    assert state == [1, 0]


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
    assert state == ML.StateGenerator.generate_state(4, 2, ML.StatePattern.SPACED)



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
            MeasurementStrategy.AMPLITUDES,
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
        MeasurementStrategy.PROBABILITIES,
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
