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

"""
Tests for the main QuantumLayer class.
"""

from collections.abc import Iterable, Sequence

import numpy as np
import perceval as pcvl
import pytest
import torch
from perceval.algorithm.sampler import Sampler

import merlin as ML
from merlin.core.computation_space import ComputationSpace

N_MODES = 8
INPUT_STATE = [1, 0, 1, 0, 1, 0, 1, 0]
N_PHOTONS = sum(INPUT_STATE)
SAMPLING_SHOTS = 200_000
SAMPLING_TOLERANCE = 0.02


def _haar_random_unitary(dim: int, seed: int) -> pcvl.Matrix:
    """Generate a deterministic Haar-random unitary using QR decomposition."""
    rng = np.random.default_rng(seed)
    components = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(components)
    diag = np.diag(r)
    phases = np.where(np.abs(diag) > 0, diag / np.abs(diag), 1.0)
    unitary = q * phases
    return pcvl.Matrix(unitary)


def _detector_from_spec(spec: str | tuple[str, dict[str, int]]) -> pcvl.Detector | None:
    """Instantiate a Perceval detector from a simple specification."""
    if spec is None:
        return None

    if isinstance(spec, tuple):
        kind, params = spec
    else:
        kind, params = spec, {}

    params = dict(params)

    if kind == "pnr":
        return pcvl.Detector.pnr()
    if kind == "threshold":
        return pcvl.Detector.threshold()
    if kind == "ppnr":
        return pcvl.Detector.ppnr(**params)
    raise ValueError(f"Unsupported detector kind: {kind}")


def _build_experiment(
    detector_specs: Sequence[str | tuple[str, dict[str, int]] | None],
) -> pcvl.Experiment:
    """Create a Perceval experiment with a shared unitary and detector configuration."""
    experiment = pcvl.Experiment()
    _haar_random_unitary_8 = _haar_random_unitary(N_MODES, seed=42)
    experiment.set_circuit(pcvl.Unitary(_haar_random_unitary_8))
    for mode, spec in enumerate(detector_specs):
        detector = _detector_from_spec(spec)
        if detector is not None:
            experiment.detectors[mode] = detector
    return experiment


def _normalize_key(key: Iterable[int] | torch.Tensor) -> tuple[int, ...]:
    """Convert detection keys to plain tuples to ease dictionary lookups."""
    if isinstance(key, torch.Tensor):
        return tuple(int(v) for v in key.tolist())
    return tuple(int(v) for v in key)


def _detector_kind(spec: str | tuple[str, dict[str, int]] | None) -> str | None:
    """Return the detector identifier string for convenience checks."""
    if spec is None:
        return None
    if isinstance(spec, tuple):
        return spec[0]
    return spec


DETECTOR_SCENARIOS: Sequence[
    tuple[str, Sequence[str | tuple[str, dict[str, int]] | None]]
] = [
    ("no_detectors", [None] * N_MODES),
    ("pnr", [("pnr", {})] * N_MODES),
    ("threshold", [("threshold", {})] * N_MODES),
    ("ppnr", [("ppnr", {"n_wires": 3})] * N_MODES),
    (
        "mixed",
        [
            ("threshold", {}),
            ("pnr", {}),
            ("ppnr", {"n_wires": 2}),
            ("threshold", {}),
            ("pnr", {}),
            ("ppnr", {"n_wires": 4}),
            ("threshold", {}),
            ("pnr", {}),
        ],
    ),
]


class TestDetectorsWithQuantumLayer:
    """Test suite for Detectors integration with QuantumLayer."""

    def test_threshold_detectors_preserve_binary_outcomes(self):
        circuit = pcvl.Circuit(3)
        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.threshold()
        experiment.detectors[2] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0, 1],
            computation_space=ComputationSpace.FOCK,
        )

        output = layer()
        keys = [tuple(key) for key in layer.output_keys]
        # All keys must be binary tuples
        assert all(all(value in (0, 1) for value in key) for key in keys)
        # Probability mass should sit entirely on the observed detection pattern
        target_index = keys.index((1, 0, 1))
        assert torch.allclose(
            output[:, target_index], torch.ones_like(output[:, target_index])
        )

    def test_amplitudes_strategy_with_detectors_is_rejected(self):
        circuit = pcvl.Circuit(2)
        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()

        with pytest.raises(
            RuntimeError,
            match="MeasurementStrategy\\.AMPLITUDES does not support experiments with detectors",
        ):
            ML.QuantumLayer(
                input_size=0,
                experiment=experiment,
                input_state=[1, 0],
                measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
                computation_space=ComputationSpace.FOCK,
            )

        # No error with MeasurementStrategy PROBABILITIES or MODE_EXPECTATIONS
        ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0],
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
            computation_space=ComputationSpace.FOCK,
        )
        ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0],
            measurement_strategy=ML.MeasurementStrategy.MODE_EXPECTATIONS,
            computation_space=ComputationSpace.FOCK,
        )

    def test_pnr_detectors_match_default_distribution(self):
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())

        default_layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 1],
            computation_space=ComputationSpace.FOCK,
        )

        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.pnr()
        experiment.detectors[1] = pcvl.Detector.pnr()

        detector_layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1],
            computation_space=ComputationSpace.FOCK,
        )

        probs_default = default_layer()
        probs_detector = detector_layer()
        assert torch.allclose(probs_detector, probs_default, atol=1e-6)
        assert [tuple(key) for key in detector_layer.output_keys] == [
            tuple(key) for key in default_layer.output_keys
        ]

    def test_interleaved_n_wire_1_equals_threshold(self):
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())
        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.ppnr(n_wires=1)

        experiment_threshold = pcvl.Experiment(circuit)
        experiment_threshold.detectors[0] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[4, 0],
            computation_space=ComputationSpace.FOCK,
        )

        layer_threshold = ML.QuantumLayer(
            input_size=0,
            experiment=experiment_threshold,
            input_state=[4, 0],
            computation_space=ComputationSpace.FOCK,
        )

        output = layer()
        output_threshold = layer_threshold()
        assert output.shape[-1] == output_threshold.shape[-1]
        assert torch.allclose(output, output_threshold)
        assert torch.allclose(output.sum(dim=1), torch.ones_like(output[:, 0]))
        assert torch.allclose(
            output_threshold.sum(dim=1), torch.ones_like(output_threshold[:, 0])
        )
        assert output.shape[-1] == len(layer.output_keys)
        assert torch.all(output >= 0)
        keys = [tuple(key) for key in layer.output_keys]
        keys_threshold = [tuple(key) for key in layer_threshold.output_keys]
        assert keys == keys_threshold
        assert all(value in (0, 1) for key in keys for value in key[:1])
        assert all(
            any(value == i for key in keys for value in key[1:]) for i in range(5)
        )

    def test_interleaved_detectors_more_wires_than_modes(self):
        experiment = pcvl.Experiment()
        experiment.set_circuit(pcvl.Unitary(pcvl.Matrix.random_unitary(4)))
        experiment.detectors[0] = pcvl.Detector.ppnr(n_wires=2)
        experiment.detectors[1] = pcvl.Detector.ppnr(n_wires=5)

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[3, 4, 1, 0],
            computation_space=ComputationSpace.FOCK,
        )

        output = layer()
        assert torch.allclose(torch.sum(output, dim=1), torch.ones_like(output[:, 0]))
        keys = [tuple(key) for key in layer.output_keys]
        assert all(key[0] in (0, 1, 2) for key in keys)
        assert all(key[1] in (0, 1, 2, 3, 4, 5) for key in keys)
        assert all(any(key[0] == i for key in keys) for i in range(3))
        assert all(any(key[1] == i for key in keys) for i in range(6))
        assert all(any(key[2] == i for key in keys) for i in range(8))
        assert all(any(key[3] == i for key in keys) for i in range(8))

    def test_mixed_detectors_probabilistic_distribution(self):
        circuit = pcvl.Circuit(4)
        circuit.add((0, 1), pcvl.BS())
        circuit.add((1, 2), pcvl.BS())
        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.pnr()
        experiment.detectors[1] = pcvl.Detector.pnr()
        experiment.detectors[2] = pcvl.Detector.threshold()
        experiment.detectors[3] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1, 1, 0],
            computation_space=ComputationSpace.FOCK,
        )

        output = layer()
        keys = [tuple(key) for key in layer.output_keys]
        assert output.shape[-1] == len(keys)
        assert torch.allclose(
            output.sum(dim=1), torch.ones_like(output[:, 0]), atol=1e-6
        )
        assert all(value in (0, 1) for key in keys for value in key[2:])
        assert all(any(key[0] == i for key in keys) for i in range(4))
        assert all(any(key[1] == i for key in keys) for i in range(4))
        assert all(0 <= key[0] + key[1] <= 3 for key in keys)

    def test_experiment_missing_detectors_default_pnr(self):
        circuit = pcvl.Circuit(2)
        circuit.add(1, pcvl.PS(torch.pi / 2))
        circuit.add((0, 1), pcvl.BS())
        experiment = pcvl.Experiment(circuit)

        layer_experiment = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1],
        )

        layer_direct = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 1],
        )

        probs_exp = layer_experiment()
        probs_direct = layer_direct()
        assert torch.allclose(probs_exp, probs_direct, atol=1e-6)

    def test_partial_detector_assignment_defaults_remaining_to_pnr(self):
        circuit = pcvl.Circuit(3)
        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[0, 0, 2],
            computation_space=ComputationSpace.FOCK,
        )

        output = layer()
        keys = [tuple(key) for key in layer.output_keys]
        assert (0, 0, 2) in keys
        idx = keys.index((0, 0, 2))
        assert torch.allclose(output[:, idx], torch.ones_like(output[:, idx]))
        assert all(value in (0, 1) for key in keys for value in key[:2])
        assert any(key[2] == 2 for key in keys)

    def test_experiment_layer_matches_perceval_pnr(self):
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())
        circuit.add(0, pcvl.PS(torch.pi / 3))

        input_state = [1, 1]

        experiment = pcvl.Experiment(circuit)
        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=input_state,
            computation_space=ComputationSpace.FOCK,
        )

        output = layer().squeeze(0)
        keys = [tuple(key) for key in layer.output_keys]
        assert output.shape[0] == len(keys)

        experiment_reference = pcvl.Experiment(circuit)
        processor = pcvl.Processor("SLOS", experiment_reference)
        processor.with_input(pcvl.BasicState(input_state))

        raw_results = processor.probs()["results"]
        probability_map = {
            tuple(int(v) for v in state): float(prob)
            for state, prob in raw_results.items()
        }

        reference = torch.tensor(
            [probability_map.get(key, 0.0) for key in keys],
            dtype=output.dtype,
        )

        assert torch.allclose(output, reference, atol=1e-6)

    def test_experiment_layer_matches_perceval_threshold(self):
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())
        circuit.add(1, pcvl.PS(torch.pi / 4))

        input_state = [1, 1]

        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=input_state,
            computation_space=ComputationSpace.FOCK,
        )

        output = layer().squeeze(0)
        keys = [tuple(key) for key in layer.output_keys]
        assert output.shape[0] == len(keys)

        processor = pcvl.Processor("SLOS", experiment)
        processor.with_input(pcvl.BasicState(input_state))
        processor.min_detected_photons_filter(1)

        raw_results = processor.probs()["results"]
        probability_map = {
            tuple(int(v) for v in state): float(prob)
            for state, prob in raw_results.items()
        }

        reference = torch.tensor(
            [probability_map.get(key, 0.0) for key in keys],
            dtype=output.dtype,
        )

        assert torch.allclose(output, reference, atol=1e-6)

    def test_detector_choice_adjusts_output_size(self):
        circuit = pcvl.Circuit(3)
        circuit.add((0, 1), pcvl.BS())
        circuit.add((1, 2), pcvl.BS())
        input_state = [3, 0, 0]

        pnr_layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=input_state,
            computation_space=ComputationSpace.FOCK,
        )

        pnr_output_size = pnr_layer.output_size
        assert pnr_output_size == len(pnr_layer.output_keys)

        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.threshold()
        experiment.detectors[2] = pcvl.Detector.threshold()

        threshold_layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=input_state,
            computation_space=ComputationSpace.FOCK,
        )

        threshold_output_size = threshold_layer.output_size
        assert threshold_output_size == len(threshold_layer.output_keys)
        assert threshold_output_size < pnr_output_size

    def test_detector_autograd_compatibility(self):
        """Detector transforms must preserve autograd support."""

        circuit = pcvl.Circuit(2)
        circuit.add(0, pcvl.PS(pcvl.P("theta_1")))
        circuit.add(0, pcvl.BS())
        circuit.add(0, pcvl.PS(pcvl.P("phi")))
        circuit.add(0, pcvl.BS())
        circuit.add(0, pcvl.PS(pcvl.P("theta_2")))
        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=1,
            experiment=experiment,
            input_state=[1, 0],
            input_parameters=["phi"],
            trainable_parameters=["theta"],
            computation_space=ComputationSpace.FOCK,
        )
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 1))

        x = torch.linspace(0.0, 0.5, steps=4, requires_grad=True)
        output = model(x.unsqueeze(1))
        target = torch.ones_like(output)
        loss = torch.sum(target - output)
        loss.backward()

        assert model[1].weight.grad is not None
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_simple_experiment_layer_detectors_vs_perceval(self):
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())
        circuit.add(1, pcvl.PS(torch.pi / 4))
        circuit.add((0, 1), pcvl.BS())

        input_state = [1, 1]

        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.pnr()
        experiment.detectors[1] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=input_state,
            computation_space=ComputationSpace.FOCK,
        )

        output = layer().squeeze(0)
        keys = layer.output_keys
        assert output.shape[0] == len(keys)

        processor = pcvl.Processor("SLOS", experiment)
        processor.with_input(pcvl.BasicState(input_state))
        processor.min_detected_photons_filter(1)

        raw_results = processor.probs()["results"]
        probability_map = {
            tuple(int(v) for v in state): float(prob)
            for state, prob in raw_results.items()
        }

        reference = torch.tensor(
            [probability_map.get(key, 0.0) for key in keys],
            dtype=output.dtype,
        )

        assert torch.allclose(output, reference, atol=1e-6)

    def test_complex_experiment_layer_detectors_vs_perceval(self):
        exp = pcvl.Experiment()
        exp.set_circuit(pcvl.Unitary(pcvl.Matrix.random_unitary(4)))
        exp.add(0, pcvl.Detector.threshold())
        exp.add(1, pcvl.Detector.pnr())
        exp.add(2, pcvl.Detector.ppnr(2))
        exp.add(3, pcvl.Detector.ppnr(3))

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=exp,
            input_state=[1, 1, 1, 1],
            computation_space=ComputationSpace.FOCK,
        )

        layer_probs = layer()
        layer_keys = layer.output_keys

        p = pcvl.Processor("SLOS", exp)
        p.with_input(pcvl.BasicState([1, 1, 1, 1]))
        p.min_detected_photons_filter(1)

        raw_results = p.probs()["results"]
        probability_map = {
            tuple(int(v) for v in state): float(prob)
            for state, prob in raw_results.items()
        }

        reference = torch.tensor(
            [probability_map.get(key, 0.0) for key in layer_keys],
            dtype=layer_probs.dtype,
        )

        assert torch.allclose(layer_probs, reference, atol=1e-6)

    def test_detector_plus_no_bunching_warning(self):
        """Using detectors with no_bunching=True should raise a warning, and ignore the detectors."""
        random_unitary = pcvl.Unitary(pcvl.Matrix.random_unitary(4))

        # Define Experiment without Detector
        experiment = pcvl.Experiment()
        experiment.set_circuit(random_unitary)
        # No detector

        # Define Experiement with pnr Detector
        experiment_pnr_detector = pcvl.Experiment()
        experiment_pnr_detector.set_circuit(random_unitary)
        experiment_pnr_detector.detectors[1] = pcvl.Detector.pnr()

        # Define Experiement with threshold Detector
        experiment_threshold_detector = pcvl.Experiment()
        experiment_threshold_detector.set_circuit(random_unitary)
        experiment_threshold_detector.detectors[3] = pcvl.Detector.threshold()

        # Define Experiement with ppnr Detertors
        experiment_ppnr_detector = pcvl.Experiment()
        experiment_ppnr_detector.set_circuit(random_unitary)
        experiment_ppnr_detector.detectors[0] = pcvl.Detector.ppnr(n_wires=2)
        experiment_ppnr_detector.detectors[2] = pcvl.Detector.ppnr(n_wires=2)

        # Define layers
        ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1, 1, 1],
            computation_space=ComputationSpace.FOCK,
        )
        layer_unbunched = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1, 1, 1],
            computation_space=ComputationSpace.UNBUNCHED,
        )
        ML.QuantumLayer(
            input_size=0,
            experiment=experiment_pnr_detector,
            input_state=[1, 1, 1, 1],
            computation_space=ComputationSpace.FOCK,
        )

        with pytest.warns(UserWarning):
            layer_pnr_unbunched = ML.QuantumLayer(
                input_size=0,
                experiment=experiment_pnr_detector,
                input_state=[1, 1, 1, 1],
                computation_space=ComputationSpace.UNBUNCHED,
            )
        ML.QuantumLayer(
            input_size=0,
            experiment=experiment_threshold_detector,
            input_state=[1, 1, 1, 1],
            computation_space=ComputationSpace.FOCK,
        )
        with pytest.warns(UserWarning):
            layer_threshold_unbunched = ML.QuantumLayer(
                input_size=0,
                experiment=experiment_threshold_detector,
                input_state=[1, 1, 1, 1],
                computation_space=ComputationSpace.UNBUNCHED,
            )
        ML.QuantumLayer(
            input_size=0,
            experiment=experiment_ppnr_detector,
            input_state=[1, 1, 1, 1],
            computation_space=ComputationSpace.FOCK,
        )
        with pytest.warns(UserWarning):
            layer_ppnr_unbunched = ML.QuantumLayer(
                input_size=0,
                experiment=experiment_ppnr_detector,
                input_state=[1, 1, 1, 1],
                computation_space=ComputationSpace.UNBUNCHED,
            )

        result_unbunched = layer_unbunched()
        result_pnr_unbunched = layer_pnr_unbunched()
        result_threshold_unbunched = layer_threshold_unbunched()
        result_ppnr_unbunched = layer_ppnr_unbunched()

        assert torch.allclose(result_unbunched, result_pnr_unbunched, atol=1e-6)
        assert torch.allclose(result_unbunched, result_threshold_unbunched, atol=1e-6)
        assert torch.allclose(result_unbunched, result_ppnr_unbunched, atol=1e-6)

    @pytest.mark.parametrize(
        "label, detector_specs",
        DETECTOR_SCENARIOS,
        ids=lambda item: item[0] if isinstance(item, tuple) else item,
    )
    def test_sampling_matches_perceval_across_detectors(
        self,
        label: str,
        detector_specs: Sequence[str | tuple[str, dict[str, int]] | None],
    ):
        """QuantumLayer sampling, probabilities, and expectations should track Perceval for diverse detectors."""
        _ = label  # id used only for parametrized display
        # Prepare Merlin layer for probability distribution
        layer_experiment = _build_experiment(detector_specs)
        layer = ML.QuantumLayer(
            input_size=0,
            experiment=layer_experiment,
            input_state=INPUT_STATE,
            computation_space=ComputationSpace.FOCK,
        )
        probabilities = layer().squeeze(0)
        keys = [_normalize_key(key) for key in layer.output_keys]
        key_to_index = {key: idx for idx, key in enumerate(keys)}

        # Build reference Perceval processor
        reference_experiment = _build_experiment(detector_specs)
        processor = pcvl.Processor("SLOS", reference_experiment)
        processor.with_input(pcvl.BasicState(INPUT_STATE))

        # Threshold / hybrid detectors can merge photons, making a strict photon-count filter unreliable.
        if any(_detector_kind(spec) not in {None, "pnr"} for spec in detector_specs):
            processor.min_detected_photons_filter(1)
        else:
            processor.min_detected_photons_filter(N_PHOTONS)

        raw_probabilities = processor.probs()["results"]
        probability_map = {
            tuple(int(v) for v in state): float(prob)
            for state, prob in raw_probabilities.items()
        }
        reference_probabilities = torch.tensor(
            [probability_map.get(key, 0.0) for key in keys],
            dtype=probabilities.dtype,
        )

        assert torch.allclose(probabilities, reference_probabilities, atol=1e-6)

        # Compare sampling statistics between Merlin and Perceval
        torch.manual_seed(12_345)
        sampled_distribution = layer(
            shots=SAMPLING_SHOTS,
            sampling_method="multinomial",
        ).squeeze(0)

        sampler = Sampler(processor)
        sample_counts = sampler.sample_count(max_samples=SAMPLING_SHOTS)["results"]
        total_samples = sum(sample_counts.values()) or 1
        perceval_sampled = torch.zeros_like(probabilities)
        for state, count in sample_counts.items():
            key = tuple(int(v) for v in state)
            index = key_to_index.get(key)
            if index is not None:
                perceval_sampled[index] = count / total_samples

        max_pairwise_delta = torch.max(
            torch.abs(sampled_distribution - perceval_sampled)
        )
        assert max_pairwise_delta < SAMPLING_TOLERANCE
        assert (
            torch.max(torch.abs(sampled_distribution - reference_probabilities))
            < SAMPLING_TOLERANCE
        )
        assert (
            torch.max(torch.abs(perceval_sampled - reference_probabilities))
            < SAMPLING_TOLERANCE
        )

        # Expectation values per mode must also coincide
        expectation_layer = ML.QuantumLayer(
            input_size=0,
            experiment=_build_experiment(detector_specs),
            input_state=INPUT_STATE,
            measurement_strategy=ML.MeasurementStrategy.MODE_EXPECTATIONS,
            computation_space=ComputationSpace.FOCK,
        )
        expectations = expectation_layer().squeeze(0)
        assert [_normalize_key(key) for key in expectation_layer.output_keys] == keys

        keys_tensor = torch.tensor(keys, dtype=probabilities.dtype)
        reference_expectations = reference_probabilities @ keys_tensor

        assert torch.allclose(expectations, reference_expectations, atol=1e-5)

    def test_threshold_detectors_preserve_unbunched_distribution(self):
        """Threshold detectors should not alter probabilities when computation space forbids bunching."""
        threshold_specs = [("threshold", {})] * N_MODES
        experiment_threshold = _build_experiment(threshold_specs)
        experiment_reference = _build_experiment([None] * N_MODES)

        with pytest.warns():
            layer_threshold = ML.QuantumLayer(
                input_size=0,
                experiment=experiment_threshold,
                input_state=INPUT_STATE,
                computation_space=ML.ComputationSpace.UNBUNCHED,
            )
        layer_reference = ML.QuantumLayer(
            input_size=0,
            experiment=experiment_reference,
            input_state=INPUT_STATE,
            computation_space=ML.ComputationSpace.UNBUNCHED,
        )

        probs_threshold = layer_threshold().squeeze(0)
        probs_reference = layer_reference().squeeze(0)
        keys_threshold = [_normalize_key(key) for key in layer_threshold.output_keys]
        keys_reference = [_normalize_key(key) for key in layer_reference.output_keys]

        assert keys_threshold == keys_reference
        assert torch.allclose(probs_threshold, probs_reference, atol=1e-6)


class TestDetectorsWithKernels:
    """Test suite for Detectors integration with Kernels."""

    def test_fidelity_kernel_with_threshold_detectors(self):
        """FidelityKernel should respect detector configuration from the experiment."""

        feature_map = ML.FeatureMap.simple(
            input_size=1,
            n_modes=2,
        )

        experiment = feature_map.experiment
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.threshold()

        kernel = ML.FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 0],
        )

        data = torch.tensor([[0.0], [1.0]], dtype=kernel.dtype)
        K = kernel(data, data)

        assert K.shape == (2, 2)
        assert torch.allclose(
            torch.diag(K), torch.ones(2, dtype=kernel.dtype), atol=1e-6
        )

    def test_fidelity_kernel_with_mixed_detectors(self):
        """FidelityKernel can combine PNR and threshold detectors seamlessly."""

        feature_map = ML.FeatureMap.simple(
            input_size=1,
            n_modes=3,
        )

        experiment = feature_map.experiment
        experiment.detectors[0] = pcvl.Detector.pnr()
        experiment.detectors[1] = pcvl.Detector.threshold()
        experiment.detectors[2] = pcvl.Detector.threshold()

        kernel = ML.FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 0, 0],
        )

        data = torch.tensor([[0.0], [0.5], [1.0]], dtype=kernel.dtype)
        K = kernel(data, data)

        assert K.shape == (3, 3)
        diag = torch.diag(K)
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)
        assert torch.allclose(K, K.T.conj(), atol=1e-6)

    def test_fidelity_kernel_with_ppnr_detectors(self):
        """FidelityKernel supports partially projected number-resolving detectors."""

        feature_map = ML.FeatureMap.simple(
            input_size=1,
            n_modes=3,
        )

        experiment = feature_map.experiment
        experiment.detectors[0] = pcvl.Detector.ppnr(n_wires=2, max_detections=1)
        experiment.detectors[1] = pcvl.Detector.pnr()
        experiment.detectors[2] = pcvl.Detector.pnr()

        kernel = ML.FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 0, 0],
        )

        data = torch.tensor([[0.0], [0.6]], dtype=kernel.dtype)
        K = kernel(data, data)

        assert K.shape == (2, 2)
        diag = torch.diag(K)
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)
        assert torch.allclose(K, K.T.conj(), atol=1e-6)
        assert torch.all(K >= 0)

    def test_fidelity_kernel_detector_choice_adjusts_key_space(self):
        """Detector selection should change the size of the kernel detection basis."""

        feature_map = ML.FeatureMap.simple(
            input_size=1,
            n_modes=3,
        )
        input_state = [1, 1, 1]

        kernel_pnr = ML.FidelityKernel(
            feature_map=feature_map,
            input_state=input_state,
        )

        feature_map_threshold = ML.FeatureMap.simple(
            input_size=1,
            n_modes=3,
        )
        experiment_threshold = feature_map_threshold.experiment
        experiment_threshold.detectors[0] = pcvl.Detector.threshold()
        experiment_threshold.detectors[1] = pcvl.Detector.threshold()
        experiment_threshold.detectors[2] = pcvl.Detector.threshold()

        kernel_threshold = ML.FidelityKernel(
            feature_map=feature_map_threshold,
            input_state=input_state,
        )

        keys_pnr = kernel_pnr._detector_transform.output_keys
        keys_threshold = kernel_threshold._detector_transform.output_keys

        assert kernel_pnr._detector_transform.output_size == len(keys_pnr)
        assert kernel_threshold._detector_transform.output_size == len(keys_threshold)
        assert len(keys_pnr) > len(keys_threshold)
        assert all(sum(key) == sum(input_state) for key in keys_pnr)
        assert any(sum(key) < sum(input_state) for key in keys_threshold)
        assert all(value in (0, 1) for key in keys_threshold for value in key)

    def test_fidelity_kernel_respects_feature_map_experiment(self):
        """FidelityKernel should inherit detector configuration provided via FeatureMap."""

        circuit = pcvl.Circuit(2)
        circuit.add(0, pcvl.PS(pcvl.P("px")))
        circuit.add((0, 1), pcvl.BS())

        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.pnr()

        feature_map = ML.FeatureMap(
            input_size=1,
            experiment=experiment,
            input_parameters=["px"],
        )

        kernel = ML.FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 1],
        )

        x_train = torch.tensor([[0.0], [0.5], [1.0]], dtype=kernel.dtype)
        x_test = torch.tensor([[0.2], [0.8]], dtype=kernel.dtype)

        k_train = kernel(x_train)
        k_test = kernel(x_test, x_train)

        assert k_train.shape == (3, 3)
        assert k_test.shape == (2, 3)
        diag = torch.diag(k_train)
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)
        assert torch.all(k_train >= 0)
        assert torch.all(k_test >= 0)
