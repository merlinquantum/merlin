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

import itertools
import math

import perceval as pcvl
import pytest
import torch

import merlin as ML
from merlin.core.computation_space import ComputationSpace


class TestPhotonLossWithQuantumLayer:
    """Test suite for photon loss integration with QuantumLayer."""

    def test_basic_photon_loss(self):
        """Single-photon experiment should produce survival and loss outcomes."""
        circuit = pcvl.Circuit(2)
        experiment = pcvl.Experiment(circuit)
        experiment.noise = pcvl.NoiseModel(
            brightness=0.6,
            transmittance=0.9,
        )

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1],
            measurement_strategy=ML.MeasurementStrategy.probs(
                computation_space=ComputationSpace.FOCK
            ),
        )

        output = layer()
        keys = layer.output_keys
        expected_keys = {(1, 1), (1, 0), (0, 0), (0, 1), (1, 0), (2, 0), (0, 2)}
        assert set(keys) == expected_keys
        assert torch.allclose(output.sum(dim=1), torch.ones_like(output[:, 0]))

        distribution = dict(zip(keys, output.squeeze(0).tolist(), strict=False))
        assert (
            pytest.approx(distribution[(1, 1)], rel=1e-6, abs=1e-6) == (0.6 * 0.9) ** 2
        )
        assert (
            pytest.approx(distribution[(0, 0)], rel=1e-6, abs=1e-6)
            == (1 - (0.6 * 0.9)) ** 2
        )
        assert pytest.approx(distribution[(1, 0)], rel=1e-6, abs=1e-6) == (
            1 - (0.6 * 0.9)
        ) * (0.6 * 0.9)
        assert pytest.approx(distribution[(0, 1)], rel=1e-6, abs=1e-6) == (
            0.6 * 0.9
        ) * (1 - (0.6 * 0.9))

        circuit = pcvl.Circuit(2)
        experiment_perfect = pcvl.Experiment(circuit)
        experiment.noise = pcvl.NoiseModel(
            brightness=1.0,
            transmittance=1.0,
        )

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment_perfect,
            input_state=[1, 1],
            measurement_strategy=ML.MeasurementStrategy.probs(
                computation_space=ComputationSpace.FOCK
            ),
        )

        output = layer()
        keys = layer.output_keys
        assert set(keys) == {(1, 1), (2, 0), (0, 2)}
        assert output.shape == (1, 3)
        assert torch.allclose(
            output.squeeze(0),
            torch.tensor([key == (1, 1) for key in keys], dtype=torch.float),
            atol=1e-6,
        )

    def test_amplitudes_strategy_rejected_with_photon_loss(self):
        """Amplitude measurement strategy must be rejected when photon loss is enabled."""
        circuit = pcvl.Circuit(2)
        experiment = pcvl.Experiment(circuit)
        experiment.noise = pcvl.NoiseModel(
            brightness=0.8,
            transmittance=1.0,
        )

        with pytest.raises(
            RuntimeError,
            match="measurement_strategy=MeasurementStrategy.AMPLITUDES cannot be used when the experiment defines a NoiseModel.",
        ):
            ML.QuantumLayer(
                input_size=0,
                experiment=experiment,
                input_state=[1, 0],
                measurement_strategy=ML.MeasurementStrategy.NONE,
            )

        prob_layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0],
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )
        expectation_layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0],
            measurement_strategy=ML.MeasurementStrategy.mode_expectations(
                computation_space=ComputationSpace.UNBUNCHED
            ),
        )
        prob_output = prob_layer()
        expectation_output = expectation_layer()
        keys = prob_layer.output_keys

        assert prob_output.shape[-1] == len(keys)
        assert expectation_output.shape[-1] == len(keys[0])

        experiment_no_noise = pcvl.Experiment(circuit)

        amplitude_layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment_no_noise,
            input_state=[1, 0],
            measurement_strategy=ML.MeasurementStrategy.NONE,
        )

        keys_amplitudes = amplitude_layer.output_keys
        assert amplitude_layer().shape[-1] == len(keys_amplitudes)

    def test_photon_loss_unbunched(self):
        """No-bunching simulations should run (unless a Detector is specified) and keep binary-valued keys after loss."""
        circuit = pcvl.Circuit(3)
        circuit.add((0, 1), pcvl.BS(theta=math.pi / 6))
        experiment = pcvl.Experiment(circuit)
        experiment.noise = pcvl.NoiseModel(brightness=0.7)

        experiment_detectors = pcvl.Experiment(circuit)
        experiment_detectors.noise = pcvl.NoiseModel(brightness=0.7)
        experiment_detectors.detectors[0] = pcvl.Detector.ppnr(n_wires=2)

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1, 1],
        )

        # Warning with a Detector present
        with pytest.warns(UserWarning):
            layer_detectors = ML.QuantumLayer(
                input_size=0,
                experiment=experiment_detectors,
                input_state=[1, 1, 1],
            )

        output = layer()
        keys = [tuple(key) for key in layer.output_keys]
        assert len(keys) == 8  # 2^3 survivals
        assert all(all(value in (0, 1) for value in key) for key in keys)
        assert torch.allclose(
            output.sum(dim=1), torch.ones_like(output[:, 0]), atol=1e-6
        )
        expected_keys = {tuple(bits) for bits in itertools.product((0, 1), repeat=3)}
        assert set(keys) == expected_keys

        output_detectors = layer_detectors()
        assert torch.allclose(output, output_detectors, atol=1e-6)

    def test_photon_loss_no_experiment(self):
        """Layers built without experiment must default to identity loss transform."""
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())

        layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 0],
        )

        output = layer()
        keys = layer.output_keys
        raw_keys = list(layer.computation_process.simulation_graph.mapped_keys)

        assert keys == raw_keys
        assert torch.allclose(
            output.sum(dim=1), torch.ones_like(output[:, 0]), atol=1e-6
        )

        layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 4],
            computation_space=ML.ComputationSpace.FOCK,
        )

        output = layer()
        keys = layer.output_keys
        raw_keys = list(layer.computation_process.simulation_graph.mapped_keys)

        assert keys == raw_keys
        assert torch.allclose(
            output.sum(dim=1), torch.ones_like(output[:, 0]), atol=1e-6
        )

    def test_photon_loss_no_noise_model(self):
        """Experiments without a noise model must match the noise-free circuit."""
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())
        experiment = pcvl.Experiment(circuit)

        layer_direct = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 0],
            computation_space=ML.ComputationSpace.FOCK,
        )
        layer_experiment = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0],
            computation_space=ML.ComputationSpace.FOCK,
        )
        layer_experiment_unbunched = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0],
        )

        assert torch.allclose(layer_direct(), layer_experiment(), atol=1e-6)
        assert torch.allclose(layer_direct(), layer_experiment_unbunched(), atol=1e-6)
        assert (
            layer_direct.output_keys
            == layer_experiment.output_keys
            == layer_experiment_unbunched.output_keys
        )

    def test_photon_loss_incomplete_noise_model(self):
        """Missing transmittance defaults to 1 while brightness controls survival and vice versa."""
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())
        base_layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 1],
            computation_space=ML.ComputationSpace.FOCK,
        )

        experiment_brightness = pcvl.Experiment(circuit)
        experiment_brightness.noise = pcvl.NoiseModel(brightness=0.4)

        loss_layer_brightness = ML.QuantumLayer(
            input_size=0,
            experiment=experiment_brightness,
            input_state=[1, 1],
            computation_space=ML.ComputationSpace.FOCK,
        )

        experiment_transmittance = pcvl.Experiment(circuit)
        experiment_transmittance.noise = pcvl.NoiseModel(transmittance=0.4)

        loss_layer_transmittance = ML.QuantumLayer(
            input_size=0,
            experiment=experiment_transmittance,
            input_state=[1, 1],
            computation_space=ML.ComputationSpace.FOCK,
        )

        keys = base_layer.output_keys
        keys_brightness = loss_layer_brightness.output_keys
        keys_transmittance = loss_layer_transmittance.output_keys

        assert all(sum(key) == 2 for key in keys)
        assert any(sum(key_b) < sum(keys[0]) for key_b in keys_brightness)
        assert any(sum(key_t) < sum(keys[0]) for key_t in keys_transmittance)
        assert set(keys_brightness) == set(keys_transmittance)

        assert torch.allclose(
            loss_layer_brightness().sum(dim=1),
            torch.ones_like(loss_layer_brightness()[:, 0]),
            atol=1e-6,
        )
        assert torch.allclose(
            loss_layer_transmittance().sum(dim=1),
            torch.ones_like(loss_layer_transmittance()[:, 0]),
            atol=1e-6,
        )

    def test_photon_loss_with_detectors(self):
        """Photon loss must compose before detector transforms."""
        circuit = pcvl.Circuit(2)
        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.threshold()
        experiment.noise = pcvl.NoiseModel(brightness=0.3, transmittance=1.0)

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0],
            computation_space=ComputationSpace.FOCK,
        )

        output = layer()
        keys = [tuple(key) for key in layer.output_keys]
        distribution = dict(zip(keys, output.squeeze(0).tolist(), strict=False))

        assert set(distribution) == {(1, 0), (0, 0), (0, 1)}
        assert torch.allclose(
            output.sum(dim=1), torch.ones_like(output[:, 0]), atol=1e-6
        )
        assert pytest.approx(distribution[(1, 0)], rel=1e-6, abs=1e-6) == 0.3
        assert pytest.approx(distribution[(0, 0)], rel=1e-6, abs=1e-6) == 0.7

        layer_4_photons = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[4, 0],
            computation_space=ComputationSpace.FOCK,
        )

        output_4 = layer_4_photons()
        keys_4 = layer_4_photons.output_keys
        prob_disapearance = 0.7**4
        assert set(keys_4) == {
            (1, 0),
            (0, 0),
            (0, 1),
            (1, 1),
        }  # Only threshold outcomes possible
        output_4 = output_4.squeeze(0)
        print(output_4.shape)
        assert output_4[keys_4.index((0, 0))] == torch.tensor(prob_disapearance)
        assert output_4[keys_4.index((1, 0))] == torch.tensor(1 - prob_disapearance)
        assert output_4[keys_4.index((0, 1))] == torch.tensor(0.0)
        assert output_4[keys_4.index((1, 1))] == torch.tensor(0.0)

    def test_photon_loss_adjusts_output_size(self):
        """Photon loss must enlarge the classical basis when losses are possible."""
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())
        base_layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 1],
            computation_space=ML.ComputationSpace.FOCK,
        )

        experiment = pcvl.Experiment(circuit)
        experiment.noise = pcvl.NoiseModel(brightness=0.8, transmittance=1.0)
        loss_layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1],
            computation_space=ML.ComputationSpace.FOCK,
        )

        base_output = base_layer()
        loss_output = loss_layer()

        assert torch.count_nonzero(base_output) < torch.count_nonzero(loss_output)
        assert len(loss_layer.output_keys) == loss_layer.output_size

        layer_unbunched = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 1],
        )
        loss_layer_unbunched = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1],
        )

        assert loss_layer_unbunched.output_size > layer_unbunched.output_size
        assert len(loss_layer_unbunched.output_keys) == loss_layer_unbunched.output_size

    def test_detector_autograd_compatibility(self):
        """Photon loss transforms must preserve autograd support."""
        circuit = pcvl.Circuit(2)
        theta = pcvl.P("phi")
        circuit.add(0, pcvl.PS(theta))
        circuit.add((0, 1), pcvl.BS())

        experiment = pcvl.Experiment(circuit)
        experiment.noise = pcvl.NoiseModel(brightness=0.9, transmittance=0.85)

        # Layer with input parameters
        layer = ML.QuantumLayer(
            input_size=1,
            experiment=experiment,
            input_state=[1, 1],
            input_parameters=["phi"],
            computation_space=ML.ComputationSpace.FOCK,
        )

        x = torch.rand(3, 1, requires_grad=True)
        layer.train()
        probabilities = layer(x)
        probabilities = probabilities
        target = torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])

        cel = torch.nn.CrossEntropyLoss()
        loss = cel(probabilities, target)
        loss.backward()

        for param in layer.parameters():
            assert param.grad is not None
            assert not torch.isclose(
                param.grad, torch.zeros_like(param.grad), atol=1e-6
            ).all()

        # Layer with trainable parameters
        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1],
            trainable_parameters=["phi"],
            computation_space=ML.ComputationSpace.FOCK,
        )

        # Check that it has trainable parameters
        trainable_params = [p for p in layer.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "Layer should have trainable parameters"

        # Test forward pass (no input needed)
        output = layer()
        assert output.shape == (1, len(layer.output_keys))
        assert torch.all(torch.isfinite(output))

        # Test gradient computation
        loss = output.sum()
        loss.backward()

        # Check that trainable parameters have gradients
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_simple_experiment_layer_photon_loss_vs_perceval(self):
        """Layer outputs must match manual photon-loss transformation for simple circuits."""
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())

        experiment = pcvl.Experiment(circuit)
        experiment.noise = pcvl.NoiseModel(brightness=0.75, transmittance=0.9)

        input_state = [1, 1]

        loss_layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=input_state,
            computation_space=ML.ComputationSpace.FOCK,
        )

        output = loss_layer()
        keys = loss_layer.output_keys

        # Perceval version
        processor = pcvl.Processor("SLOS", experiment)
        processor.with_input(pcvl.BasicState(input_state))
        processor.min_detected_photons_filter(0)

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

    def test_complex_experiment_layer_photon_loss_vs_perceval(self):
        """Photon loss must match manual transform on multi-photon, multi-mode circuits."""
        circuit = pcvl.Circuit(3)
        circuit.add((0, 1), pcvl.BS())
        circuit.add(0, pcvl.PS(math.pi / 4))
        circuit.add((1, 2), pcvl.BS())

        input_state = [2, 1, 0]

        experiment = pcvl.Experiment(circuit)
        experiment.noise = pcvl.NoiseModel(brightness=0.9, transmittance=1.0)
        experiment.detectors[1] = pcvl.Detector.threshold()

        loss_layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=input_state,
            computation_space=ML.ComputationSpace.FOCK,
        )

        output = loss_layer()
        keys = loss_layer.output_keys

        # Perceval version
        processor = pcvl.Processor("SLOS", experiment)
        processor.with_input(pcvl.BasicState(input_state))
        processor.min_detected_photons_filter(0)

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


class TestPhotonLossWithFidelityKernel:
    """Test suite covering photon loss integration within FidelityKernel."""

    def test_kernel_reflects_photon_survival(self):
        """Kernel value must drop according to the survival probability."""
        circuit = pcvl.Circuit(1)
        circuit.add(0, pcvl.PS(pcvl.P("x")))
        experiment = pcvl.Experiment(circuit)
        experiment.noise = pcvl.NoiseModel(brightness=0.8, transmittance=0.9)

        feature_map = ML.FeatureMap(
            experiment=experiment,
            input_size=1,
            input_parameters="x",
        )
        kernel = ML.FidelityKernel(
            feature_map=feature_map,
            input_state=[1],
            computation_space=ComputationSpace.FOCK,
        )

        x = torch.tensor([0.0])
        value = kernel(x, x)
        # kernel_matrix = [0.72, 1 - 0.72] where 0.72 = (0.8 * 0.9)
        # and value = kernel_matrix * kernel_matrix^T
        expected = (0.72) ** 2 + (1 - (0.72)) ** 2
        assert pytest.approx(expected, rel=1e-6, abs=1e-6) == value
        assert kernel.has_custom_noise_model

    def test_kernel_matches_noise_free_case(self):
        """Removing the noise model restores unit kernel values."""
        circuit = pcvl.Circuit(1)
        circuit.add(0, pcvl.PS(pcvl.P("x")))

        feature_map = ML.FeatureMap(
            circuit=circuit,
            input_size=1,
            input_parameters="x",
        )
        kernel = ML.FidelityKernel(
            feature_map=feature_map,
            input_state=[1],
            computation_space=ComputationSpace.FOCK,
        )

        x = torch.tensor([0.0])
        value = kernel(x, x)
        assert pytest.approx(1.0, rel=1e-6, abs=1e-6) == value
        assert not kernel.has_custom_noise_model

        experiment = pcvl.Experiment(circuit)
        experiment.noise = pcvl.NoiseModel(brightness=1.0, transmittance=1.0)

        feature_map_noiseless = ML.FeatureMap(
            experiment=experiment,
            input_size=1,
            input_parameters="x",
        )
        kernel_noiseless = ML.FidelityKernel(
            feature_map=feature_map_noiseless,
            input_state=[1],
            computation_space=ComputationSpace.FOCK,
        )

        x = torch.tensor([0.0])
        value_noiseless = kernel_noiseless(x, x)

        assert pytest.approx(value, rel=1e-6, abs=1e-6) == value_noiseless
        assert kernel_noiseless.has_custom_noise_model

    def test_fidelity_kernel_photon_loss_choice_adjusts_key_space(self):
        """Noise model selection should change the size of the kernel detection basis."""

        feature_map = ML.FeatureMap.simple(
            input_size=1,
            n_modes=3,
        )
        input_state = [1, 1, 1]

        kernel = ML.FidelityKernel(
            feature_map=feature_map,
            input_state=input_state,
        )

        feature_map_noise = ML.FeatureMap.simple(
            input_size=1,
            n_modes=3,
        )
        experiment_noise = feature_map_noise.experiment
        experiment_noise.noise = pcvl.NoiseModel(brightness=0.8, transmittance=1.0)

        kernel_noise = ML.FidelityKernel(
            feature_map=feature_map_noise,
            input_state=input_state,
        )

        keys = kernel._detector_transform.output_keys
        keys_noise = kernel_noise._detector_transform.output_keys

        assert kernel._detector_transform.output_size == len(keys)
        assert kernel_noise._detector_transform.output_size == len(keys_noise)
        assert len(keys) < len(keys_noise)
        assert all(sum(key) == sum(input_state) for key in keys)
        assert any(sum(key) < sum(input_state) for key in keys_noise)
        assert all(sum(key) <= sum(input_state) for key in keys_noise)

    def test_fidelity_kernel_respects_feature_map_experiment(self):
        """FidelityKernel should inherit noise model and detector configuration provided via FeatureMap."""

        circuit = pcvl.Circuit(2)
        circuit.add(0, pcvl.PS(pcvl.P("px")))
        circuit.add((0, 1), pcvl.BS())

        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.pnr()
        experiment.noise = pcvl.NoiseModel(brightness=0.85, transmittance=1.0)

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
