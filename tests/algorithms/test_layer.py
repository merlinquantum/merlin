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

import math

import numpy as np
import perceval as pcvl
import pytest
import torch
import torch.nn as nn
from perceval import FFCircuitProvider

import merlin as ML
from merlin.core.computation_space import ComputationSpace
from merlin.core.partial_measurement import (
    PartialMeasurement,
)
from merlin.core.probability_distribution import ProbabilityDistribution
from merlin.core.state_vector import StateVector


class TestQuantumLayer:
    """Test suite for QuantumLayer."""

    def test_experiment_unitary_initialization(self):
        """QuantumLayer should accept a unitary experiment."""

        circuit = pcvl.Circuit(1)
        experiment = pcvl.Experiment(circuit)

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1],
        )

        output = layer()
        assert torch.allclose(
            output.sum(), torch.tensor(1.0, dtype=output.dtype), atol=1e-6
        )

    def test_experiment_non_unitary_rejected(self):
        """A non-unitary experiment should be rejected."""

        circuit = pcvl.Circuit(1)
        experiment = pcvl.Experiment(circuit)
        experiment.add(0, pcvl.TD(1))
        assert experiment.is_unitary is False

        with pytest.raises(ValueError, match="must be unitary"):
            ML.QuantumLayer(
                input_size=0,
                experiment=experiment,
                input_state=[1],
            )

    def test_experiment_min_photons_filter_error(self):
        """A min_photons_filter configured on the experiment should raise an error (unsupported)."""

        circuit = pcvl.Circuit(1)
        experiment = pcvl.Experiment(circuit)
        experiment.min_detected_photons_filter(1)

        with pytest.raises(ValueError):
            ML.QuantumLayer(
                input_size=0,
                experiment=experiment,
                input_state=[1],
            )

    def test_experiment_sequence_collapses_to_single_unitary(self):
        """Experiments composed of multiple unitary components should collapse to a single circuit."""

        experiment = pcvl.Experiment()
        experiment.add(0, pcvl.BS())
        experiment.add(0, pcvl.PS(pcvl.P("phi1")))
        experiment.add(0, pcvl.BS())
        experiment.add(0, pcvl.PS(pcvl.P("phi2")))

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0],
            trainable_parameters=["phi"],
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        expected = pcvl.Circuit(2)
        expected.add(0, pcvl.BS())
        expected.add(0, pcvl.PS(pcvl.P("phi1")))
        expected.add(0, pcvl.BS())
        expected.add(0, pcvl.PS(pcvl.P("phi2")))

        for pname, val in {"phi1": 0.1, "phi2": 0.2}.items():
            layer.circuit.param(pname).set_value(val)
            expected.param(pname).set_value(val)

        combined = np.array(layer.circuit.compute_unitary(), dtype=np.complex128)
        target = np.array(expected.compute_unitary(), dtype=np.complex128)
        assert np.allclose(combined, target, atol=1e-6)

    def test_experiment_with_feedforward_not_supported(self):
        """Experiments containing feed-forward components should be rejected."""

        experiment = pcvl.Experiment()
        experiment.add(0, pcvl.BS())
        experiment.add(0, pcvl.Detector.pnr())
        ff = FFCircuitProvider(1, 0, pcvl.Circuit(1))
        experiment.add(0, ff)

        with pytest.raises(
            ValueError,
            match="Feed-forward components are not supported inside a QuantumLayer experiment",
        ):
            ML.QuantumLayer(
                input_size=0,
                experiment=experiment,
                input_state=[1, 0],
                measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
            )

    def test_amplitude_encoding_rejects_input_size(self):
        """Amplitude encoding forbids explicit input_size."""
        circuit = pcvl.Circuit(2)

        with pytest.raises(ValueError, match="amplitude_encoding"):
            ML.QuantumLayer(
                input_size=2,
                circuit=circuit,
                amplitude_encoding=True,
                n_photons=1,
            )

    def test_amplitude_encoding_requires_n_photons(self):
        """Amplitude encoding requires n_photons."""
        circuit = pcvl.Circuit(2)

        with pytest.raises(ValueError, match="n_photons"):
            ML.QuantumLayer(
                input_size=None,
                circuit=circuit,
                amplitude_encoding=True,
            )

    def test_amplitude_encoding_rejects_input_parameters(self):
        """Amplitude encoding cannot be combined with classical input parameters."""
        circuit = pcvl.Circuit(2)

        with pytest.raises(ValueError, match="input parameters"):
            ML.QuantumLayer(
                circuit=circuit,
                amplitude_encoding=True,
                n_photons=1,
                input_parameters=["x"],
            )

    def test_experiment_input_state_overrides_warns(self):
        """Experiment input_state should override user input_state with a warning."""
        circuit = pcvl.Circuit(2)
        experiment = pcvl.Experiment(circuit)
        experiment.with_input(pcvl.BasicState([1, 0]))

        with pytest.warns(UserWarning, match="experiment.input_state"):
            layer = ML.QuantumLayer(
                input_size=0,
                experiment=experiment,
                input_state=[0, 1],
            )

        assert layer.input_state == [1, 0]

    def test_statevector_empty_rejected(self):
        """Empty StateVector inputs should be rejected."""
        circuit = pcvl.Circuit(2)
        empty_state = pcvl.StateVector()

        with pytest.raises(ValueError, match="StateVector cannot be empty"):
            ML.QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=empty_state,
            )

    def test_amplitudes_reject_custom_detectors(self):
        """Amplitude readout is incompatible with custom detectors."""
        circuit = pcvl.Circuit(2)
        experiment = pcvl.Experiment(circuit)
        experiment._add_detector(mode=0, detector=pcvl.Detector.threshold())

        with pytest.raises(
            RuntimeError, match="does not support experiments with detectors"
        ):
            ML.QuantumLayer(
                input_size=0,
                experiment=experiment,
                input_state=[1, 0],
                computation_space=ML.ComputationSpace.FOCK,
                measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
            )

    def test_builder_based_layer_creation(self):
        """Test creating a layer from an builder."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1, 3], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=3,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        assert layer.input_size == 3
        assert layer.thetas[0].shape[0] == 2 * 4 * (
            4 - 1
        )  # 24 trainable parameters from U1 and U2

    @pytest.mark.parametrize("names", [("input", "input"), ("input_a", "input_b")])
    def test_multiple_angle_encodings_validate_input_size(self, names):
        builder = ML.CircuitBuilder(n_modes=5)
        builder.add_angle_encoding(modes=[0, 1], name=names[0])
        builder.add_angle_encoding(modes=[2, 3, 4], name=names[1])

        layer = ML.QuantumLayer(
            input_size=5,
            input_state=[1, 0, 0, 0, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        pcvl.pdisplay(layer.circuit, output_format=pcvl.Format.TEXT)

        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        dummy_input = torch.rand(1, 5)
        output = model(dummy_input)
        assert output.shape == (1, 3), "Output shape mismatch"
        assert layer.input_size == 5, "Input size should match number of encoded modes"
        assert not torch.isnan(output).any(), "Output should not contain NaNs"

    def test_forward_pass_batched(self):
        """Test forward pass with batched input."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        # Test with batch
        x = torch.rand(10, 2)
        output = model(x)

        assert output.shape == (10, 3)
        assert torch.all(output >= -1e6)  # More reasonable bounds for quantum outputs

    def test_prepare_amplitude_input_updates_state_and_splits_inputs(self):
        """Amplitude input helper should capture state and return remaining inputs."""
        circuit = pcvl.Circuit(2)
        layer = ML.QuantumLayer(
            circuit=circuit,
            n_photons=1,
            amplitude_encoding=True,
            measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
            trainable_parameters=[],
            input_parameters=[],
        )
        # TODO: will need to be updated to StateVector when implemented
        original_state = torch.tensor([0.0])
        layer.computation_process.input_state = original_state

        amplitude = torch.rand(len(layer.output_keys))
        remaining_input = torch.rand(2)
        amplitude_out, remaining, saved_state = layer._prepare_amplitude_input([
            amplitude,
            remaining_input,
        ])

        assert saved_state is original_state
        assert remaining[0] is remaining_input
        assert torch.allclose(amplitude_out, amplitude)
        assert torch.allclose(layer.computation_process.input_state, original_state)

        with layer._temporary_input_state(amplitude_out, saved_state):
            assert torch.allclose(layer.computation_process.input_state, amplitude_out)
        assert torch.allclose(layer.computation_process.input_state, original_state)

    def test_prepare_classical_parameters_detects_batch_mismatch(self):
        """Classical parameter helper should reject mismatched batch sizes."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_angle_encoding(modes=[0, 1], name="input_a")
        builder.add_angle_encoding(modes=[2, 3], name="input_b")

        layer = ML.QuantumLayer(
            input_size=4,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        with pytest.raises(ValueError, match="Inconsistent batch dimensions"):
            layer._prepare_classical_parameters([torch.rand(2, 2), torch.rand(3, 2)])

    def test_prepare_classical_parameters_reports_batch_dim(self):
        """Classical parameter helper should report batch size when consistent."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input_a")
        builder.add_angle_encoding(modes=[2, 3], name="input_b")

        layer = ML.QuantumLayer(
            input_size=4,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        params, batch_dim = layer._prepare_classical_parameters([
            torch.rand(2, 2),
            torch.rand(2, 2),
        ])

        assert batch_dim == 2
        assert len(params) >= 2

    def test_amplitude_encoding_rejects_classical_input_parameters(self):
        """Amplitude encoding should not allow classical input parameters."""
        # TODO: to remove when dual encoding will be implemented (>0.4.x)
        circuit = pcvl.Circuit(2)
        with pytest.raises(
            ValueError,
            match="Amplitude encoding cannot be combined with classical input parameters.",
        ):
            ML.QuantumLayer(
                circuit=circuit,
                n_photons=1,
                amplitude_encoding=True,
                input_parameters=["px"],
                trainable_parameters=[],
                measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
            )

    def test_amplitude_encoding_requires_amplitude_input(self):
        """Amplitude encoding should require an amplitude tensor at call time."""
        circuit = pcvl.Circuit(2)
        layer = ML.QuantumLayer(
            circuit=circuit,
            n_photons=1,
            amplitude_encoding=True,
            measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
            trainable_parameters=[],
            input_parameters=[],
        )

        with pytest.raises(ValueError, match="expects an amplitude tensor input"):
            layer()

    def test_multiple_classical_inputs_forward(self):
        """Classical encoding should accept one tensor per input prefix."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input_a")
        builder.add_angle_encoding(modes=[2, 3], name="input_b")

        layer = ML.QuantumLayer(
            input_size=4,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        input_a = torch.rand(2, 2)
        input_b = torch.rand(2, 2)
        output = layer(input_a, input_b)
        assert output.shape == (2, layer.output_size)

        prefixes = list(layer.computation_process.input_parameters)
        assert prefixes == ["input_a", "input_b"]
        params = layer.prepare_parameters([input_a, input_b])
        encoded_a = layer._prepare_input_encoding(input_a, prefixes[0])
        encoded_b = layer._prepare_input_encoding(input_b, prefixes[1])
        assert torch.allclose(params[-2], encoded_a)
        assert torch.allclose(params[-1], encoded_b)

    def test_builder_infers_input_size_for_backward_compat(self):
        """Builder-based layers should infer input_size when omitted."""
        builder = ML.CircuitBuilder(n_modes=3)
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U1")

        layer = ML.QuantumLayer(
            builder=builder,
            input_state=[1, 0, 0],
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        assert layer.input_size == 2
        output = layer(torch.rand(1, 2))
        assert output.shape == (1, layer.output_size)

    def test_renormalize_distribution_and_amplitudes_applies_for_unbunched(self):
        """UNBUNCHED computation space should renormalize amplitudes and distribution."""
        circuit = pcvl.Circuit(2)
        layer = ML.QuantumLayer(
            circuit=circuit,
            input_state=[1, 0],
            computation_space=ML.ComputationSpace.UNBUNCHED,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        amplitudes = torch.tensor([[2.0 + 0j, 0.0 + 0j]], dtype=torch.cfloat)
        distribution, normalized = layer._renormalize_distribution_and_amplitudes(
            amplitudes
        )

        assert torch.allclose(distribution, torch.tensor([[1.0, 0.0]]))
        assert torch.allclose(
            normalized, torch.tensor([[1.0 + 0j, 0.0 + 0j]], dtype=torch.cfloat)
        )

    def test_renormalize_distribution_and_amplitudes_skips_for_fock(self):
        """FOCK computation space should not renormalize amplitudes."""
        circuit = pcvl.Circuit(2)
        layer = ML.QuantumLayer(
            circuit=circuit,
            input_state=[1, 0],
            computation_space=ML.ComputationSpace.FOCK,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        amplitudes = torch.tensor([[2.0 + 0j, 0.0 + 0j]], dtype=torch.cfloat)
        distribution, normalized = layer._renormalize_distribution_and_amplitudes(
            amplitudes
        )

        assert torch.allclose(distribution, torch.tensor([[4.0, 0.0]]))
        assert torch.allclose(normalized, amplitudes)

    def test_forward_pass_single(self):
        """Test forward pass with single input."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 0, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        # Test with single sample
        x = torch.rand(1, 2)
        output = model(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 3

    def test_default_input_state_even_distribution(self):
        """Omitted input_state should evenly distribute photons across modes."""
        circuit = pcvl.Circuit(5)

        layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            n_photons=2,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        expected_state = ML.StateGenerator.generate_state(
            circuit.m, 2, ML.StatePattern.SPACED
        )
        assert layer.input_state == expected_state

    def test_gradient_computation(self):
        """Test that gradients flow through the layer."""

        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 1, 0, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        x = torch.rand(5, 2, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None

        # Check that layer parameters have gradients
        has_trainable_params = False
        for param in model.parameters():
            if param.requires_grad:
                has_trainable_params = True
                assert param.grad is not None

        assert has_trainable_params, "Model should have trainable parameters"

    def test_sampling_configuration(self):
        """Sampling is configured per-call via forward(); training disables it automatically."""
        # Build a tiny circuit
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        # Compose with a linear head (as in the old test)
        _ = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        # There is no layer-level sampling state anymore
        assert not hasattr(layer, "shots")
        assert not hasattr(layer, "sampling_method")

        # Prepare a batch of inputs (B, features)
        x = torch.rand(4, 2)

        # ---------- EVAL: no sampling (default) ----------
        layer.eval()
        y_no_sampling = layer(x)  # shots defaults to None/0 -> no sampling path
        assert isinstance(y_no_sampling, torch.Tensor)
        assert y_no_sampling.shape[0] == x.shape[0]

        # ---------- EVAL: enable sampling by passing shots ----------
        y_sampled = layer(x, shots=100, sampling_method="multinomial")
        assert isinstance(y_sampled, torch.Tensor)
        assert y_sampled.shape[0] == x.shape[0]

        # ---------- TRAIN: sampling request is overridden (no sampling during training) ----------
        layer.train()
        # Request sampling, but autodiff backend should turn it off for differentiability
        with pytest.warns():
            y_train = layer(x, shots=100, sampling_method="multinomial")
            loss = y_train.sum()
            loss.backward()  # should succeed with gradients flowing (no sampling taken)
            # At least one trainable parameter should have a gradient
            assert any(
                p.grad is not None for p in layer.parameters() if p.requires_grad
            )

        # ---------- Invalid sampling method should error ----------
        with pytest.raises(ValueError):
            _ = layer(x, shots=10, sampling_method="invalid")

    def test_simple_wrapper_forwards_sampling_args(self):
        """The .simple() wrapper should accept shots/sampling_method and forward them to the quantum layer."""
        model = ML.QuantumLayer.simple(input_size=2)
        x = torch.rand(3, 2)

        # Works without sampling
        y = model(x)
        assert y.shape[0] == x.shape[0]

        # Works with sampling (multinomial default in the wrapper)
        model.eval()
        y2 = model(x, shots=50)
        assert y2.shape[0] == x.shape[0]

    def test_reservoir_mode(self):
        """Test reservoir computing mode."""
        # Test normal mode first
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer_normal = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        model_normal = torch.nn.Sequential(
            layer_normal, torch.nn.Linear(layer_normal.output_size, 3)
        )

        layer_reservoir = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        model_reservoir = torch.nn.Sequential(
            layer_reservoir, torch.nn.Linear(layer_reservoir.output_size, 3)
        )

        model_reservoir.requires_grad_(False)
        assert any(p.requires_grad for p in model_normal.parameters())
        assert all(not p.requires_grad for p in model_reservoir.parameters())

        normal_trainable = sum(
            p.numel() for p in model_normal.parameters() if p.requires_grad
        )

        reservoir_trainable = sum(
            p.numel() for p in model_reservoir.parameters() if p.requires_grad
        )

        # Reservoir mode should freeze all parameters while keeping the normal layer trainable.
        assert normal_trainable > 0
        assert reservoir_trainable == 0

        # Test that reservoir layer still works
        x = torch.rand(3, 2)
        output = model_reservoir(x)
        assert output.shape == (3, 3)

    def test_measurement_strategies(self):
        """Test different measurement strategies and grouping policies."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        configs = [
            {
                "measurement_strategy": ML.MeasurementStrategy.PROBABILITIES,
                "grouping_policy": None,
            },
            {
                "measurement_strategy": ML.MeasurementStrategy.PROBABILITIES,
                "grouping_policy": ML.LexGrouping,
            },
            {
                "measurement_strategy": ML.MeasurementStrategy.PROBABILITIES,
                "grouping_policy": ML.ModGrouping,
            },
        ]

        for cfg in configs:
            if cfg["grouping_policy"] is None:
                layer = ML.QuantumLayer(
                    input_size=2,
                    input_state=[1, 0, 1, 0],
                    builder=builder,
                    measurement_strategy=cfg["measurement_strategy"],
                )

                model = torch.nn.Sequential(
                    layer, torch.nn.Linear(layer.output_size, 4)
                )

                x = torch.rand(3, 2)
                output = model(x)
                assert output.shape == (3, 4)
                assert torch.all(torch.isfinite(output))

            else:
                layer = ML.QuantumLayer(
                    input_size=2,
                    input_state=[1, 0, 1, 0],
                    builder=builder,
                    measurement_strategy=cfg["measurement_strategy"],
                )

                model = torch.nn.Sequential(
                    layer, cfg["grouping_policy"](layer.output_size, 4)
                )

                x = torch.rand(3, 2)
                output = model(x)
                assert output.shape == (3, 4)
                assert torch.all(torch.isfinite(output))

    def test_probabilities_grouping_return_object(self):
        """Grouped probabilities with return_object should yield ProbabilityDistribution of grouped size."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(
                ComputationSpace.UNBUNCHED, grouping=ML.ModGrouping(6, 4)
            ),
            return_object=True,
        )
        assert layer.output_size == 6
        x = torch.rand(3, 2)
        output = layer(x)
        assert isinstance(output, ProbabilityDistribution)
        assert output.tensor.shape == (3, 4)

    def test_string_representation(self):
        """Test string representation of the layer."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1, 2], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=3,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        layer_str = str(layer)
        print(f"Layer string representation:\n{layer_str}")
        assert "QuantumLayer" in layer_str
        assert "modes=4" in layer_str
        assert "input_size=3" in layer_str

    def test_invalid_configurations(self):
        """Test that invalid configurations raise appropriate errors."""
        # this tests include builder, simple and circuit-based API
        with pytest.raises(
            ValueError,
            match="Provide exactly one of 'circuit', 'builder', or 'experiment'.",
        ):
            ML.QuantumLayer(input_size=3)

        # Test invalid experiment configuration
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        # Input size mismatch between declaration and builder-produced features
        with pytest.raises(
            ValueError,
            match="Input size \\(3\\) must equal the number of encoded input features generated by the circuit \\(2\\)\\.",
        ):
            ML.QuantumLayer(
                input_size=3,
                input_state=[1, 0, 1, 0],
                builder=builder,
                measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
            )

        with pytest.raises(ValueError):
            ML.QuantumLayer(
                input_size=2,
                n_photons=5,  # more photons than modes
                builder=builder,
                measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
            )

        with pytest.raises(ValueError):
            ML.QuantumLayer.simple(input_size=21)

    def test_subset_combinations_respected(self):
        """Ensure subset combinations expose more parameters without breaking input size checks."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(
            modes=[0, 1, 2], name="input", subset_combinations=True
        )
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=3,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        assert layer.input_size == 3

    def test_none_output_mapping_with_correct_size(self):
        """Test NONE output mapping with correct size matching."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        temp_layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
        )

        # Get actual distribution size
        dummy_input = torch.rand(1, 2)
        with torch.no_grad():
            _temp_output = temp_layer(dummy_input)

        # Now create NONE strategy with correct size
        layer_none = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
        )

        x = torch.rand(2, 2)
        output = layer_none(x)

        # Output should be amplitudes
        assert torch.allclose(
            torch.sum(output.abs() ** 2, dim=1), torch.ones(2), atol=1e-6
        )
        assert output.shape[0] == 2

    def test_simple_perceval_circuit_no_input(self):
        """Test QuantumLayer with simple perceval circuit and no input parameters."""
        # Create a simple perceval circuit with no input parameters
        circuit = pcvl.Circuit(3)  # 3 modes
        circuit.add(0, pcvl.BS())  # Beam splitter on modes 0,1
        circuit.add(
            0, pcvl.PS(pcvl.P("phi1"))
        )  # Phase shifter with trainable parameter
        circuit.add(1, pcvl.BS())  # Beam splitter on modes 1,2
        circuit.add(1, pcvl.PS(pcvl.P("phi2")))  # Another phase shifter

        # Define input state (where photons are placed)
        input_state = pcvl.BasicState([1, 0, 0])  # 1 photon in first mode

        # Create QuantumLayer with custom circuit
        layer = ML.QuantumLayer(
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=["phi"],  # Parameters to train (by prefix)
            input_parameters=[],  # No input parameters
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        output_size = math.comb(3, sum(input_state))  # Calculate output size
        with pytest.raises(
            ValueError,
            match="Input size \\(2\\) must equal the number of input parameters generated by the circuit \\(0\\)\\.",
        ):
            layer = ML.QuantumLayer(
                input_size=2,  # input_size > nb of input_parameters
                circuit=circuit,
                input_state=input_state,
                trainable_parameters=["phi"],  # Parameters to train (by prefix)
                input_parameters=None,  # No input parameters
                measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
            )

        # Test layer properties
        assert layer.input_size == 0
        assert layer.output_size == output_size
        # Check that it has trainable parameters
        trainable_params = [p for p in layer.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "Layer should have trainable parameters"

        # Test forward pass (no input needed)
        output = layer()
        assert output.shape == (1, 3)
        assert torch.all(torch.isfinite(output))

        # Test gradient computation
        loss = output.sum()
        loss.backward()

        # Check that trainable parameters have gradients
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_simple_perceval_circuit_no_trainable_parameter(self):
        """Test QuantumLayer with simple perceval circuit and no trainable parameters."""
        # Create a simple perceval circuit with no input parameters
        circuit = pcvl.Circuit(3)  # 3 modes
        circuit.add(0, pcvl.BS())  # Beam splitter on modes 0,1
        circuit.add(
            0, pcvl.PS(pcvl.P("phi1"))
        )  # Phase shifter with trainable parameter
        circuit.add(1, pcvl.BS())  # Beam splitter on modes 1,2
        circuit.add(1, pcvl.PS(pcvl.P("phi2")))  # Another phase shifter

        # Define input state (where photons are placed)
        input_state = [1, 0, 0]  # 1 photon in first mode

        # Create QuantumLayer with custom circuit
        layer = ML.QuantumLayer(
            input_size=2,  # 2 input parameters
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=[],  # No trainable parameters
            input_parameters=["phi"],  # No input parameters
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        dummy_input = torch.rand(1, 2)

        math.comb(3, sum(input_state))  # Calculate output size
        # Test layer properties
        assert layer.input_size == 2
        assert model[1].out_features == 3
        # Check that it has trainable parameters (only in Linear layer)
        trainable_params_layer = [p for p in layer.parameters() if p.requires_grad]
        assert len(trainable_params_layer) == 0, (
            "Layer should have no trainable parameters"
        )
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "Model should have trainable parameters"

        # Test forward pass (no input needed)
        output = model(dummy_input)
        assert output.shape == (1, 3)
        assert torch.all(torch.isfinite(output))

        # Test gradient computation
        loss = output.sum()
        loss.backward()

        # Check that trainable parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    @pytest.mark.parametrize(
        ("computation_space"),
        [
            ML.ComputationSpace.UNBUNCHED,
            ML.ComputationSpace.DUAL_RAIL,
            ML.ComputationSpace.FOCK,
        ],
    )
    def test_computation_space_normalized_output(self, computation_space):
        """Test QuantumLayer with simple perceval circuit and no trainable parameters."""
        # Create a simple perceval circuit with no input parameters
        m = 8
        n = 4
        batch_size = 5
        circuit = pcvl.Circuit(m)
        circuit.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)))
        for i in range(m):
            circuit.add(i, pcvl.PS(pcvl.P(f"phi{i}")))
        circuit.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)))

        layer = ML.QuantumLayer(
            input_size=m,
            n_photons=n,
            circuit=circuit,
            input_parameters=["phi"],  # No input parameters
            measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
            computation_space=computation_space,
        )

        o = layer.forward(torch.rand(batch_size, m))

        assert torch.allclose(torch.sum(o.abs() ** 2, dim=1), torch.ones(batch_size))

    def test_basicstate_input(self):
        bs1 = pcvl.BasicState("|1,0,1>")
        ML.QuantumLayer(
            circuit=pcvl.Circuit(bs1.m),
            computation_space=ML.ComputationSpace.FOCK,
            input_state=bs1,
        )
        # An annotated BasicState should raise as annotations are not supported
        bs_annot = pcvl.BasicState("|{a:0},0,1>")
        with pytest.raises(
            ValueError, match="BasicState with annotations is not supported"
        ):
            _ = ML.QuantumLayer(
                circuit=pcvl.Circuit(bs_annot.m),
                computation_space=ML.ComputationSpace.FOCK,
                input_state=bs_annot,
            )

    # TODO Change the default returns when the default measurement strategy will be changed. Also, uncomment the partial_measurment tests when it is ready
    def test_forward_output_objects(self):
        # MS:None, ro:false
        builder = ML.CircuitBuilder(5)
        builder.add_entangling_layer()
        qlayer = ML.QuantumLayer(
            input_size=0,
            builder=builder,
            input_state=[0, 1, 0, 1, 0],
        )
        res_no_obj = qlayer()

        assert isinstance(res_no_obj, torch.Tensor)

        # MS:None, ro:true
        qlayer.return_object = True
        res_obj = qlayer()
        assert isinstance(res_obj, ProbabilityDistribution)
        assert isinstance(res_obj.tensor, torch.Tensor)
        assert np.allclose(res_no_obj.detach().numpy(), res_obj.tensor.detach().numpy())

        # -------------------------------------------------------------------------------#

        # MS:amplitudes, ro:false
        builder = ML.CircuitBuilder(5)
        builder.add_entangling_layer()
        qlayer = ML.QuantumLayer(
            input_size=0,
            builder=builder,
            input_state=[0, 1, 0, 1, 0],
            measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
        )

        res_no_obj = qlayer()

        assert isinstance(qlayer(), torch.Tensor)

        # MS:amplitudes, ro:true
        qlayer.return_object = True
        res_obj = qlayer()

        assert isinstance(qlayer(), StateVector)
        assert isinstance(res_obj.tensor, torch.Tensor)
        assert np.allclose(res_no_obj.detach().numpy(), res_obj.tensor.detach().numpy())

        # -------------------------------------------------------------------------------#

        # MS:probs, ro:false
        builder = ML.CircuitBuilder(5)
        builder.add_entangling_layer()
        qlayer = ML.QuantumLayer(
            input_size=0,
            builder=builder,
            input_state=[0, 1, 0, 1, 0],
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        res_no_obj = qlayer()

        assert isinstance(res_no_obj, torch.Tensor)

        # MS:probs, ro:true
        qlayer.return_object = True
        res_obj = qlayer()

        assert isinstance(res_obj, ProbabilityDistribution)
        assert isinstance(res_obj.tensor, torch.Tensor)
        assert np.allclose(res_no_obj.detach().numpy(), res_obj.tensor.detach().numpy())

        # -------------------------------------------------------------------------------#

        # MS:mode_expectation, ro:false
        builder = ML.CircuitBuilder(5)
        builder.add_entangling_layer()
        qlayer = ML.QuantumLayer(
            input_size=0,
            builder=builder,
            input_state=[0, 1, 0, 1, 0],
            measurement_strategy=ML.MeasurementStrategy.MODE_EXPECTATIONS,
        )

        res_no_obj = qlayer()

        assert isinstance(res_no_obj, torch.Tensor)

        # MS:mode_expectation, ro:true
        qlayer.return_object = True
        res_obj = qlayer()

        assert isinstance(res_obj, torch.Tensor)
        assert np.allclose(res_obj.detach().numpy(), res_obj.detach().numpy())

        # -------------------------------------------------------------------------------#

        # TODO uncomment when partial is ready
        # MS:partial, ro:false
        builder = ML.CircuitBuilder(5)
        builder.add_entangling_layer()
        qlayer = ML.QuantumLayer(
            input_size=0,
            builder=builder,
            input_state=[0, 1, 0, 1, 0],
            measurement_strategy=ML.MeasurementStrategy.partial(
                modes=[0, 1],
            ),
        )

        res_no_obj = qlayer()
        assert isinstance(res_no_obj, PartialMeasurement)

        # MS:partial, ro:true
        qlayer.return_object = True

        res_obj = qlayer()
        assert isinstance(res_obj, PartialMeasurement)
        assert isinstance(res_obj.tensor, torch.Tensor)
        assert np.allclose(
            res_no_obj.tensor.detach().numpy(), res_obj.tensor.detach().numpy()
        )

    def test_forward_output_objects_new_api(self):
        builder = ML.CircuitBuilder(4)
        builder.add_entangling_layer()
        input_state = [0, 1, 0, 1]

        # PROBABILITIES, return_object=False
        qlayer = ML.QuantumLayer(
            input_size=0,
            builder=builder,
            input_state=input_state,
            measurement_strategy=ML.MeasurementStrategy.probs(
                ComputationSpace.UNBUNCHED
            ),
        )
        res_no_obj = qlayer()
        assert isinstance(res_no_obj, torch.Tensor)

        # PROBABILITIES, return_object=True
        qlayer.return_object = True
        res_obj = qlayer()
        assert isinstance(res_obj, ProbabilityDistribution)
        assert isinstance(res_obj.tensor, torch.Tensor)
        assert np.allclose(res_no_obj.detach().numpy(), res_obj.tensor.detach().numpy())

        # AMPLITUDES, return_object=False
        qlayer = ML.QuantumLayer(
            input_size=0,
            builder=builder,
            input_state=input_state,
            measurement_strategy=ML.MeasurementStrategy.amplitudes(),
        )
        res_no_obj = qlayer()
        assert isinstance(res_no_obj, torch.Tensor)

        # AMPLITUDES, return_object=True
        qlayer.return_object = True
        res_obj = qlayer()
        assert isinstance(res_obj, StateVector)
        assert isinstance(res_obj.tensor, torch.Tensor)
        assert np.allclose(res_no_obj.detach().numpy(), res_obj.tensor.detach().numpy())

        # MODE_EXPECTATIONS, return_object=False
        qlayer = ML.QuantumLayer(
            input_size=0,
            builder=builder,
            input_state=input_state,
            measurement_strategy=ML.MeasurementStrategy.mode_expectations(
                ComputationSpace.UNBUNCHED
            ),
        )
        res_no_obj = qlayer()
        assert isinstance(res_no_obj, torch.Tensor)

        # MODE_EXPECTATIONS, return_object=True (still a tensor)
        qlayer.return_object = True
        res_obj = qlayer()
        assert isinstance(res_obj, torch.Tensor)
        assert np.allclose(res_no_obj.detach().numpy(), res_obj.detach().numpy())

    def test_gradient_through_typed_objects_ProbabilityDistribution(self):
        """Test that gradients flow through the layer."""

        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        class custom_layer(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.qlayer = ML.QuantumLayer(
                    input_size=2,
                    input_state=[1, 1, 0, 0],
                    builder=builder,
                    measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
                    return_object=True,
                )
                self.clayer = torch.nn.Linear(
                    self.qlayer.output_size,
                    3,
                )

            def forward(self, x):
                output_q = self.qlayer(x)
                return self.clayer(output_q.tensor)

        model = custom_layer()
        x = torch.rand(5, 2, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None

        # Check that layer parameters have gradients
        has_trainable_params = False
        for param in model.parameters():
            if param.requires_grad:
                has_trainable_params = True
                assert param.grad is not None

        assert has_trainable_params, "Model should have trainable parameters"

    def test_gradient_through_typed_objects_StateVector(self):
        """Test that gradients flow through the layer."""

        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        class custom_layer(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.qlayer = ML.QuantumLayer(
                    input_size=2,
                    input_state=[1, 1, 0, 0],
                    builder=builder,
                    measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
                    return_object=True,
                )
                self.clayer = torch.nn.Linear(
                    self.qlayer.output_size,
                    3,
                )

            def forward(self, x):
                output_q = self.qlayer(x)
                return self.clayer(output_q.tensor.abs())

        model = custom_layer()
        x = torch.rand(5, 2, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None

        # Check that layer parameters have gradients
        has_trainable_params = False
        for param in model.parameters():
            if param.requires_grad:
                has_trainable_params = True
                assert param.grad is not None

        assert has_trainable_params, "Model should have trainable parameters"

    # TODO Define test_gradient_through_typed_objects_PartialMeasurement when MeasurementStrategy is completed.

    def test_gradient_through_typed_objects_outputs_tensor(self):
        """Test that gradients flow through the layer."""

        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        to_test = [
            [ML.MeasurementStrategy.MODE_EXPECTATIONS, False],
            [ML.MeasurementStrategy.MODE_EXPECTATIONS, True],
            [ML.MeasurementStrategy.PROBABILITIES, False],
            [ML.MeasurementStrategy.AMPLITUDES, False],
        ]
        for strategy, typed_object in to_test:

            class custom_layer(nn.Module):
                def __init__(
                    self,
                    *args,
                    _strategy=strategy,
                    _typed_object=typed_object,
                    **kwargs,
                ):
                    super().__init__(*args, **kwargs)
                    self.qlayer = ML.QuantumLayer(
                        input_size=2,
                        input_state=[1, 1, 0, 0],
                        builder=builder,
                        measurement_strategy=_strategy,
                        return_object=_typed_object,
                    )
                    self.clayer = torch.nn.Linear(
                        self.qlayer.output_size,
                        3,
                    )

                def forward(self, x):
                    output_q = self.qlayer(x)
                    return self.clayer(output_q.abs())

            model = custom_layer()
            x = torch.rand(5, 2, requires_grad=True)
            output = model(x)
            loss = output.sum()
            loss.backward()

            # Check that input gradients exist
            assert x.grad is not None

            # Check that layer parameters have gradients
            has_trainable_params = False
            for param in model.parameters():
                if param.requires_grad:
                    has_trainable_params = True
                    assert param.grad is not None

            assert has_trainable_params, "Model should have trainable parameters"


def test_simple_num_photons_modes_and_input_state():
    for i in range(1, 15):
        ql = ML.QuantumLayer.simple(input_size=i)
        if i < 2:
            assert ql.quantum_layer.n_photons == 1
            assert ql.quantum_layer.input_state == [0, 1]
        else:
            assert ql.quantum_layer.n_photons == (i) // 2
            assert np.sum(ql.quantum_layer.input_state) == (i) // 2
            assert len(ql.quantum_layer.input_state) == i

            input_state = [0] * (i)
            for j in range(len(input_state)):
                if j % 2 == 1:
                    input_state[j] = 1
            assert ql.quantum_layer.input_state == input_state


def test_simple_parameters():
    for i in range(1, 15):
        ql = ML.QuantumLayer.simple(input_size=i)
        params = list(ql.quantum_layer.parameters())
        named_params = [k[0] for k in ql.quantum_layer.named_parameters()]
        if i < 2:
            assert params[0].numel() == 2
            assert params[0].numel() == 2
            assert len(params) == 2
            assert "LI_simple" in named_params
            assert "RI_simple" in named_params
        else:
            assert params[0].numel() == i * (i - 1)
            assert params[1].numel() == i * (i - 1)
            assert len(params) == 2
            assert "LI_simple" in named_params
            assert "RI_simple" in named_params
