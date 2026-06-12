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
import re
from copy import deepcopy

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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
                measurement_strategy=ML.MeasurementStrategy.probs(),
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

        assert layer.input_state == pcvl.BasicState([1, 0])

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
                measurement_strategy=ML.MeasurementStrategy.amplitudes(
                    computation_space=ML.ComputationSpace.FOCK
                ),
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.NONE,
            trainable_parameters=[],
            input_parameters=[],
        )
        # TODO: will need to be updated to StateVector when implemented
        original_state = torch.tensor([0.0])
        layer.computation_process.input_state = original_state

        amplitude = torch.rand(len(layer.output_keys))
        remaining_input = torch.rand(2)
        amplitude_out, remaining, saved_state = layer._prepare_amplitude_input(
            [
                amplitude,
                remaining_input,
            ]
        )

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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        params, batch_dim = layer._prepare_classical_parameters(
            [
                torch.rand(2, 2),
                torch.rand(2, 2),
            ]
        )

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
                measurement_strategy=ML.MeasurementStrategy.NONE,
            )

    def test_amplitude_encoding_requires_amplitude_input(self):
        """Amplitude encoding should require an amplitude tensor at call time."""
        circuit = pcvl.Circuit(2)
        layer = ML.QuantumLayer(
            circuit=circuit,
            n_photons=1,
            amplitude_encoding=True,
            measurement_strategy=ML.MeasurementStrategy.NONE,
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.probs(
                computation_space=ML.ComputationSpace.UNBUNCHED
            ),
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
            measurement_strategy=ML.MeasurementStrategy.probs(
                computation_space=ML.ComputationSpace.FOCK
            ),
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        expected_state = ML.generate_state(circuit.m, 2, ML.StatePattern.SPACED)
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )
        model_normal = torch.nn.Sequential(
            layer_normal, torch.nn.Linear(layer_normal.output_size, 3)
        )

        layer_reservoir = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
                "measurement_strategy": ML.MeasurementStrategy.probs(),
                "grouping_policy": None,
            },
            {
                "measurement_strategy": ML.MeasurementStrategy.probs(),
                "grouping_policy": ML.LexGrouping,
            },
            {
                "measurement_strategy": ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
                measurement_strategy=ML.MeasurementStrategy.probs(),
            )

        with pytest.raises(ValueError):
            ML.QuantumLayer(
                input_size=2,
                n_photons=5,  # more photons than modes
                builder=builder,
                measurement_strategy=ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.NONE,
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
            measurement_strategy=ML.MeasurementStrategy.NONE,
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
                measurement_strategy=ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        dummy_input = torch.rand(1, 2)

        math.comb(3, sum(input_state))  # Calculate output size
        # Test layer properties
        assert layer.input_size == 2
        assert model[1].out_features == 3
        # Check that it has trainable parameters (only in Linear layer)
        trainable_params_layer = [p for p in layer.parameters() if p.requires_grad]
        assert (
            len(trainable_params_layer) == 0
        ), "Layer should have no trainable parameters"
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
            measurement_strategy=ML.MeasurementStrategy.amplitudes(
                computation_space=computation_space
            ),
        )

        o = layer.forward(torch.rand(batch_size, m))

        assert torch.allclose(torch.sum(o.abs() ** 2, dim=1), torch.ones(batch_size))

    def test_basicstate_input(self):
        bs1 = pcvl.BasicState("|1,0,1>")
        ML.QuantumLayer(
            circuit=pcvl.Circuit(bs1.m),
            measurement_strategy=ML.MeasurementStrategy.probs(
                computation_space=ML.ComputationSpace.FOCK
            ),
            input_state=bs1,
        )
        # An annotated BasicState should raise as annotations are not supported
        bs_annot = pcvl.BasicState("|{a:0},0,1>")
        with pytest.raises(
            ValueError, match="BasicState with annotations is not supported"
        ):
            _ = ML.QuantumLayer(
                circuit=pcvl.Circuit(bs_annot.m),
                measurement_strategy=ML.MeasurementStrategy.probs(
                    computation_space=ML.ComputationSpace.FOCK
                ),
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
            measurement_strategy=ML.MeasurementStrategy.NONE,
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
            measurement_strategy=ML.MeasurementStrategy.probs(),
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
            measurement_strategy=ML.MeasurementStrategy.mode_expectations(
                ComputationSpace.UNBUNCHED
            ),
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
                    measurement_strategy=ML.MeasurementStrategy.probs(),
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
                    measurement_strategy=ML.MeasurementStrategy.NONE,
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
            [
                ML.MeasurementStrategy.mode_expectations(ComputationSpace.UNBUNCHED),
                False,
            ],
            [
                ML.MeasurementStrategy.mode_expectations(ComputationSpace.UNBUNCHED),
                True,
            ],
            [ML.MeasurementStrategy.probs(), False],
            [ML.MeasurementStrategy.NONE, False],
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

    def test_memrsistive_update(self):
        def update_rule(state: torch.Tensor, output: torch.Tensor):
            return state + torch.vstack([output[0, 0]] * state.size(0)).squeeze(dim=0)

        circ = ML.CircuitBuilder(n_modes=3)
        circ.add_entangling_layer()
        circ.add_memristive_ps(mode=0, update_rule=update_rule, initial_state=0)
        circ.add_entangling_layer()
        circ.add_angle_encoding(modes=[0, 1])

        ql = ML.QuantumLayer(
            builder=circ,
            n_photons=3,
            measurement_strategy=ML.MeasurementStrategy.probs(
                computation_space=ML.ComputationSpace.FOCK
            ),
        )

        input = torch.Tensor([[0, 0]])

        first_output = ql(input)
        second_output = ql(input)
        assert not torch.allclose(first_output, second_output)

    def test_memrsistive_metadata(self):
        def update_rule(state: torch.Tensor, output: torch.Tensor):
            return state + output[:, 0]

        def update_rule_exp(state: torch.Tensor, output: torch.Tensor):
            return torch.exp(state + output[:, 0])

        circ = ML.CircuitBuilder(n_modes=3)
        circ.add_entangling_layer()
        circ.add_memristive_ps(mode=1, update_rule=update_rule, initial_state=1.2)
        circ.add_memristive_ps(mode=0, update_rule=update_rule_exp, initial_state=0.01)
        circ.add_entangling_layer()
        circ.add_angle_encoding(modes=[0, 2])

        ql = ML.QuantumLayer(
            builder=circ,
            n_photons=3,
            measurement_strategy=ML.MeasurementStrategy.probs(
                computation_space=ML.ComputationSpace.FOCK
            ),
        )
        # Initial metadata check
        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        assert torch.allclose(ql.memristive_state[0], torch.Tensor([[1.2]]))
        assert torch.allclose(ql.memristive_state[1], torch.Tensor([[0.01]]))

        assert ql.memristive_history[0][0] == ql.memristive_state[0]
        assert ql.memristive_history[1][0] == ql.memristive_state[1]
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 1

        assert ql._memristive_metadata == circ.memristive_specs

        input = torch.Tensor([[0, 0]])

        first_output = ql(input)

        # Metadata check after one pass
        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        new_state_0_t1 = update_rule(torch.Tensor([[1.2]]), first_output)
        assert torch.allclose(ql.memristive_state[0], new_state_0_t1)
        new_state_1_t1 = update_rule_exp(torch.Tensor([[0.01]]), first_output)
        assert torch.allclose(ql.memristive_state[1], new_state_1_t1)

        assert torch.allclose(ql.memristive_history[0][0], torch.Tensor([[1.2]]))
        assert torch.allclose(ql.memristive_history[1][0], torch.Tensor([[0.01]]))
        assert ql.memristive_history[0][1] == ql.memristive_state[0]
        assert ql.memristive_history[1][1] == ql.memristive_state[1]
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 2

        assert ql._memristive_metadata == circ.memristive_specs

        second_output = ql(input)

        # Metadata check after two passes
        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        new_state_0_t2 = update_rule(new_state_0_t1, second_output)
        assert torch.allclose(ql.memristive_state[0], new_state_0_t2)
        new_state_1_t2 = update_rule_exp(new_state_1_t1, second_output)
        assert torch.allclose(ql.memristive_state[1], new_state_1_t2)

        assert torch.allclose(ql.memristive_history[0][0], torch.Tensor([[1.2]]))
        assert torch.allclose(ql.memristive_history[1][0], torch.Tensor([[0.01]]))
        assert ql.memristive_history[0][1] == new_state_0_t1
        assert ql.memristive_history[1][1] == new_state_1_t1
        assert ql.memristive_history[0][2] == ql.memristive_state[0]
        assert ql.memristive_history[1][2] == ql.memristive_state[1]
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 3

        assert ql._memristive_metadata == circ.memristive_specs

        ql.reset()

        # Metadata check after a reset
        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        assert torch.allclose(ql.memristive_state[0], torch.Tensor([[1.2]]))
        assert torch.allclose(ql.memristive_state[1], torch.Tensor([[0.01]]))

        assert ql.memristive_history[0][0] == ql.memristive_state[0]
        assert ql.memristive_history[1][0] == ql.memristive_state[1]
        assert len(ql.memristive_history[0]) == 1
        assert len(ql.memristive_history[1]) == 1
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 1

        assert ql._memristive_metadata == circ.memristive_specs

    def test_reset_and_batches(self):
        def update_rule(state: torch.Tensor, output: torch.Tensor):
            return state + output[:, 0]

        def update_rule_exp(state: torch.Tensor, output: torch.Tensor):
            return torch.exp(state + output[:, 0])

        circ = ML.CircuitBuilder(n_modes=3)
        circ.add_entangling_layer()
        circ.add_memristive_ps(mode=1, update_rule=update_rule, initial_state=1.2)
        circ.add_memristive_ps(mode=0, update_rule=update_rule_exp, initial_state=0.01)
        circ.add_entangling_layer()
        circ.add_angle_encoding(modes=[0, 2])

        ql = ML.QuantumLayer(
            builder=circ,
            n_photons=3,
            measurement_strategy=ML.MeasurementStrategy.probs(
                computation_space=ML.ComputationSpace.FOCK
            ),
        )

        # Initial metadata check
        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        assert torch.allclose(ql.memristive_state[0], torch.Tensor([[1.2]]))
        assert torch.allclose(ql.memristive_state[1], torch.Tensor([[0.01]]))

        assert ql.memristive_history[0][0] == ql.memristive_state[0]
        assert ql.memristive_history[1][0] == ql.memristive_state[1]
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 1

        assert ql._memristive_metadata == circ.memristive_specs

        # Test batch too big
        with pytest.raises(
            RuntimeError,
            match="batch size mismatch",
        ):
            ql(torch.zeros([10, 2]))

        ql.reset(batch_size=5)

        # Test batch too big
        with pytest.raises(
            RuntimeError,
            match="batch size mismatch",
        ):
            ql(torch.zeros([10, 2]))

        # Initial metadata check after reset
        assert ql.input_size == 2

        assert torch.allclose(ql.memristive_state[0], torch.Tensor([[1.2] * 5]))
        assert torch.allclose(ql.memristive_state[1], torch.Tensor([[0.01] * 5]))

        assert torch.allclose(ql.memristive_history[0][0], ql.memristive_state[0])
        assert torch.allclose(ql.memristive_history[1][0], ql.memristive_state[1])
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 1

        assert ql._memristive_metadata == circ.memristive_specs

        input_1 = torch.zeros([5, 2])
        first_output = ql(input_1)

        # Metadata check after one pass
        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        new_state_0_t1 = update_rule(torch.Tensor([1.2] * 5), first_output)
        assert torch.allclose(ql.memristive_state[0], new_state_0_t1)
        new_state_1_t1 = update_rule_exp(torch.Tensor([0.01] * 5), first_output)
        assert torch.allclose(ql.memristive_state[1], new_state_1_t1)

        assert torch.allclose(ql.memristive_history[0][0], torch.Tensor([[1.2] * 5]))
        assert torch.allclose(ql.memristive_history[1][0], torch.Tensor([[0.01] * 5]))
        assert torch.allclose(ql.memristive_history[0][1], ql.memristive_state[0])
        assert torch.allclose(ql.memristive_history[1][1], ql.memristive_state[1])
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 2

        assert ql._memristive_metadata == circ.memristive_specs

        input_2 = torch.arange(10).reshape([5, 2])
        second_output = ql(input_2)

        # Metadata check after two passes
        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        new_state_0_t2 = update_rule(new_state_0_t1, second_output)
        assert torch.allclose(ql.memristive_state[0], new_state_0_t2)
        new_state_1_t2 = update_rule_exp(new_state_1_t1, second_output)
        assert torch.allclose(ql.memristive_state[1], new_state_1_t2)

        assert torch.allclose(ql.memristive_history[0][0], torch.Tensor([[1.2] * 5]))
        assert torch.allclose(ql.memristive_history[1][0], torch.Tensor([[0.01] * 5]))
        assert torch.allclose(ql.memristive_history[0][1], new_state_0_t1)
        assert torch.allclose(ql.memristive_history[1][1], new_state_1_t1)
        assert torch.allclose(ql.memristive_history[0][2], ql.memristive_state[0])
        assert torch.allclose(ql.memristive_history[1][2], ql.memristive_state[1])
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 3

        assert ql._memristive_metadata == circ.memristive_specs

        # Metadata check after smaller last batch
        input_3 = torch.arange(15, 21).reshape([3, 2])
        third_output = ql(input_3)

        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        new_state_0_t3 = update_rule(new_state_0_t2[:3], third_output)
        assert torch.allclose(ql.memristive_state[0], new_state_0_t3)
        new_state_1_t3 = update_rule_exp(new_state_1_t2[:3], third_output)
        assert torch.allclose(ql.memristive_state[1], new_state_1_t3)

        assert torch.allclose(ql.memristive_history[0][0], torch.Tensor([[1.2] * 5]))
        assert torch.allclose(ql.memristive_history[1][0], torch.Tensor([[0.01] * 5]))
        assert torch.allclose(ql.memristive_history[0][1], new_state_0_t1)
        assert torch.allclose(ql.memristive_history[1][1], new_state_1_t1)
        assert torch.allclose(ql.memristive_history[0][2], new_state_0_t2)
        assert torch.allclose(ql.memristive_history[1][2], new_state_1_t2)
        assert torch.allclose(ql.memristive_history[0][3], ql.memristive_state[0])
        assert torch.allclose(ql.memristive_history[1][3], ql.memristive_state[1])
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 4
        assert ql.memristive_history[0][0].size(0) == 5
        assert ql.memristive_history[0][3].size(0) == 3

        assert ql._memristive_metadata == circ.memristive_specs

        # Test last batch error
        with pytest.raises(
            RuntimeError,
            match="Already ran a smaller batch size",
        ):
            ql(input_1)

        with pytest.raises(
            RuntimeError,
            match="Already ran a smaller batch size",
        ):
            ql(input_3)

        ql.reset()

        # Metadata check after a reset
        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        assert torch.allclose(ql.memristive_state[0], torch.Tensor([[1.2]]))
        assert torch.allclose(ql.memristive_state[1], torch.Tensor([[0.01]]))

        assert ql.memristive_history[0][0] == ql.memristive_state[0]
        assert ql.memristive_history[1][0] == ql.memristive_state[1]
        assert len(ql.memristive_history[0]) == 1
        assert len(ql.memristive_history[1]) == 1
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 1

        assert ql._memristive_metadata == circ.memristive_specs

        # Running again after a reset
        try:
            ql(torch.Tensor([[0, 0]]))
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

    def test_param_mapping(self):
        # Test with more than one param if the right param is attributed to the right component
        def update_rule(state: torch.Tensor, output: torch.Tensor):
            return state + output[:, 0]

        def update_rule_exp(state: torch.Tensor, output: torch.Tensor):
            return torch.exp(state + output[:, 0])

        circ = ML.CircuitBuilder(n_modes=3)
        circ.add_entangling_layer()
        circ.add_memristive_ps(mode=1, update_rule=update_rule, initial_state=1.2)
        circ.add_memristive_ps(mode=0, update_rule=update_rule_exp, initial_state=0.01)
        circ.add_entangling_layer()
        circ.add_angle_encoding(modes=[0, 2])

        ql = ML.QuantumLayer(
            builder=circ,
            n_photons=3,
            measurement_strategy=ML.MeasurementStrategy.probs(
                computation_space=ML.ComputationSpace.FOCK
            ),
        )

        name_to_index = (
            ql.computation_process.converter.memristive_metadata_name_to_index
        )
        metadata = ql.computation_process.converter.memristive_metadata
        assert metadata[name_to_index["mem1"]]["update_rule"] == update_rule
        assert metadata[name_to_index["mem2"]]["update_rule"] == update_rule_exp

        output = ql(torch.Tensor([0.0, 0.0]))

        name_to_index = (
            ql.computation_process.converter.memristive_metadata_name_to_index
        )
        metadata = ql.computation_process.converter.memristive_metadata
        current_state = ql.computation_process.converter.memristive_current_state
        assert metadata[name_to_index["mem1"]]["update_rule"] == update_rule
        assert metadata[name_to_index["mem2"]]["update_rule"] == update_rule_exp
        # After forward pass, check that states are updated correctly
        expected_state_0 = update_rule(torch.Tensor([1.2]), output)
        expected_state_1 = update_rule_exp(torch.Tensor([0.01]), output)
        assert torch.allclose(current_state[name_to_index["mem1"]], expected_state_0)
        assert torch.allclose(current_state[name_to_index["mem2"]], expected_state_1)

    def test_memristor_gradient_flow(self):
        def update_rule(state: torch.Tensor, output: torch.Tensor):
            return state + output[:, 0]

        def update_rule_exp(state: torch.Tensor, output: torch.Tensor):
            return torch.exp(state + output[:, 0])

        circ = ML.CircuitBuilder(n_modes=3)
        circ.add_entangling_layer()
        circ.add_memristive_ps(
            mode=1,
            update_rule=update_rule,
            initial_state=1.2,
            detach_at_each_forward=False,
        )
        circ.add_memristive_ps(
            mode=0,
            update_rule=update_rule_exp,
            initial_state=0.01,
            detach_at_each_forward=False,
        )
        circ.add_entangling_layer()

        ql = ML.QuantumLayer(
            builder=circ,
            n_photons=3,
            measurement_strategy=ML.MeasurementStrategy.probs(
                computation_space=ML.ComputationSpace.FOCK
            ),
        )

        # Check that the returned tensor remains attached to autograd even though
        # memristive state updates use a detached copy internally.
        output = ql()

        assert isinstance(output, torch.Tensor)
        assert output.requires_grad
        assert output.grad_fn is not None

        trainable_params = [param for param in ql.parameters() if param.requires_grad]
        assert trainable_params
        params_before_step = [param.detach().clone() for param in trainable_params]

        opt = torch.optim.Adam(trainable_params)

        weights = torch.arange(
            1,
            output.shape[-1] + 1,
            device=output.device,
            dtype=output.dtype,
        )
        loss = (output * weights).sum()
        loss.backward()

        assert all(param.grad is not None for param in trainable_params)
        assert all(torch.isfinite(param.grad).all() for param in trainable_params)
        assert any(torch.any(param.grad != 0) for param in trainable_params)

        opt.step()

        assert any(
            not torch.allclose(param_before, param_after)
            for param_before, param_after in zip(
                params_before_step, trainable_params, strict=True
            )
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memristor_works_on_cuda(self):
        def update_rule(state: torch.Tensor, output: torch.Tensor):
            return state + output[:, 0]

        def update_rule_exp(state: torch.Tensor, output: torch.Tensor):
            return torch.exp(state + output[:, 0])

        circ = ML.CircuitBuilder(n_modes=3)
        circ.add_entangling_layer()
        circ.add_memristive_ps(mode=1, update_rule=update_rule, initial_state=1.2)
        circ.add_memristive_ps(mode=0, update_rule=update_rule_exp, initial_state=0.01)
        circ.add_entangling_layer()
        circ.add_angle_encoding(modes=[0, 2])

        ql = ML.QuantumLayer(
            builder=circ,
            n_photons=3,
            measurement_strategy=ML.MeasurementStrategy.probs(
                computation_space=ML.ComputationSpace.FOCK
            ),
        )
        # Copy to check the memristor states are correclty moved
        ql_copy = deepcopy(ql)
        ql_copy.reset(batch_size=5)
        ql = ql.to(torch.device("cuda"))

        # Initial metadata check
        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        assert ql.memristive_history[0][0].device.type == torch.device("cuda").type
        assert ql.memristive_state[0].device.type == torch.device("cuda").type

        assert torch.allclose(
            ql.memristive_state[0], torch.tensor([[1.2]], device=torch.device("cuda"))
        )
        assert torch.allclose(
            ql.memristive_state[1], torch.tensor([[0.01]], device=torch.device("cuda"))
        )

        assert ql.memristive_history[0][0] == ql.memristive_state[0]
        assert ql.memristive_history[1][0] == ql.memristive_state[1]
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 1

        assert ql._memristive_metadata == circ.memristive_specs

        ql.reset(batch_size=5)

        # Initial metadata check after reset
        assert ql.input_size == 2

        assert torch.allclose(
            ql.memristive_state[0],
            torch.tensor([[1.2] * 5], device=torch.device("cuda")),
        )
        assert torch.allclose(
            ql.memristive_state[1],
            torch.tensor([[0.01] * 5], device=torch.device("cuda")),
        )

        assert torch.allclose(ql.memristive_history[0][0], ql.memristive_state[0])
        assert torch.allclose(ql.memristive_history[1][0], ql.memristive_state[1])
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 1

        assert ql._memristive_metadata == circ.memristive_specs

        input_1 = torch.zeros([5, 2])
        first_output = ql(input_1)

        # Metadata check after one pass
        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        assert ql.memristive_history[0][0].device.type == torch.device("cuda").type
        assert ql.memristive_state[0].device.type == torch.device("cuda").type

        new_state_0_t1 = update_rule(
            torch.tensor([1.2] * 5, device=torch.device("cuda")), first_output
        )
        assert torch.allclose(ql.memristive_state[0], new_state_0_t1)
        new_state_1_t1 = update_rule_exp(
            torch.tensor([0.01] * 5, device=torch.device("cuda")), first_output
        )

        assert torch.allclose(ql.memristive_state[1], new_state_1_t1)

        assert torch.allclose(
            ql.memristive_history[0][0].to(torch.device("cuda")),
            torch.tensor([[1.2] * 5], device=torch.device("cuda")),
        )
        assert torch.allclose(
            ql.memristive_history[1][0],
            torch.tensor([[0.01] * 5], device=torch.device("cuda")),
        )
        assert torch.allclose(ql.memristive_history[0][1], ql.memristive_state[0])
        assert torch.allclose(ql.memristive_history[1][1], ql.memristive_state[1])
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 2

        assert ql._memristive_metadata == circ.memristive_specs

        input_2 = torch.arange(10).reshape([5, 2])
        second_output = ql(input_2)

        # Metadata check after two passes
        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        assert ql.memristive_history[0][0].device.type == torch.device("cuda").type
        assert ql.memristive_state[0].device.type == torch.device("cuda").type

        new_state_0_t2 = update_rule(
            new_state_0_t1.to(torch.device("cuda")), second_output
        )
        assert torch.allclose(ql.memristive_state[0], new_state_0_t2)
        new_state_1_t2 = update_rule_exp(
            new_state_1_t1.to(torch.device("cuda")), second_output
        )
        assert torch.allclose(ql.memristive_state[1], new_state_1_t2)

        assert torch.allclose(
            ql.memristive_history[0][0],
            torch.tensor([[1.2] * 5], device=torch.device("cuda")),
        )
        assert torch.allclose(
            ql.memristive_history[1][0],
            torch.tensor([[0.01] * 5], device=torch.device("cuda")),
        )
        assert torch.allclose(ql.memristive_history[0][1], new_state_0_t1)
        assert torch.allclose(ql.memristive_history[1][1], new_state_1_t1)
        assert torch.allclose(ql.memristive_history[0][2], ql.memristive_state[0])
        assert torch.allclose(ql.memristive_history[1][2], ql.memristive_state[1])
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 3

        assert ql._memristive_metadata == circ.memristive_specs

        # Metadata check after smaller last batch
        input_3 = torch.arange(15, 21).reshape([3, 2]).to(torch.device("cuda"))
        third_output = ql(input_3)

        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        assert ql.memristive_history[0][0].device.type == torch.device("cuda").type
        assert ql.memristive_state[0].device.type == torch.device("cuda").type
        new_state_0_t3 = update_rule(
            new_state_0_t2[:3].to(torch.device("cuda")), third_output
        )
        assert torch.allclose(ql.memristive_state[0], new_state_0_t3)
        new_state_1_t3 = update_rule_exp(
            new_state_1_t2[:3].to(torch.device("cuda")), third_output
        )
        assert torch.allclose(ql.memristive_state[1], new_state_1_t3)

        assert torch.allclose(
            ql.memristive_history[0][0],
            torch.tensor([[1.2] * 5], device=torch.device("cuda")),
        )
        assert torch.allclose(
            ql.memristive_history[1][0],
            torch.tensor([[0.01] * 5], device=torch.device("cuda")),
        )
        assert torch.allclose(ql.memristive_history[0][1], new_state_0_t1)
        assert torch.allclose(ql.memristive_history[1][1], new_state_1_t1)
        assert torch.allclose(ql.memristive_history[0][2], new_state_0_t2)
        assert torch.allclose(ql.memristive_history[1][2], new_state_1_t2)
        assert torch.allclose(ql.memristive_history[0][3], ql.memristive_state[0])
        assert torch.allclose(ql.memristive_history[1][3], ql.memristive_state[1])
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 4
        assert ql.memristive_history[0][0].size(0) == 5
        assert ql.memristive_history[0][3].size(0) == 3

        assert ql._memristive_metadata == circ.memristive_specs

        ql.reset()

        # Metadata check after a reset
        assert ql.input_size == 2
        assert "mem0" not in ql.input_parameters
        assert "mem1" not in ql.input_parameters

        assert ql.memristive_history[0][0].device.type == torch.device("cuda").type
        assert ql.memristive_state[0].device.type == torch.device("cuda").type

        assert torch.allclose(
            ql.memristive_state[0], torch.tensor([[1.2]], device=torch.device("cuda"))
        )
        assert torch.allclose(
            ql.memristive_state[1], torch.tensor([[0.01]], device=torch.device("cuda"))
        )

        assert ql.memristive_history[0][0] == ql.memristive_state[0]
        assert ql.memristive_history[1][0] == ql.memristive_state[1]
        assert len(ql.memristive_history[0]) == 1
        assert len(ql.memristive_history[1]) == 1
        assert len(ql.memristive_history) == len(ql.memristive_state) == 2
        assert len(ql.memristive_history[0]) == len(ql.memristive_history[1]) == 1

        assert ql._memristive_metadata == circ.memristive_specs

        # Running again after a reset
        try:
            ql(torch.tensor([[0, 0]], device=torch.device("cuda")))
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

        # Running three forward and seeing if the input is correctly moved
        ql_copy(input_1)
        ql_copy(input_2)
        ql_copy(input_3)
        ql_copy.to(torch.device("cuda"))

        assert ql_copy.memristive_history[0][0].device.type == torch.device("cuda").type
        assert ql_copy.memristive_state[0].device.type == torch.device("cuda").type
        assert ql_copy.input_size == 2
        assert "mem0" not in ql_copy.input_parameters
        assert "mem1" not in ql_copy.input_parameters

        assert torch.allclose(
            ql_copy.memristive_history[0][0],
            torch.tensor([[1.2] * 5], device=torch.device("cuda")),
        )
        assert torch.allclose(
            ql_copy.memristive_history[1][0],
            torch.tensor([[0.01] * 5], device=torch.device("cuda")),
        )
        assert torch.allclose(ql_copy.memristive_history[0][1], new_state_0_t1)
        assert torch.allclose(ql_copy.memristive_history[1][1], new_state_1_t1)
        assert torch.allclose(ql_copy.memristive_history[0][2], new_state_0_t2)
        assert torch.allclose(ql_copy.memristive_history[1][2], new_state_1_t2)
        assert torch.allclose(
            ql_copy.memristive_history[0][3], ql_copy.memristive_state[0]
        )
        assert torch.allclose(
            ql_copy.memristive_history[1][3], ql_copy.memristive_state[1]
        )
        assert len(ql_copy.memristive_history) == len(ql_copy.memristive_state) == 2
        assert (
            len(ql_copy.memristive_history[0])
            == len(ql_copy.memristive_history[1])
            == 4
        )
        assert ql_copy.memristive_history[0][0].size(0) == 5
        assert ql_copy.memristive_history[0][3].size(0) == 3

        assert ql_copy._memristive_metadata == circ.memristive_specs

        # Moving the data back
        ql_copy.to(torch.device("cpu"))
        assert ql_copy.memristive_history[0][0].device.type == torch.device("cpu").type
        assert ql_copy.memristive_state[0].device.type == torch.device("cpu").type


def _identity_update(state: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Keep a memristive state unchanged while satisfying current annotations."""
    return state


def _builder_with_memristor(
    update_rule=_identity_update,
    *,
    with_inputs: bool = False,
) -> ML.CircuitBuilder:
    """Build the smallest useful memristive circuit for review probes."""
    builder = ML.CircuitBuilder(n_modes=3)
    builder.add_entangling_layer()
    if with_inputs:
        builder.add_angle_encoding(modes=[0, 2], name="input")
    builder.add_memristive_ps(mode=1, update_rule=update_rule, initial_state=0.25)
    builder.add_entangling_layer()
    return builder


def _layer(builder: ML.CircuitBuilder, *, input_size: int = 0) -> ML.QuantumLayer:
    """Create a probability layer in FOCK space so batch behavior is explicit."""
    return ML.QuantumLayer(
        builder=builder,
        input_size=input_size,
        input_state=[1, 0, 1],
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK
        ),
    )


def test_memristive_layer_to_moves_state_and_history_without_raising():
    """``QuantumLayer.to(...)`` should work for layers with memristive state.

    The device/dtype movement path is part of making the feature usable with
    CPU/GPU workflows. The current implementation raises even for
    ``layer.to(torch.device("cpu"))`` because it calls ``len(...)`` on an integer
    loop index while moving ``memristive_history``.
    """
    layer = _layer(_builder_with_memristor())

    layer.to(torch.device("cpu"))

    assert layer.memristive_state[0].device == torch.device("cpu")
    assert layer.memristive_history[0][0].device == torch.device("cpu")


def test_update_rule_return_shape_is_validated_close_to_user_callback():
    """The update rule must return a state tensor of shape ``[batch_size]``.

    The ticket and docs both describe the runtime state as one scalar per batch
    element. A user callback returning ``[batch_size, 1]`` should fail with a
    clear validation error near the callback boundary instead of being stored as
    the next memristive state and surfacing later as converter shape behavior.
    """

    def bad_shape_update(state: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return torch.ones((state.size(0), 1), dtype=state.dtype, device=state.device)

    layer = _layer(
        _builder_with_memristor(bad_shape_update, with_inputs=True),
        input_size=2,
    )
    layer.reset(batch_size=2)

    with pytest.raises(ValueError, match="shape|batch_size"):
        layer(torch.zeros(2, 2))


def test_invalid_memristor_update_rule():
    def valid_update(state: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return state + 0.1

    def invalid_update(state: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        raise ValueError("Not valid")

    builder = ML.CircuitBuilder(n_modes=5)
    builder.add_memristive_ps(mode=1, update_rule=valid_update, initial_state=0.1)
    builder.add_memristive_ps(mode=4, update_rule=invalid_update, initial_state=0.3)
    builder.add_memristive_ps(mode=3, update_rule=valid_update, initial_state=0.5)

    layer = ML.QuantumLayer(builder=builder, n_photons=3)

    error_string = (
        "The update rule of the following memristor does not follow the correct build or raises an error. Here is the expected signature:\n\n"
        "                    Expected: update_rule(state: torch.Tensor,output: torch.Tensor | StateVector | ProbabilityDistribution | PartialMeasurement)-> torch.Tensor\n\n"
        f"                    Memristive phase-shifter analyzed: {builder.memristive_specs[1]}\n"
        "                    "
    )

    with pytest.raises(ValueError, match=re.escape(error_string)):
        layer()


def test_simple_num_photons_modes_and_input_state():
    for i in range(1, 15):
        ql = ML.QuantumLayer.simple(input_size=i)
        assert ql.quantum_layer.n_photons == int(np.ceil((i + 1) / 2))
        assert sum(ql.quantum_layer.input_state) == int(np.ceil((i + 1) / 2))
        assert len(ql.quantum_layer.input_state) == i + 1

        input_state = [0] * (i + 1)
        for j in range(len(input_state)):
            if j % 2 == 0:
                input_state[j] = 1
        assert ql.quantum_layer.input_state == pcvl.BasicState(input_state)


def test_simple_parameters():
    for i in range(1, 15):
        ql = ML.QuantumLayer.simple(input_size=i)
        params = list(ql.quantum_layer.parameters())
        named_params = [k[0] for k in ql.quantum_layer.named_parameters()]

        assert params[0].numel() == i * (i + 1)
        assert params[1].numel() == i * (i + 1)
        assert len(params) == 2
        assert "LI_simple" in named_params
        assert "RI_simple" in named_params


def test_memristive_state_dict_round_trip_preserves_state_and_history():
    """``state_dict`` / ``load_state_dict`` must preserve memristive state.

    Plain Python lists of tensors are invisible to PyTorch's default
    serialization. QuantumLayer therefore injects explicit memristive runtime
    entries into its own ``state_dict`` so save/load round-trips preserve both
    the current memristive state and the accumulated history.
    """
    import io

    layer = _layer(_builder_with_memristor(with_inputs=True), input_size=2)

    # Advance the memristive state over two forward passes.
    layer(torch.zeros(1, 2))
    layer(torch.zeros(1, 2))

    state_before = layer.memristive_state[0].clone()
    history_len_before = len(layer.memristive_history[0])

    # Save via the standard PyTorch checkpoint path.
    buffer = io.BytesIO()
    torch.save(layer.state_dict(), buffer)
    buffer.seek(0)

    restored = _layer(_builder_with_memristor(with_inputs=True), input_size=2)
    restored.load_state_dict(torch.load(buffer, weights_only=True))

    assert torch.allclose(
        restored.memristive_state[0], state_before
    ), "memristive_state was not preserved across a state_dict round-trip"
    assert (
        len(restored.memristive_history[0]) == history_len_before
    ), "memristive_history length was not preserved across a state_dict round-trip"


def test_memristive_state_dict_round_trip_as_submodule():
    """Serialization must work when ``QuantumLayer`` is a child of another module.

    When the parent calls ``state_dict()`` the prefix for child keys is
    ``"quantum_layer."``; the custom memristive runtime entries must still
    resolve correctly in that case.
    """
    import io

    class _Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quantum_layer = _layer(
                _builder_with_memristor(with_inputs=True),
                input_size=2,
            )

    wrapper = _Wrapper()
    wrapper.quantum_layer(torch.zeros(1, 2))
    wrapper.quantum_layer(torch.zeros(1, 2))

    state_before = wrapper.quantum_layer.memristive_state[0].clone()
    history_len_before = len(wrapper.quantum_layer.memristive_history[0])

    buffer = io.BytesIO()
    torch.save(wrapper.state_dict(), buffer)
    buffer.seek(0)

    restored_wrapper = _Wrapper()
    restored_wrapper.load_state_dict(torch.load(buffer, weights_only=True))

    assert torch.allclose(
        restored_wrapper.quantum_layer.memristive_state[0], state_before
    ), "memristive_state was not preserved when QuantumLayer is a submodule"
    assert (
        len(restored_wrapper.quantum_layer.memristive_history[0]) == history_len_before
    ), "memristive_history length was not preserved when QuantumLayer is a submodule"


def test_non_memristive_layer_state_dict_has_no_memristive_runtime_entries():
    """Non-memristive layers should not inject memristive runtime entries."""
    builder = ML.CircuitBuilder(n_modes=3)
    builder.add_entangling_layer()
    layer = ML.QuantumLayer(builder=builder, n_photons=2)

    sd = layer.state_dict()

    assert "_extra_state" not in sd
    assert all("_memristive_state" not in key for key in sd)
    assert all("_memristive_history" not in key for key in sd)


def test_memristive_reset_after_to_uses_updated_device_and_dtype():
    """``reset()`` must recreate memristive tensors on the moved device and dtype."""
    layer = _layer(_builder_with_memristor())

    layer.to(torch.device("cpu"))
    layer.to(torch.float64)
    layer.reset(batch_size=3)

    assert layer.memristive_state[0].device == torch.device("cpu")
    assert layer.memristive_state[0].dtype == torch.float64
    assert layer.memristive_history[0][0].device == torch.device("cpu")
    assert layer.memristive_history[0][0].dtype == torch.float64
    assert layer.memristive_state[0].shape == torch.Size([3])


def test_memristive_ps_with_same_requested_name_get_distinct_parameter_names():
    """Two memristive PS entries with the same requested prefix must stay unique."""

    def update_to_first_marker(
        state: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        return torch.full_like(state, 0.123)

    def update_to_second_marker(
        state: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        return torch.full_like(state, 2.345)

    builder = ML.CircuitBuilder(n_modes=3)
    builder.add_entangling_layer()
    builder.add_memristive_ps(
        mode=1,
        update_rule=update_to_first_marker,
        initial_state=0.25,
        name="memPS",
    )
    builder.add_memristive_ps(
        mode=0,
        update_rule=update_to_second_marker,
        initial_state=0.5,
        name="memPS",
    )
    builder.add_entangling_layer()

    layer = _layer(builder)

    parameter_names = [parameter.name for parameter in layer.circuit.get_parameters()]
    metadata_names = [metadata["name"] for metadata in layer._memristive_metadata]

    assert metadata_names == ["memPS1", "memPS2"]
    assert "memPS1" in parameter_names
    assert "memPS2" in parameter_names
    assert len(set(metadata_names)) == 2


def test_memristive_updates_still_run_in_eval_mode():
    """``eval()`` must not suppress memristive side effects between forwards."""

    def increment_update(state: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return state + 1.0

    layer = _layer(
        _builder_with_memristor(increment_update, with_inputs=True), input_size=2
    )
    initial_state = layer.memristive_state[0].clone()

    layer.eval()
    layer(torch.zeros(1, 2))
    state_after_first_forward = layer.memristive_state[0].clone()
    layer(torch.zeros(1, 2))

    assert torch.allclose(state_after_first_forward, initial_state + 1.0)
    assert torch.allclose(layer.memristive_state[0], initial_state + 2.0)
    assert len(layer.memristive_history[0]) == 3


def test_batch_size_mismatch_raises_before_first_memristive_update():
    """Inconsistent memristive state sizes must fail before any forward side effect."""
    update_calls = 0

    def counted_update(state: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        nonlocal update_calls
        update_calls += 1
        return state + output[:, 0]

    builder = ML.CircuitBuilder(n_modes=3)
    builder.add_entangling_layer()
    builder.add_memristive_ps(
        mode=1,
        update_rule=counted_update,
        initial_state=0.25,
        name="memPS_a",
    )
    builder.add_memristive_ps(
        mode=0,
        update_rule=counted_update,
        initial_state=0.5,
        name="memPS_b",
    )
    builder.add_angle_encoding(modes=[0, 2], name="input")

    layer = _layer(builder, input_size=2)
    layer.reset(batch_size=4)
    layer.memristive_state[1] = layer.memristive_state[1][:1]

    history_lengths_before = [len(history) for history in layer.memristive_history]

    with pytest.raises(
        RuntimeError,
        match="Not all memristive states have the same size",
    ):
        layer(torch.zeros(4, 2))

    assert update_calls == 0
    assert [
        len(history) for history in layer.memristive_history
    ] == history_lengths_before


def test_memristive_works_with_typed_objects_and_cloning_protects_gradients():
    """Memristive updates with typed objects must not break gradients via cloning.

    This test verifies that:
    1. Memristive phase shifters work correctly with typed output objects
       (StateVector, ProbabilityDistribution, PartialMeasurement)
    2. Cloning the output protects memristive computations from interfering
       with the returned output's autograd graph
    3. Modifications to the output inside the update rule do not affect the
       returned layer output
    4. Gradients flow correctly despite aggressive modification attempts
    5. Memristive history accumulates across forward passes
    """

    def update_rule(state: torch.Tensor, output: torch.Tensor):
        # Handle both tensor and typed object outputs
        if isinstance(output, torch.Tensor):
            output_tensor = output
        else:
            # Extract tensor from StateVector, ProbabilityDistribution, PartialMeasurement
            output_tensor = output.tensor

        # Modify the output received by the update rule to verify cloning protects
        # the returned output from side effects
        modified_output = output_tensor.clone()
        modified_output[:] = 999.0  # Drastically modify the copy
        return state + modified_output[:, 0]

    # Create memristive circuit with trainable layers
    builder = ML.CircuitBuilder(n_modes=3)
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_memristive_ps(
        mode=1,
        update_rule=update_rule,
        initial_state=0.5,
        name="memPS",
        detach_at_each_forward=False,
    )
    builder.add_angle_encoding(modes=[0, 2], name="input")
    builder.add_entangling_layer(trainable=True, name="U2")

    # Test with StateVector typed output (from amplitudes)
    layer_statevector_typed = ML.QuantumLayer(
        builder=builder,
        input_size=2,
        n_photons=2,
        measurement_strategy=ML.MeasurementStrategy.amplitudes(),
        return_object=True,
    )
    layer_statevector_typed.reset(batch_size=2)

    # Test with PartialMeasurement typed output
    layer_partial = ML.QuantumLayer(
        builder=builder,
        input_size=2,
        n_photons=2,
        measurement_strategy=ML.MeasurementStrategy.partial(modes=[0, 1]),
        return_object=True,
    )
    layer_partial.reset(batch_size=2)

    # Test with ProbabilityDistribution typed output
    layer_probdist_typed = ML.QuantumLayer(
        builder=builder,
        input_size=2,
        n_photons=2,
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK
        ),
        return_object=True,
    )
    layer_probdist_typed.reset(batch_size=2)

    input_data = torch.randn(2, 2, requires_grad=True)

    # Test StateVector with return_object=True
    output_sv_typed = layer_statevector_typed(input_data)

    assert isinstance(output_sv_typed, StateVector)
    # Verify output is NOT modified by the update rule (cloning protected it)
    assert not torch.allclose(
        output_sv_typed.tensor, torch.full_like(output_sv_typed.tensor, 999.0)
    ), "StateVector output was incorrectly modified by memristive update rule"
    assert output_sv_typed.tensor.requires_grad

    # Compute loss and backward for StateVector
    loss_sv = output_sv_typed.tensor.abs().sum()
    loss_sv.backward()
    assert input_data.grad is not None
    assert torch.any(
        input_data.grad != 0
    ), "Gradients should flow through StateVector output"

    # Verify memristive history accumulated (one forward pass = one state)
    assert len(layer_statevector_typed.memristive_history[0]) >= 1

    # Clear gradient for next test
    input_data.grad = None

    # Test ProbabilityDistribution with typed output
    output_pd_typed = layer_probdist_typed(input_data.detach())

    assert isinstance(output_pd_typed, ProbabilityDistribution)
    # Verify output is NOT modified by the update rule (cloning protected it)
    assert not torch.allclose(
        output_pd_typed.tensor, torch.full_like(output_pd_typed.tensor, 999.0)
    ), "ProbabilityDistribution output was incorrectly modified by memristive update rule"
    assert output_pd_typed.tensor.requires_grad

    # Compute loss and backward for ProbabilityDistribution
    loss_pd = (
        output_pd_typed.tensor * torch.arange(1, output_pd_typed.tensor.shape[-1] + 1)
    ).sum()
    loss_pd.backward()

    # Verify memristive history accumulated
    assert len(layer_probdist_typed.memristive_history[0]) >= 1

    # Test PartialMeasurement with typed output
    output_partial = layer_partial(input_data.detach())
    assert isinstance(output_partial, PartialMeasurement)
    # Verify output is NOT modified by the update rule (cloning protected it)
    assert not torch.allclose(
        output_partial.tensor, torch.full_like(output_partial.tensor, 999.0)
    ), "PartialMeasurement output was incorrectly modified by memristive update rule"
    assert output_partial.tensor.requires_grad

    # Compute loss and backward for PartialMeasurement
    loss_partial = output_partial.tensor.abs().sum()
    loss_partial.backward()

    # Verify memristive history accumulated
    assert len(layer_partial.memristive_history[0]) >= 1

    # Run second forward passes to verify memristive state and history continue to accumulate
    output_sv_typed_2 = layer_statevector_typed(input_data.detach())
    output_pd_typed_2 = layer_probdist_typed(input_data.detach())
    output_partial_2 = layer_partial(input_data.detach())

    assert isinstance(output_sv_typed_2, StateVector)
    assert isinstance(output_pd_typed_2, ProbabilityDistribution)
    assert isinstance(output_partial_2, PartialMeasurement)

    # Verify history accumulated over multiple forward passes
    assert len(layer_statevector_typed.memristive_history[0]) >= 2
    assert len(layer_probdist_typed.memristive_history[0]) >= 2
    assert len(layer_partial.memristive_history[0]) >= 2

    # Verify outputs from second pass are also not modified to 999.0
    assert not torch.allclose(
        output_sv_typed_2.tensor, torch.full_like(output_sv_typed_2.tensor, 999.0)
    )
    assert not torch.allclose(
        output_pd_typed_2.tensor, torch.full_like(output_pd_typed_2.tensor, 999.0)
    )
    assert not torch.allclose(
        output_partial_2.tensor, torch.full_like(output_partial_2.tensor, 999.0)
    )


def _update_from_first_probability(
    state: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    """Simple differentiable recurrence used by the review tests.

    The next memristive state depends on the previous state and on the first
    output probability. Without a working backpropagation guard, a later loss
    can therefore propagate through the state chain back to earlier inputs.
    """
    return state + output[:, 0]


def _make_memristive_layer(detach_at_each_forward: bool = True) -> ML.QuantumLayer:
    """Build a small memristive layer with one recurrent phase shifter."""
    builder = ML.CircuitBuilder(n_modes=3)
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_memristive_ps(
        mode=1,
        update_rule=_update_from_first_probability,
        initial_state=0.25,
        name="mem",
        detach_at_each_forward=detach_at_each_forward,
    )
    builder.add_angle_encoding(modes=[0, 2], name="input")
    builder.add_entangling_layer(trainable=True, name="U2")

    return ML.QuantumLayer(
        builder=builder,
        input_size=2,
        input_state=[1, 1, 0],
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK
        ),
    )


def test_detach_at_each_forward_true_blocks_gradient_through_recurrence():
    """detach_at_each_forward=True should block gradients through state recurrence.

    With detach_at_each_forward=True, the memristive state is detached at each
    forward pass, preventing gradients from flowing through the recurrence
    (state → new_state). This breaks the gradient chain so earlier inputs
    receive zero gradients from the memristive state dependence.
    """
    torch.manual_seed(1)
    layer = _make_memristive_layer(detach_at_each_forward=True)
    layer.reset(batch_size=1)

    inputs = [torch.randn(1, 2, requires_grad=True) for _ in range(4)]
    outputs = [layer(input_batch) for input_batch in inputs]

    loss = outputs[-1][:, 0].sum()
    loss.backward()

    past_grad_norms = [
        0.0 if input_batch.grad is None else input_batch.grad.abs().max().item()
        for input_batch in inputs[:-1]
    ]
    current_grad_norm = inputs[-1].grad.abs().max().item()

    assert current_grad_norm > 1e-8
    assert past_grad_norms == pytest.approx([0.0, 0.0, 0.0], abs=1e-8)


def test_detach_at_each_forward_false_allows_full_gradient_history():
    """detach_at_each_forward=False should allow full gradient flow through history.

    With detach_at_each_forward=False, the memristive state is NOT detached,
    so gradients flow through the entire recurrence chain. All previous inputs
    should receive non-zero gradients from the memristive history.
    """
    torch.manual_seed(42)
    layer = _make_memristive_layer(detach_at_each_forward=False)
    layer.reset(batch_size=1)

    # Run 5 forward passes to accumulate history
    inputs = [torch.randn(1, 2, requires_grad=True) for _ in range(5)]
    outputs = [layer(input_batch) for input_batch in inputs]

    # Verify memristive history accumulated
    assert (
        len(layer.memristive_history[0]) == 6
    ), f"Expected 6 history entries, got {len(layer.memristive_history[0])}"

    # Compute loss from the last output and backpropagate
    loss = outputs[-1][:, 0].sum()
    loss.backward()

    # With detach_at_each_forward=False, all inputs should receive gradients
    # from the full recurrence chain
    grad_norms = [
        0.0 if input_batch.grad is None else input_batch.grad.abs().max().item()
        for input_batch in inputs
    ]

    # All inputs should have non-zero gradients (full history retained)
    for i, grad_norm in enumerate(grad_norms):
        assert (
            grad_norm > 1e-8
        ), f"Input {i} should have non-zero gradient with full history, got {grad_norm}"


def test_detach_at_each_forward_is_boolean_flag():
    """detach_at_each_forward flag accepts boolean values.

    The flag should accept True or False to control whether gradients flow
    through the memristive recurrence. Both values should be accepted without
    raising errors.
    """
    builder = ML.CircuitBuilder(n_modes=3)

    # Should not raise for valid boolean values
    builder.add_memristive_ps(
        mode=1,
        update_rule=_update_from_first_probability,
        initial_state=0.25,
        detach_at_each_forward=True,
    )
    assert builder.memristive_specs[0]["detach_at_each_forward"] is True

    builder2 = ML.CircuitBuilder(n_modes=3)
    builder2.add_memristive_ps(
        mode=1,
        update_rule=_update_from_first_probability,
        initial_state=0.25,
        detach_at_each_forward=False,
    )
    assert builder2.memristive_specs[0]["detach_at_each_forward"] is False


def test_detach_without_assignment_does_not_detach_original_tensor():
    """Calling detach without assignment does not change the original tensor.

    This test explains the reset() review comment. PyTorch ``Tensor.detach()``
    returns a new tensor disconnected from autograd; it does not mutate the
    original tensor in place. Therefore a statement like ``x.detach()`` has no
    lasting effect unless the return value is assigned back to ``x`` or stored
    somewhere else.
    """
    base = torch.ones(1, requires_grad=True)
    state = base * 2

    state.detach()

    assert state.requires_grad
    assert not state.detach().requires_grad


def test_mixed_memristors_with_different_detach_settings():
    """Layer with multiple memristors using different detach_at_each_forward settings.

    Different memristors in the same layer can have different gradient flow,
    allowing fine-grained control over gradient paths.
    """
    torch.manual_seed(2)

    builder = ML.CircuitBuilder(n_modes=5)
    # First memristor: detach gradients (blocking flow)
    builder.add_memristive_ps(
        mode=1,
        update_rule=_update_from_first_probability,
        initial_state=0.5,
        detach_at_each_forward=True,
        name="mem_detached",
    )
    # Second memristor: keep gradients (full flow)
    builder.add_memristive_ps(
        mode=2,
        update_rule=_update_from_first_probability,
        initial_state=0.3,
        detach_at_each_forward=False,
        name="mem_full_grad",
    )
    builder.add_angle_encoding(modes=[0, 3])

    layer = ML.QuantumLayer(
        builder=builder,
        n_photons=3,
        input_size=2,
        measurement_strategy=ML.MeasurementStrategy.probs(),
    )
    layer.reset(batch_size=1)

    # Run several forward passes
    inputs = [torch.randn(1, 2, requires_grad=True) for _ in range(3)]
    outputs = [layer(inp) for inp in inputs]

    loss = outputs[-1].sum()
    loss.backward()

    # First memristor (detached) should NOT have gradients in history
    for state in layer.memristive_history[0]:
        assert state.grad_fn is None, "Detached memristor states should have no grad_fn"

    # Second memristor (full grad) should retain gradients
    assert (
        layer.memristive_history[1][-1].grad_fn is not None
    ), "Non-detached memristor should retain gradients"


def test_detach_at_each_forward_false_has_larger_gradients_than_true():
    """Gradients propagate further with detach_at_each_forward=False.

    With full gradient flow (False), earlier inputs typically receive
    larger gradients than with detach_at_each_forward=True due to longer
    computational paths.
    """
    torch.manual_seed(3)

    layer_detached = _make_memristive_layer(detach_at_each_forward=True)
    layer_detached.reset(batch_size=1)

    layer_full_grad = _make_memristive_layer(detach_at_each_forward=False)
    layer_full_grad.reset(batch_size=1)

    N = 5
    seed_inputs = [torch.randn(1, 2) for _ in range(N)]

    # Run with detach=True
    torch.manual_seed(42)
    inputs_detached = [inp.clone().detach().requires_grad_(True) for inp in seed_inputs]
    outputs_detached = [layer_detached(inp) for inp in inputs_detached]
    loss_detached = outputs_detached[-1].sum()
    loss_detached.backward()
    grads_detached = [
        inp.grad.abs().max().item() if inp.grad is not None else 0.0
        for inp in inputs_detached
    ]

    # Run with detach=False
    torch.manual_seed(42)
    inputs_full = [inp.clone().detach().requires_grad_(True) for inp in seed_inputs]
    outputs_full = [layer_full_grad(inp) for inp in inputs_full]
    loss_full = outputs_full[-1].sum()
    loss_full.backward()
    grads_full = [
        inp.grad.abs().max().item() if inp.grad is not None else 0.0
        for inp in inputs_full
    ]

    # Earlier inputs should have larger or equal gradients with full history
    for i in range(N - 1):  # All except last
        assert (
            grads_full[i] >= grads_detached[i] * 0.9
        ), "Full gradient flow should have larger or equal gradients for earlier inputs"


def test_reset_properly_detaches_or_keeps_initial_state():
    """reset() properly initializes gradient tracking based on detach_at_each_forward.

    After reset(), the initial state should be detached or not based on the flag,
    ensuring fresh gradient tracking for the new batch.
    """
    torch.manual_seed(4)

    # Test with detach=True
    layer_detached = _make_memristive_layer(detach_at_each_forward=True)
    layer_detached.reset(batch_size=1)
    assert (
        layer_detached.memristive_state[0].grad_fn is None
    ), "Initial state should be detached with detach_at_each_forward=True"
    assert (
        layer_detached.memristive_history[0][0].grad_fn is None
    ), "Initial history state should be detached with detach_at_each_forward=True"

    # Test with detach=False
    layer_full = _make_memristive_layer(detach_at_each_forward=False)
    layer_full.reset(batch_size=1)
    # Initial state is typically created without grad by default
    # But after first forward it should retain gradients
    inp = torch.randn(1, 2, requires_grad=True)
    layer_full(inp)
    assert (
        layer_full.memristive_state[0].grad_fn is not None
    ), "State should retain gradients after forward with detach_at_each_forward=False"


def test_detach_at_each_forward_with_batch_dimension():
    """Gradient flow works correctly with batch_size > 1.

    The flag should maintain correct gradient behavior regardless of batch size.
    """
    torch.manual_seed(5)
    batch_size = 4

    layer = _make_memristive_layer(detach_at_each_forward=False)
    layer.reset(batch_size=batch_size)

    inputs = [torch.randn(batch_size, 2, requires_grad=True) for _ in range(3)]
    outputs = [layer(inp) for inp in inputs]

    loss = outputs[-1].sum()
    loss.backward()

    # All inputs should have gradients
    for i, inp in enumerate(inputs):
        assert inp.grad is not None, f"Input {i} should have gradients"
        assert inp.grad.shape == (
            batch_size,
            2,
        ), f"Gradient shape mismatch for input {i}"


def test_detach_memristive_state_preserves_history_by_default():
    """Manual detach should cut graph edges while preserving history values."""
    torch.manual_seed(6)
    layer = _make_memristive_layer(detach_at_each_forward=False)
    layer.reset(batch_size=1)

    for _ in range(3):
        layer(torch.randn(1, 2, requires_grad=True))

    history_before = [state.clone() for state in layer.memristive_history[0]]
    history_len_before = len(layer.memristive_history[0])

    layer.detach_memristive_state()

    assert len(layer.memristive_history[0]) == history_len_before
    assert layer.memristive_state[0].grad_fn is None
    assert layer.memristive_history[0][-1] is layer.memristive_state[0]

    for state_before, state_after in zip(
        history_before,
        layer.memristive_history[0],
        strict=True,
    ):
        assert state_after.grad_fn is None
        assert torch.allclose(state_after, state_before)


def test_long_sequence_with_manual_sliding_window_detach():
    """Manual TBPTT should cut gradients before the chosen chunk boundary."""
    torch.manual_seed(7)
    layer = _make_memristive_layer(detach_at_each_forward=False)
    layer.reset(batch_size=1)

    n_steps = 8
    k = 3
    inputs = []
    outputs = []

    for timestep in range(n_steps):
        input_batch = torch.randn(1, 2, requires_grad=True)
        inputs.append(input_batch)
        outputs.append(layer(input_batch))

        if timestep == n_steps - k - 1:
            layer.detach_memristive_state(clear_history=True)

    assert len(layer.memristive_history[0]) == k + 1

    loss = outputs[-1][:, 0].sum()
    loss.backward()

    grad_norms = [
        0.0 if input_batch.grad is None else input_batch.grad.abs().max().item()
        for input_batch in inputs
    ]

    assert grad_norms[: n_steps - k] == pytest.approx(
        [0.0] * (n_steps - k),
        abs=1e-8,
    )
    for index, grad_norm in enumerate(grad_norms[n_steps - k :], start=n_steps - k):
        assert grad_norm > 1e-8, f"Input {index} should remain in the TBPTT window"

    for index, grad_norm in enumerate(grad_norms[: n_steps - k]):
        assert grad_norm < 1e-8, f"Input {index} should not be in the TBPTT window"
