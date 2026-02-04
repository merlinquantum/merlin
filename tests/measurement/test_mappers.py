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
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import merlin as ML


class TestOutputMapper:
    def test_linear_mapping_creation(self):
        """Test creation of linear output mapping."""
        fock_distribution = ML.OutputMapper.create_mapping(
            ML.MeasurementStrategy.probs()
        )
        mapping = torch.nn.Sequential(fock_distribution, nn.Linear(6, 3))
        assert isinstance(mapping[0], ML.Probabilities)
        assert isinstance(mapping[-1], nn.Linear)
        assert mapping[-1].in_features == 6
        assert mapping[-1].out_features == 3

    def test_fock_distribution_mapping_creation(self):
        fock_distribution = ML.OutputMapper.create_mapping(
            ML.MeasurementStrategy.probs()
        )
        assert isinstance(fock_distribution, ML.Probabilities)

    def test_state_vector_mapping_creation_valid(self):
        """Test creation of state vector mapping with matching sizes."""
        mapping = ML.OutputMapper.create_mapping(ML.MeasurementStrategy.amplitudes())
        batch_size = 4
        input_amps = torch.rand(batch_size, 5)
        output_amps = mapping(input_amps)
        assert isinstance(mapping, ML.Amplitudes)
        assert torch.allclose(input_amps, output_amps, atol=1e-6)

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""

        class FakeStrategy:
            pass

        with pytest.raises(ValueError, match="Unknown measurement strategy"):
            ML.OutputMapper.create_mapping(FakeStrategy())

    def test_mode_expectations_requires_keys(self):
        """ModeExpectations mapping should require keys."""
        with pytest.raises(
            ValueError,
            match="When using ModeExpectations measurement strategy, keys must be provided.",
        ):
            ML.OutputMapper.create_mapping(
                ML.MeasurementStrategy.mode_expectations(
                    computation_space=ML.ComputationSpace.UNBUNCHED
                )
            )


class TestOutputMappingIntegration:
    """Integration tests for output mapping with QuantumLayer."""

    def test_linear_mapping_integration(self):
        """Test linear mapping integration with QuantumLayer."""

        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            n_photons=2,
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )

        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        x = torch.rand(5, 2)
        output = model(x)

        assert output.shape == (5, 3)
        assert torch.all(torch.isfinite(output))

    def test_mapping_gradient_flow(self):
        """Test gradient flow through different mapping strategies."""

        strategies = [
            ML.MeasurementStrategy.probs(),
            ML.MeasurementStrategy.mode_expectations(
                computation_space=ML.ComputationSpace.UNBUNCHED
            ),
            ML.MeasurementStrategy.amplitudes(),
        ]

        builder = ML.CircuitBuilder(n_modes=6)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        for strategy in strategies:
            layer = ML.QuantumLayer(
                input_size=2,
                n_photons=2,
                builder=builder,
                measurement_strategy=strategy,
            )

            model = (
                torch.nn.Sequential(
                    layer, torch.nn.Linear(layer.output_size, 3, dtype=torch.float32)
                )
                if strategy is not strategies[-1]
                else torch.nn.Sequential(
                    layer, torch.nn.Linear(layer.output_size, 3, dtype=torch.complex64)
                )
            )

            x = torch.rand(2, 2, requires_grad=True)
            output = model(x)

            # mse_loss is a real loss so let us work on real
            output = output.real

            # Use MSE loss instead of sum for better gradient flow
            target = torch.ones_like(output)
            loss = F.mse_loss(output, target)
            loss.backward()

            # Input should have gradients
            assert x.grad is not None, f"No gradients for strategy {strategy}"
            assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), (
                f"Zero gradients for strategy {strategy}"
            )

    def test_mapping_output_bounds(self):
        """Test that different mappings produce reasonable output bounds."""

        x = torch.rand(5, 2)

        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            n_photons=2,
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))
        output_linear = model(x)
        assert torch.all(torch.isfinite(output_linear))

    def test_dtype_consistency_in_mappings(self):
        """Test that output mappings respect input dtypes."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        for dtype in [torch.float32, torch.float64]:
            layer = ML.QuantumLayer(
                input_size=2,
                n_photons=2,
                builder=builder,
                measurement_strategy=ML.MeasurementStrategy.probs(),
                dtype=dtype,
            )
            model = torch.nn.Sequential(
                layer, torch.nn.Linear(layer.output_size, 3, dtype=dtype)
            )

            x = torch.rand(3, 2, dtype=dtype)
            output = model(x)

            # Output should be finite and have reasonable values
            assert torch.all(torch.isfinite(output))
            assert output.shape == (3, 3)

    def test_edge_case_single_dimension(self):
        """Test edge case with single input/output dimensions."""
        builder = ML.CircuitBuilder(n_modes=3)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=1,
            n_photons=1,
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 1))

        x = torch.rand(5, 1)
        output = model(x)

        assert output.shape == (5, 1)
        assert torch.all(torch.isfinite(output))


class TestProbabilitiesMapping:
    """Unit tests for Probabilities mapper."""

    def test_amplitudes_are_squared(self):
        """Amplitude inputs should be converted to probabilities."""
        mapper = ML.Probabilities()
        amplitude = torch.tensor(
            [1 / math.sqrt(2) + 0j, 1j / math.sqrt(2)], dtype=torch.complex64
        )

        prob = mapper(amplitude)
        expected = torch.tensor([0.5, 0.5], dtype=torch.float32)
        assert prob.shape == (2,)
        assert torch.allclose(prob, expected, atol=1e-6)

    def test_probabilities_are_preserved(self):
        """Already probabilistic inputs should be left untouched."""
        mapper = ML.Probabilities()
        probabilities = torch.tensor([[0.2, 0.8], [0.3, 0.7]], dtype=torch.float32)

        prob = mapper(probabilities)
        assert prob.shape == probabilities.shape
        assert torch.allclose(prob, probabilities, atol=1e-6)


class TestModeExpectationsMapping:
    """Unit tests for ModeExpectations mapper."""

    def test_no_bunching_probability(self):
        """no_bunching=True should compute per-mode occupancy probability."""
        keys = [(1, 0), (0, 1), (1, 1)]
        mapper = ML.ModeExpectations(
            computation_space=ML.ComputationSpace.UNBUNCHED, keys=keys
        )

        probability_distribution = torch.tensor([0.2, 0.3, 0.5])
        expected = torch.tensor([0.7, 0.8])

        output = mapper(probability_distribution)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_expectation_counts(self):
        """no_bunching=False should compute expected photon counts per mode."""
        keys = [(2, 0), (0, 2), (1, 1)]
        mapper = ML.ModeExpectations(
            computation_space=ML.ComputationSpace.FOCK, keys=keys
        )

        probability_distribution = torch.tensor([[0.1, 0.2, 0.7]])
        expected = torch.tensor([[0.9, 1.1]])

        output = mapper(probability_distribution)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_invalid_keys(self):
        """Empty or mismatched keys should raise."""
        with pytest.raises(ValueError, match="Keys list cannot be empty"):
            ML.ModeExpectations(
                computation_space=ML.ComputationSpace.UNBUNCHED, keys=[]
            )

        with pytest.raises(ValueError, match="All keys must have the same length"):
            ML.ModeExpectations(
                computation_space=ML.ComputationSpace.UNBUNCHED,
                keys=[(1, 0), (0, 1, 0)],
            )
