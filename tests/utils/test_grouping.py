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

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import merlin as ML
from merlin import LexGrouping, ModGrouping


def test_groupings_creation():
    """Test creation of default grouping mappings."""
    lex_grouping = LexGrouping(input_size=8, output_size=4)
    mod_grouping = ModGrouping(input_size=8, output_size=4)
    assert isinstance(lex_grouping, LexGrouping)
    assert lex_grouping.input_size == 8
    assert lex_grouping.output_size == 4
    assert isinstance(mod_grouping, ModGrouping)
    assert mod_grouping.input_size == 8
    assert mod_grouping.output_size == 4


class TestLexGrouping:
    """Test suite for LexGrouping."""

    def test_lexgrouping_input_size_mismatch(self):
        """LexGrouping should raise when the input size does not match."""
        lex_mapper = LexGrouping(input_size=4, output_size=2)

        with pytest.raises(ValueError, match="Input tensor's last dimension"):
            lex_mapper(torch.rand(3))

        with pytest.raises(ValueError, match="Input tensor's last dimension"):
            lex_mapper(torch.rand(2, 3))

    def test_lexgrouping_exact_division(self):
        """Test lexicographical grouping with exact division."""
        lex_mapper = LexGrouping(input_size=8, output_size=4)

        # Input: 8 elements, should group into 4 buckets of 2 each
        input_dist = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.05, 0.15, 0.05, 0.05])

        output = lex_mapper(input_dist)

        assert output.shape == (4,)

        # Check grouping: [0.1+0.2, 0.3+0.1, 0.05+0.15, 0.05+0.05]
        expected = torch.tensor([0.3, 0.4, 0.2, 0.1])
        assert torch.allclose(output, expected, atol=1e-6)

    def test_lexgrouping_with_padding(self):
        """Test lexicographical grouping with padding."""
        lex_mapper = LexGrouping(input_size=7, output_size=4)

        # Input: 7 elements, needs 1 padding to make 8, then group into 4 buckets
        input_dist = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1])

        output = lex_mapper(input_dist)

        assert output.shape == (4,)
        assert torch.allclose(output.sum(), input_dist.sum(), atol=1e-6)

    def test_lexgrouping_batched(self):
        """Test lexicographical grouping with batched input."""
        lex_mapper = LexGrouping(input_size=6, output_size=3)

        batch_size = 4
        # Equivalent of counts as input distribution
        input_dist = torch.rand(batch_size, 6)
        # Grouped
        output = lex_mapper(input_dist)

        assert output.shape == (batch_size, 3)

    def test_lexgrouping_no_padding_needed(self):
        """Test lexicographical grouping when no padding is needed."""
        lex_mapper = LexGrouping(input_size=6, output_size=3)

        input_dist = torch.tensor([0.1, 0.2, 0.15, 0.25, 0.2, 0.1])

        output = lex_mapper(input_dist)

        # Should group as [0.1+0.2, 0.15+0.25, 0.2+0.1]
        expected = torch.tensor([0.3, 0.4, 0.3])
        assert torch.allclose(output, expected, atol=1e-6)

    def test_lexgrouping_single_input(self):
        """Test lexicographical grouping with single input dimension."""
        lex_mapper = LexGrouping(input_size=1, output_size=1)

        # Equivalent of count as input distribution
        input_dist = torch.tensor([0.8])
        # (not grouped here since single value)
        output = lex_mapper(input_dist)

        assert output.shape == (1,)

    def test_lexgrouping_gradient_flow(self):
        """Test that gradients flow through lexicographical grouping."""
        lex_mapper = LexGrouping(input_size=6, output_size=3)

        input_dist = torch.rand(2, 6, requires_grad=True)

        # Use MSE loss instead of sum for better gradient flow
        output = lex_mapper(input_dist)
        target = torch.ones_like(output)
        loss = F.mse_loss(output, target)
        loss.backward()

        assert input_dist.grad is not None
        assert not torch.allclose(input_dist.grad, torch.zeros_like(input_dist.grad))

    def test_lexgrouping_with_padding_batched(self):
        """Batched inputs with padding should preserve mass per batch."""
        lex_mapper = LexGrouping(input_size=5, output_size=4)

        input_dist = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ])

        output = lex_mapper(input_dist)
        expected = torch.tensor([[3.0, 7.0, 5.0, 0.0], [9.0, 5.0, 1.0, 0.0]])

        assert output.shape == (2, 4)
        assert torch.allclose(output, expected, atol=1e-6)


class TestModGrouping:
    """Test suite for ModGrouping."""

    def test_modgrouping_input_size_mismatch(self):
        """ModGrouping should raise when the input size does not match."""
        mod_mapper = ModGrouping(input_size=5, output_size=2)

        with pytest.raises(ValueError, match="Input tensor's last dimension"):
            mod_mapper(torch.rand(4))

        with pytest.raises(ValueError, match="Input tensor's last dimension"):
            mod_mapper(torch.rand(2, 4))

    def test_modgrouping_basic(self):
        """Test basic modulo grouping."""
        mod_mapper = ModGrouping(input_size=6, output_size=3)

        # Indices: 0,1,2,3,4,5 -> groups: 0,1,2,0,1,2
        input_dist = torch.tensor([0.1, 0.2, 0.3, 0.15, 0.1, 0.15])

        output = mod_mapper(input_dist)

        assert output.shape == (3,)

        # Group 0: indices 0,3 -> 0.1+0.15 = 0.25
        # Group 1: indices 1,4 -> 0.2+0.1 = 0.3
        # Group 2: indices 2,5 -> 0.3+0.15 = 0.45
        expected = torch.tensor([0.25, 0.3, 0.45])
        assert torch.allclose(output, expected, atol=1e-6)

    def test_modgrouping_larger_output_than_input(self):
        """Test modulo grouping when output size > input size."""
        mod_mapper = ModGrouping(input_size=3, output_size=5)

        input_dist = torch.tensor([0.3, 0.4, 0.3])

        output = mod_mapper(input_dist)

        assert output.shape == (5,)

        # Should pad with zeros: [0.3, 0.4, 0.3, 0.0, 0.0]
        expected = torch.tensor([0.3, 0.4, 0.3, 0.0, 0.0])
        assert torch.allclose(output, expected, atol=1e-6)

    def test_modgrouping_larger_output_than_input_batched(self):
        """Batched modulo grouping should pad correctly."""
        mod_mapper = ModGrouping(input_size=2, output_size=4)

        input_dist = torch.tensor([[0.2, 0.8], [0.6, 0.4]])

        output = mod_mapper(input_dist)
        expected = torch.tensor([[0.2, 0.8, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0]])

        assert output.shape == (2, 4)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_modgrouping_batched(self):
        """Test modulo grouping with batched input."""
        mod_mapper = ModGrouping(input_size=8, output_size=3)

        batch_size = 3
        # Equivalent of counts as input distribution
        input_dist = torch.rand(batch_size, 8)
        # Grouped
        output = mod_mapper(input_dist)

        assert output.shape == (batch_size, 3)

    def test_modgrouping_single_output(self):
        """Test modulo grouping with single output dimension."""
        mod_mapper = ModGrouping(input_size=5, output_size=1)

        input_dist = torch.tensor([0.2, 0.3, 0.1, 0.25, 0.15])

        output = mod_mapper(input_dist)

        assert output.shape == (1,)
        # Should sum all inputs
        assert torch.allclose(output, input_dist.sum().unsqueeze(0), atol=1e-6)

    def test_modgrouping_equal_sizes(self):
        """Test modulo grouping when input and output sizes are equal."""
        mod_mapper = ModGrouping(input_size=4, output_size=4)

        input_dist = torch.tensor([0.25, 0.35, 0.2, 0.2])

        output = mod_mapper(input_dist)

        assert output.shape == (4,)
        # Should be identity mapping when sizes are equal
        assert torch.allclose(output, input_dist, atol=1e-6)

    def test_modgrouping_gradient_flow(self):
        """Test that gradients flow through modulo grouping."""
        mod_mapper = ModGrouping(input_size=6, output_size=3)

        input_dist = torch.rand(4, 6, requires_grad=True)

        output = mod_mapper(input_dist)
        target = torch.ones(4, 3)
        loss = (target - output).pow(2).sum()
        loss.backward()

        assert input_dist.grad is not None
        assert not torch.allclose(input_dist.grad, torch.zeros_like(input_dist.grad))


def test_lexgrouping_mapping_integration():
    """Test lexicographical grouping integration with QuantumLayer."""

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
    model = nn.Sequential(layer, LexGrouping(layer.output_size, 4))

    x = torch.rand(3, 2)
    output = model(x)

    assert output.shape == (3, 4)
    assert torch.all(output >= 0)  # Should be non-negative (probabilities)


def test_modgrouping_mapping_integration():
    """Test modulo grouping integration with QuantumLayer."""

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
    model = nn.Sequential(layer, ModGrouping(layer.output_size, 3))

    x = torch.rand(4, 2)
    output = model(x)

    assert output.shape == (4, 3)
    assert torch.all(output >= 0)


def test_mapping_gradient_flow():
    """Test gradient flow through different mapping strategies."""

    builder = ML.CircuitBuilder(n_modes=6)
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(modes=[0, 1], name="input")
    builder.add_entangling_layer(trainable=True, name="U2")

    strategies = [LexGrouping, ModGrouping]

    for grouping_policy in strategies:
        layer = ML.QuantumLayer(
            input_size=2,
            n_photons=2,
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )
        model = torch.nn.Sequential(layer, grouping_policy(layer.output_size, 3))

        x = torch.rand(2, 2, requires_grad=True)
        output = model(x)

        # Use MSE loss instead of sum for better gradient flow
        target = torch.ones_like(output)
        loss = F.mse_loss(output, target)
        loss.backward()

        # Input should have gradients
        assert x.grad is not None, f"No gradients for strategy {grouping_policy}"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), (
            f"Zero gradients for strategy {grouping_policy}"
        )


def test_mapping_output_bounds():
    """Test that different mappings produce reasonable output bounds."""

    builder = ML.CircuitBuilder(n_modes=4)
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(modes=[0, 1], name="input")
    builder.add_entangling_layer(trainable=True, name="U2")

    x = torch.rand(4, 2)

    # LEXGROUPING mapping - should preserve probability mass
    layer_lex = ML.QuantumLayer(
        input_size=2,
        n_photons=2,
        builder=builder,
        measurement_strategy=ML.MeasurementStrategy.probs(),
    )
    model_lex = nn.Sequential(layer_lex, LexGrouping(layer_lex.output_size, 3))
    output_lex = model_lex(x)
    assert torch.all(output_lex >= 0)  # Should be non-negative

    # MODGROUPING mapping - should preserve probability mass
    layer_mod = ML.QuantumLayer(
        input_size=2,
        n_photons=2,
        builder=builder,
        measurement_strategy=ML.MeasurementStrategy.probs(),
    )
    model_mod = nn.Sequential(layer_mod, ModGrouping(layer_mod.output_size, 3))
    output_mod = model_mod(x)
    assert torch.all(output_mod >= 0)  # Should be non-negative


def test_large_dimension_mappings():
    """Test mappings with larger dimensions."""
    # Create a larger quantum system

    builder = ML.CircuitBuilder(n_modes=6)
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(modes=[0, 1, 2, 3], name="input")
    builder.add_entangling_layer(trainable=True, name="U2")

    layer = ML.QuantumLayer(
        input_size=4,
        n_photons=3,
        builder=builder,
        measurement_strategy=ML.MeasurementStrategy.probs(),
    )
    model = nn.Sequential(layer, LexGrouping(layer.output_size, 8))

    x = torch.rand(10, 4)
    output = model(x)

    assert output.shape == (10, 8)
    assert torch.all(torch.isfinite(output))
    assert torch.all(output >= 0)  # Non-negative for grouping strategies


def test_mapping_determinism():
    """Test that mappings are deterministic."""

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
    model = nn.Sequential(layer, ModGrouping(layer.output_size, 3))

    x = torch.rand(3, 2)

    # Run multiple times - should get identical results
    outputs = []
    for _ in range(5):
        with torch.no_grad():
            output = model(x)
            outputs.append(output)

    # All outputs should be identical (deterministic)
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i], atol=1e-6), (
            f"Output {i} differs from output 0"
        )
