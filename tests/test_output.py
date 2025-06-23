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
import merlin as ML



class TestOutputMapper:
    """Test suite for OutputMapper factory."""

    def test_linear_mapping_creation(self):
        """Test creation of linear output mapping."""
        mapping = ML.OutputMapper.create_mapping(
            ML.OutputMappingStrategy.LINEAR,
            input_size=6,
            output_size=3
        )

        assert isinstance(mapping, nn.Linear)
        assert mapping.in_features == 6
        assert mapping.out_features == 3

    def test_lexgrouping_mapping_creation(self):
        """Test creation of lexicographical grouping mapping."""
        mapping = ML.OutputMapper.create_mapping(
            ML.OutputMappingStrategy.LEXGROUPING,
            input_size=8,
            output_size=4
        )

        assert isinstance(mapping, ML.LexGroupingMapper)
        assert mapping.input_size == 8
        assert mapping.output_size == 4

    def test_modgrouping_mapping_creation(self):
        """Test creation of modulo grouping mapping."""
        mapping = ML.OutputMapper.create_mapping(
            ML.OutputMappingStrategy.MODGROUPING,
            input_size=10,
            output_size=3
        )

        assert isinstance(mapping, ML.ModGroupingMapper)
        assert mapping.input_size == 10
        assert mapping.output_size == 3

    def test_none_mapping_creation_valid(self):
        """Test creation of identity mapping with matching sizes."""
        mapping = ML.OutputMapper.create_mapping(
            ML.OutputMappingStrategy.NONE,
            input_size=5,
            output_size=5
        )

        assert isinstance(mapping, nn.Identity)

    def test_none_mapping_creation_invalid(self):
        """Test that NONE strategy with mismatched sizes raises error."""
        with pytest.raises(ValueError, match="Distribution size .* must equal output size"):
            ML.OutputMapper.create_mapping(
                ML.OutputMappingStrategy.NONE,
                input_size=5,
                output_size=3
            )

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unknown output mapping strategy"):
            ML.OutputMapper.create_mapping(
                "invalid_strategy",
                input_size=5,
                output_size=3
            )


class TestLexGroupingMapper:
    """Test suite for LexGroupingMapper."""

    def test_lexgrouping_exact_division(self):
        """Test lexicographical grouping with exact division."""
        mapper = ML.LexGroupingMapper(input_size=8, output_size=4)

        # Input: 8 elements, should group into 4 buckets of 2 each
        input_dist = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.05, 0.15, 0.05, 0.05])

        output = mapper(input_dist)

        assert output.shape == (4,)

        # Check grouping: [0.1+0.2, 0.3+0.1, 0.05+0.15, 0.05+0.05]
        expected = torch.tensor([0.3, 0.4, 0.2, 0.1])
        assert torch.allclose(output, expected, atol=1e-6)

    def test_lexgrouping_with_padding(self):
        """Test lexicographical grouping with padding."""
        mapper = ML.LexGroupingMapper(input_size=7, output_size=4)

        # Input: 7 elements, needs 1 padding to make 8, then group into 4 buckets
        input_dist = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1])

        output = mapper(input_dist)

        assert output.shape == (4,)
        assert torch.allclose(output.sum(), input_dist.sum(), atol=1e-6)

    def test_lexgrouping_batched(self):
        """Test lexicographical grouping with batched input."""
        mapper = ML.LexGroupingMapper(input_size=6, output_size=3)

        batch_size = 4
        input_dist = torch.rand(batch_size, 6)

        output = mapper(input_dist)

        assert output.shape == (batch_size, 3)

        # Each batch should preserve total probability
        for i in range(batch_size):
            assert torch.allclose(
                output[i].sum(),
                input_dist[i].sum(),
                atol=1e-6
            )

    def test_lexgrouping_no_padding_needed(self):
        """Test lexicographical grouping when no padding is needed."""
        mapper = ML.LexGroupingMapper(input_size=6, output_size=3)

        input_dist = torch.tensor([0.1, 0.2, 0.15, 0.25, 0.2, 0.1])

        output = mapper(input_dist)

        # Should group as [0.1+0.2, 0.15+0.25, 0.2+0.1]
        expected = torch.tensor([0.3, 0.4, 0.3])
        assert torch.allclose(output, expected, atol=1e-6)

    def test_lexgrouping_single_input(self):
        """Test lexicographical grouping with single input dimension."""
        mapper = ML.LexGroupingMapper(input_size=1, output_size=1)

        input_dist = torch.tensor([0.8])
        output = mapper(input_dist)

        assert output.shape == (1,)
        assert torch.allclose(output, input_dist, atol=1e-6)

    def test_lexgrouping_gradient_flow(self):
        """Test that gradients flow through lexicographical grouping."""
        mapper = ML.LexGroupingMapper(input_size=6, output_size=3)

        input_dist = torch.rand(2, 6, requires_grad=True)

        output = mapper(input_dist)
        loss = output.sum()
        loss.backward()

        assert input_dist.grad is not None
        assert not torch.allclose(input_dist.grad, torch.zeros_like(input_dist.grad))


class TestModGroupingMapper:
    """Test suite for ModGroupingMapper."""

    def test_modgrouping_basic(self):
        """Test basic modulo grouping."""
        mapper = ML.ModGroupingMapper(input_size=6, output_size=3)

        # Indices: 0,1,2,3,4,5 -> groups: 0,1,2,0,1,2
        input_dist = torch.tensor([0.1, 0.2, 0.3, 0.15, 0.1, 0.15])

        output = mapper(input_dist)

        assert output.shape == (3,)

        # Group 0: indices 0,3 -> 0.1+0.15 = 0.25
        # Group 1: indices 1,4 -> 0.2+0.1 = 0.3
        # Group 2: indices 2,5 -> 0.3+0.15 = 0.45
        expected = torch.tensor([0.25, 0.3, 0.45])
        assert torch.allclose(output, expected, atol=1e-6)

    def test_modgrouping_larger_output_than_input(self):
        """Test modulo grouping when output size > input size."""
        mapper = ML.ModGroupingMapper(input_size=3, output_size=5)

        input_dist = torch.tensor([0.3, 0.4, 0.3])

        output = mapper(input_dist)

        assert output.shape == (5,)

        # Should pad with zeros: [0.3, 0.4, 0.3, 0.0, 0.0]
        expected = torch.tensor([0.3, 0.4, 0.3, 0.0, 0.0])
        assert torch.allclose(output, expected, atol=1e-6)

    def test_modgrouping_batched(self):
        """Test modulo grouping with batched input."""
        mapper = ML.ModGroupingMapper(input_size=8, output_size=3)

        batch_size = 3
        input_dist = torch.rand(batch_size, 8)

        output = mapper(input_dist)

        assert output.shape == (batch_size, 3)

        # Check each batch individually
        for i in range(batch_size):
            # Total probability should be preserved
            assert torch.allclose(
                output[i].sum(),
                input_dist[i].sum(),
                atol=1e-6
            )

    def test_modgrouping_single_output(self):
        """Test modulo grouping with single output dimension."""
        mapper = ML.ModGroupingMapper(input_size=5, output_size=1)

        input_dist = torch.tensor([0.2, 0.3, 0.1, 0.25, 0.15])

        output = mapper(input_dist)

        assert output.shape == (1,)
        # Should sum all inputs
        assert torch.allclose(output, input_dist.sum().unsqueeze(0), atol=1e-6)

    def test_modgrouping_equal_sizes(self):
        """Test modulo grouping when input and output sizes are equal."""
        mapper = ML.ModGroupingMapper(input_size=4, output_size=4)

        input_dist = torch.tensor([0.25, 0.35, 0.2, 0.2])

        output = mapper(input_dist)

        assert output.shape == (4,)
        # Should be identity mapping when sizes are equal
        assert torch.allclose(output, input_dist, atol=1e-6)

    def test_modgrouping_gradient_flow(self):
        """Test that gradients flow through modulo grouping."""
        mapper = ML.ModGroupingMapper(input_size=6, output_size=3)

        input_dist = torch.rand(4, 6, requires_grad=True)

        output = mapper(input_dist)
        loss = output.sum()
        loss.backward()

        assert input_dist.grad is not None
        assert not torch.allclose(input_dist.grad, torch.zeros_like(input_dist.grad))


class TestOutputMappingIntegration:
    """Integration tests for output mapping with QuantumLayer."""

    def test_linear_mapping_integration(self):
        """Test linear mapping integration with QuantumLayer."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=4,
            n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=3,
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        x = torch.rand(5, 2)
        output = layer(x)

        assert output.shape == (5, 3)
        assert torch.all(torch.isfinite(output))

    def test_lexgrouping_mapping_integration(self):
        """Test lexicographical grouping integration with QuantumLayer."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=4,
            n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=4,
            output_mapping_strategy=ML.OutputMappingStrategy.LEXGROUPING
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        x = torch.rand(3, 2)
        output = layer(x)

        assert output.shape == (3, 4)
        assert torch.all(output >= 0)  # Should be non-negative (probabilities)

    def test_modgrouping_mapping_integration(self):
        """Test modulo grouping integration with QuantumLayer."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=4,
            n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=3,
            output_mapping_strategy=ML.OutputMappingStrategy.MODGROUPING
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        x = torch.rand(4, 2)
        output = layer(x)

        assert output.shape == (4, 3)
        assert torch.all(output >= 0)


    def test_mapping_gradient_flow(self):
        """Test gradient flow through different mapping strategies."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=6,
            n_photons=2
        )

        strategies = [
            ML.OutputMappingStrategy.LINEAR,
            ML.OutputMappingStrategy.LEXGROUPING,
            ML.OutputMappingStrategy.MODGROUPING
        ]

        for strategy in strategies:
            ansatz = ML.AnsatzFactory.create(
                PhotonicBackend=experiment,
                input_size=2,
                output_size=3,
                output_mapping_strategy=strategy
            )

            layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

            x = torch.rand(2, 2, requires_grad=True)
            output = layer(x)
            loss = output.sum()
            loss.backward()

            # Input should have gradients
            assert x.grad is not None, f"No gradients for strategy {strategy}"
            assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), \
                f"Zero gradients for strategy {strategy}"

    def test_mapping_output_bounds(self):
        """Test that different mappings produce reasonable output bounds."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=4,
            n_photons=2
        )

        x = torch.rand(5, 2)

        # LINEAR mapping - can have any range
        ansatz_linear = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=3,
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR
        )
        layer_linear = ML.QuantumLayer(input_size=2, ansatz=ansatz_linear)
        output_linear = layer_linear(x)
        assert torch.all(torch.isfinite(output_linear))

        # LEXGROUPING mapping - should preserve probability mass
        ansatz_lex = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=4,
            output_mapping_strategy=ML.OutputMappingStrategy.LEXGROUPING
        )
        layer_lex = ML.QuantumLayer(input_size=2, ansatz=ansatz_lex)
        output_lex = layer_lex(x)
        assert torch.all(output_lex >= 0)  # Should be non-negative

        # MODGROUPING mapping - should preserve probability mass
        ansatz_mod = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=3,
            output_mapping_strategy=ML.OutputMappingStrategy.MODGROUPING
        )
        layer_mod = ML.QuantumLayer(input_size=2, ansatz=ansatz_mod)
        output_mod = layer_mod(x)
        assert torch.all(output_mod >= 0)  # Should be non-negative

    def test_dtype_consistency_in_mappings(self):
        """Test that output mappings respect input dtypes."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=4,
            n_photons=2
        )

        for dtype in [torch.float32, torch.float64]:
            ansatz = ML.AnsatzFactory.create(
                PhotonicBackend=experiment,
                input_size=2,
                output_size=3,
                output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
                dtype=dtype
            )

            layer = ML.QuantumLayer(input_size=2, ansatz=ansatz, dtype=dtype)

            x = torch.rand(3, 2, dtype=dtype)
            output = layer(x)

            # Output should be finite and have reasonable values
            assert torch.all(torch.isfinite(output))
            assert output.shape == (3, 3)

    def test_large_dimension_mappings(self):
        """Test mappings with larger dimensions."""
        # Create a larger quantum system
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=6,
            n_photons=3
        )

        # Test with larger input/output dimensions
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=4,
            output_size=8,
            output_mapping_strategy=ML.OutputMappingStrategy.LEXGROUPING
        )

        layer = ML.QuantumLayer(input_size=4, ansatz=ansatz)

        x = torch.rand(10, 4)
        output = layer(x)

        assert output.shape == (10, 8)
        assert torch.all(torch.isfinite(output))
        assert torch.all(output >= 0)  # Non-negative for grouping strategies

    def test_edge_case_single_dimension(self):
        """Test edge case with single input/output dimensions."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL,
            n_modes=3,
            n_photons=1
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=1,
            output_size=1,
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR
        )

        layer = ML.QuantumLayer(input_size=1, ansatz=ansatz)

        x = torch.rand(5, 1)
        output = layer(x)

        assert output.shape == (5, 1)
        assert torch.all(torch.isfinite(output))

    def test_mapping_determinism(self):
        """Test that mappings are deterministic."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=4,
            n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=3,
            output_mapping_strategy=ML.OutputMappingStrategy.MODGROUPING
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        x = torch.rand(3, 2)

        # Run multiple times - should get identical results
        outputs = []
        for _ in range(5):
            with torch.no_grad():
                output = layer(x)
                outputs.append(output)

        # All outputs should be identical (deterministic)
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6), \
                f"Output {i} differs from output 0"

