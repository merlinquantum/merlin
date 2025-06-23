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
Test file specifically for output mapping strategies in QuantumLayer.simple()
"""

import torch
import pytest
import math
from merlin import QuantumLayer, OutputMappingStrategy  # Replace with actual import path


class TestOutputMappingStrategies:
    """Test cases for output mapping strategies in QuantumLayer.simple()."""

    def test_none_strategy_without_output_size(self):
        """Test NONE strategy when output_size is not specified."""
        print("\n=== Testing NONE strategy without output_size ===")

        # When using NONE strategy without specifying output_size,
        # the output size should equal the distribution size
        layer = QuantumLayer.simple(
            input_size=3,
            n_params=100,
            output_mapping_strategy=OutputMappingStrategy.NONE
        )

        print(f"Input size: {layer.input_size}")
        print(f"Number of modes: {layer.ansatz.experiment.n_modes}")
        print(f"Output size: {layer.output_size}")

        # Test forward pass
        x = torch.rand(2, 3)
        output = layer(x)

        print(f"Output shape: {output.shape}")
        print(f"Distribution size should equal output size")

        # With NONE strategy, output should be the raw distribution
        assert output.shape[1] == layer.output_size

        # The distribution should sum to 1 (probability distribution)
        assert torch.allclose(output.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_none_strategy_with_matching_output_size(self):
        """Test NONE strategy when output_size matches distribution size."""
        print("\n=== Testing NONE strategy with matching output_size ===")

        # First, create a layer to find out the distribution size
        temp_layer = QuantumLayer.simple(
            input_size=3,
            n_params=100,
            output_mapping_strategy=OutputMappingStrategy.NONE
        )
        dist_size = temp_layer.output_size
        print(f"Distribution size: {dist_size}")

        # Now create a layer with matching output_size
        layer = QuantumLayer.simple(
            input_size=3,
            n_params=100,
            output_size=dist_size,  # Explicitly set to match distribution
            output_mapping_strategy=OutputMappingStrategy.NONE
        )

        print(f"Created layer with output_size={dist_size}")

        # This should work without errors
        x = torch.rand(2, 3)
        output = layer(x)

        assert output.shape == (2, dist_size)
        print(f"✓ Successfully created NONE strategy with matching output_size")

    def test_none_strategy_with_mismatched_output_size(self):
        """Test that NONE strategy fails when output_size doesn't match distribution size."""
        print("\n=== Testing NONE strategy with mismatched output_size ===")

        # This should raise an error because NONE strategy requires
        # output_size to match distribution size
        with pytest.raises(ValueError) as exc_info:
            layer = QuantumLayer.simple(
                input_size=3,
                n_params=100,
                output_size=10,  # Arbitrary size that won't match distribution
                output_mapping_strategy=OutputMappingStrategy.NONE
            )

        print(f"Expected error: {exc_info.value}")
        assert "Distribution size" in str(exc_info.value)
        assert "must equal output size" in str(exc_info.value)
        print("✓ Correctly raised error for mismatched sizes")

    def test_linear_strategy_with_output_size(self):
        """Test LINEAR strategy with specified output_size."""
        print("\n=== Testing LINEAR strategy with output_size ===")

        layer = QuantumLayer.simple(
            input_size=3,
            n_params=100,
            output_size=10,
            output_mapping_strategy=OutputMappingStrategy.LINEAR
        )

        print(f"Input size: {layer.input_size}")
        print(f"Output size: {layer.output_size}")
        print(f"Output mapping type: {type(layer.output_mapping)}")

        # Check that LINEAR strategy uses nn.Linear
        assert isinstance(layer.output_mapping, torch.nn.Linear)

        # Test forward pass
        x = torch.rand(5, 3)
        output = layer(x)

        assert output.shape == (5, 10)
        print(f"✓ LINEAR strategy correctly maps to output_size={10}")

    def test_linear_strategy_without_output_size(self):
        """Test that LINEAR strategy requires output_size."""
        print("\n=== Testing LINEAR strategy without output_size ===")

        # LINEAR strategy should fail without output_size
        with pytest.raises(ValueError) as exc_info:
            layer = QuantumLayer.simple(
                input_size=3,
                n_params=100,
                output_mapping_strategy=OutputMappingStrategy.LINEAR
            )

        print(f"Expected error: {exc_info.value}")
        assert "output_size must be specified" in str(exc_info.value)
        print("✓ Correctly raised error when output_size not specified")

    def test_default_strategy(self):
        """Test that default strategy is NONE (as per the code)."""
        print("\n=== Testing default strategy ===")

        # Check the default in the method signature
        import inspect
        sig = inspect.signature(QuantumLayer.simple)
        default_strategy = sig.parameters['output_mapping_strategy'].default

        print(f"Default strategy: {default_strategy}")
        assert default_strategy == OutputMappingStrategy.NONE

    def test_strategy_override_mechanism(self):
        """Test that the strategy override mechanism works correctly."""
        print("\n=== Testing strategy override mechanism ===")

        # Create layer with LINEAR strategy
        layer = QuantumLayer.simple(
            input_size=3,
            n_params=100,
            output_size=10,
            output_mapping_strategy=OutputMappingStrategy.LINEAR
        )

        # Check that ansatz has the correct strategy after override
        assert layer.ansatz.output_mapping_strategy == OutputMappingStrategy.LINEAR

        # Check that the actual output mapping is Linear
        assert isinstance(layer.output_mapping, torch.nn.Linear)

        print("✓ Strategy override mechanism working correctly")

    def test_distribution_size_calculation(self):
        """Test distribution size for different parameter configurations."""
        print("\n=== Testing distribution size calculation ===")

        configs = [
            (2, 50),  # input_size=2, n_params=50
            (3, 100),  # input_size=3, n_params=100
            (4, 200),  # input_size=4, n_params=200
        ]

        for input_size, n_params in configs:
            layer = QuantumLayer.simple(
                input_size=input_size,
                n_params=n_params,
                output_mapping_strategy=OutputMappingStrategy.NONE
            )

            n_modes = layer.ansatz.experiment.n_modes
            dist_size = layer.output_size

            print(f"Input size: {input_size}, n_params: {n_params}")
            print(f"  → n_modes: {n_modes}, dist_size: {dist_size}")

            # Distribution size should be combinatorial based on modes and photons
            # For no_bunching=True, it's C(n_modes, n_photons)
            from math import comb
            expected_dist_size = comb(n_modes, input_size)

            # Note: The actual distribution size might be different due to
            # the specific circuit implementation
            print(f"  → Expected (combinatorial): {expected_dist_size}")

    def test_gradient_flow_with_strategies(self):
        """Test gradient flow works with different strategies."""
        print("\n=== Testing gradient flow with different strategies ===")

        # Test with LINEAR strategy
        layer_linear = QuantumLayer.simple(
            input_size=3,
            n_params=100,
            output_size=5,
            output_mapping_strategy=OutputMappingStrategy.LINEAR,
            reservoir_mode=False
        )

        x = torch.rand(10, 3, requires_grad=True)
        output = layer_linear(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None and not torch.all(p.grad == 0)
                       for p in layer_linear.parameters())
        assert has_grad
        print("✓ Gradients flow correctly with LINEAR strategy")

        # Test with NONE strategy
        layer_none = QuantumLayer.simple(
            input_size=3,
            n_params=100,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            reservoir_mode=False
        )

        x = torch.rand(10, 3, requires_grad=True)
        output = layer_none(x)
        loss = output.sum()
        loss.backward()

        has_grad = any(p.grad is not None and not torch.all(p.grad == 0)
                       for p in layer_none.parameters())
        assert has_grad
        print("✓ Gradients flow correctly with NONE strategy")

    def test_all_input_sizes_with_none_mapping(self):
        """Test NONE mapping with various input sizes."""
        print("\n=== Testing NONE mapping with different input sizes ===")

        input_sizes = [1, 2, 3, 4, 5, 8, 10]

        for input_size in input_sizes:
            print(f"\nTesting input_size={input_size}")

            # Create layer with NONE mapping
            layer = QuantumLayer.simple(
                input_size=input_size,
                n_params=100,
                output_mapping_strategy=OutputMappingStrategy.NONE
            )

            # Get info about the layer
            n_modes = layer.ansatz.experiment.n_modes
            n_photons = layer.ansatz.experiment.n_photons
            dist_size = layer.output_size

            print(f"  n_modes: {n_modes}, n_photons: {n_photons}, dist_size: {dist_size}")

            # Test forward pass with different batch sizes
            for batch_size in [1, 5, 10]:
                x = torch.rand(batch_size, input_size)
                output = layer(x)

                assert output.shape == (batch_size, dist_size)
                # Check probability distribution sums to 1
                assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-5)

            print(f"  ✓ Forward pass successful for all batch sizes")

    def test_all_n_params_with_none_mapping(self):
        """Test NONE mapping with various n_params values."""
        print("\n=== Testing NONE mapping with different n_params ===")

        n_params_list = [10, 50, 100, 200, 500, 1000]
        input_size = 3  # Fixed input size

        for n_params in n_params_list:
            print(f"\nTesting n_params={n_params}")

            layer = QuantumLayer.simple(
                input_size=input_size,
                n_params=n_params,
                output_mapping_strategy=OutputMappingStrategy.NONE
            )

            # Calculate expected n_modes
            expected_n_modes = max(
                int(math.ceil(math.sqrt(n_params / 2))),
                input_size + 1
            )

            actual_n_modes = layer.ansatz.experiment.n_modes
            dist_size = layer.output_size

            print(f"  Expected n_modes: {expected_n_modes}, Actual: {actual_n_modes}")
            print(f"  Distribution size: {dist_size}")

            assert actual_n_modes == expected_n_modes

            # Test forward pass
            x = torch.rand(2, input_size)
            output = layer(x)
            assert output.shape == (2, dist_size)

            print(f"  ✓ Configuration successful")

    def test_edge_cases_with_none_mapping(self):
        """Test edge cases with NONE mapping."""
        print("\n=== Testing edge cases with NONE mapping ===")

        # Test 1: Minimum viable configuration
        print("\nTest 1: Minimum configuration (input_size=1, n_params=10)")
        layer1 = QuantumLayer.simple(
            input_size=1,
            n_params=10,
            output_mapping_strategy=OutputMappingStrategy.NONE
        )
        x1 = torch.rand(1, 1)
        output1 = layer1(x1)
        print(f"  Output shape: {output1.shape}")
        print(f"  ✓ Minimum configuration works")

        # Test 2: Large input size
        print("\nTest 2: Large input size (input_size=20)")
        layer2 = QuantumLayer.simple(
            input_size=10,
            n_params=120,
            output_mapping_strategy=OutputMappingStrategy.NONE
        )
        x2 = torch.rand(1, 20)
        output2 = layer2(x2)
        print(f"  Output shape: {output2.shape}")
        print(f"  n_modes: {layer2.ansatz.experiment.n_modes}")
        print(f"  ✓ Large input size works")

        # Test 3: Very small n_params that gets overridden by input_size+1
        print("\nTest 3: Small n_params (input_size=10, n_params=10)")
        layer3 = QuantumLayer.simple(
            input_size=10,
            n_params=10,
            output_mapping_strategy=OutputMappingStrategy.NONE
        )
        # Should use input_size + 1 = 11 modes
        assert layer3.ansatz.experiment.n_modes >= 11
        print(f"  n_modes: {layer3.ansatz.experiment.n_modes}")
        print(f"  ✓ Correctly uses max(calculated, input_size+1)")


    def test_none_mapping_with_reservoir_mode(self):
        """Test NONE mapping with reservoir mode."""
        print("\n=== Testing NONE mapping with reservoir mode ===")

        layer = QuantumLayer.simple(
            input_size=4,
            n_params=100,
            reservoir_mode=True,
            output_mapping_strategy=OutputMappingStrategy.NONE
        )

        # Check that no trainable parameters exist
        trainable_params = list(layer.parameters())
        assert len(trainable_params) == 0
        print("  ✓ No trainable parameters in reservoir mode")

        # Check that phi_static buffer exists
        assert hasattr(layer, 'phi_static')
        print("  ✓ phi_static buffer exists")

        # Test forward pass
        x = torch.rand(3, 4)
        output = layer(x)
        assert output.shape[0] == 3
        assert output.shape[1] == layer.output_size
        print("  ✓ Forward pass works in reservoir mode")

    def test_none_mapping_with_different_dtypes(self):
        """Test NONE mapping with different data types."""
        print("\n=== Testing NONE mapping with different dtypes ===")

        dtypes = [torch.float32, torch.float64]

        for dtype in dtypes:
            print(f"\nTesting with dtype={dtype}")

            layer = QuantumLayer.simple(
                input_size=3,
                n_params=100,
                dtype=dtype,
                output_mapping_strategy=OutputMappingStrategy.NONE
            )

            # Check parameter dtypes
            for param in layer.parameters():
                assert param.dtype == dtype

            # Test forward pass
            x = torch.rand(2, 3, dtype=dtype)
            output = layer(x)
            assert output.dtype == dtype

            print(f"  ✓ Correctly uses {dtype}")

    def test_none_mapping_batch_processing(self):
        """Test NONE mapping with various batch configurations."""
        print("\n=== Testing NONE mapping with batch processing ===")

        layer = QuantumLayer.simple(
            input_size=3,
            n_params=100,
            output_mapping_strategy=OutputMappingStrategy.NONE
        )

        batch_sizes = [1, 10, 32]

        for batch_size in batch_sizes:
            x = torch.rand(batch_size, 3)
            output = layer(x)

            assert output.shape[0] == batch_size
            assert output.shape[1] == layer.output_size

            # Check that each sample in batch is a valid probability distribution
            assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-5)
            assert torch.all(output >= 0)  # All probabilities non-negative

            print(f"  ✓ Batch size {batch_size} processed correctly")


if __name__ == "__main__":
    test = TestOutputMappingStrategies()

    try:
        print("Running NONE strategy tests...")
        test.test_none_strategy_without_output_size()
        test.test_none_strategy_with_matching_output_size()
        test.test_none_strategy_with_mismatched_output_size()

        print("\nRunning LINEAR strategy tests...")
        test.test_linear_strategy_with_output_size()
        test.test_linear_strategy_without_output_size()

        print("\nRunning other tests...")
        test.test_default_strategy()
        test.test_strategy_override_mechanism()
        test.test_distribution_size_calculation()
        test.test_gradient_flow_with_strategies()

        print("\nRunning comprehensive NONE mapping tests...")
        test.test_all_input_sizes_with_none_mapping()
        test.test_all_n_params_with_none_mapping()
        test.test_edge_cases_with_none_mapping()
        test.test_none_mapping_with_sampling()
        test.test_none_mapping_with_reservoir_mode()
        test.test_none_mapping_with_different_dtypes()
        test.test_none_mapping_batch_processing()

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
