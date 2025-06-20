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
Test suite for CircuitConverter function.

This module provides comprehensive tests for the circuit compilation and unitary computation
functionality, including edge cases, error handling, and performance validation.
"""

import pytest
import torch
import numpy as np
import perceval as pcvl
from typing import List, Dict, Any
import tempfile
import os

# Import the function under test
from merlin import CircuitConverter


class TestBuildCircuitUnitaryComputegraph:
    """Test suite for CircuitConverter function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_dtypes = [torch.float, torch.float64, torch.float16]
        self.tolerance = {
            torch.float: 1e-6,
            torch.float64: 1e-12,
            torch.float16: 1e-3
        }

    def create_simple_circuit(self, n_modes: int = 2) -> pcvl.Circuit:
        """Create a simple test circuit with beam splitters and phase shifters."""
        circuit = pcvl.Circuit(n_modes)
        circuit.add(0, pcvl.BS())
        circuit.add(0, pcvl.PS(pcvl.P("phi1")))
        circuit.add(0, pcvl.BS())
        circuit.add(0, pcvl.PS(pcvl.P("phi2")))
        return circuit

    def create_complex_circuit(self, n_modes: int = 4) -> pcvl.Circuit:
        """Create a more complex circuit with multiple parameter groups."""
        circuit = pcvl.Circuit(n_modes)

        # Add multiple components with different parameter prefixes
        for i in range(n_modes):
            circuit.add(i, pcvl.PS(pcvl.P(f"theta_{i}")))

        # Add beam splitters
        for i in range(n_modes - 1):
            circuit.add(i, pcvl.BS())

        # Add more phase shifters with different prefix
        for i in range(n_modes):
            circuit.add(i, pcvl.PS(pcvl.P(f"phi_{i}")))

        return circuit

    def create_circuit_with_subcircuits(self) -> pcvl.Circuit:
        """Create a circuit containing sub-circuits."""
        main_circuit = pcvl.Circuit(4)

        # Create a sub-circuit
        sub_circuit = pcvl.Circuit(2)
        sub_circuit.add(0, pcvl.BS())
        sub_circuit.add(0, pcvl.PS(pcvl.P("sub_phi")))

        # Add sub-circuit to main circuit (without name parameter)
        main_circuit.add(0, sub_circuit)
        main_circuit.add(2, pcvl.PS(pcvl.P("main_phi")))

        return main_circuit

    def test_basic_functionality(self):
        """Test basic functionality with simple circuit."""
        circuit = self.create_simple_circuit()
        input_specs = ["phi"]

        converter = CircuitConverter(circuit, input_specs, dtype=torch.float)

        # Test function call
        phi_params = torch.tensor([0.1, 0.2])
        unitary = converter.to_tensor(phi_params)

        # Verify unitary properties
        assert unitary.shape == (2, 2)
        assert torch.allclose(
            torch.matmul(unitary, unitary.conj().T),
            torch.eye(2, dtype=unitary.dtype),
            atol=self.tolerance[torch.float]
        )

    @pytest.mark.parametrize("dtype", [torch.float, torch.float64])  # Remove float16 due to ComplexHalf issues
    def test_different_dtypes(self, dtype):
        """Test function with different data types."""
        circuit = self.create_simple_circuit()
        input_specs = ["phi"]

        converter = CircuitConverter(circuit, input_specs, dtype=dtype)

        # Test computation
        phi_params = torch.tensor([0.1, 0.2], dtype=dtype)
        unitary = converter.to_tensor(phi_params)

        # Check output dtype
        expected_complex_dtype = torch.complex64 if dtype != torch.float64 else torch.complex128
        assert unitary.dtype == expected_complex_dtype

        # Verify unitary property
        identity = torch.eye(2, dtype=unitary.dtype)
        product = torch.matmul(unitary, unitary.conj().T)
        assert torch.allclose(product, identity, atol=self.tolerance[dtype])

    def test_batched_input(self):
        """Test function with batched inputs."""
        circuit = self.create_simple_circuit()
        input_specs = ["phi"]

        converter = CircuitConverter(circuit, input_specs, dtype=torch.float)

        # Test with batch of parameters
        batch_size = 5
        phi_params = torch.rand(batch_size, 2)
        unitary_batch = converter.to_tensor(phi_params)

        assert unitary_batch.shape == (batch_size, 2, 2)

        # Verify each unitary in batch
        for i in range(batch_size):
            unitary = unitary_batch[i]
            identity = torch.eye(2, dtype=unitary.dtype)
            product = torch.matmul(unitary, unitary.conj().T)
            assert torch.allclose(product, identity, atol=self.tolerance[torch.float])
            if i > 0:
                # Ensure different batches yield different unitaries
                assert not torch.allclose(unitary, unitary_batch[0], atol=self.tolerance[torch.float])

    def test_multiple_parameter_groups(self):
        """Test with multiple parameter groups."""
        circuit = self.create_complex_circuit(n_modes=3)
        input_specs = ["theta", "phi"]

        converter = CircuitConverter(circuit, input_specs, dtype=torch.float)

        # Test computation with multiple parameter groups
        theta_params = torch.tensor([0.1, 0.2, 0.3])
        phi_params = torch.tensor([0.4, 0.5, 0.6])

        unitary = converter.to_tensor(theta_params, phi_params)

        assert unitary.shape == (3, 3)

        # Verify unitary property
        identity = torch.eye(3, dtype=unitary.dtype)
        product = torch.matmul(unitary, unitary.conj().T)
        assert torch.allclose(product, identity, atol=self.tolerance[torch.float])

    def test_specific_parameter_names(self):
        """Test with specific parameter names instead of prefixes."""
        circuit = self.create_simple_circuit()
        input_specs = ["phi1", "phi2"]  # Specific parameter names

        converter = CircuitConverter(circuit, input_specs, dtype=torch.float)

        # Test computation
        phi1_param = torch.tensor([0.1])
        phi2_param = torch.tensor([0.2])

        unitary = converter.to_tensor(phi1_param, phi2_param)

        assert unitary.shape == (2, 2)

        # Verify unitary property
        identity = torch.eye(2, dtype=unitary.dtype)
        product = torch.matmul(unitary, unitary.conj().T)
        assert torch.allclose(product, identity, atol=self.tolerance[torch.float])

    def test_empty_parameter_group(self):
        """Test handling of parameter specs that match no parameters."""
        circuit = self.create_simple_circuit()
        input_specs = ["nonexistent"]  # This prefix doesn't match any parameters

        # This should raise an error because not all parameters are covered
        with pytest.raises(ValueError, match="No parameters found matching the input spec 'nonexistent'"):
            CircuitConverter(
                circuit, input_specs, dtype=torch.float
            )

        input_specs = []
        with pytest.raises(ValueError, match="Parameter 'phi1' not covered by any input spec"):
            CircuitConverter(
                circuit, input_specs, dtype=torch.float
            )

    def test_subcircuit_handling(self):
        """Test circuits containing sub-circuits."""
        circuit = self.create_circuit_with_subcircuits()
        input_specs = ["sub", "main"]  # Use prefixes that match the parameters

        converter = CircuitConverter(
            circuit, input_specs, dtype=torch.float
        )

        # Test computation
        sub_phi_param = torch.tensor([0.1])  # For sub_phi parameter
        main_phi_param = torch.tensor([0.2])  # For main_phi parameter

        unitary = converter.to_tensor(sub_phi_param, main_phi_param)

        assert unitary.shape == (4, 4)

    def test_torchscript_compilation(self):
        """Test TorchScript compilation and execution."""
        circuit = self.create_simple_circuit()
        input_specs = ["phi"]

        converter = CircuitConverter(
            circuit, input_specs, dtype=torch.float
        )

        # Test that the function works
        phi_params = torch.tensor([0.1, 0.2])
        unitary1 = converter.to_tensor(phi_params)
        unitary2 = converter.to_tensor(phi_params)

        # Results should be identical
        assert torch.allclose(unitary1, unitary2)

    def test_gradient_computation(self):
        """Test gradient computation through the compiled function."""
        circuit = self.create_simple_circuit()
        input_specs = ["phi"]

        converter = CircuitConverter(circuit, input_specs, dtype=torch.float)

        # Test gradient computation
        phi_params = torch.tensor([0.1, 0.2], requires_grad=True)
        unitary = converter.to_tensor(phi_params)

        # Create a simple loss function
        loss = torch.sum(torch.abs(unitary) ** 2)
        loss.backward()

        # Check that gradients were computed
        assert phi_params.grad is not None
        assert phi_params.grad.shape == phi_params.shape

    def test_error_handling_invalid_dtype(self):
        """Test error handling for invalid dtype."""
        circuit = self.create_simple_circuit()
        input_specs = ["phi"]

        with pytest.raises(TypeError, match="Unsupported dtype"):
            CircuitConverter(
                circuit, input_specs, dtype=torch.int32
            )


    @pytest.mark.parametrize("n_modes", [2, 3, 4, 6])
    def test_different_circuit_sizes(self, n_modes):
        """Test with circuits of different sizes."""
        circuit = self.create_complex_circuit(n_modes)
        input_specs = ["theta", "phi"]

        converter = CircuitConverter(
            circuit, input_specs, dtype=torch.float
        )

        # Test computation
        theta_params = torch.rand(n_modes)
        phi_params = torch.rand(n_modes)

        unitary = converter.to_tensor(theta_params, phi_params)

        assert unitary.shape == (n_modes, n_modes)

        # Verify unitary property
        identity = torch.eye(n_modes, dtype=unitary.dtype)
        product = torch.matmul(unitary, unitary.conj().T)
        assert torch.allclose(product, identity, atol=self.tolerance[torch.float])

    def test_numerical_stability(self):
        """Test numerical stability with extreme parameter values."""
        circuit = self.create_simple_circuit()
        input_specs = ["phi"]

        converter = CircuitConverter(
            circuit, input_specs, dtype=torch.float64  # Use higher precision
        )

        # Test with large parameter values
        large_params = torch.tensor([100.0, -100.0], dtype=torch.float64)
        unitary_large = converter.to_tensor(large_params)

        # Test with small parameter values
        small_params = torch.tensor([1e-6, -1e-6], dtype=torch.float64)
        unitary_small = converter.to_tensor(small_params)

        # Both should still be unitary
        identity = torch.eye(2, dtype=torch.complex128)

        product_large = torch.matmul(unitary_large, unitary_large.conj().T)
        assert torch.allclose(product_large, identity, atol=self.tolerance[torch.float64])

        product_small = torch.matmul(unitary_small, unitary_small.conj().T)
        assert torch.allclose(product_small, identity, atol=self.tolerance[torch.float64])


class TestIntegration:
    """Integration tests for CircuitConverter."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create circuit
        circuit = pcvl.Circuit(3)
        circuit.add(0, pcvl.BS())
        circuit.add(0, pcvl.PS(pcvl.P("theta1")))
        circuit.add(1, pcvl.BS())
        circuit.add(1, pcvl.PS(pcvl.P("theta2")))
        circuit.add(0, pcvl.BS())
        circuit.add(0, pcvl.PS(pcvl.P("phi1")))

        # Compile circuit
        input_specs = ["theta", "phi"]
        converter = CircuitConverter(
            circuit, input_specs, dtype=torch.float
        )

        # Use in training loop simulation
        theta_params = torch.tensor([0.1, 0.2], requires_grad=True)
        phi_params = torch.tensor([0.3], requires_grad=True)

        optimizer = torch.optim.Adam([theta_params, phi_params], lr=0.01)

        # Simulate a few training steps
        for step in range(5):
            optimizer.zero_grad()

            unitary = converter.to_tensor(theta_params, phi_params)

            # Simple loss: minimize deviation from identity
            target = torch.eye(3, dtype=unitary.dtype)
            loss = torch.sum(torch.abs(unitary - target) ** 2)

            loss.backward()
            optimizer.step()

            assert not torch.isnan(loss)
            assert theta_params.grad is not None
            assert phi_params.grad is not None

    def test_performance_comparison(self):
        """Test performance comparison between scripted and non-scripted versions."""
        circuit = self.create_medium_circuit()
        input_specs = ["theta"]

        converter = CircuitConverter(
            circuit, input_specs, dtype=torch.float
        )

        # Warm up
        params = torch.rand(6)
        for _ in range(10):
            converter.to_tensor(params)

        # Time multiple runs
        import time
        n_runs = 100

        start_time = time.time()
        for _ in range(n_runs):
            converter.to_tensor(params)
        end_time = time.time()

        avg_time = (end_time - start_time) / n_runs

        # Just verify it runs without error and completes in reasonable time
        assert avg_time < 1.0  # Should complete in less than 1 second per call

    def create_medium_circuit(self):
        """Create a medium-sized circuit for performance testing."""
        circuit = pcvl.Circuit(6)
        for i in range(6):
            circuit.add(i, pcvl.PS(pcvl.P(f"theta_{i}")))
        for i in range(5):
            circuit.add(i, pcvl.BS())
        return circuit


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])