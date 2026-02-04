"""
Low-level tests for the compute_ebs_simultaneously kernel and related plumbing.

These scenarios concentrate on:
* Numerical equivalence between the batch EBS kernel and the single-state paths.
* Gradient propagation and dtype handling for the ComputationProcess internals.
* Validation of superposition dimensions under different computation spaces.

By keeping these checks isolated we can evolve the high-level QuantumLayer API
without losing coverage of the specialised batch kernel behaviours.
"""

import math

import perceval as pcvl
import pytest
import torch

from merlin import QuantumLayer
from merlin.core.computation_space import ComputationSpace
from merlin.core.process import ComputationProcess
from merlin.measurement.strategies import MeasurementStrategy


def classical_method_ebs(layer, input_state):
    """Classical method for computing superposition states using individual state computations."""
    dtype = (
        layer.computation_process.simulation_graph.prev_amplitudes.dtype
        if layer.computation_process.simulation_graph.prev_amplitudes is not None
        else torch.complex128
    )
    device = layer.device or torch.device("cpu")
    output_classical = torch.zeros(1, layer.output_size, dtype=dtype, device=device)

    for key, value in input_state.items():
        layer.computation_process.input_state = key
        _ = layer()
        coeff = value.to(device=output_classical.device, dtype=dtype)
        prev_amplitudes = layer.computation_process.simulation_graph.prev_amplitudes
        if prev_amplitudes.device != output_classical.device:
            prev_amplitudes = prev_amplitudes.to(output_classical.device)
        if prev_amplitudes.dtype != dtype:
            prev_amplitudes = prev_amplitudes.to(dtype)
        output_classical += coeff * prev_amplitudes

    output_classical = output_classical / torch.norm(
        output_classical, p=2, dim=1, keepdim=True
    )

    distribution = output_classical.real**2 + output_classical.imag**2

    return distribution


class TestComputeEbsSimultaneously:
    """Test cases for compute_ebs_simultaneously method in ComputationProcess."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple circuit for testing
        self.circuit = pcvl.components.GenericInterferometer(
            6,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        # Create superposition input state
        expected_states = math.comb(6, 2)
        self.input_state_tensor = torch.rand(2, expected_states, dtype=torch.float64)
        sum_values = self.input_state_tensor.abs().pow(2).sum(dim=1).sqrt().unsqueeze(1)
        self.input_state_tensor = self.input_state_tensor / sum_values
        # Set up parameters
        self.trainable_parameters = ["phi"]
        self.input_parameters = []
        self.n_photons = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer = QuantumLayer(
            input_size=0,
            circuit=self.circuit,
            n_photons=self.n_photons,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED
            ),
            input_state=self.input_state_tensor,
            trainable_parameters=self.trainable_parameters,
            input_parameters=self.input_parameters,
            dtype=torch.float64,
            device=device,
        )
        # Create computation process
        self.process = self.layer.computation_process
        self.process.input_state = self.input_state_tensor

        # Create test parameters
        self.test_parameters = self.layer.prepare_parameters([])

    def test_basic_functionality(self):
        """Test basic functionality of compute_ebs_simultaneously."""
        # Test with simultaneous_processes=1 (default)
        result = self.process.compute_ebs_simultaneously(
            self.test_parameters, simultaneous_processes=1
        )

        # Check output shape and dtype
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.complex128  # Should be complex for float64 input
        assert len(result.shape) == 2

        # Result should not be all zeros
        assert not torch.allclose(result, torch.zeros_like(result))

    def test_different_batch_sizes(self):
        """Test compute_ebs_simultaneously with different batch sizes."""
        # Test with different simultaneous_processes values
        batch_sizes = [1, 2, 4]
        results = []

        for batch_size in batch_sizes:
            result = self.process.compute_ebs_simultaneously(
                self.test_parameters, simultaneous_processes=batch_size
            )
            results.append(result)

        # All results should be the same regardless of batch size
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], rtol=1e-12, atol=1e-14), (
                f"Results differ between batch_size=1 and batch_size={batch_sizes[i]}"
            )

    @pytest.mark.skip(
        reason="compute_superposition state is broken but not sure it is necessary - not called anywhere anyway"
    )
    def test_comparison_with_superposition_state(self):
        """Test that compute_ebs_simultaneously gives same results as compute_superposition_state."""
        # Compute using both methods
        result_ebs = self.process.compute_ebs_simultaneously(
            self.test_parameters, simultaneous_processes=2
        )
        result_superposition = self.process.compute_superposition_state(
            self.test_parameters
        )

        # Results should be identical (or very close due to numerical precision)
        assert torch.allclose(
            result_ebs, result_superposition, rtol=1e-12, atol=1e-14
        ), (
            "compute_ebs_simultaneously and compute_superposition_state give different results"
        )

    def test_edge_case_single_state(self):
        """Test with input state that has only one non-zero component."""
        # Create input state with only one non-zero component
        expected_states = math.comb(self.circuit.m, self.n_photons)
        single_state = torch.zeros(1, expected_states, dtype=torch.float64)
        single_state[0, 0] = 1.0

        process_single = ComputationProcess(
            circuit=self.circuit,
            input_state=single_state,
            trainable_parameters=self.trainable_parameters,
            input_parameters=self.input_parameters,
            n_photons=self.n_photons,
            dtype=torch.float64,
            no_bunching=True,
        )

        result = process_single.compute_ebs_simultaneously(
            self.test_parameters, simultaneous_processes=1
        )

        # Should still produce valid output
        assert isinstance(result, torch.Tensor)
        assert not torch.allclose(result, torch.zeros_like(result))

    def test_dtype_consistency(self):
        """Test that dtype is handled correctly."""
        # Test with float32
        input_state_f32 = self.input_state_tensor.to(torch.float32)
        process_f32 = ComputationProcess(
            circuit=self.circuit,
            input_state=input_state_f32,
            trainable_parameters=self.trainable_parameters,
            input_parameters=self.input_parameters,
            n_photons=self.n_photons,
            dtype=torch.float32,
            no_bunching=True,
        )

        params_f32 = [p.to(torch.float32) for p in self.test_parameters]
        result_f32 = process_f32.compute_ebs_simultaneously(
            params_f32, simultaneous_processes=2
        )

        # Should be complex64 for float32 input
        assert result_f32.dtype == torch.complex64

    def test_invalid_superposition_dimension_no_bunching(self):
        """Input state with mismatched dimension should raise a ValueError."""
        invalid_state = torch.rand(
            self.input_state_tensor.shape[0],
            self.input_state_tensor.shape[1] - 1,
            dtype=self.input_state_tensor.dtype,
            device=self.input_state_tensor.device,
        )
        self.process.input_state = invalid_state

        with pytest.raises(ValueError, match="Input state dimension mismatch"):
            self.process.compute_ebs_simultaneously(
                self.test_parameters, simultaneous_processes=1
            )

    def test_invalid_superposition_dimension_fock(self):
        """Input state dimension mismatch in Fock space raises a ValueError."""
        expected_fock_states = math.comb(
            self.circuit.m + self.n_photons - 1, self.n_photons
        )
        valid_state = torch.rand(1, expected_fock_states, dtype=torch.float64)
        valid_state = (
            valid_state / valid_state.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
        )

        process_fock = ComputationProcess(
            circuit=self.circuit,
            input_state=valid_state,
            trainable_parameters=self.trainable_parameters,
            input_parameters=self.input_parameters,
            n_photons=self.n_photons,
            dtype=torch.float64,
            no_bunching=False,
        )

        invalid_state = torch.rand(
            1, expected_fock_states + 1, dtype=torch.float64, device=valid_state.device
        )
        process_fock.input_state = invalid_state

        params = [p.clone() for p in self.test_parameters]

        with pytest.raises(ValueError, match="Input state dimension mismatch"):
            process_fock.compute_ebs_simultaneously(params, simultaneous_processes=1)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with invalid input state type
        with pytest.raises(TypeError, match="Input state should be a tensor"):
            process_invalid = ComputationProcess(
                circuit=self.circuit,
                input_state=[1, 1, 0, 0, 0, 0],  # List instead of tensor
                trainable_parameters=self.trainable_parameters,
                input_parameters=self.input_parameters,
                n_photons=self.n_photons,
                dtype=torch.float64,
                no_bunching=True,
            )
            process_invalid.compute_ebs_simultaneously(
                self.test_parameters, simultaneous_processes=1
            )

    def test_large_batch_size(self):
        """Test with batch size larger than number of input states."""
        # This should still work - it just means some batches will be smaller
        result = self.process.compute_ebs_simultaneously(
            self.test_parameters, simultaneous_processes=100
        )

        # Should give same result as smaller batch sizes
        result_small = self.process.compute_ebs_simultaneously(
            self.test_parameters, simultaneous_processes=1
        )

        assert torch.allclose(result, result_small, rtol=1e-12, atol=1e-14)

    def test_gradient_flow(self):
        """Test that gradients flow through compute_ebs_simultaneously."""
        # Create parameters that require gradients
        params_with_grad = [
            p.clone().detach().requires_grad_(True) for p in self.test_parameters
        ]

        result = self.process.compute_ebs_simultaneously(
            params_with_grad, simultaneous_processes=2
        )

        # Compute a simple loss and backpropagate
        loss = result.abs().sum()
        loss.backward()

        # Check that gradients were computed
        for param in params_with_grad:
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_quantum_layer_benchmark(self, benchmark):
        """Test that compute_ebs_simultaneously gives same results as QuantumLayer with superposition."""
        print("\n=== Testing Batch Method vs QuantumLayer Comparison ===")

        # Create QuantumLayer with the same configuration
        layer = self.layer

        # Run the QuantumLayer (uses compute_superposition_state internally)
        # output_quantum_layer = benchmark(layer, batch=True, simultaneous_processes=6)
        output_quantum_layer = benchmark(layer)
        # Run our batch method
        result_batch = self.process.compute_ebs_simultaneously(
            self.test_parameters, simultaneous_processes=4
        )

        # Convert to probabilities for comparison
        probs_batch = result_batch.real**2 + result_batch.imag**2
        sum_probs = probs_batch.sum(dim=1, keepdim=True)

        # Only normalize when sum > 0 to avoid division by zero
        valid_entries = sum_probs > 0
        if valid_entries.any():
            probs_batch = torch.where(
                valid_entries,
                probs_batch
                / torch.where(valid_entries, sum_probs, torch.ones_like(sum_probs)),
                probs_batch,
            )

        # Results should be identical (or very close due to numerical precision)
        assert torch.allclose(
            output_quantum_layer[1], probs_batch[1], rtol=3e-4, atol=1e-7
        ), "Batch method and QuantumLayer give different results"

    @pytest.mark.skip(
        reason="this test use superposition_state input which is deprecated"
    )
    def test_classical_method_benchmark(self, benchmark):
        """Benchmark classical method (individual state computation) vs batch method."""
        print("\n=== Testing Classical Method Benchmark ===")

        # Create QuantumLayer for classical method
        layer = self.layer

        # Create input state dictionary for classical method
        input_state_superposed = {
            layer.computation_process.simulation_graph.mapped_keys[
                k
            ]: self.input_state_tensor[0, k]
            for k in range(len(self.input_state_tensor[0]))
            if self.input_state_tensor[0, k].abs()
            > 1e-10  # Only include non-zero entries
        }

        # Run classical method with benchmark
        output_classical = benchmark(
            classical_method_ebs, layer, input_state_superposed
        )
        self.process.input_state = self.input_state_tensor
        # Run batch method
        result_batch = self.process.compute_ebs_simultaneously(
            self.test_parameters, simultaneous_processes=4
        )

        # Convert to probabilities for comparison
        probs_batch = result_batch.real**2 + result_batch.imag**2

        # Results should be identical (or very close due to numerical precision)
        classical_probs = output_classical.to("cpu")
        batch_probs = probs_batch[0].to("cpu")

        assert torch.allclose(classical_probs, batch_probs, rtol=3e-4, atol=1e-7), (
            "Classical method and batch method give different results"
        )

    def test_batch_benchmark(self, benchmark):
        """Test that compute_ebs_simultaneously gives same results as QuantumLayer with superposition."""
        print("\n=== Testing Batch Method vs QuantumLayer Comparison ===")

        # Create QuantumLayer with the same configuration
        layer = self.layer

        # Run the QuantumLayer (uses compute_superposition_state internally)
        output_quantum_layer = layer()

        # Run our batch method
        result_batch = benchmark(
            self.process.compute_ebs_simultaneously,
            self.test_parameters,
            simultaneous_processes=4,
        )

        # Convert to probabilities for comparison
        probs_batch = result_batch.real**2 + result_batch.imag**2
        sum_probs = probs_batch.sum(dim=1, keepdim=True)

        # Only normalize when sum > 0 to avoid division by zero
        valid_entries = sum_probs > 0
        if valid_entries.any():
            probs_batch = torch.where(
                valid_entries,
                probs_batch
                / torch.where(valid_entries, sum_probs, torch.ones_like(sum_probs)),
                probs_batch,
            )

        # Results should be identical (or very close due to numerical precision)
        assert torch.allclose(
            output_quantum_layer[1], probs_batch[1], rtol=3e-4, atol=1e-7
        ), "Batch method and QuantumLayer give different results"
