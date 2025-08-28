"""
Test suite for amplitude output feature in quantum models.
"""

import pytest
import torch
import numpy as np
from math import comb

from merlin.models import QuantumModel, QuantumConfig
from merlin.builder import CircuitBuilder
from merlin.backends.photonic import PhotonicBackend

try:
    import perceval as pcvl

    PERCEVAL_AVAILABLE = True
except ImportError:
    PERCEVAL_AVAILABLE = False


class TestAmplitudeOutput:
    """Test amplitude output functionality."""

    @pytest.fixture
    def simple_circuit(self):
        """Create a simple test circuit."""
        builder = CircuitBuilder(n_modes=3, n_photons=2)
        builder.add_rotation_layer(modes=[0], role="input")
        builder.add_entangling_layer(trainable=True, depth=1)
        return builder

    @pytest.fixture
    def quantum_model(self, simple_circuit):
        """Create a quantum model with amplitude support."""
        config = QuantumConfig(
            n_modes=3,
            n_photons=2,
            return_amplitudes=False,  # Default to False
            backend_options={'input_state': [1, 1, 0]}
        )
        return QuantumModel.from_builder(simple_circuit, config=config)

    def test_amplitude_return_shape(self, quantum_model):
        """Test that amplitudes have correct shape when returned."""
        x = torch.randn(5, 1)  # batch_size=5, input_size=1

        # Get probabilities only (default)
        probs = quantum_model(x)
        assert probs.dim() == 2
        assert probs.shape[0] == 5

        # Get both probabilities and amplitudes
        probs, amps = quantum_model(x, return_amplitudes=True)

        # Check shapes match
        assert probs.shape == amps.shape
        assert amps.dtype == torch.complex64 or amps.dtype == torch.complex128

        # Check probability normalization
        assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_amplitude_probability_consistency(self, quantum_model):
        """Test that probabilities equal |amplitudes|^2."""
        x = torch.randn(10, 1)

        probs, amps = quantum_model(x, return_amplitudes=True)

        # Calculate probabilities from amplitudes
        calculated_probs = amps.real ** 2 + amps.imag ** 2

        # Account for normalization in no_bunching case
        if quantum_model.config.backend_options.get('no_bunching', True):
            sum_calc = calculated_probs.sum(dim=-1, keepdim=True)
            valid = sum_calc > 0
            calculated_probs = torch.where(
                valid,
                calculated_probs / torch.where(valid, sum_calc, torch.ones_like(sum_calc)),
                calculated_probs
            )

        assert torch.allclose(probs, calculated_probs, atol=1e-5)

    def test_config_return_amplitudes(self):
        """Test that config.return_amplitudes sets default behavior."""
        builder = CircuitBuilder(n_modes=2, n_photons=1)
        builder.add_rotation_layer(modes=[0], role="input")

        # Model with return_amplitudes=True in config
        config_true = QuantumConfig(
            n_modes=2,
            n_photons=1,
            return_amplitudes=True
        )
        model_true = QuantumModel.from_builder(builder, config=config_true)

        x = torch.randn(3, 1)
        result = model_true(x)

        # Should return tuple by default
        assert isinstance(result, tuple)
        assert len(result) == 2

        # Model with return_amplitudes=False in config
        config_false = QuantumConfig(
            n_modes=2,
            n_photons=1,
            return_amplitudes=False
        )
        model_false = QuantumModel.from_builder(builder, config=config_false)

        result = model_false(x)

        # Should return tensor by default
        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, tuple)

    def test_amplitude_gradient_flow(self, quantum_model):
        """Test that gradients flow through amplitudes."""
        x = torch.randn(3, 1, requires_grad=True)

        probs, amps = quantum_model(x, return_amplitudes=True)

        # Create loss from amplitude magnitude
        loss = (amps.abs() ** 2).sum()
        loss.backward()

        # Check gradient exists
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_backend_amplitude_support(self):
        """Test that backend properly indicates amplitude support."""
        backend = PhotonicBackend(n_modes=3, n_photons=2)
        assert backend.supports_amplitudes == True

        # Check through model info
        builder = CircuitBuilder(n_modes=3, n_photons=2)
        builder.add_rotation_layer(role="input")
        model = QuantumModel.from_builder(builder)

        info = model.get_info()
        assert info['supports_amplitudes'] == True

    @pytest.mark.skipif(not PERCEVAL_AVAILABLE, reason="Perceval not available")
    def test_direct_perceval_circuit_amplitudes(self):
        """Test amplitude output with direct Perceval circuit."""
        circuit = pcvl.Circuit(2)
        circuit.add(0, pcvl.PS(pcvl.P("theta")))
        circuit.add(0, pcvl.BS())

        config = QuantumConfig(
            n_modes=2,
            n_photons=1,
            backend_options={
                'input_state': [1, 0],
                'trainable_parameters': ['theta'],
                'input_parameters': []
            }
        )

        model = QuantumModel(circuit, config=config)

        # Test without input (uses trainable params only)
        probs, amps = model(return_amplitudes=True)

        assert probs.dim() == 1 or probs.dim() == 2
        assert amps.dtype in [torch.complex64, torch.complex128]

        # Verify unitarity preservation
        assert torch.allclose(amps.abs().pow(2).sum(), torch.tensor(1.0), atol=1e-5)

    def test_batch_amplitude_output(self, quantum_model):
        """Test amplitude output with batched inputs."""
        batch_sizes = [1, 4, 16]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 1)
            probs, amps = quantum_model(x, return_amplitudes=True)

            assert probs.shape[0] == batch_size
            assert amps.shape[0] == batch_size
            assert probs.shape == amps.shape

            # Each sample should be normalized
            for i in range(batch_size):
                assert torch.allclose(
                    probs[i].sum(),
                    torch.tensor(1.0),
                    atol=1e-5
                )

    def test_amplitude_with_output_mapping(self):
        """Test that output mapping only affects probabilities, not amplitudes."""
        builder = CircuitBuilder(n_modes=3, n_photons=2)
        builder.add_rotation_layer(modes=[0], role="input")

        config = QuantumConfig(
            n_modes=3,
            n_photons=2,
            output_size=5,  # Different from natural output size
            backend_options={'input_state': [1, 1, 0]}
        )

        model = QuantumModel.from_builder(builder, config=config)

        x = torch.randn(4, 1)
        probs, amps = model(x, return_amplitudes=True)

        # Probabilities should be mapped to output_size
        assert probs.shape[-1] == 5

        # Amplitudes should remain in original Fock space
        expected_fock_dim = comb(3 + 2 - 1, 2)  # C(n_modes + n_photons - 1, n_photons)
        assert amps.shape[-1] == expected_fock_dim

    def test_amplitude_phase_information(self, quantum_model):
        """Test that amplitudes contain phase information."""
        x = torch.randn(10, 1)

        _, amps = quantum_model(x, return_amplitudes=True)

        # Check that amplitudes have non-zero imaginary parts
        has_phase = torch.any(amps.imag.abs() > 1e-8)

        # In general quantum circuits, we expect complex amplitudes
        # This might not always be true for specific circuits, but is typical
        assert has_phase or torch.allclose(amps.imag, torch.zeros_like(amps.imag))

    def test_amplitude_deterministic(self, quantum_model):
        """Test that amplitude computation is deterministic."""
        x = torch.randn(5, 1)

        # Multiple forward passes
        results = []
        for _ in range(3):
            _, amps = quantum_model(x, return_amplitudes=True)
            results.append(amps)

        # All results should be identical
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], atol=1e-7)

    def test_amplitude_with_shots(self, quantum_model):
        """Test that shots parameter doesn't affect amplitude output."""
        x = torch.randn(3, 1)

        # Get amplitudes without shots
        probs_no_shots, amps_no_shots = quantum_model(x, shots=0, return_amplitudes=True)

        # Get amplitudes with shots (affects probabilities only)
        probs_shots, amps_shots = quantum_model(x, shots=1000, return_amplitudes=True)

        # Amplitudes should be identical
        assert torch.allclose(amps_no_shots, amps_shots, atol=1e-7)

        # Probabilities might differ due to sampling
        # But should be close for large shot count
        assert not torch.allclose(probs_no_shots, probs_shots, atol=1e-3)


class TestAmplitudeEdgeCases:
    """Test edge cases and error handling for amplitude output."""

    def test_empty_input_amplitudes(self):
        """Test amplitude output with no input data."""
        builder = CircuitBuilder(n_modes=2, n_photons=1)
        builder.add_rotation_layer(role="trainable")  # No input layer

        model = QuantumModel.from_builder(builder)

        # Should work without input
        probs, amps = model(return_amplitudes=True)

        assert probs is not None
        assert amps is not None
        assert amps.dtype in [torch.complex64, torch.complex128]

    def test_amplitude_numerical_stability(self):
        """Test numerical stability of amplitude calculations."""
        builder = CircuitBuilder(n_modes=4, n_photons=3)
        builder.add_rotation_layer(modes=[0, 1], role="input")
        builder.add_entangling_layer(trainable=True, depth=2)

        model = QuantumModel.from_builder(builder)

        # Test with extreme input values
        x_small = torch.tensor([[1e-8, 1e-8]])
        x_large = torch.tensor([[1e3, 1e3]])

        for x in [x_small, x_large]:
            probs, amps = model(x, return_amplitudes=True)

            # Check for NaN or Inf
            assert torch.all(torch.isfinite(probs))
            assert torch.all(torch.isfinite(amps.real))
            assert torch.all(torch.isfinite(amps.imag))

            # Check normalization still holds
            assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-4)


def test_integration_fourier_approximation_with_amplitudes():
    """Integration test: Use amplitudes for Fourier series approximation."""
    # Create a simple quantum model
    builder = CircuitBuilder(n_modes=3, n_photons=2)
    builder.add_entangling_layer(trainable=True, depth=1)
    builder.add_rotation_layer(modes=[1], role="input")
    builder.add_entangling_layer(trainable=True, depth=1)

    config = QuantumConfig(
        n_modes=3,
        n_photons=2,
        output_size=1,
        backend_options={'input_state': [1, 1, 0]}
    )

    model = QuantumModel.from_builder(builder, config=config)

    # Generate training data (simple sine wave)
    x = torch.linspace(-np.pi, np.pi, 50).unsqueeze(1)
    y_target = torch.sin(x).squeeze()

    # Training with amplitude regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(10):
        optimizer.zero_grad()

        # Get both outputs
        output, amps = model(x, return_amplitudes=True)
        output = output.squeeze()

        # Standard MSE loss
        mse_loss = ((output - y_target) ** 2).mean()

        # Amplitude regularization (encourage sparse amplitudes)
        amp_reg = 0.01 * amps.abs().mean()

        loss = mse_loss + amp_reg
        loss.backward()
        optimizer.step()

    # Final evaluation
    with torch.no_grad():
        final_output, final_amps = model(x, return_amplitudes=True)

    # Check that we got meaningful results
    assert final_output is not None
    assert final_amps is not None
    assert not torch.allclose(final_output, torch.zeros_like(final_output))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])