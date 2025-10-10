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
Tests for sampling and autodiff utilities.
"""

import warnings

import pytest
import torch

import merlin as ML


class TestSamplingProcess:
    """Test suite for SamplingProcess."""

    def test_no_sampling_with_zero_shots(self):
        """Test that no sampling occurs with shots=0."""
        sampler = ML.SamplingProcess()

        # Original distribution
        dist = torch.tensor([0.3, 0.4, 0.2, 0.1])

        # Should return unchanged
        result = sampler.pcvl_sampler(dist, shots=0)

        assert torch.allclose(result, dist)

    def test_multinomial_sampling_1d(self):
        """Test multinomial sampling with 1D distribution."""
        sampler = ML.SamplingProcess()

        dist = torch.tensor([0.3, 0.4, 0.2, 0.1])
        shots = 1000

        result = sampler.pcvl_sampler(dist, shots=shots, method="multinomial")

        # Check output is valid probability distribution
        assert torch.all(result >= 0)
        assert torch.allclose(result.sum(), torch.tensor(1.0), atol=1e-6)

        # Check it's different from original (sampling noise)
        assert not torch.allclose(result, dist, atol=1e-3)

    def test_multinomial_sampling_batched(self):
        """Test multinomial sampling with batched distribution."""
        sampler = ML.SamplingProcess()

        batch_size = 5
        dist_size = 4
        dist = torch.rand(batch_size, dist_size)
        dist = dist / dist.sum(dim=1, keepdim=True)  # Normalize

        shots = 500
        result = sampler.pcvl_sampler(dist, shots=shots, method="multinomial")

        assert result.shape == dist.shape
        # Each row should sum to 1
        assert torch.allclose(result.sum(dim=1), torch.ones(batch_size), atol=1e-6)
        assert torch.all(result >= 0)

    def test_gaussian_sampling(self):
        """Test Gaussian sampling method."""
        sampler = ML.SamplingProcess()

        dist = torch.tensor([0.4, 0.3, 0.2, 0.1])
        shots = 1000

        result = sampler.pcvl_sampler(dist, shots=shots, method="gaussian")

        # Check output is valid probability distribution
        assert torch.all(result >= 0)
        assert torch.allclose(result.sum(), torch.tensor(1.0), atol=1e-6)

        # Should be different from original
        assert not torch.allclose(result, dist, atol=1e-3)

    def test_binomial_sampling(self):
        """Test binomial sampling method."""
        sampler = ML.SamplingProcess()

        dist = torch.tensor([0.4, 0.3, 0.2, 0.1])
        shots = 1000

        result = sampler.pcvl_sampler(dist, shots=shots, method="binomial")

        # Check output is valid probability distribution
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

        # Should be different from original
        assert not torch.allclose(result, dist, atol=1e-3)

    def test_invalid_sampling_method(self):
        """Test that invalid sampling methods raise errors."""
        sampler = ML.SamplingProcess()

        dist = torch.tensor([0.4, 0.3, 0.2, 0.1])

        with pytest.raises(ValueError, match="Invalid sampling method"):
            sampler.pcvl_sampler(dist, shots=100, method="invalid_method")

    def test_sampling_with_small_shots(self):
        """Test sampling behavior with small number of shots."""
        sampler = ML.SamplingProcess()

        dist = torch.tensor([0.5, 0.3, 0.2])
        shots = 10

        result = sampler.pcvl_sampler(dist, shots=shots, method="multinomial")

        # Should still be valid distribution
        assert torch.all(result >= 0)
        assert torch.allclose(result.sum(), torch.tensor(1.0), atol=1e-6)


class TestAutoDiffProcess:
    """Test suite for AutoDiffProcess."""

    def test_autodiff_no_gradients_no_sampling(self):
        """Test autodiff when no gradients needed and no sampling requested."""
        autodiff = ML.AutoDiffProcess()

        apply_sampling, shots = autodiff.autodiff_backend(
            needs_gradient=False, apply_sampling=False, shots=0
        )

        assert apply_sampling is False
        assert shots == 0

    def test_autodiff_no_gradients_with_sampling(self):
        """Test autodiff when no gradients needed but sampling requested."""
        autodiff = ML.AutoDiffProcess()

        apply_sampling, shots = autodiff.autodiff_backend(
            needs_gradient=False, apply_sampling=True, shots=100
        )

        assert apply_sampling is True
        assert shots == 100

    def test_autodiff_gradients_with_sampling_warning(self):
        """Test that sampling is disabled during gradient computation with warning."""
        autodiff = ML.AutoDiffProcess()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            apply_sampling, shots = autodiff.autodiff_backend(
                needs_gradient=True, apply_sampling=True, shots=100
            )

            # Should disable sampling
            assert apply_sampling is False
            assert shots == 0

            # Should have warned
            assert len(w) == 1
            assert "Sampling was requested but is disabled" in str(w[0].message)

    def test_autodiff_gradients_with_shots_warning(self):
        """Test that shots>0 is disabled during gradient computation."""
        autodiff = ML.AutoDiffProcess()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            apply_sampling, shots = autodiff.autodiff_backend(
                needs_gradient=True, apply_sampling=False, shots=100
            )

            # Should disable sampling
            assert apply_sampling is False
            assert shots == 0

            # Should have warned
            assert len(w) == 1
            assert "Sampling was requested but is disabled" in str(w[0].message)

    def test_autodiff_gradients_no_sampling(self):
        """Test autodiff when gradients needed and no sampling requested."""
        autodiff = ML.AutoDiffProcess()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            apply_sampling, shots = autodiff.autodiff_backend(
                needs_gradient=True, apply_sampling=False, shots=0
            )

            # Should remain unchanged
            assert apply_sampling is False
            assert shots == 0

            # Should not warn
            assert len(w) == 0


class TestSamplingIntegration:
    """Integration tests for sampling with QuantumLayer."""

    def test_layer_sampling_during_training(self):
        """Test that sampling is disabled during training mode."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz, shots=100)

        # Set to training mode
        layer.train()

        x = torch.rand(3, 2, requires_grad=True)

        # Should not apply sampling during training with gradients
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            output = layer(x, apply_sampling=True, shots=100)
            loss = output.sum()
            loss.backward()

            # Should have warned about disabled sampling
            warning_found = any(
                "Sampling was requested but is disabled" in str(warning.message)
                for warning in w
            )
            assert warning_found

    def test_layer_sampling_during_evaluation(self):
        """Test that sampling works during evaluation mode."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz, shots=100)

        # Set to evaluation mode
        layer.eval()

        x = torch.rand(3, 2)

        # Get clean output
        clean_output = layer(x)

        # Get sampled output
        sampled_output = layer(x, apply_sampling=True, shots=100)

        # Should be different due to sampling noise
        assert not torch.allclose(clean_output, sampled_output, atol=1e-3)

        # Both should be valid
        assert torch.all(torch.isfinite(clean_output))
        assert torch.all(torch.isfinite(sampled_output))

    def test_layer_sampling_config_update(self):
        """Test updating sampling configuration on layer."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        # Initial config
        assert layer.shots == 0
        assert layer.sampling_method == "multinomial"

        # Update config
        layer.set_sampling_config(shots=200, method="gaussian")

        assert layer.shots == 200
        assert layer.sampling_method == "gaussian"

        # Test invalid updates
        with pytest.raises(ValueError):
            layer.set_sampling_config(shots=-1)

        with pytest.raises(ValueError):
            layer.set_sampling_config(method="invalid")

    def test_different_sampling_methods_produce_different_results(self):
        """Test that different sampling methods produce different results."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        layer.eval()

        x = torch.rand(5, 2)

        results = {}
        methods = ["multinomial", "gaussian", "binomial"]

        for method in methods:
            layer.set_sampling_config(shots=100, method=method)
            output = layer(x, apply_sampling=True, shots=100)
            results[method] = output

        # All results should be different from each other
        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1 :]:
                assert not torch.allclose(
                    results[method1], results[method2], atol=1e-3
                ), f"Methods {method1} and {method2} produced identical results"
