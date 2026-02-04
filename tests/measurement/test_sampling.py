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
    """Integration tests for sampling with QuantumLayer (modern API)."""

    def _make_layer(self):
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input", subset_combinations=True)
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )
        return layer

    def test_layer_sampling_during_training(self):
        """Sampling requests are ignored during training to keep the path differentiable."""
        layer = self._make_layer()
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        # Training mode
        model.train()

        x = torch.rand(3, 2, requires_grad=True)

        # Baseline (no sampling)
        clean_out = model(x)

        # Request sampling, but backend should disable it while training
        # (effectively shots -> 0)
        with pytest.warns():
            x_out_req = layer(x, shots=100, sampling_method="multinomial")
        sampled_out_train = model[1](x_out_req)

        # Should be (near-)identical to no-sampling output in training
        assert torch.allclose(clean_out, sampled_out_train, atol=1e-7, rtol=1e-6)

        # Backprop should work
        loss = sampled_out_train.sum()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_layer_sampling_during_evaluation(self):
        """Sampling works during evaluation mode and produces noisy outputs."""
        layer = self._make_layer()
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        # Evaluation mode
        model.eval()

        x = torch.rand(3, 2)

        # Clean (no sampling)
        clean_output = model(x)

        # Sampled output (enable sampling via shots)
        x_out = layer(x, shots=100, sampling_method="multinomial")
        sampled_output = model[1](x_out)

        # Should typically differ due to sampling noise
        # (allow the possibility of equality in rare cases by using a small tolerance)
        assert not torch.allclose(clean_output, sampled_output, atol=1e-4, rtol=1e-4)

        # Both should be finite
        assert torch.all(torch.isfinite(clean_output))
        assert torch.all(torch.isfinite(sampled_output))

    def test_invalid_sampling_method_raises(self):
        """Invalid sampling method should raise ValueError at call time."""
        layer = self._make_layer()
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))
        model.eval()

        x = torch.rand(2, 2)

        with pytest.raises(ValueError):
            # Forward with invalid method
            _ = layer(x, shots=10, sampling_method="invalid")

    def test_different_sampling_methods_produce_different_results(self):
        """Different sampling methods should (generally) yield different outputs in eval."""
        layer = self._make_layer()
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))
        model.eval()

        # Fix seed for reproducibility of comparisons within this test
        torch.manual_seed(42)

        x = torch.rand(5, 2)

        methods = ["multinomial", "gaussian", "binomial"]
        outputs = {}

        for method in methods:
            # Request sampling per call
            x_out = layer(x, shots=200, sampling_method=method)
            outputs[method] = model[1](x_out)

        # Pairwise compare: expect differences (allow tolerance)
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1 :]:
                assert not torch.allclose(
                    outputs[m1], outputs[m2], atol=1e-4, rtol=1e-4
                ), f"Methods {m1} and {m2} produced indistinguishable results"
