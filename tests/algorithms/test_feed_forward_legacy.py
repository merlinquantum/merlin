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
Tests for the FeedForward class.
"""

import pytest
import torch

from merlin.algorithms.feed_forward_legacy import (
    FeedForwardBlockLegacy,
    PoolingFeedForwardLegacy,
    define_layer_no_input,
)


class TestFeedForwardBlockLegacy:
    """Comprehensive test suite for FeedForwardBlockLegacy."""

    def test_init_single_mode(self):
        """Test initialization with single conditional mode."""
        ff = FeedForwardBlockLegacy(
            input_size=6, m=6, n=2, depth=3, conditional_modes=[0]
        )
        assert ff.m == 6
        assert ff.n_photons == 2
        assert ff.conditional_modes == [0]
        assert ff.depth == 3
        assert isinstance(ff.layers, dict)

    def test_init_multi_mode(self):
        """Test initialization with multiple conditional modes."""
        ff = FeedForwardBlockLegacy(
            input_size=6, m=6, n=2, depth=2, conditional_modes=[0, 1]
        )
        assert ff.n_cond == 2
        assert all(isinstance(k, tuple) for k in ff.layers.keys())

    def test_generate_possible_tuples(self):
        """Ensure generated tuples are non-empty and valid."""
        ff = FeedForwardBlockLegacy(input_size=4, n=2, m=4, conditional_modes=[0, 1])
        tuples = ff.generate_possible_tuples()
        assert isinstance(tuples, list)
        assert len(tuples) > 0
        assert all(isinstance(t, tuple) for t in tuples)

    def test_parameters_method(self):
        """Check that parameters() returns a non-empty iterable."""
        ff = FeedForwardBlockLegacy(
            input_size=4, n=2, m=4, depth=2, conditional_modes=[0]
        )
        params = list(ff.parameters())
        assert all(isinstance(p, torch.Tensor) for p in params)

    def test_forward_output_shape_and_sum(self):
        """Check forward pass produces valid probabilities summing to 1."""
        ff = FeedForwardBlockLegacy(
            input_size=4, n=2, m=4, depth=2, conditional_modes=[0]
        )
        x = torch.rand(3, 4)  # batch of 3
        output = ff(x)
        assert isinstance(output, torch.Tensor)
        assert output.ndim == 2  # (batch_size, n_outputs)
        # Each row should sum approximately to 1
        probs_sum = output.sum(dim=1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-3)

    def test_forward_multi_mode(self):
        """Test forward with multiple conditional modes and output normalization."""
        ff = FeedForwardBlockLegacy(
            input_size=3, n=2, m=4, depth=2, conditional_modes=[0, 1]
        )
        x = torch.rand(2, 3)
        output = ff(x)
        assert output.shape[0] == 2  # batch size preserved
        assert output.shape[1] > 0
        assert torch.allclose(output.sum(dim=1), torch.ones(2), atol=1e-3)

    def test_backward_pass(self):
        """Ensure gradients propagate correctly through the quantum layers."""
        ff = FeedForwardBlockLegacy(
            input_size=4, n=2, m=4, depth=2, conditional_modes=[0]
        )
        x = torch.rand(1, 4, requires_grad=True)
        output = ff(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_indices_by_values_and_match_indices(self):
        """Test low-level index mapping helpers."""
        ff = FeedForwardBlockLegacy(
            input_size=2, n=2, m=4, depth=2, conditional_modes=[0, 1]
        )
        keys = [(0, 1, 0), (1, 0, 1), (0, 0, 1)]
        idx_dict = ff._indices_by_values(keys, [0, 1])
        assert all(isinstance(v, torch.Tensor) for v in idx_dict.values())

        data_out = [(1, 0), (0, 1)]
        idx = ff._match_indices_multi(keys, data_out, [0, 1], (0, 1))
        assert isinstance(idx, torch.Tensor)

    def test_integration_with_optimizer(self):
        """Ensure block integrates cleanly with PyTorch optimizers."""
        ff = FeedForwardBlockLegacy(
            input_size=4, n=2, m=4, depth=2, conditional_modes=[0]
        )
        x = torch.rand(1, 4)
        optimizer = torch.optim.Adam(ff.parameters(), lr=1e-3)

        output = ff(x)
        loss = output.pow(2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        assert True  # No runtime errors

    def test_output_keys_consistency(self):
        """Ensure output keys are consistent after forward pass."""
        ff = FeedForwardBlockLegacy(
            input_size=3, n=2, m=3, depth=2, conditional_modes=[0]
        )
        _ = ff(torch.rand(1, 3))
        keys = ff.output_keys
        print(keys)
        assert isinstance(keys, list)
        assert len(keys) > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available for GPU tests"
)
class TestFeedForwardBlockGPU:
    """GPU-only tests for FeedForwardBlock forward/backward behavior."""

    def test_forward_on_gpu(self):
        """Ensure forward pass works on GPU and outputs valid probabilities."""
        device = torch.device("cuda")
        ff = FeedForwardBlockLegacy(
            input_size=4, n=2, m=4, depth=2, conditional_modes=[0]
        ).to(device)
        x = torch.rand(8, 4, device=device)
        output = ff(x)
        assert output.device.type == "cuda"
        assert torch.allclose(
            output.sum(dim=1), torch.ones(8, device=device), atol=1e-3
        )

    def test_backward_on_gpu(self):
        """Ensure gradients flow correctly on GPU."""
        device = torch.device("cuda")
        ff = FeedForwardBlockLegacy(
            input_size=4, n=2, m=4, depth=2, conditional_modes=[0]
        ).to(device)
        x = torch.rand(4, 4, device=device, requires_grad=True)
        output = ff(x)
        loss = output.sum()
        loss.backward()

        # Verify gradients computed and reside on GPU
        assert x.grad is not None
        assert x.grad.device.type == "cuda"
        assert not torch.isnan(x.grad).any()

    def test_forward_backward_multi_mode_gpu(self):
        """Test feedforward & gradient with multiple conditional modes on GPU."""
        device = torch.device("cuda")
        ff = FeedForwardBlockLegacy(
            input_size=6, n=3, m=6, depth=2, conditional_modes=[0, 1]
        ).to(device)
        x = torch.rand(2, 6, device=device, requires_grad=True)

        output = ff(x)
        assert output.device.type == "cuda"
        assert output.shape[0] == 2
        assert torch.allclose(
            output.sum(dim=1), torch.ones(2, device=device), atol=1e-3
        )

        loss = output.mean()
        loss.backward()
        assert x.grad is not None
        assert x.grad.device.type == "cuda"

    def test_gpu_optimizer_step(self):
        """Ensure GPU-based optimization loop runs correctly."""
        device = torch.device("cuda")
        ff = FeedForwardBlockLegacy(
            input_size=4, n=2, m=4, depth=2, conditional_modes=[0]
        ).to(device)
        x = torch.rand(2, 4, device=device)
        optimizer = torch.optim.Adam(ff.parameters(), lr=1e-3)

        # Forward, backward, and optimizer step
        output = ff(x)
        loss = output.pow(2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Ensure no NaNs or device mismatch
        assert not torch.isnan(loss).any()
        assert all(p.device.type == "cuda" for p in ff.parameters())


class TestPoolingFeedForwardLegacy:
    """Test suite for PoolingFeedForwardLegacy class."""

    def test_init_default_pooling(self):
        """Test PoolingFeedForwardLegacy initialization with default pooling modes."""
        pff = PoolingFeedForwardLegacy(n_modes=16, n_photons=2, n_output_modes=8)
        assert pff.n_modes == 16
        assert isinstance(pff.match_indices, torch.Tensor)
        assert isinstance(pff.exclude_indices, torch.Tensor)
        assert isinstance(pff.keys_out, list)

    def test_init_custom_pooling(self):
        """Test PoolingFeedForwardLegacy initialization with custom pooling modes."""
        pooling_modes = [[0, 1], [2, 3], [4, 5], [6, 7]]
        pff = PoolingFeedForwardLegacy(
            n_modes=8, n_photons=2, n_output_modes=4, pooling_modes=pooling_modes
        )
        assert pff.n_modes == 8
        assert isinstance(pff.match_indices, torch.Tensor)
        assert isinstance(pff.exclude_indices, torch.Tensor)

    def test_init_with_default_space(self):
        """Test PoolingFeedForwardLegacy initialization with default computation space."""
        pff = PoolingFeedForwardLegacy(n_modes=8, n_photons=2, n_output_modes=4)
        assert pff.n_modes == 8
        assert isinstance(pff.match_indices, torch.Tensor)

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        pff = PoolingFeedForwardLegacy(n_modes=16, n_photons=2, n_output_modes=8)
        batch_size = 4
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        amplitudes = torch.rand(batch_size, n_input_states)
        output = pff(amplitudes)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == batch_size
        assert output.shape[1] == len(pff.keys_out)

    def test_forward_pass_dtype_preservation(self):
        """Test that forward pass preserves dtype."""
        pff = PoolingFeedForwardLegacy(n_modes=8, n_photons=2, n_output_modes=4)
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)

        # Test float32
        amplitudes_f32 = torch.rand(2, n_input_states, dtype=torch.float32)
        output_f32 = pff(amplitudes_f32)
        assert output_f32.dtype == torch.float32

        # Test float64
        amplitudes_f64 = torch.rand(2, n_input_states, dtype=torch.float64)
        output_f64 = pff(amplitudes_f64)
        assert output_f64.dtype == torch.float64

    def test_forward_pass_device_preservation(self):
        """Test that forward pass preserves device."""
        pff = PoolingFeedForwardLegacy(n_modes=8, n_photons=2, n_output_modes=4)
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        amplitudes = torch.rand(2, n_input_states)
        output = pff(amplitudes)
        assert output.device == amplitudes.device

    def test_match_tuples_method(self):
        """Test match_tuples method."""
        pff = PoolingFeedForwardLegacy(n_modes=8, n_photons=2, n_output_modes=4)
        keys_in = [(0, 0, 1, 1, 0, 0, 0, 0), (1, 0, 1, 0, 0, 0, 0, 0)]
        keys_out = [(0, 1, 0, 0), (1, 1, 0, 0)]
        pooling_modes = [[0, 1], [2, 3], [4, 5], [6, 7]]
        match_indices, exclude_indices = pff.match_tuples(
            keys_in, keys_out, pooling_modes
        )
        assert isinstance(match_indices, list)
        assert isinstance(exclude_indices, list)

    def test_backward_pass_integration(self):
        """Test backward pass through PoolingFeedForwardLegacy."""
        pff = PoolingFeedForwardLegacy(n_modes=8, n_photons=2, n_output_modes=4)
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        amplitudes = torch.rand(2, n_input_states, requires_grad=True)
        output = pff(amplitudes)
        loss = output.sum()
        loss.backward()
        assert amplitudes.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_pass_on_cuda(self):
        """Test that forward pass runs correctly on CUDA."""
        device = torch.device("cuda")
        pff = PoolingFeedForwardLegacy(n_modes=16, n_photons=2, n_output_modes=8).to(
            device
        )
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        amplitudes = torch.rand(4, n_input_states, device=device)
        output = pff(amplitudes)
        assert output.is_cuda
        assert output.shape[0] == 4
        assert output.shape[1] == len(pff.keys_out)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_pass_on_cuda(self):
        """Test backward pass with CUDA tensors."""
        device = torch.device("cuda")
        pff = PoolingFeedForwardLegacy(n_modes=8, n_photons=2, n_output_modes=4).to(
            device
        )
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        amplitudes = torch.rand(2, n_input_states, requires_grad=True, device=device)
        output = pff(amplitudes)
        loss = output.pow(2).sum()
        loss.backward()
        assert amplitudes.grad is not None
        assert amplitudes.grad.is_cuda

    def test_integration_with_quantum_layer(self):
        """Test integration with quantum layers (as in the main script)."""
        pff = PoolingFeedForwardLegacy(n_modes=16, n_photons=2, n_output_modes=8)
        pre_layer = define_layer_no_input(16, 2)
        post_layer = define_layer_no_input(8, 2)
        amplitudes = pre_layer()
        amplitudes = pff(amplitudes)
        post_layer.set_input_state(amplitudes)
        res = post_layer()
        assert isinstance(res, torch.Tensor)
        assert res.requires_grad

    def test_optimization_with_layers(self):
        """Test optimization loop with PoolingFeedForwardLegacy between layers."""
        from itertools import chain

        pff = PoolingFeedForwardLegacy(n_modes=8, n_photons=2, n_output_modes=4)
        pre_layer = define_layer_no_input(8, 2)
        post_layer = define_layer_no_input(4, 2)
        params = chain(pre_layer.parameters(), post_layer.parameters())
        optimizer = torch.optim.Adam(params)

        for _ in range(3):
            amplitudes = pre_layer()
            amplitudes = pff(amplitudes)
            post_layer.set_input_state(amplitudes)
            res = post_layer().abs().pow(2).sum()
            res.backward()
            optimizer.step()
            optimizer.zero_grad()
        assert True

    def test_different_pooling_ratios(self):
        """Test various pooling ratios."""
        test_cases = [(16, 8), (16, 4), (12, 6), (9, 3)]
        for n_modes, n_output_modes in test_cases:
            pff = PoolingFeedForwardLegacy(
                n_modes=n_modes, n_photons=2, n_output_modes=n_output_modes
            )
            n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
            amplitudes = torch.rand(2, n_input_states)
            output = pff(amplitudes)
            assert output.shape[1] == len(pff.keys_out)

    def test_single_photon_case(self):
        """Test PoolingFeedForwardLegacy with a single photon."""
        pff = PoolingFeedForwardLegacy(n_modes=8, n_photons=1, n_output_modes=4)
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        amplitudes = torch.rand(2, n_input_states)
        output = pff(amplitudes)
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2

    def test_batch_size_one(self):
        """Test PoolingFeedForwardLegacy with batch size of 1."""
        pff = PoolingFeedForwardLegacy(n_modes=8, n_photons=2, n_output_modes=4)
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        amplitudes = torch.rand(1, n_input_states)
        output = pff(amplitudes)
        assert output.shape[0] == 1

    def test_large_batch_size(self):
        """Test PoolingFeedForwardLegacy with large batch size."""
        pff = PoolingFeedForwardLegacy(n_modes=8, n_photons=2, n_output_modes=4)
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        batch_size = 32
        amplitudes = torch.rand(batch_size, n_input_states)
        output = pff(amplitudes)
        assert output.shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__])
