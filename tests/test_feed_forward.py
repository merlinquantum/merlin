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

from merlin.core.feed_forward import FeedForwardBlock


class TestFeedForward:
    """Test suite for FeedForward class."""

    def test_init(self):
        """Test FeedForward initialization."""
        ff = FeedForwardBlock(input_size=6, m=6, n=2, depth=3, conditional_mode=0)
        assert ff.m == 6
        assert ff.n_photons == 2
        assert ff.conditional_mode == 0
        assert ff.depth == 3
        assert isinstance(ff.layers, dict)

    def test_generate_possible_tuples(self):
        """Test possible tuples generation."""
        ff = FeedForwardBlock(input_size=2, n=2, m=4, conditional_mode=0)
        tuples = ff.generate_possible_tuples()
        assert isinstance(tuples, list)
        assert len(tuples) > 0

    def test_parameters_method(self):
        """Test parameters() method returns generator."""
        ff = FeedForwardBlock(input_size=4, n=2, m=4, depth=2, conditional_mode=0)
        params = list(ff.parameters())
        assert len(params) > 0
        assert all(isinstance(p, torch.Tensor) for p in params)

    def test_forward_pass(self):
        """Test forward pass execution."""
        ff = FeedForwardBlock(input_size=4, n=2, m=4, depth=2, conditional_mode=0)
        x = torch.rand(1, 4)
        output = ff(x)
        assert isinstance(output, torch.Tensor)
        assert output.requires_grad

    def test_backward_pass(self):
        """Test backward pass execution."""
        ff = FeedForwardBlock(input_size=4, n=2, m=4, depth=2, conditional_mode=0)
        x = torch.rand(1, 4, requires_grad=True)

        output = ff(x)
        loss = output.sum()

        # Should not raise an error
        loss.backward()

        # Check gradients are computed
        assert x.grad is not None

    def test_indices_by_value(self):
        """Test indices_by_value method."""
        ff = FeedForwardBlock(input_size=2, n=2, m=4, depth=2, conditional_mode=0)
        keys = [(0, 1, 0), (1, 0, 1), (0, 0, 1)]
        idx_0, idx_1 = ff._indices_by_value(keys, 0)

        assert len(idx_0) == 2  # positions where first element is 0
        assert len(idx_1) == 1  # positions where first element is 1

    def test_match_indices(self):
        """Test match_indices method."""
        ff = FeedForwardBlock(input_size=2, n=2, m=4, depth=2, conditional_mode=0)
        data = [(0, 1, 0), (1, 0, 1), (0, 0, 1)]
        data_out = [(1, 0), (0, 1), (0, 1)]

        idx = ff._match_indices(data, data_out, k=0, k_value=0)
        assert isinstance(idx, torch.Tensor)

    def test_integration_with_optimizer(self):
        """Test integration with PyTorch optimizer."""
        ff = FeedForwardBlock(input_size=4, n=2, m=4, depth=2, conditional_mode=0)
        x = torch.rand(1, 4)
        optimizer = torch.optim.Adam(ff.parameters())

        # Forward pass
        output = ff(x)
        loss = output.pow(2).sum()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Should complete without errors
        assert True


if __name__ == "__main__":
    pytest.main([__file__])
