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
Quantum measurement sampling utilities.
"""

import torch


class SamplingProcess:
    """Handles quantum measurement sampling with different methods."""

    def __init__(self):
        self.gradient_method = "exact"  # Always use exact for gradients

    def pcvl_sampler(self, distribution: torch.Tensor, shots: int,
                     method: str = 'multinomial') -> torch.Tensor:
        """Apply sampling noise to a probability distribution."""
        if shots <= 0:
            return distribution

        # Validate method
        valid_methods = ['multinomial', 'binomial', 'gaussian']
        if method not in valid_methods:
            raise ValueError(f"Invalid sampling method: {method}. Valid options are: {valid_methods}")

        if method == 'multinomial':
            if distribution.dim() == 1:
                sampled_counts = torch.multinomial(
                    distribution, num_samples=shots, replacement=True
                )
                noisy_dist = torch.zeros_like(distribution)
                for idx in sampled_counts:
                    noisy_dist[idx] += 1
                return noisy_dist / shots
            else:
                batch_size = distribution.shape[0]
                noisy_dists = []
                for i in range(batch_size):
                    sampled_counts = torch.multinomial(
                        distribution[i], num_samples=shots, replacement=True
                    )
                    noisy_dist = torch.zeros_like(distribution[i])
                    for idx in sampled_counts:
                        noisy_dist[idx] += 1
                    noisy_dists.append(noisy_dist / shots)
                return torch.stack(noisy_dists)

        elif method == 'binomial':
            return torch.distributions.Binomial(shots, distribution).sample() / shots

        elif method == 'gaussian':
            std_dev = torch.sqrt(distribution * (1 - distribution) / shots)
            noise = torch.randn_like(distribution) * std_dev
            noisy_dist = distribution + noise
            noisy_dist = torch.clamp(noisy_dist, 0, 1)
            noisy_dist = noisy_dist / noisy_dist.sum(dim=-1, keepdim=True)
            return noisy_dist