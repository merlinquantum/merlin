"""Sampling and autodiff support for quantum measurements."""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class SamplingNoise(nn.Module):
    """Apply sampling noise to probability distributions."""

    def __init__(self, method: str = 'multinomial'):
        super().__init__()
        self.method = method

    def pcvl_sampler(self,
                    distribution: torch.Tensor,
                    shots: int,
                    method: Optional[str] = None) -> torch.Tensor:
        """
        Apply sampling to distribution.

        Matches original QuantumLayer interface.
        """
        method = method or self.method

        if method == 'multinomial':
            return self._multinomial_sample(distribution, shots)
        elif method == 'gaussian':
            return self._gaussian_sample(distribution, shots)
        elif method == 'binomial':
            return self._binomial_sample(distribution, shots)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _multinomial_sample(self, probs: torch.Tensor, shots: int) -> torch.Tensor:
        """Multinomial sampling."""
        if probs.dim() == 1:
            indices = torch.multinomial(probs, shots, replacement=True)
            sampled = torch.zeros_like(probs)
            sampled.index_add_(0, indices, torch.ones_like(indices, dtype=probs.dtype))
            sampled /= shots
            return sampled
        else:
            # Batch sampling
            batch_size = probs.shape[0]
            sampled = torch.zeros_like(probs)
            for i in range(batch_size):
                indices = torch.multinomial(probs[i], shots, replacement=True)
                sampled[i].index_add_(0, indices, torch.ones_like(indices, dtype=probs.dtype))
            sampled /= shots
            return sampled

    def _gaussian_sample(self, probs: torch.Tensor, shots: int) -> torch.Tensor:
        """Gaussian noise approximation."""
        std = torch.sqrt(probs * (1 - probs) / shots)
        noise = torch.randn_like(probs) * std
        sampled = probs + noise
        sampled = torch.clamp(sampled, min=0)
        sampled = sampled / sampled.sum(dim=-1, keepdim=True)
        return sampled

    def _binomial_sample(self, probs: torch.Tensor, shots: int) -> torch.Tensor:
        """Binomial sampling."""
        sampled = torch.distributions.Binomial(total_count=shots, probs=probs).sample()
        return sampled / shots


class AutoDiffProcess:
    """Handle gradient flow through sampling."""

    def __init__(self, sampling_method: str = 'multinomial'):
        self.sampling_noise = SamplingNoise(sampling_method)
        self.sampling_method = sampling_method

    def autodiff_backend(self,
                        needs_gradient: bool,
                        apply_sampling: bool,
                        shots: int) -> Tuple[bool, int]:
        """
        Determine sampling strategy for gradients.

        Returns:
            (should_sample, num_shots)
        """
        if not apply_sampling or shots <= 0:
            return False, 0

        # Allow sampling with gradients (straight-through estimator)
        return True, shots

    def pcvl_sampler(self,
                    distribution: torch.Tensor,
                    shots: int,
                    method: Optional[str] = None) -> torch.Tensor:
        """Wrapper for sampling."""
        return self.sampling_noise.pcvl_sampler(distribution, shots, method)
