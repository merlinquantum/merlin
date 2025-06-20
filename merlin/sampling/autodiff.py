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
Automatic differentiation handling for sampling.
"""

import warnings
from typing import Tuple
from .process import SamplingProcess


class AutoDiffProcess:
    """Handles automatic differentiation backend and sampling noise integration."""

    def __init__(self):
        self.sampling_noise = SamplingProcess()

    def autodiff_backend(self, needs_gradient: bool, apply_sampling: bool,
                         shots: int) -> Tuple[bool, int]:
        """Determine sampling configuration based on gradient requirements."""
        if needs_gradient and (apply_sampling or shots > 0):
            warnings.warn(
                "Sampling was requested but is disabled because gradients are being computed. "
                "Sampling during gradient computation would lead to incorrect gradients.",
                UserWarning
            )
            return False, 0
        return apply_sampling, shots