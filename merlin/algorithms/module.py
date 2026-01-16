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

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from ..utils.dtypes import complex_dtype_for


class MerlinModule(nn.Module):
    """Generic MerLin module with shared utility functions

    Merlin remote execution policy:
      - `_force_simulation` (bool) defaults to False. When True, the layer MUST run locally.
        The variable is set with property (getter and setter): `force_local`.
      - `supports_offload()` reports whether remote offload is possible (via `export_config()`).
      - `should_offload(processor, shots)` encapsulates the current offload policy:
            return supports_offload() and not force_local
      - `as_simulation()` provide local context forcing use as simulation
    """

    # -------------------- Execution policy & helpers --------------------

    @property
    def force_local(self) -> bool:
        """When True, this layer must run locally (Merlin will not offload it)."""
        return self._force_simulation

    @force_local.setter
    def force_local(self, value: bool) -> None:
        self._force_simulation = bool(value)

    @contextmanager
    def as_simulation(self):
        """Temporarily force local simulation within the context."""
        prev = self.force_local
        self.force_local = True
        try:
            yield self
        finally:
            self.force_local = prev

    # Offload capability & policy (queried by MerlinProcessor)
    def supports_offload(self) -> bool:
        """Return True if this layer is technically offloadable."""
        return hasattr(self, "export_config") and callable(self.export_config)

    def should_offload(self, _processor=None, _shots=None) -> bool:
        """Return True if this layer should be offloaded under current policy."""
        return self.supports_offload() and not self.force_local

    def __init__(self) -> None:
        super().__init__()

        # execution policy: when True, always simulate locally (do not offload)
        self._force_simulation: bool = False

    @staticmethod
    def setup_device_and_dtype(
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> tuple[torch.device | None, torch.dtype, torch.dtype]:
        """Normalize device/dtype to final forms."""
        resolved_dtype = dtype or torch.float32
        if resolved_dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "dtype must be torch.float32 or torch.float64 for Merlin modules."
            )
        resolved_complex = complex_dtype_for(resolved_dtype)
        return device, resolved_dtype, resolved_complex
