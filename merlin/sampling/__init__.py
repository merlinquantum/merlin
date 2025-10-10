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

"""Sampling and autodiff utilities."""

from typing import TYPE_CHECKING

from .autodiff import AutoDiffProcess
from .process import SamplingProcess
from .strategies import OutputMappingStrategy

if TYPE_CHECKING:
    from ..torch_utils.torch_codes import (
        LexGroupingMapper,
        ModGroupingMapper,
        OutputMapper,
    )


def __getattr__(name):
    """Lazy import for backward compatibility"""
    import warnings

    if name in ("LexGroupingMapper", "ModGroupingMapper", "OutputMapper"):
        warnings.warn(
            f"Importing {name} from this location is deprecated. "
            f"Use 'from merlin.torch_utils.torch_codes import {name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..torch_utils.torch_codes import (  # noqa: F401
            LexGroupingMapper,
            ModGroupingMapper,
            OutputMapper,
        )

        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OutputMappingStrategy",
    "SamplingProcess",
    "AutoDiffProcess",
    "LexGroupingMapper",  # noqa: F401, F822
    "ModGroupingMapper",  # noqa: F401, F822
    "OutputMapper",  # noqa: F401, F822
]
