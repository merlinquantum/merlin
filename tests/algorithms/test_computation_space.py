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

"""Lightweight regression tests for the ComputationSpace enum."""

import pytest

from merlin import ComputationSpace


def test_computation_space_members():
    """Enum members surface the expected string identifiers."""
    assert list(ComputationSpace) == [
        ComputationSpace.FOCK,
        ComputationSpace.UNBUNCHED,
        ComputationSpace.DUAL_RAIL,
    ]
    assert ComputationSpace.FOCK.value == "fock"
    assert ComputationSpace.UNBUNCHED.value == "unbunched"
    assert ComputationSpace.DUAL_RAIL.value == "dual_rail"


def test_computation_space_coerce_accepts_enum_members():
    """Enum inputs are returned unchanged."""
    assert ComputationSpace.coerce(ComputationSpace.FOCK) is ComputationSpace.FOCK
    assert (
        ComputationSpace.coerce(ComputationSpace.UNBUNCHED)
        is ComputationSpace.UNBUNCHED
    )


def test_computation_space_coerce_accepts_strings():
    """Case-insensitive string inputs are normalized into enum members."""
    assert ComputationSpace.coerce("fock") is ComputationSpace.FOCK
    assert ComputationSpace.coerce("unbunched") is ComputationSpace.UNBUNCHED
    assert ComputationSpace.coerce("Dual_Rail".lower()) is ComputationSpace.DUAL_RAIL


def test_computation_space_coerce_rejects_unknown():
    """Invalid inputs raise a ValueError with supported members listed."""
    with pytest.raises(ValueError, match="Invalid computation_space 'invalid'"):
        ComputationSpace.coerce("invalid")
