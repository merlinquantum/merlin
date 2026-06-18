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

"""Computation space definitions controlling logical basis selection."""

from enum import Enum


class ComputationSpace(str, Enum):
    """Enumeration of supported computational subspaces."""

    FOCK = "fock"
    UNBUNCHED = "unbunched"
    DUAL_RAIL = "dual_rail"

    @classmethod
    def default(cls, *, no_bunching: bool) -> "ComputationSpace":
        """Derive the default computation space from the legacy flag.

        Parameters
        ----------
        no_bunching : bool
            Legacy flag indicating whether bunching should be disallowed.

        Returns
        -------
        ComputationSpace
            Default computation space matching the legacy behavior.
        """
        return cls.UNBUNCHED if no_bunching else cls.FOCK

    @classmethod
    def coerce(cls, value: "ComputationSpace | str") -> "ComputationSpace":
        """Normalize user-provided values (enum instances or case-insensitive strings).

        Parameters
        ----------
        value : ComputationSpace | str
            Enum instance or case-insensitive string value.

        Returns
        -------
        ComputationSpace
            Normalized computation space value.

        Raises
        ------
        ValueError
            If ``value`` does not match a supported computation space.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.lower()
            for space in cls:
                if normalized == space.value:
                    return space
        supported = sorted(space.value for space in cls)
        raise ValueError(
            f"Invalid computation_space '{value}'. Supported values are {supported}."
        )
