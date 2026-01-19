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

"""Combinatorial ranking and unranking for optical Fock states.

This module implements the :class:`Combinadics` utility class, providing a unified
interface to enumerate photonic occupation patterns under different constraints.

The implementation complexity is bounded O(n*m) for both ranking and unranking, where n is the
number of photons and m is the number of modes.
Could be optimized further using numba, and caching of binomial coefficients.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from typing import overload

TupleInt = tuple[int, ...]


class Combinadics:
    """Rank/unrank Fock states in descending lexicographic order.

    Parameters
    ----------
    scheme : str
        Enumeration strategy. Supported values are ``"fock"``, ``"unbunched"``,
        and ``"dual_rail"``.
    n : int
        Number of photons. Must be non-negative.
    m : int
        Number of modes. Must be at least one.

    Raises
    ------
    ValueError
        If an unsupported scheme is provided or the parameters violate the
        constraints of the selected scheme.
    """

    def __init__(self, scheme: str, n: int, m: int) -> None:
        scheme_lower = scheme.lower()
        if scheme_lower not in {"fock", "unbunched", "dual_rail"}:
            raise ValueError('scheme must be one of {"fock","unbunched","dual_rail"}')
        if n < 0:
            raise ValueError("n must be non-negative")
        if m <= 0:
            raise ValueError("m must be strictly positive")

        if scheme_lower == "dual_rail":
            if m % 2 != 0:
                raise ValueError("dual_rail requires even m")
            if n != m // 2:
                raise ValueError("dual_rail requires n = m // 2")
        if scheme_lower == "unbunched" and n > m:
            raise ValueError("unbunched requires n <= m")

        self.scheme = scheme_lower
        self.n = n
        self.m = m

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def fock_to_index(self, counts: Iterable[int]) -> int:
        """Map a Fock state to its index under the configured scheme.

        Parameters
        ----------
        counts : Iterable[int]
            Photon counts per mode.

        Returns
        -------
        int
            Rank of the Fock state in descending lexicographic order.

        Raises
        ------
        ValueError
            If ``counts`` violates the scheme-specific constraints.
        """
        state = self._validate_counts(counts, self.n, self.m)
        if self.scheme == "fock":
            return self._rank_fock(state)
        if self.scheme == "unbunched":
            return self._rank_unbunched(state)
        return self._rank_dualrail(state)

    def index_to_fock(self, index: int) -> TupleInt:
        """Map an index to its corresponding Fock state.

        Parameters
        ----------
        index : int
            Rank to decode.

        Returns
        -------
        Tuple[int, ...]
            Photon counts per mode.

        Raises
        ------
        ValueError
            If ``index`` is outside the valid range for the configured scheme.
        """
        if self.scheme == "fock":
            return self._unrank_fock(index)
        if self.scheme == "unbunched":
            return self._unrank_unbunched(index)
        return self._unrank_dualrail(index)

    @overload
    def __getitem__(self, key: int) -> TupleInt: ...

    @overload
    def __getitem__(self, key: Iterable[int]) -> int: ...

    def __getitem__(self, key: int | Iterable[int]) -> int | TupleInt:
        """Symmetric lookup: rank -> state or state -> rank.

        Parameters
        ----------
        key : int | Iterable[int]
            Integer rank to decode, or photon counts to encode.

        Returns
        -------
        tuple[int, ...] | int
            State when given an ``int``; rank when given counts.

        Raises
        ------
        TypeError
            If ``key`` is neither an int nor an iterable of ints.
        """

        if isinstance(key, int):
            return self.index_to_fock(key)
        if isinstance(key, Iterable) and not isinstance(key, (str, bytes)):
            return self.fock_to_index(key)
        raise TypeError("Combinadics key must be an int or iterable of ints")

    def index(self, state: Iterable[int]) -> int:
        """Return the rank for a given state (``tuple.index`` compatible).

        Parameters
        ----------
        state : Iterable[int]
            Photon counts per mode.

        Returns
        -------
        int
            Rank of the provided state.

        Raises
        ------
        ValueError
            If the state violates the configured scheme.
        """

        return self.fock_to_index(state)

    def compute_space_size(self) -> int:
        """Return the number of admissible Fock states for this configuration.

        Returns
        -------
        int
            Cardinality of the state space.
        """
        if self.scheme == "fock":
            return math.comb(self.n + self.m - 1, self.m - 1)
        if self.scheme == "unbunched":
            return math.comb(self.m, self.n)
        return 1 << (self.m // 2)

    def __len__(self) -> int:
        """Alias for :meth:`compute_space_size` to mirror sequence semantics."""

        return self.compute_space_size()

    def __iter__(self) -> Iterator[TupleInt]:
        """Yield states in descending lexicographic order (iterator protocol)."""

        return self.iter_states()

    def iter_states(self) -> Iterator[TupleInt]:
        """Yield admissible states in descending lexicographic order.

        Returns
        -------
        Iterator[Tuple[int, ...]]
            Generator over states matching the configured scheme.
        """

        if self.scheme == "fock":
            yield from self._iter_fock(self.n, 0, ())
        elif self.scheme == "unbunched":
            yield from self._iter_unbunched(self.n, 0, ())
        else:
            yield from self._iter_dualrail()

    def enumerate_states(self) -> list[TupleInt]:
        """Return all admissible states in descending lexicographic order.

        Returns
        -------
        list[Tuple[int, ...]]
            State list matching :meth:`iter_states`.
        """

        return list(self.iter_states())

    # --------------------------------------------------------------------- #
    # Validation helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _validate_counts(
        counts: Iterable[int],
        n: int,
        m: int,
    ) -> TupleInt:
        state = tuple(counts)
        if len(state) != m:
            raise ValueError(f"counts must have length m={m}")
        if any(c < 0 for c in state):
            raise ValueError("counts must be nonnegative")
        if sum(state) != n:
            raise ValueError(f"sum(counts) must be n={n}")
        return state

    @staticmethod
    def _binom(n: int, k: int) -> int:
        return math.comb(n, k) if 0 <= k <= n else 0

    # --------------------------------------------------------------------- #
    # Fock (weak compositions)
    # --------------------------------------------------------------------- #
    def _rank_fock(self, counts: TupleInt) -> int:
        rank = 0
        remaining = self.n
        for i in range(self.m - 1):
            ni = counts[i]
            for a in range(remaining, ni, -1):
                rank += self._binom((remaining - a) + (self.m - i - 2), self.m - i - 2)
            remaining -= ni
        return rank

    def _unrank_fock(self, index: int) -> TupleInt:
        total = self.compute_space_size()
        if not (0 <= index < total):
            raise ValueError(f"index out of range [0, {total - 1}] for fock scheme")

        counts = [0] * self.m
        r = index
        remaining = self.n
        for i in range(self.m - 1):
            for a in range(remaining, -1, -1):
                block = self._binom((remaining - a) + (self.m - i - 2), self.m - i - 2)
                if r < block:
                    counts[i] = a
                    remaining -= a
                    break
                r -= block
        counts[-1] = remaining
        return tuple(counts)

    def _iter_fock(
        self, remaining: int, position: int, prefix: TupleInt
    ) -> Iterator[TupleInt]:
        if position == self.m - 1:
            yield prefix + (remaining,)
            return
        for a in range(remaining, -1, -1):
            yield from self._iter_fock(remaining - a, position + 1, prefix + (a,))

    # --------------------------------------------------------------------- #
    # Unbunched (collision free)
    # --------------------------------------------------------------------- #
    def _check_unbunched(self, counts: TupleInt) -> None:
        if any(c not in (0, 1) for c in counts):
            raise ValueError("unbunched requires counts in {0, 1}")

    def _rank_unbunched(self, counts: TupleInt) -> int:
        self._check_unbunched(counts)
        rank = 0
        ones_used = 0
        for i, xi in enumerate(counts, start=1):
            remaining_positions = self.m - i
            if xi == 0:
                rank += self._binom(remaining_positions, self.n - (ones_used + 1))
            else:
                ones_used += 1
        return rank

    def _unrank_unbunched(self, index: int) -> TupleInt:
        total = self.compute_space_size()
        if not (0 <= index < total):
            raise ValueError(
                f"index out of range [0, {total - 1}] for unbunched scheme"
            )

        counts = [0] * self.m
        r = index
        ones_used = 0
        for i in range(1, self.m + 1):
            remaining_positions = self.m - i
            with_one = self._binom(remaining_positions, self.n - (ones_used + 1))
            if ones_used < self.n and r < with_one:
                counts[i - 1] = 1
                ones_used += 1
            elif ones_used < self.n:
                r -= with_one
        return tuple(counts)

    def _iter_unbunched(
        self, ones_left: int, position: int, prefix: TupleInt
    ) -> Iterator[TupleInt]:
        if position == self.m:
            if ones_left == 0:
                yield prefix
            return

        remaining_positions = self.m - position
        if ones_left > remaining_positions:
            return

        if ones_left > 0:
            yield from self._iter_unbunched(ones_left - 1, position + 1, prefix + (1,))
        if remaining_positions - 1 >= ones_left:
            yield from self._iter_unbunched(ones_left, position + 1, prefix + (0,))

    # --------------------------------------------------------------------- #
    # Dual rail (exactly one photon per pair)
    # --------------------------------------------------------------------- #
    def _check_dualrail(self, counts: TupleInt) -> None:
        for k in range(0, self.m, 2):
            a, b = counts[k], counts[k + 1]
            if a not in (0, 1) or b not in (0, 1) or a + b != 1:
                raise ValueError(
                    f"invalid dual-rail pair at modes {k + 1},{k + 2}: {(a, b)}"
                )

    def _rank_dualrail(self, counts: TupleInt) -> int:
        self._check_dualrail(counts)
        value = 0
        pairs = self.m // 2
        for k in range(pairs):
            bit = 1 if counts[2 * k] == 1 else 0
            value = (value << 1) | bit
        return (1 << pairs) - 1 - value

    def _unrank_dualrail(self, index: int) -> TupleInt:
        total = self.compute_space_size()
        if not (0 <= index < total):
            raise ValueError(
                f"index out of range [0, {total - 1}] for dual_rail scheme"
            )

        pairs = self.m // 2
        value = (total - 1) - index
        counts = [0] * self.m
        for k in range(pairs - 1, -1, -1):
            bit = (value >> (pairs - 1 - k)) & 1
            i = 2 * k
            if bit == 1:
                counts[i], counts[i + 1] = 1, 0
            else:
                counts[i], counts[i + 1] = 0, 1
        return tuple(counts)

    def _iter_dualrail(self) -> Iterator[TupleInt]:
        total = self.compute_space_size()
        pairs = self.m // 2
        for value in range(total - 1, -1, -1):
            counts = [0] * self.m
            for k in range(pairs - 1, -1, -1):
                bit = (value >> (pairs - 1 - k)) & 1
                i = 2 * k
                if bit == 1:
                    counts[i], counts[i + 1] = 1, 0
                else:
                    counts[i], counts[i + 1] = 0, 1
            yield tuple(counts)
