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

"""StateVector with combinatorial metadata and conversions.

This module provides a lightweight :class:`StateVector` wrapper that keeps the
Fock-space metadata (number of modes, number of photons, basis ordering) tied to
its amplitude tensor. It supports dense and sparse tensors, Fock ordering via
:class:`~merlin.utils.combinadics.Combinadics`, and conversion to/from
:class:`perceval.StateVector`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from numbers import Number

import perceval as pcvl
import torch

from ..utils.combinadics import Combinadics
from ..utils.dtypes import complex_dtype_for

Basis = Combinadics


@cache
def _basis_for(n_modes: int, n_photons: int) -> Basis:
    return Combinadics("fock", n_photons, n_modes)


@cache
def _basis_size(n_modes: int, n_photons: int) -> int:
    return Combinadics("fock", n_photons, n_modes).compute_space_size()


def _to_complex_dense(
    tensor: torch.Tensor, *, dtype: torch.dtype | None, device: torch.device | None
) -> torch.Tensor:
    target_device = device or tensor.device
    target_dtype = dtype or (
        tensor.dtype if tensor.is_complex() else complex_dtype_for(torch.float32)
    )
    if tensor.is_complex():
        return tensor.to(device=target_device, dtype=target_dtype)
    if tensor.is_floating_point() or tensor.dtype in (
        torch.int32,
        torch.int64,
        torch.int16,
        torch.int8,
        torch.uint8,
    ):
        real = tensor.to(
            device=target_device, dtype=torch.promote_types(tensor.dtype, torch.float32)
        )
        imag = torch.zeros_like(real)
        return torch.complex(real, imag).to(dtype=target_dtype, device=target_device)
    raise TypeError(
        "Tensor dtype is not supported for complex conversion; expected real or complex inputs."
    )


def _to_complex(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if tensor.is_sparse:
        coalesced = tensor.coalesce()
        values = _to_complex_dense(coalesced.values(), dtype=dtype, device=device)
        return torch.sparse_coo_tensor(
            coalesced.indices(),
            values,
            coalesced.shape,
            device=device or coalesced.device,
        )
    return _to_complex_dense(tensor, dtype=dtype, device=device)


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_sparse:
        coalesced = tensor.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        if tensor.ndim == 1:
            norm_sq = torch.sum(values.abs().pow(2))
            if norm_sq == 0:
                return tensor
            norm = torch.sqrt(norm_sq)
            new_values = values / norm
            return torch.sparse_coo_tensor(
                indices, new_values, tensor.shape, device=tensor.device
            )

        nnz = values.shape[0]
        norm_map: dict[tuple[int, ...], torch.Tensor] = {}
        for col in range(nnz):
            batch_coords = tuple(int(v) for v in indices[:-1, col].tolist())
            contrib = values[col].abs().pow(2)
            norm_map[batch_coords] = (
                norm_map.get(
                    batch_coords,
                    torch.tensor(0.0, device=values.device, dtype=values.dtype),
                )
                + contrib
            )

        norm_map = {k: torch.sqrt(v) for k, v in norm_map.items()}
        scaled_values: list[torch.Tensor] = []
        for col in range(nnz):
            batch_coords = tuple(int(v) for v in indices[:-1, col].tolist())
            norm = norm_map.get(batch_coords)
            if norm is None or norm == 0:
                scaled_values.append(values[col])
            else:
                scaled_values.append(values[col] / norm)
        new_values_tensor = torch.stack(scaled_values)
        return torch.sparse_coo_tensor(
            indices, new_values_tensor, tensor.shape, device=tensor.device
        )

    norm = torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)
    norm_safe = torch.where(norm == 0, torch.ones_like(norm), norm)
    return tensor / norm_safe


def _ensure_last_dim(tensor: torch.Tensor, expected: int) -> None:
    if tensor.shape[-1] != expected:
        raise ValueError(
            f"Tensor last dimension {tensor.shape[-1]} does not match basis size {expected}."
        )


def _basic_state_counts(state: Sequence[int] | pcvl.BasicState) -> tuple[int, ...]:
    if isinstance(state, pcvl.BasicState):
        return tuple(int(x) for x in state)
    return tuple(int(x) for x in state)


def _basis_index_map(basis: Basis) -> dict[tuple[int, ...], int]:
    return {state: idx for idx, state in enumerate(basis)}


def _basic_state_tuple(state: Sequence[int] | pcvl.BasicState) -> tuple[int, ...]:
    if isinstance(state, pcvl.BasicState):
        return tuple(int(x) for x in state)
    return tuple(int(x) for x in state)


@dataclass
class StateVector:
    """Amplitude tensor bundled with its Fock metadata.

    Keeps ``n_modes`` / ``n_photons`` and combinadics basis ordering alongside the
    underlying PyTorch tensor (dense or sparse).
    """

    tensor: torch.Tensor
    n_modes: int
    n_photons: int
    _normalized: bool = field(default=False)

    def __setattr__(self, name: str, value) -> None:
        if name in ("n_modes", "n_photons") and name in self.__dict__:
            raise AttributeError("n_modes and n_photons are immutable once set")
        super().__setattr__(name, value)

    @property
    def is_normalized(self) -> bool:
        return self._normalized

    def _normalized_tensor(self) -> torch.Tensor:
        if self._normalized:
            return self.tensor
        normalized = _normalize_tensor(self.tensor)
        self.tensor = normalized
        self._normalized = True
        return normalized

    @property
    def basis(self) -> Basis:
        """Lazy combinadics basis for ``(n_modes, n_photons)`` in Fock ordering."""
        return _basis_for(self.n_modes, self.n_photons)

    @property
    def is_sparse(self) -> bool:
        """Return True if the underlying tensor uses a sparse layout."""
        return self.tensor.is_sparse

    @property
    def basis_size(self) -> int:
        """Return the number of basis states for ``(n_modes, n_photons)``."""
        return _basis_size(self.n_modes, self.n_photons)

    def memory_bytes(self) -> int:
        """Approximate memory footprint (bytes) of the underlying tensor data."""
        if self.tensor.is_sparse:
            coalesced = self._tensor_coalesced()
            idx = coalesced.indices()
            vals = coalesced.values()
            return int(
                idx.numel() * idx.element_size() + vals.numel() * vals.element_size()
            )
        return int(self.tensor.numel() * self.tensor.element_size())

    def _tensor_coalesced(self) -> torch.Tensor:
        if not self.tensor.is_sparse:
            return self.tensor
        if self.tensor.is_coalesced():
            return self.tensor
        coalesced = self.tensor.coalesce()
        self.tensor = coalesced
        return coalesced

    def _extract_single_state(self) -> tuple[tuple[int, ...], torch.Tensor] | None:
        """Detect one-hot vectors (single non-zero amplitude) and return (state, amplitude)."""
        if self.tensor.ndim != 1:
            return None
        if self.is_sparse:
            coalesced = self._tensor_coalesced()
            if coalesced._nnz() != 1:
                return None
            idx = int(coalesced.indices()[0, 0].item())
            return self.basis[idx], coalesced.values()[0]
        non_zero = torch.nonzero(self.tensor.abs(), as_tuple=False)
        if non_zero.numel() != 1:
            return None
        idx = int(non_zero[0].item())
        return self.basis[idx], self.tensor[idx]

    def to_perceval(self) -> pcvl.StateVector | list[pcvl.StateVector]:
        """Convert to ``pcvl.StateVector``.

        Args:
            None

        Returns:
            pcvl.StateVector | list[pcvl.StateVector]: A Perceval state for 1D tensors,
            or a list for batched tensors, with amplitudes preserved (no extra
            renormalization).
        """
        basis = self.basis
        if self.tensor.ndim == 1:
            return self._perceval_from_1d(self.tensor, basis)
        if self.is_sparse:
            return self._perceval_from_sparse_batch(self.tensor, basis)
        flat = self.tensor.reshape(-1, self.tensor.shape[-1])
        result: list[pcvl.StateVector] = []
        for row in flat:
            result.append(self._perceval_from_1d(row, basis))
        return result

    @staticmethod
    def _perceval_from_1d(vector: torch.Tensor, basis: Basis) -> pcvl.StateVector:
        if vector.is_sparse:
            coalesced = vector.coalesce()
            entries = zip(
                coalesced.indices().flatten().tolist(),
                coalesced.values().tolist(),
                strict=False,
            )
        else:
            entries = enumerate(vector.tolist())
        mapping = [
            (pcvl.BasicState(basis[idx]), complex(val))
            for idx, val in entries
            if val != 0 and val != 0.0
        ]
        acc: pcvl.StateVector | None = None
        for bs, amp in mapping:
            term = pcvl.StateVector(bs)
            if amp != 1:
                term = term * amp
            acc = term if acc is None else acc + term
        return acc if acc is not None else pcvl.StateVector()

    @staticmethod
    def _perceval_from_sparse_batch(
        tensor: torch.Tensor, basis: Basis
    ) -> list[pcvl.StateVector]:
        if tensor.ndim < 2:
            raise ValueError("Expected batched tensor for sparse batch conversion.")
        coalesced = tensor.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        if indices.shape[0] != tensor.ndim:
            raise ValueError("Sparse indices rank does not match tensor rank.")
        batch_shape = tensor.shape[:-1]
        basis_dim = tensor.shape[-1]
        if basis_dim != len(basis):
            raise ValueError("Basis size mismatch in sparse batch conversion.")

        batch_maps: dict[tuple[int, ...], dict[int, complex]] = {}
        nnz = values.shape[0]
        for col in range(nnz):
            batch_coords = tuple(int(v) for v in indices[:-1, col].tolist())
            basis_idx = int(indices[-1, col].item())
            amp = complex(values[col].item())
            if amp == 0 or amp == 0.0:
                continue
            bucket = batch_maps.setdefault(batch_coords, {})
            bucket[basis_idx] = amp

        total_batches = 1
        for dim in batch_shape:
            total_batches *= dim

        def _coords_from_linear(linear: int) -> tuple[int, ...]:
            coords: list[int] = []
            rem = linear
            for dim in reversed(batch_shape):
                rem, idx = divmod(rem, dim)
                coords.append(idx)
            coords.reverse()
            return tuple(coords)

        result: list[pcvl.StateVector] = []
        for linear_idx in range(total_batches):
            coords = _coords_from_linear(linear_idx)
            entries = batch_maps.get(coords, {})
            if not entries:
                result.append(pcvl.StateVector())
                continue
            acc: pcvl.StateVector | None = None
            for basis_idx, amp in entries.items():
                if amp == 0 or amp == 0.0:
                    continue
                term = pcvl.StateVector(pcvl.BasicState(basis[basis_idx]))
                if amp != 1:
                    term = term * complex(amp)
                acc = term if acc is None else acc + term
            result.append(acc if acc is not None else pcvl.StateVector())
        return result

    @classmethod
    def from_perceval(
        cls,
        state_vector: pcvl.StateVector,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        sparse: bool | None = None,
    ) -> StateVector:
        """Build from a ``pcvl.StateVector``.

        Args:
            state_vector: Perceval state to wrap.
            dtype: Optional target dtype.
            device: Optional target device.
            sparse: Force sparse/dense; if None use density heuristic (<=30%).

        Returns:
            StateVector: Merlin wrapper with metadata and preserved amplitudes.

        Raises:
            ValueError: If the Perceval state is empty or has inconsistent photon/mode counts.
        """
        items = list(state_vector)
        if not items:
            raise ValueError("Perceval StateVector is empty.")
        n_modes = len(items[0][0])
        n_photons = sum(int(v) for v in items[0][0])
        for basic, _ in items[1:]:
            if len(basic) != n_modes:
                raise ValueError("Inconsistent mode count in perceval StateVector.")
            if sum(int(v) for v in basic) != n_photons:
                raise ValueError(
                    "Perceval StateVector must have uniform photon number."
                )
        basis = _basis_for(n_modes, n_photons)
        index_map = _basis_index_map(basis)
        if sparse is None:
            basis_size = _basis_size(n_modes, n_photons)
            sparse = (len(items) / basis_size) <= 0.3
        if sparse:
            indices_list: list[int] = []
            values_list: list[complex] = []
            for basic, amplitude in items:
                idx = index_map.get(tuple(int(v) for v in basic))
                if idx is None:
                    continue
                amp_complex = complex(amplitude)
                if amp_complex == 0 or amp_complex == 0.0:
                    continue
                indices_list.append(idx)
                values_list.append(amp_complex)
            if not indices_list:
                zero = torch.zeros(
                    _basis_size(n_modes, n_photons),
                    device=device or torch.device("cpu"),
                )
                return cls(
                    _to_complex(zero, dtype=dtype, device=device), n_modes, n_photons
                )
            indices = torch.tensor([indices_list], dtype=torch.long, device=device)
            values = torch.tensor(
                values_list, dtype=complex_dtype_for(torch.float32), device=device
            )
            tensor = torch.sparse_coo_tensor(
                indices, values, (_basis_size(n_modes, n_photons),), device=device
            )
            tensor = _to_complex(tensor, dtype=dtype, device=device)
            return cls(tensor, n_modes, n_photons, _normalized=False)
        dense = torch.zeros(
            _basis_size(n_modes, n_photons),
            dtype=complex_dtype_for(torch.float32),
            device=device,
        )
        for basic, amplitude in items:
            idx = index_map.get(tuple(int(v) for v in basic))
            if idx is None:
                continue
            dense[idx] = complex(amplitude)
        dense = _to_complex(dense, dtype=dtype, device=device)
        return cls(dense, n_modes, n_photons, _normalized=False)

    @classmethod
    def from_basic_state(
        cls,
        state: Sequence[int] | pcvl.BasicState,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        sparse: bool = True,
    ) -> StateVector:
        """Create a one-hot state from a Fock occupation list/BasicState.

        Args:
            state: Occupation numbers per mode.
            dtype: Optional target dtype.
            device: Optional target device.
            sparse: Build sparse layout when True.

        Returns:
            StateVector: One-hot state.
        """
        counts = _basic_state_counts(state)
        n_modes = len(counts)
        n_photons = sum(counts)
        comb = Combinadics("fock", n_photons, n_modes)
        index = comb.fock_to_index(counts)
        basis_size = comb.compute_space_size()
        if sparse:
            indices = torch.tensor([[index]], dtype=torch.long, device=device)
            values = torch.ones(
                1, dtype=complex_dtype_for(torch.float32), device=device
            )
            tensor = torch.sparse_coo_tensor(
                indices, values, (basis_size,), device=device
            )
        else:
            tensor = torch.zeros(
                basis_size, dtype=complex_dtype_for(torch.float32), device=device
            )
            tensor[index] = 1.0
        tensor = _to_complex(tensor, dtype=dtype, device=device)
        return cls(tensor, n_modes, n_photons, _normalized=True)

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        n_modes: int,
        n_photons: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> StateVector:
        """Wrap an existing tensor with explicit metadata.

        Args:
            tensor: Dense or sparse amplitude tensor.
            n_modes: Number of modes.
            n_photons: Total photons.
            dtype: Optional target dtype.
            device: Optional target device.

        Returns:
            StateVector: Wrapped tensor.

        Raises:
            ValueError: If the last dimension does not match the basis size.
        """
        basis_size = _basis_size(n_modes, n_photons)
        _ensure_last_dim(tensor, basis_size)
        normalized = _to_complex(tensor, dtype=dtype, device=device)
        return cls(normalized, n_modes, n_photons, _normalized=False)

    def tensor_product(
        self,
        other: StateVector | Sequence[int] | pcvl.BasicState,
        *,
        sparse: bool | None = None,
    ) -> StateVector:
        """Tensor product of two states with metadata propagation.

        If any operand is dense, the result is dense. Supports one-hot fast path.
        The resulting state is normalized before returning.

        Args:
            other: Another StateVector or a BasicState/occupation list.
            sparse: Override sparsity of the result; default keeps dense if any input dense.

        Returns:
            StateVector: Combined state with summed modes/photons (normalized).

        Raises:
            ValueError: If tensors are not 1D.
        """
        if not isinstance(other, StateVector):
            other = StateVector.from_basic_state(
                other,
                device=self.tensor.device,
                dtype=self.tensor.dtype,
                sparse=self.is_sparse if sparse is None else sparse,
            )
        if self.tensor.ndim != 1 or other.tensor.ndim != 1:
            raise ValueError("tensor_product currently supports 1D state tensors only.")
        m_total = self.n_modes + other.n_modes
        n_total = self.n_photons + other.n_photons
        basis_total = _basis_for(m_total, n_total)
        basis_left = self.basis
        basis_right = other.basis
        left_index = _basis_index_map(basis_left)
        right_index = _basis_index_map(basis_right)
        size_total = len(basis_total)

        if sparse is None:
            sparse = self.is_sparse and other.is_sparse

        left_single = self._extract_single_state()
        right_single = other._extract_single_state()
        if left_single is not None:
            return self._product_with_basic(
                left_single, other, basic_on_left=True, sparse=sparse
            )
        if right_single is not None:
            return self._product_with_basic(
                right_single, self, basic_on_left=False, sparse=sparse
            )

        left_dense = self.to_dense()
        right_dense = other.to_dense()
        if right_dense.device != left_dense.device:
            right_dense = right_dense.to(left_dense.device)
        if right_dense.dtype != left_dense.dtype:
            right_dense = right_dense.to(left_dense.dtype)
        return self._dense_product(
            left_dense,
            right_dense,
            basis_total,
            left_index,
            right_index,
            size_total,
            m_split=self.n_modes,
            n_modes_total=m_total,
            n_photons_total=n_total,
        )

    def _product_with_basic(
        self,
        basic_entry: tuple[tuple[int, ...], torch.Tensor],
        other: StateVector,
        *,
        basic_on_left: bool,
        sparse: bool | None,
    ) -> StateVector:
        basic_state, basic_amp = basic_entry
        device = other.tensor.device
        dtype = other.tensor.dtype
        amp_scalar = basic_amp.to(device=device, dtype=dtype)
        m_total = len(basic_state) + other.n_modes
        n_total = sum(basic_state) + other.n_photons
        comb_total = Combinadics("fock", n_total, m_total)
        size_total = comb_total.compute_space_size()
        basis_other = other.basis

        use_sparse = sparse
        if use_sparse:
            coalesced = (
                other.tensor.coalesce() if other.is_sparse else other.tensor.to_sparse()
            )
            idx_list: list[int] = []
            val_list: list[torch.Tensor] = []
            flat_indices = coalesced.indices().flatten().tolist()
            values = coalesced.values().to(device=device, dtype=dtype)
            for pos, val in zip(flat_indices, values, strict=False):
                state_other = basis_other[pos]
                combined = (
                    basic_state + state_other
                    if basic_on_left
                    else state_other + basic_state
                )
                idx_total = comb_total.fock_to_index(combined)
                idx_list.append(idx_total)
                val_list.append(amp_scalar * val)
            if not idx_list:
                zero = torch.zeros(size_total, dtype=dtype, device=device)
                return StateVector(zero, m_total, n_total)
            indices = torch.tensor([idx_list], dtype=torch.long, device=device)
            values_tensor = torch.stack(val_list)
            tensor = torch.sparse_coo_tensor(
                indices, values_tensor, (size_total,), device=device
            )
            return StateVector(
                _normalize_tensor(tensor), m_total, n_total, _normalized=True
            )

        other_dense = other.to_dense().to(device=device, dtype=dtype)
        output = torch.zeros(size_total, dtype=dtype, device=device)
        for idx_other, state_other in enumerate(basis_other):
            combined = (
                basic_state + state_other
                if basic_on_left
                else state_other + basic_state
            )
            idx_total = comb_total.fock_to_index(combined)
            output[idx_total] = amp_scalar * other_dense[idx_other]
        return StateVector(
            _normalize_tensor(output), m_total, n_total, _normalized=True
        )

    def __add__(self, other: StateVector) -> StateVector:
        """Add two states without renormalization (lazy norm, like Perceval).

        Args:
            other: StateVector with matching metadata.

        Returns:
            StateVector: Sum with raw amplitudes preserved.

        Raises:
            ValueError: If metadata mismatches.
        """
        if not isinstance(other, StateVector):
            return NotImplemented
        if self.n_modes != other.n_modes or self.n_photons != other.n_photons:
            raise ValueError(
                "StateVector addition requires matching n_modes and n_photons."
            )
        target_sparse = self.is_sparse and other.is_sparse
        if target_sparse:
            summed = self._tensor_coalesced() + other.tensor.coalesce()
            return StateVector(summed, self.n_modes, self.n_photons, _normalized=False)
        left = self.tensor.to_dense() if self.is_sparse else self.tensor
        right = other.tensor.to_dense() if other.is_sparse else other.tensor
        if right.device != left.device:
            right = right.to(left.device)
        if right.dtype != left.dtype:
            right = right.to(left.dtype)
        summed = left + right
        return StateVector(summed, self.n_modes, self.n_photons, _normalized=False)

    def __sub__(self, other: StateVector) -> StateVector:
        """Subtract two states without renormalization (lazy norm).

        Args:
            other: StateVector with matching metadata.

        Returns:
            StateVector: Difference with raw amplitudes preserved.

        Raises:
            ValueError: If metadata mismatches.
        """
        if not isinstance(other, StateVector):
            return NotImplemented
        if self.n_modes != other.n_modes or self.n_photons != other.n_photons:
            raise ValueError(
                "StateVector subtraction requires matching n_modes and n_photons."
            )
        target_sparse = self.is_sparse and other.is_sparse
        if target_sparse:
            diff = self._tensor_coalesced() - other.tensor.coalesce()
            return StateVector(diff, self.n_modes, self.n_photons, _normalized=False)
        left = self.tensor.to_dense() if self.is_sparse else self.tensor
        right = other.tensor.to_dense() if other.is_sparse else other.tensor
        if right.device != left.device:
            right = right.to(left.device)
        if right.dtype != left.dtype:
            right = right.to(left.dtype)
        diff = left - right
        return StateVector(diff, self.n_modes, self.n_photons, _normalized=False)

    def __mul__(self, scalar: Number) -> StateVector:
        """Scale amplitudes by a scalar (no renormalization)."""
        if not isinstance(scalar, Number):
            return NotImplemented
        if self.is_sparse:
            return StateVector(
                self.tensor * scalar, self.n_modes, self.n_photons, _normalized=False
            )
        return StateVector(
            self.tensor * scalar, self.n_modes, self.n_photons, _normalized=False
        )

    def __rmul__(self, scalar: Number) -> StateVector:
        """Right scalar multiplication delegation."""
        return self.__mul__(scalar)

    def __matmul__(
        self, other: StateVector | Sequence[int] | pcvl.BasicState
    ) -> StateVector:
        """Tensor product operator alias for ``tensor_product``."""
        if isinstance(other, (StateVector, Sequence, pcvl.BasicState)):
            return self.tensor_product(other)
        return NotImplemented

    def __rmatmul__(
        self, other: StateVector | Sequence[int] | pcvl.BasicState
    ) -> StateVector:
        """Right tensor product to support BasicState/sequence @ StateVector."""
        if isinstance(other, StateVector):
            return other.tensor_product(self)
        if isinstance(other, (Sequence, pcvl.BasicState)):
            left = StateVector.from_basic_state(
                other,
                device=self.tensor.device,
                dtype=self.tensor.dtype,
                sparse=self.is_sparse,
            )
            return left.tensor_product(self)
        return NotImplemented

    def index(self, state: Sequence[int] | pcvl.BasicState) -> int | None:
        """Return basis index for the given Fock state.

        Args:
            state: Occupation list or BasicState.

        Returns:
            int | None: Basis index, or None if not present (or zero in sparse tensor).
        """
        target = _basic_state_tuple(state)
        basis = self.basis
        try:
            idx = basis.index(target)
        except ValueError:
            return None
        if self.is_sparse:
            coalesced = self._tensor_coalesced()
            positions = (coalesced.indices()[-1] == idx).nonzero(as_tuple=False)
            return idx if positions.numel() > 0 else None
        return idx

    def __getitem__(self, state: Sequence[int] | pcvl.BasicState) -> torch.Tensor:
        """Amplitude lookup for a given Fock state.

        Args:
            state: Occupation list or BasicState.

        Returns:
            torch.Tensor: Amplitude (scalar or batch-aligned tensor).

        Raises:
            KeyError: If the state is outside the basis.
        """
        target = _basic_state_tuple(state)
        basis = self.basis
        try:
            idx = basis.index(target)
        except ValueError:
            raise KeyError("State not in basis") from None

        normalized = self._normalized_tensor()

        if normalized.ndim == 1:
            if normalized.is_sparse:
                coalesced = normalized.coalesce()
                mask = coalesced.indices()[0] == idx
                positions = mask.nonzero(as_tuple=False)
                if positions.numel() == 0:
                    return torch.zeros(
                        (), dtype=normalized.dtype, device=normalized.device
                    )
                return coalesced.values()[positions[0, 0]]
            return normalized[idx]

        # Batched: gather amplitudes for each batch entry
        if normalized.is_sparse:
            coalesced = normalized.coalesce()
            indices = coalesced.indices()
            values = coalesced.values()
            batch_shape = normalized.shape[:-1]
            batch_size = 1
            for dim in batch_shape:
                batch_size *= dim
            out = torch.zeros(
                batch_size, dtype=normalized.dtype, device=normalized.device
            )
            nnz = values.shape[0]
            for col in range(nnz):
                if int(indices[-1, col].item()) != idx:
                    continue
                # map batch coords to linear
                coords = [int(v) for v in indices[:-1, col].tolist()]
                linear = 0
                for d, s in zip(coords, batch_shape, strict=False):
                    linear = linear * s + d
                out[linear] = values[col]
            return out.view(*batch_shape)

        flat = normalized.reshape(-1, normalized.shape[-1])
        gathered = flat[..., idx]
        return gathered.view(*normalized.shape[:-1])

    def _dense_product(
        self,
        left_tensor: torch.Tensor,
        right_tensor: torch.Tensor,
        basis_total: Basis,
        left_index: dict[tuple[int, ...], int],
        right_index: dict[tuple[int, ...], int],
        size_total: int,
        *,
        m_split: int,
        n_modes_total: int,
        n_photons_total: int,
    ) -> StateVector:
        device = left_tensor.device
        dtype = left_tensor.dtype
        output = torch.zeros(size_total, dtype=dtype, device=device)
        for idx_total, state in enumerate(basis_total):
            left_state = state[:m_split]
            right_state = state[m_split:]
            idx_left = left_index.get(left_state)
            idx_right = right_index.get(right_state)
            if idx_left is None or idx_right is None:
                continue
            output[idx_total] = left_tensor[idx_left] * right_tensor[idx_right]
        return StateVector(
            _normalize_tensor(output), n_modes_total, n_photons_total, _normalized=True
        )

    def to_dense(self) -> torch.Tensor:
        """Return a dense, normalized tensor view of the amplitudes."""
        normalized = self._normalized_tensor()
        return normalized.to_dense() if normalized.is_sparse else normalized

    def normalize(self) -> StateVector:
        """Normalize this state in-place and return self."""
        if self._normalized:
            return self
        normalized_tensor = _normalize_tensor(self.tensor)
        self.tensor = normalized_tensor
        self._normalized = True
        return self

    def normalized_str(self) -> str:
        """Human-friendly string of the normalized state (forces normalization for display)."""
        normalized = self.normalize()
        return f"StateVector(n_modes={normalized.n_modes}, n_photons={normalized.n_photons}, tensor={normalized.tensor})"

    def __str__(self) -> str:  # pragma: no cover - simple wrapper
        return self.normalized_str()


__all__ = ["StateVector"]
