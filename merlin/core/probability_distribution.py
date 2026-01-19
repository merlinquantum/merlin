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

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Union, cast

import perceval as pcvl
import torch

from ..utils.combinadics import Combinadics
from .computation_space import ComputationSpace

Basis = Union[Combinadics, "FilteredBasis", tuple[tuple[int, ...], ...]]


class FilteredBasis:
    """Lazy subset view over a base basis with bidirectional lookup."""

    def __init__(self, base: Basis, kept: Iterable[int]) -> None:
        self._base = base
        self._kept = tuple(int(i) for i in kept)
        self._state_to_idx = {self._base[i]: pos for pos, i in enumerate(self._kept)}

    def __len__(self) -> int:
        return len(self._kept)

    def __iter__(self) -> Iterable[tuple[int, ...]]:
        for pos in range(len(self._kept)):
            yield cast(tuple[int, ...], self._base[self._kept[pos]])

    def __getitem__(self, key: int | Iterable[int]) -> int | tuple[int, ...]:
        if isinstance(key, int):
            if key < 0 or key >= len(self._kept):
                raise IndexError("FilteredBasis index out of range")
            return self._base[self._kept[key]]
        state = tuple(int(x) for x in key)
        idx = self._state_to_idx.get(state)
        if idx is None:
            raise ValueError("State not in filtered basis")
        return idx

    def index(self, state: Iterable[int]) -> int:
        state_tuple = tuple(int(x) for x in state)
        idx = self._state_to_idx.get(state_tuple)
        if idx is None:
            raise ValueError("State not in filtered basis")
        return idx


@cache
def _basis_for_space(space: ComputationSpace, n_modes: int, n_photons: int) -> Basis:
    return Combinadics(space.value, n_photons, n_modes)


@cache
def _basis_size(space: ComputationSpace, n_modes: int, n_photons: int) -> int:
    return len(_basis_for_space(space, n_modes, n_photons))


def _ensure_last_dim(tensor: torch.Tensor, expected: int) -> None:
    if tensor.shape[-1] != expected:
        raise ValueError(
            f"Tensor last dimension {tensor.shape[-1]} does not match basis size {expected}."
        )


def _to_real(
    tensor: torch.Tensor, *, dtype: torch.dtype | None, device: torch.device | None
) -> torch.Tensor:
    target_device = device or tensor.device
    target_dtype = dtype or torch.promote_types(tensor.dtype, torch.float32)
    if tensor.is_sparse:
        coalesced = tensor.coalesce()
        values = coalesced.values().to(dtype=target_dtype, device=target_device)
        return torch.sparse_coo_tensor(
            coalesced.indices(), values, coalesced.shape, device=target_device
        )
    return tensor.to(dtype=target_dtype, device=target_device)


@dataclass
class ProbabilityDistribution:
    """Probability tensor bundled with Fock metadata and post-filter tracking.

    Parameters
    ----------
    tensor:
        Dense or sparse probabilities; leading dimensions are treated as batch axes.
    n_modes:
        Number of modes in the Fock space.
    n_photons:
        Total photon number represented by the distribution.
    computation_space:
        Basis enumeration used to order amplitudes (``fock``, ``unbunched``, ``dual_rail``).
    logical_performance:
        Optional per-batch scalar tracking kept/total probability after filtering.

    Notes
    -----
    Instances are normalized on construction; arithmetic-style temporary
    unnormalized states are not supported (unlike ``StateVector``).
    Only ``shape``, ``device``, ``dtype``, and ``requires_grad`` are delegated to
    the underlying ``torch.Tensor``; tensor-like helpers ``to``, ``clone``,
    ``detach``, and ``requires_grad_`` mirror tensor semantics while keeping
    metadata and logical performance aligned. Layout-changing tensor operations
    should be done on ``tensor`` directly, then wrapped again via ``from_tensor``
    to maintain a consistent basis.
    """

    tensor: torch.Tensor
    n_modes: int
    n_photons: int
    computation_space: ComputationSpace = field(default=ComputationSpace.FOCK)
    logical_performance: torch.Tensor | None = field(default=None)
    _custom_basis: Basis | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Probability distributions are always kept normalized on construction.
        self.normalize()

    def __setattr__(self, name: str, value) -> None:
        if name in ("n_modes", "n_photons") and name in self.__dict__:
            raise AttributeError("n_modes and n_photons are immutable once set")
        super().__setattr__(name, value)

    def __getattr__(self, name: str):
        allowed = {"shape", "device", "dtype", "requires_grad"}
        tensor = self.__dict__.get("tensor")
        if tensor is not None and name in allowed and hasattr(tensor, name):
            return getattr(tensor, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!s}")

    @property
    def basis(self) -> Basis:
        return (
            self._custom_basis
            if self._custom_basis is not None
            else _basis_for_space(self.computation_space, self.n_modes, self.n_photons)
        )

    @property
    def basis_size(self) -> int:
        return len(self.basis)

    def to(self, *args, **kwargs) -> ProbabilityDistribution:
        """Return a new ``ProbabilityDistribution`` with tensor (and logical_performance) moved/cast via ``torch.Tensor.to``."""
        new_tensor = self.tensor.to(*args, **kwargs)
        new_lp = None
        if self.logical_performance is not None:
            new_lp = self.logical_performance.to(*args, **kwargs)
        return ProbabilityDistribution(
            new_tensor,
            self.n_modes,
            self.n_photons,
            computation_space=self.computation_space,
            logical_performance=new_lp,
            _custom_basis=self._custom_basis,
        )

    def clone(self) -> ProbabilityDistribution:
        """Return a cloned ``ProbabilityDistribution`` with metadata and logical performance copied."""
        new_lp = None
        if self.logical_performance is not None:
            new_lp = self.logical_performance.clone()
        return ProbabilityDistribution(
            self.tensor.clone(),
            self.n_modes,
            self.n_photons,
            computation_space=self.computation_space,
            logical_performance=new_lp,
            _custom_basis=self._custom_basis,
        )

    def detach(self) -> ProbabilityDistribution:
        """Return a detached ``ProbabilityDistribution`` sharing data without gradients."""
        new_lp = None
        if self.logical_performance is not None:
            new_lp = self.logical_performance.detach()
        return ProbabilityDistribution(
            self.tensor.detach(),
            self.n_modes,
            self.n_photons,
            computation_space=self.computation_space,
            logical_performance=new_lp,
            _custom_basis=self._custom_basis,
        )

    def requires_grad_(self, requires_grad: bool = True) -> ProbabilityDistribution:
        """Set ``requires_grad`` on underlying tensors and return self."""
        self.tensor.requires_grad_(requires_grad)
        if self.logical_performance is not None:
            self.logical_performance.requires_grad_(requires_grad)
        return self

    @property
    def is_sparse(self) -> bool:
        return self.tensor.is_sparse

    @property
    def is_normalized(self) -> bool:
        return True

    def _tensor_coalesced(self) -> torch.Tensor:
        if not self.tensor.is_sparse:
            return self.tensor
        if self.tensor.is_coalesced():
            return self.tensor
        coalesced = self.tensor.coalesce()
        self.tensor = coalesced
        return coalesced

    def memory_bytes(self) -> int:
        """Return the tensor's approximate memory footprint in bytes."""
        if self.tensor.is_sparse:
            coalesced = self._tensor_coalesced()
            idx = coalesced.indices()
            vals = coalesced.values()
            return int(
                idx.numel() * idx.element_size() + vals.numel() * vals.element_size()
            )
        return int(self.tensor.numel() * self.tensor.element_size())

    def normalize(self) -> ProbabilityDistribution:
        """In-place normalization; safe for zero-mass batches.

        Returns
        -------
        ProbabilityDistribution
            The same instance, normalized along the basis dimension.
        """
        if self.tensor.is_sparse:
            coalesced = self._tensor_coalesced()
            sums = torch.zeros(
                coalesced.shape[:-1], device=coalesced.device, dtype=coalesced.dtype
            )
            indices = coalesced.indices()
            values = coalesced.values()
            nnz = values.shape[0]
            if nnz == 0:
                # Nothing to renormalize; keep an empty sparse tensor with the same shape
                self.tensor = torch.sparse_coo_tensor(
                    indices, values, coalesced.shape, device=coalesced.device
                )
                return self
            for col in range(nnz):
                batch_coords = tuple(int(v) for v in indices[:-1, col].tolist())
                sums[batch_coords] += values[col]
            safe = torch.where(sums == 0, torch.ones_like(sums), sums)
            scaled_vals: list[torch.Tensor] = []
            for col in range(nnz):
                batch_coords = tuple(int(v) for v in indices[:-1, col].tolist())
                scaled_vals.append(values[col] / safe[batch_coords])
            new_vals = torch.stack(scaled_vals)
            self.tensor = torch.sparse_coo_tensor(
                indices, new_vals, coalesced.shape, device=coalesced.device
            )
            return self
        totals = self.tensor.sum(dim=-1, keepdim=True)
        safe = torch.where(totals == 0, torch.ones_like(totals), totals)
        self.tensor = self.tensor / safe
        return self

    def to_dense(self) -> torch.Tensor:
        """Return a dense, normalized tensor representation."""
        normalized = self.normalize().tensor
        return normalized.to_dense() if normalized.is_sparse else normalized

    def probabilities(self) -> torch.Tensor:
        """Alias for :meth:`to_dense` for readability."""
        return self.to_dense()

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        n_modes: int,
        n_photons: int,
        computation_space: ComputationSpace | str | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> ProbabilityDistribution:
        """Build a distribution from an explicit probability tensor.

        Parameters
        ----------
        tensor:
            Dense or sparse probability tensor; last dimension must match the basis size.
        n_modes / n_photons:
            Metadata for basis construction.
        computation_space:
            Optional basis scheme; defaults to ``fock``.
        dtype / device:
            Optional overrides for output tensor placement and precision.

        Raises
        ------
        ValueError
            If the last dimension does not match the expected basis size.
        """
        space = (
            ComputationSpace.coerce(computation_space)
            if computation_space
            else ComputationSpace.FOCK
        )
        basis_size = _basis_size(space, n_modes, n_photons)
        _ensure_last_dim(tensor, basis_size)
        real = _to_real(tensor, dtype=dtype, device=device)
        return cls(real, n_modes, n_photons, computation_space=space)

    @classmethod
    def from_state_vector(
        cls,
        state_vector,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        computation_space: ComputationSpace | str | None = None,
    ) -> ProbabilityDistribution:
        """Convert a ``StateVector`` to a probability distribution.

        Parameters
        ----------
        state_vector:
            Source amplitudes; must expose ``to_dense``, ``n_modes``, and ``n_photons``.
        dtype / device:
            Optional overrides for output tensor placement and precision.
        computation_space:
            Optional basis scheme; defaults to ``fock``.
        """
        dense = state_vector.to_dense()
        probs = dense.abs().pow(2)
        probs = _to_real(probs, dtype=dtype, device=device)
        space = (
            ComputationSpace.coerce(computation_space)
            if computation_space
            else ComputationSpace.FOCK
        )
        return cls(
            probs, state_vector.n_modes, state_vector.n_photons, computation_space=space
        )

    @classmethod
    def from_perceval(
        cls,
        distribution: pcvl.BSDistribution,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        sparse: bool | None = None,
    ) -> ProbabilityDistribution:
        """Construct from a Perceval ``BSDistribution``.

        Validates that all entries share the same photon number and mode count.

        Parameters
        ----------
        distribution:
            Input Perceval distribution.
        dtype / device:
            Optional overrides for output tensor placement and precision.
        sparse:
            Force dense or sparse output; default auto-selects based on fill ratio.

        Raises
        ------
        ValueError
            If the distribution is empty or inconsistent in shape/photon number.
        """
        keys = list(distribution)
        if not keys:
            raise ValueError("Perceval BSDistribution is empty.")
        items = [(key, distribution[key]) for key in keys]
        n_modes = len(items[0][0])
        n_photons = sum(int(v) for v in items[0][0])
        for basic, _ in items[1:]:
            if len(basic) != n_modes:
                raise ValueError("Inconsistent mode count in BSDistribution.")
            if sum(int(v) for v in basic) != n_photons:
                raise ValueError("BSDistribution must have uniform photon number.")
        basis = _basis_for_space(ComputationSpace.FOCK, n_modes, n_photons)
        iter_basis = cast(Iterable[tuple[int, ...]], basis)
        index_map = {state: idx for idx, state in enumerate(iter_basis)}
        basis_size = len(basis)
        if sparse is None:
            sparse = (len(items) / basis_size) <= 0.3
        if sparse:
            idxs: list[int] = []
            vals: list[float] = []
            for basic, prob in items:
                idx = index_map.get(tuple(int(v) for v in basic))
                if idx is None:
                    continue
                p_val = float(prob)
                if p_val == 0.0:
                    continue
                idxs.append(idx)
                vals.append(p_val)
            if not idxs:
                zero = torch.zeros(
                    basis_size,
                    dtype=torch.float32,
                    device=device or torch.device("cpu"),
                )
                return cls(
                    _to_real(zero, dtype=dtype, device=device), n_modes, n_photons
                )
            indices = torch.tensor([idxs], dtype=torch.long, device=device)
            values = torch.tensor(vals, dtype=torch.float32, device=device)
            tensor = torch.sparse_coo_tensor(
                indices, values, (basis_size,), device=device
            )
            tensor = _to_real(tensor, dtype=dtype, device=device)
            return cls(
                tensor, n_modes, n_photons, computation_space=ComputationSpace.FOCK
            )
        dense = torch.zeros(basis_size, dtype=torch.float32, device=device)
        for basic, prob in items:
            idx = index_map.get(tuple(int(v) for v in basic))
            if idx is None:
                continue
            dense[idx] = float(prob)
        dense = _to_real(dense, dtype=dtype, device=device)
        return cls(dense, n_modes, n_photons, computation_space=ComputationSpace.FOCK)

    def to_perceval(self):
        """Convert to Perceval ``BSDistribution`` (single) or list for batches."""
        basis = self.basis
        tensor = self.normalize().tensor
        if tensor.ndim == 1:
            return self._to_pcvl_single(tensor, basis)
        if tensor.is_sparse:
            return self._to_pcvl_sparse_batch(tensor, basis)
        flat = tensor.reshape(-1, tensor.shape[-1])
        result: list[pcvl.BSDistribution] = []
        for row in flat:
            result.append(self._to_pcvl_single(row, basis))
        return result

    @staticmethod
    def _to_pcvl_single(vector: torch.Tensor, basis: Basis) -> pcvl.BSDistribution:
        dist = pcvl.BSDistribution()
        entries: Iterable[tuple[int, float]]
        if vector.is_sparse:
            coalesced = vector.coalesce()
            entries = zip(
                coalesced.indices().flatten().tolist(),
                coalesced.values().tolist(),
                strict=False,
            )
        else:
            dense_list = vector.tolist()
            entries = ((i, float(val)) for i, val in enumerate(dense_list))
        for idx, prob in entries:
            if prob == 0 or prob == 0.0:
                continue
            dist[pcvl.BasicState(basis[idx])] = float(prob)
        return dist

    @staticmethod
    def _to_pcvl_sparse_batch(
        tensor: torch.Tensor, basis: Basis
    ) -> list[pcvl.BSDistribution]:
        coalesced = tensor.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        batch_shape = tensor.shape[:-1]
        total_batches = 1
        for dim in batch_shape:
            total_batches *= dim

        def coords_from_linear(linear: int) -> tuple[int, ...]:
            coords: list[int] = []
            rem = linear
            for dim in reversed(batch_shape):
                rem, idx = divmod(rem, dim)
                coords.append(idx)
            coords.reverse()
            return tuple(coords)

        buckets: dict[tuple[int, ...], dict[int, float]] = {}
        nnz = values.shape[0]
        for col in range(nnz):
            batch_coords = tuple(int(v) for v in indices[:-1, col].tolist())
            basis_idx = int(indices[-1, col].item())
            amp = float(values[col].item())
            if amp == 0.0:
                continue
            bucket = buckets.setdefault(batch_coords, {})
            bucket[basis_idx] = amp

        result: list[pcvl.BSDistribution] = []
        for lin in range(total_batches):
            coords = coords_from_linear(lin)
            mapping = buckets.get(coords, {})
            dist = pcvl.BSDistribution()
            for idx, prob in mapping.items():
                dist[pcvl.BasicState(basis[idx])] = prob
            result.append(dist)
        return result

    def filter(
        self,
        rule: ComputationSpace
        | str
        | Callable[[tuple[int, ...]], bool]
        | Iterable[Sequence[int]]
        | tuple[ComputationSpace | str, Callable[[tuple[int, ...]], bool]],
    ) -> ProbabilityDistribution:
        """Apply post-selection filter and renormalize probabilities.

        logical_performance records kept_mass / original_mass per batch.

        Parameters
        ----------
        rule:
            Computation space alias (``fock``, ``unbunched``, ``dual_rail``), a predicate,
            an explicit iterable of allowed states, or a tuple ``(space, predicate)``
            to combine a computation-space constraint with an additional predicate.

        Returns
        -------
        ProbabilityDistribution
            A new, normalized distribution; may shrink its basis when filtering to
            ``unbunched`` or ``dual_rail`` in the dense case.

        Raises
        ------
        ValueError
            If ``dual_rail`` is selected with incompatible ``n_modes``/``n_photons`` or
            an unknown computation space is requested.
        """
        basis = self.basis
        target_space: ComputationSpace | None = None
        predicate: Callable[[tuple[int, ...]], bool]
        extra_predicate: Callable[[tuple[int, ...]], bool] | None = None

        # allow (space, predicate) tuple to combine constraints
        if (
            isinstance(rule, (tuple, list))
            and len(rule) == 2
            and isinstance(rule[0], (str, ComputationSpace))
            and callable(rule[1])
        ):
            extra_predicate = rule[1]
            rule = rule[0]

        if isinstance(rule, (str, ComputationSpace)):
            normalized = rule.lower() if isinstance(rule, str) else rule.value
            if normalized == "dual_rail":
                normalized = ComputationSpace.DUAL_RAIL.value
            space = ComputationSpace.coerce(normalized)
            target_space = space
            if space is ComputationSpace.FOCK:

                def predicate(state: tuple[int, ...]) -> bool:
                    return True

            elif space is ComputationSpace.UNBUNCHED:

                def predicate(state: tuple[int, ...]) -> bool:
                    return all(x <= 1 for x in state)

            elif space is ComputationSpace.DUAL_RAIL:
                if self.n_modes % 2 != 0 or self.n_photons * 2 != self.n_modes:
                    raise ValueError(
                        "DUAL_RAIL requires even mode count with one photon per rail pair."
                    )
                pair_count = self.n_modes // 2

                def dual_rail_ok(state: tuple[int, ...]) -> bool:
                    for i in range(pair_count):
                        a, b = state[2 * i], state[2 * i + 1]
                        if a + b != 1:
                            return False
                    return True

                predicate = dual_rail_ok
            else:
                raise ValueError("Unknown computation space filter")
        elif callable(rule):
            predicate = cast(Callable[[tuple[int, ...]], bool], rule)
        else:
            allowed_states = cast(Iterable[Sequence[int]], rule)
            allowed = {tuple(int(x) for x in state) for state in allowed_states}

            def predicate(state: tuple[int, ...]) -> bool:
                return tuple(state) in allowed

        if extra_predicate is not None:
            base_pred = predicate

            def combined(state: tuple[int, ...]) -> bool:
                return base_pred(state) and extra_predicate(state)

            predicate = combined

        iter_basis = cast(Iterable[tuple[int, ...]], basis)

        if self.is_sparse:
            coalesced = self._tensor_coalesced()
            idx = coalesced.indices()
            vals = coalesced.values()
            keep_cols: list[int] = []
            kept_basis_indices: list[int] = []
            for col in range(vals.shape[0]):
                basis_idx = int(idx[-1, col].item())
                state = cast(tuple[int, ...], basis[basis_idx])
                if predicate(state):
                    keep_cols.append(col)
                    kept_basis_indices.append(basis_idx)
            if not keep_cols:
                new_shape = coalesced.shape[:-1] + (0,)
                zero = torch.zeros(
                    new_shape, dtype=coalesced.dtype, device=coalesced.device
                )
                perf = torch.zeros(
                    coalesced.shape[:-1], dtype=coalesced.dtype, device=coalesced.device
                )
                return ProbabilityDistribution(
                    zero,
                    self.n_modes,
                    self.n_photons,
                    computation_space=self.computation_space,
                    logical_performance=perf,
                    _custom_basis=FilteredBasis(basis, ()),
                )

            kept_idx = idx[:, keep_cols]
            kept_vals = vals[keep_cols]
            unique_kept = sorted(set(kept_basis_indices))
            remap = {old: new for new, old in enumerate(unique_kept)}

            # compute performance per batch
            kept_sums = torch.zeros(
                coalesced.shape[:-1], device=coalesced.device, dtype=coalesced.dtype
            )
            total_sums = torch.zeros_like(kept_sums)
            for col in range(vals.shape[0]):
                batch_coords = tuple(int(v) for v in idx[:-1, col].tolist())
                total_sums[batch_coords] += vals[col]
            for col in keep_cols:
                batch_coords = tuple(int(v) for v in idx[:-1, col].tolist())
                kept_sums[batch_coords] += vals[col]
            perf = torch.where(
                total_sums == 0,
                torch.zeros_like(total_sums),
                kept_sums
                / torch.where(total_sums == 0, torch.ones_like(total_sums), total_sums),
            )

            # renormalize kept values per batch and reindex last dim
            safe = torch.where(kept_sums == 0, torch.ones_like(kept_sums), kept_sums)
            scaled_vals: list[torch.Tensor] = []
            new_last_indices: list[int] = []
            for col, kept_val in zip(keep_cols, kept_vals, strict=False):
                batch_coords = tuple(int(v) for v in idx[:-1, col].tolist())
                scaled_vals.append(kept_val / safe[batch_coords])
                old_basis_idx = int(idx[-1, col].item())
                new_last_indices.append(remap[old_basis_idx])
            new_vals = torch.stack(scaled_vals)
            batch_part = kept_idx[:-1, :]
            last_part = torch.tensor(
                [new_last_indices], dtype=kept_idx.dtype, device=kept_idx.device
            )
            new_indices = torch.cat([batch_part, last_part], dim=0)
            new_shape = coalesced.shape[:-1] + (len(unique_kept),)
            tensor = torch.sparse_coo_tensor(
                new_indices, new_vals, new_shape, device=coalesced.device
            )
            return ProbabilityDistribution(
                tensor,
                self.n_modes,
                self.n_photons,
                computation_space=self.computation_space,
                logical_performance=perf,
                _custom_basis=FilteredBasis(basis, unique_kept),
            )

        dense = self.to_dense()
        if target_space in (ComputationSpace.UNBUNCHED, ComputationSpace.DUAL_RAIL):
            target_basis = _basis_for_space(target_space, self.n_modes, self.n_photons)
            src_index = {state: idx for idx, state in enumerate(iter_basis)}
            gather_src: list[int] = []
            kept_positions: list[int] = []
            target_iter = cast(Iterable[tuple[int, ...]], target_basis)
            for pos, state in enumerate(target_iter):
                if predicate(state):
                    kept_positions.append(pos)
                    gather_src.append(src_index[state])
            if not gather_src:
                shape = dense.shape[:-1] + (0,)
                zero = torch.zeros(shape, dtype=dense.dtype, device=dense.device)
                perf = torch.zeros(
                    dense.shape[:-1], dtype=dense.dtype, device=dense.device
                )
                return ProbabilityDistribution(
                    zero,
                    self.n_modes,
                    self.n_photons,
                    computation_space=target_space,
                    logical_performance=perf,
                    _custom_basis=FilteredBasis(target_basis, ()),
                )
            reduced = torch.stack([dense[..., i] for i in gather_src], dim=-1)
            total = dense.sum(dim=-1, keepdim=True)
            kept = reduced.sum(dim=-1, keepdim=True)
            perf = torch.where(
                total == 0,
                torch.zeros_like(total),
                kept / torch.where(total == 0, torch.ones_like(total), total),
            )
            safe = torch.where(kept == 0, torch.ones_like(kept), kept)
            renorm = reduced / safe
            return ProbabilityDistribution(
                renorm,
                self.n_modes,
                self.n_photons,
                computation_space=target_space,
                logical_performance=perf.squeeze(-1),
                _custom_basis=FilteredBasis(target_basis, kept_positions),
            )

        gather_indices: list[int] = []
        for i, state in enumerate(iter_basis):
            if predicate(cast(tuple[int, ...], state)):
                gather_indices.append(i)
        if not gather_indices:
            shape = dense.shape[:-1] + (0,)
            zero = torch.zeros(shape, dtype=dense.dtype, device=dense.device)
            perf = torch.zeros(dense.shape[:-1], dtype=dense.dtype, device=dense.device)
            return ProbabilityDistribution(
                zero,
                self.n_modes,
                self.n_photons,
                computation_space=self.computation_space,
                logical_performance=perf,
                _custom_basis=FilteredBasis(basis, ()),
            )
        reduced = torch.stack([dense[..., i] for i in gather_indices], dim=-1)
        total = dense.sum(dim=-1, keepdim=True)
        kept = reduced.sum(dim=-1, keepdim=True)
        perf = torch.where(
            total == 0,
            torch.zeros_like(total),
            kept / torch.where(total == 0, torch.ones_like(total), total),
        )
        safe = torch.where(kept == 0, torch.ones_like(kept), kept)
        renorm = reduced / safe
        return ProbabilityDistribution(
            renorm,
            self.n_modes,
            self.n_photons,
            computation_space=self.computation_space,
            logical_performance=perf.squeeze(-1),
            _custom_basis=FilteredBasis(basis, gather_indices),
        )

    def __getitem__(self, state: Sequence[int] | pcvl.BasicState) -> torch.Tensor:
        """Return probability for a specific Fock state.

        Raises
        ------
        KeyError
            If the requested state is not part of the current basis.
        """
        target = tuple(int(x) for x in state)
        try:
            idx = self.basis.index(target)
        except ValueError:
            raise KeyError("State not in basis") from None
        dense = self.to_dense()
        return dense[..., idx]


__all__ = ["ProbabilityDistribution"]
