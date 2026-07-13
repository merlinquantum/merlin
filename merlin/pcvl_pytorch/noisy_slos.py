"""Noisy SLOS probability graphs for source-indistinguishability models.

This module implements a probability-only simulation backend for source noise
in Merlin's SLOS pipeline. The main entry point,
``NoisySLOSComputeGraph``, caches one noisy subgraph per input Fock state in
``_slos_graph_per_input`` and reuses those cached subgraphs across repeated
evaluations.

The implementation follows the Orthogonal Bad Bits model: each input state is
expanded into partitions of fully indistinguishable and distinguishable photon
subsets, and the corresponding probability distributions are convolved back
together to obtain the final noisy output distribution.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from functools import reduce
from itertools import combinations
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from merlin.algorithms.layer_utils import NoiseGroups
from merlin.core.computation_space import ComputationSpace
from merlin.core.sectored_distribution import SectoredDistribution, SectorResult
from merlin.utils.combinadics import Combinadics
from merlin.utils.dtypes import resolve_float_complex

if TYPE_CHECKING:
    from merlin.pcvl_pytorch.slos_torchscript import SLOSComputeGraph


class NoisyG2SLOSComputeGraph:
    def __init__(
        self,
        noise_groups: NoiseGroups | None,
        m: int,
        n_photons: int,
        computation_space: ComputationSpace = ComputationSpace.FOCK,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float,
    ) -> None:
        if noise_groups is None:
            raise RuntimeError(
                "The NoisyG2SLOSComputeGraph should only be used if there is g2 noise in the circuit."
            )
        if noise_groups.source is None:
            raise RuntimeError(
                "The NoisyG2SLOSComputeGraph should only be used if there is g2 noise in the circuit."
            )
        if "g2" not in noise_groups.source:
            raise RuntimeError(
                "The NoisyG2SLOSComputeGraph requires source noise containing a g2 entry."
            )

        self.noise_groups = noise_groups
        self.indistinguishability = noise_groups.source.get("indistinguishability", 1.0)

        self.g2_distinguishable = noise_groups.source.get("g2_distinguishable", False)
        if self.g2_distinguishable is None:
            self.g2_distinguishable = False
        self.g2 = noise_groups.source.get("g2", 0.0)

        self.m = m
        self.n_photons = n_photons

        self.device = device
        self.dtype = dtype

        self.computation_space = computation_space
        if not self.computation_space == ComputationSpace.FOCK:
            warnings.warn(
                "Noisy simulations with source noise currently use ComputationSpace.FOCK. Other computation spaces are not yet supported for noise models.",
                UserWarning,
                stacklevel=2,
            )
            self.computation_space = ComputationSpace.FOCK

        from .slos_torchscript import (
            build_slos_distribution_computegraph as build_slos_graph,
        )

        self._slos_graphs: NoisySLOSComputeGraph | list[NoisySLOSComputeGraph]
        if self.g2_distinguishable:
            self._slos_graphs = NoisySLOSComputeGraph(
                noise_groups=noise_groups,
                m=self.m,
                n_photons=self.n_photons,
                computation_space=self.computation_space,
                keep_keys=True,
                device=device,
                dtype=dtype,
            )
        else:
            regular_slos_graphs: list[SLOSComputeGraph] = cast(
                list["SLOSComputeGraph"],
                [
                    build_slos_graph(
                        self.m,
                        n_i,
                        computation_space=self.computation_space,
                        device=device,
                        dtype=dtype,
                    )
                    for n_i in range(1, (2 * self.n_photons) + 1)
                ],
            )
            self._slos_graphs = [
                NoisySLOSComputeGraph(
                    noise_groups=noise_groups,
                    m=self.m,
                    n_photons=self.n_photons + i,
                    computation_space=self.computation_space,
                    keep_keys=False,
                    device=device,
                    dtype=dtype,
                    _slos_graphs=regular_slos_graphs[: n_photons + i],
                )
                for i in range(self.n_photons + 1)
            ]

        # All fock states associated with each photon number n
        self._fock_states_per_n = {
            i: torch.tensor(Combinadics("fock", n=i, m=self.m).enumerate_states()).to(
                device
            )
            for i in range(1, 2 * self.n_photons + 1)
        }

        self.mapped_keys = [
            [
                tuple(state)
                for state in Combinadics(
                    self.computation_space.casefold(), n=self.n_photons + i, m=self.m
                ).enumerate_states()
            ]
            for i in range(self.n_photons + 1)
        ]

    def _get_extra_photon_combinations(
        self,
        input_state: list[int] | tuple[int, ...],
    ) -> list[list[tuple[int, ...]]]:
        """Return source-slot combinations that emit extra g2 photons.

        The g2 model treats each intended input photon as one source slot. For
        bunched inputs, a mode with occupation greater than one contributes one
        slot per photon in that mode. For example, input state ``[2, 0]`` has
        two source slots in mode 0 and can emit zero, one, or two extra photons.

        Parameters
        ----------
        input_state : list[int] | tuple[int, ...]
            Fock input state whose occupied photons define the g2 source slots.

        Returns
        -------
        list[list[tuple[int, ...]]]
            Extra-photon combinations grouped by the number of extra photons.
        """
        num_photons = sum(input_state)
        output: list[list[tuple[int, ...]]] = [[]]

        # Convert to tensor if not already
        if not isinstance(input_state, Tensor):
            input_state_tensor = torch.tensor(
                list(input_state),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            input_state_tensor = input_state.to(dtype=torch.int32)
            if self.device is not None:
                input_state_tensor = input_state_tensor.to(self.device)

        positions = torch.arange(
            len(input_state_tensor),
            dtype=torch.long,
            device=input_state_tensor.device,
        )
        photon_positions = torch.repeat_interleave(positions, input_state_tensor)

        # g2 sectors are indexed by how many extra photons were emitted. For
        # sector k, enumerate every multiset of k source positions that can add
        # one extra photon; repeated positions are possible when the input
        # already has multiple photons in that mode.
        for i in range(1, num_photons + 1):
            output.append(list(combinations(photon_positions.tolist(), i)))

        return output

    def compute_probs(
        self,
        unitary: torch.Tensor,
        input_state: list[int] | tuple[int, ...],
    ) -> SectoredDistribution:

        sector_outputs = []

        if unitary.size(0) == unitary.size(1) and unitary.ndim == 2:
            unitary = unitary.unsqueeze(0)

        # Getting the non-g2 probs and the one hot runs for g2_distinguishable photons
        if self.g2_distinguishable:
            # Cast for mypy: _slos_graphs is single NoisySLOSComputeGraph when g2_distinguishable is True
            single_graph = cast(NoisySLOSComputeGraph, self._slos_graphs)
            keys_regular, probs_regular = single_graph.compute_probs(
                unitary, input_state
            )
            # Generate one-hot states for each active mode and compute their probs
            one_hot_slos_graphs = {}
            for mode_idx in range(len(input_state)):
                if input_state[mode_idx] > 0:  # Only for active modes
                    one_hot_state = [0] * len(input_state)
                    one_hot_state[mode_idx] = 1
                    keys_one_hot, probs_one_hot = single_graph._slos_graphs[
                        0
                    ].compute_probs(unitary, one_hot_state)
                    one_hot_slos_graphs[mode_idx] = (keys_one_hot, probs_one_hot)
        else:
            # Cast for mypy: _slos_graphs is list when g2_distinguishable is False
            slos_graphs_list = cast(list[NoisySLOSComputeGraph], self._slos_graphs)
            probs_regular = slos_graphs_list[0].compute_probs(unitary, input_state)

        # Group possible extra emissions by sector. Entry k contains every
        # source-mode combination that produces n_photons + k output photons.
        extra_photons_combinations = self._get_extra_photon_combinations(input_state)

        num_input_photons = sum(input_state)
        # Convert g2 = g^(2)(0) second-order coherence to per-source emission
        # probability p.  The two are related by g^(2)(0) = 2p/(1+p)^2, so:
        #   p = ((1 - g2) - sqrt(1 - 2*g2)) / g2,  valid for g2 in [0, 0.5].
        # For small g2, p ≈ g2/2 (L'Hôpital).  At g2=0 the limit is p=0.
        _g2 = torch.as_tensor(self.g2, device=unitary.device, dtype=self.dtype)
        _disc = (1.0 - 2.0 * _g2).clamp(min=0.0)
        p_emit = ((1.0 - _g2) - _disc.sqrt()) / _g2.clamp(min=1e-15)
        for num_photons_added in range(len(extra_photons_combinations)):
            weight_k = (p_emit**num_photons_added) * (
                (1 - p_emit) ** (num_input_photons - num_photons_added)
            )

            # Each g2 emission count lives in a different photon-number sector,
            # so probabilities cannot be accumulated in one flat Fock basis.
            sector = SectorResult(
                torch.zeros(
                    Combinadics(
                        scheme="fock", n=self.n_photons + num_photons_added, m=self.m
                    ).compute_space_size(),
                    dtype=self.dtype,
                    device=unitary.device,
                ),
                n_modes=self.m,
                n_photons=self.n_photons + num_photons_added,
            )

            if num_photons_added == 0:
                sector.tensor = sector.tensor + weight_k * probs_regular
            else:
                # Sum all ways to choose which sources emitted the extra
                # photons. Distinguishable g2 photons are convolved as
                # independent one-hot distributions; indistinguishable extra
                # photons are simulated directly in the higher-photon SLOS graph.
                for combination in extra_photons_combinations[num_photons_added]:
                    if self.g2_distinguishable:
                        distributions_to_convolve = [
                            one_hot_slos_graphs[photon] for photon in combination
                        ]
                        # Convolve probs_regular with all one-hot distributions
                        all_distributions = [
                            (keys_regular, probs_regular)
                        ] + distributions_to_convolve
                        keys_list, probs_list = zip(*all_distributions, strict=True)
                        keys, probs = convolve_distributions(keys_list, *probs_list)

                        # Reorder probs to match Fock order
                        fock_states = self._fock_states_per_n[
                            self.n_photons + num_photons_added
                        ]
                        fock_states_list = [
                            tuple(state.tolist()) for state in fock_states
                        ]
                        keys_as_tuples = (
                            [tuple(k.tolist()) for k in keys]
                            if isinstance(keys, torch.Tensor)
                            else [tuple(k) for k in keys]
                        )
                        key_to_idx = {
                            key: idx for idx, key in enumerate(keys_as_tuples)
                        }

                        reordered_probs = torch.zeros_like(probs)
                        for fock_idx, fock_state in enumerate(fock_states_list):
                            if fock_state in key_to_idx:
                                conv_idx = key_to_idx[fock_state]
                                if probs.ndim == 1:
                                    reordered_probs[fock_idx] = probs[conv_idx]
                                else:
                                    reordered_probs[:, fock_idx] = probs[:, conv_idx]
                        probs = reordered_probs

                    else:
                        input_state_to_run = list(input_state)
                        for photon in combination:
                            input_state_to_run[photon] += 1
                        probs = cast(
                            torch.Tensor,
                            slos_graphs_list[num_photons_added].compute_probs(
                                unitary, input_state_to_run
                            ),
                        )

                    sector.tensor = sector.tensor + weight_k * probs

            sector_outputs.append(sector)
        return SectoredDistribution(tuple(sector_outputs))

    def to(self, device: str | torch.device) -> NoisyG2SLOSComputeGraph:
        """Move cached tensors and subgraphs to a specific device.

        Parameters
        ----------
        device : str | torch.device
            Target device.

        Returns
        -------
        NoisyG2SLOSComputeGraph
            The graph instance moved to ``device``.

        Raises
        ------
        TypeError
            If ``device`` is neither a string nor a ``torch.device``.
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(
                f"Expected a string or torch.device, but got {type(device).__name__}"
            )

        # Move fock states tensors to device
        self._fock_states_per_n = {
            n: states.to(self.device) for n, states in self._fock_states_per_n.items()
        }

        # Move SLOS graphs to device (handle both single graph and list cases)
        if isinstance(self._slos_graphs, NoisySLOSComputeGraph):
            self._slos_graphs.to(self.device)
        else:
            for graph in self._slos_graphs:
                graph.to(self.device)

        return self

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save the noisy g2 SLOS graph configuration to disk.

        Parameters
        ----------
        path : str | os.PathLike[str]
            Destination path for the serialized graph metadata.
        """
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        metadata = {
            "graph_type": "noisy_g2_slos",
            "noise_groups": self.noise_groups,
            "m": self.m,
            "n_photons": self.n_photons,
            "computation_space": self.computation_space.value,
            "g2_distinguishable": self.g2_distinguishable,
            "g2": float(self.g2),
            "dtype_str": str(self.dtype),
        }

        torch.save({"metadata": metadata}, path)


class NoisySLOSComputeGraph:
    """Probability-only SLOS graph with source noise.

    The graph caches one ``_InputStateNoisySLOSComputeGraph`` per input
    Fock state and one shared list of noiseless SLOS subgraphs in
    ``_slos_graphs`` indexed by photon number. Each cached input-state helper
    expands its input according to the Orthogonal Bad Bits model and uses the
    shared subgraphs to produce the final noisy distribution in Fock space.

    Parameters
    ----------
    noise_groups : NoiseGroups | None
        Noise configuration extracted from the layer or experiment. Source noise
        must be present.
    m : int
        Number of optical modes.
    n_photons : int
        Total photon number represented by the graph.
    computation_space : ComputationSpace
        Requested computation space. Source-noise simulations currently operate
        in Fock space only. Default is ``ComputationSpace.FOCK``.
    keep_keys : bool
        If True, return output basis keys together with probabilities. Default
        is True.
    device : str | torch.device | None
        Target device for cached tensors and subgraphs. Default is None.
    dtype : torch.dtype
        Real dtype used by the probability graph. Default is ``torch.float``.

    Raises
    ------
    RuntimeError
        If ``noise_groups`` is missing or does not contain source noise.
    """

    def __init__(
        self,
        noise_groups: NoiseGroups | None,
        m: int,
        n_photons: int,
        computation_space: ComputationSpace = ComputationSpace.FOCK,
        keep_keys: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float,
        _slos_graphs: list[SLOSComputeGraph] | None = None,
    ) -> None:
        if noise_groups is None:
            raise RuntimeError(
                "The NoisySLOSComputeGraph should only be used if there is source noise in the circuit."
            )
        if noise_groups.source is None:
            raise RuntimeError(
                "The NoisySLOSComputeGraph should only be used if there is source noise in the circuit."
            )

        self.noise_groups = noise_groups
        self.indistinguishability = noise_groups.source.get("indistinguishability", 1.0)

        self.g2_distinguishable = noise_groups.source.get("g2_distinguishable", None)
        self._slos_graph_per_input: dict[
            tuple[int, ...], _InputStateNoisySLOSComputeGraph
        ] = {}

        self.m = m
        self.n_photons = n_photons
        self.computation_space = computation_space
        # TODO Change with post-selection if it applies
        if not self.computation_space == ComputationSpace.FOCK:
            warnings.warn(
                "Noisy simulations with source noise currently use ComputationSpace.FOCK. Other computation spaces are not yet supported for noise models.",
                UserWarning,
                stacklevel=2,
            )
            self.computation_space = ComputationSpace.FOCK

        self.keep_keys = keep_keys
        self.device = device
        self.dtype = dtype
        self.cdtype = resolve_float_complex(dtype)[1]

        self.mapped_keys = [
            tuple(state)
            for state in Combinadics(
                self.computation_space.casefold(), n=self.n_photons, m=self.m
            ).enumerate_states()
        ]

        from .slos_torchscript import (
            build_slos_distribution_computegraph as build_slos_graph,
        )

        self._slos_graphs: list[SLOSComputeGraph] = (
            cast(
                list["SLOSComputeGraph"],
                [
                    build_slos_graph(
                        self.m,
                        n_i,
                        computation_space=self.computation_space,
                        device=device,
                        dtype=dtype,
                    )
                    for n_i in range(1, self.n_photons + 1)
                ],
            )
            if _slos_graphs is None
            else _slos_graphs
        )

    def compute_probs(
        self,
        unitary: torch.Tensor,
        input_state: list[int] | tuple[int, ...],
    ) -> tuple[list[tuple[int, ...]], torch.Tensor] | torch.Tensor:
        """Compute noisy output probabilities for one input Fock state.

        Parameters
        ----------
        unitary : torch.Tensor
            Circuit unitary with shape ``[m, m]`` or batched shape
            ``[batch_size, m, m]``. Its dtype must match the complex dtype
            associated with ``self.dtype``.
        input_state : list[int] | tuple[int, ...]
            Input Fock occupation numbers.

        Returns
        -------
        tuple[list[tuple[int, ...]], torch.Tensor] | torch.Tensor
            If ``keep_keys`` is True, returns the Fock output keys and a tensor
            of probabilities with shape ``[batch_size, n_output_states]``.
            Otherwise returns the probability tensor directly.

        Raises
        ------
        ValueError
            If the unitary shape is invalid, the dtype is incompatible, or the
            input state contains negative occupations or no photons.
        """

        if len(unitary.shape) == 2:
            unitary = unitary.unsqueeze(0)  # Add batch dimension [1 x m x m]
        else:
            pass

        batch_size, m, m2 = unitary.shape
        if m != m2 or m != self.m:
            raise ValueError(
                f"Unitary matrix must be square with dimension {self.m}x{self.m}"
            )

        if unitary.dtype != self.cdtype:
            # Raise an error instead of just warning and converting
            raise ValueError(
                f"Unitary dtype {unitary.dtype} doesn't match the expected complex dtype {self.cdtype} "
                f"for the graph built with dtype {self.dtype}. Please provide a unitary with the correct dtype "
                f"or rebuild the graph with a compatible dtype."
            )

        input_state = tuple(input_state)
        if any(n < 0 for n in input_state) or sum(input_state) == 0:
            raise ValueError("Photon numbers cannot be negative or all zeros")

        if input_state not in self._slos_graph_per_input:
            slos_graph = _InputStateNoisySLOSComputeGraph(
                input_state,
                self.indistinguishability,
                self.computation_space,
                self.device,
                self.dtype,
            )
            self.computation_space = slos_graph.computation_space
            self._slos_graph_per_input[input_state] = slos_graph
        else:
            slos_graph = self._slos_graph_per_input[input_state]

        keys, probs = slos_graph.compute_probs(unitary, self._slos_graphs)

        if self.keep_keys:
            return keys, probs
        return probs

    def to(self, device: str | torch.device) -> NoisySLOSComputeGraph:
        """Move cached tensors and subgraphs to a specific device.

        Parameters
        ----------
        device : str | torch.device
            Target device.

        Returns
        -------
        NoisySLOSComputeGraph
            The graph instance moved to ``device``.

        Raises
        ------
        TypeError
            If ``device`` is neither a string nor a ``torch.device``.
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(
                f"Expected a string or torch.device, but got {type(device).__name__}"
            )

        for slos_graph in self._slos_graph_per_input.values():
            slos_graph.device = self.device
            slos_graph._obb_input_states = slos_graph._obb_input_states.to(self.device)
            slos_graph._weights = [
                weight.to(self.device) for weight in slos_graph._weights
            ]
            slos_graph._partitions = [
                [partition[0].to(self.device), partition[1].to(self.device)]
                for partition in slos_graph._partitions
            ]
            slos_graph._fock_states_per_n = {
                n: states.to(self.device)
                for n, states in slos_graph._fock_states_per_n.items()
            }

        for graph in self._slos_graphs:
            graph.to(self.device)

        return self

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save the noisy SLOS graph configuration to disk.

        Parameters
        ----------
        path : str | os.PathLike[str]
            Destination path for the serialized graph metadata.
        """
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        metadata = {
            "noise_groups": self.noise_groups,
            "m": self.m,
            "n_photons": self.n_photons,
            "computation_space": self.computation_space.value,
            "keep_keys": self.keep_keys,
            "dtype_str": str(self.dtype),
            "has_output_map_func": False,
        }

        torch.save({"metadata": metadata}, path)


class _InputStateNoisySLOSComputeGraph:
    """Noisy SLOS graph specialized to one fixed input Fock state.

    This helper precomputes the Orthogonal Bad Bits partitions for one fixed
    input state. It does not own SLOS subgraphs; instead, the caller provides
    the shared SLOS subgraph list when computing probabilities.
    """

    def __init__(
        self,
        input_state: list[int] | tuple[int, ...],
        indistinguishability: float,
        computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float,
    ) -> None:
        """Initialize the cached noisy graph for one input state.

        Parameters
        ----------
        input_state : list[int] | tuple[int, ...]
            Fixed input Fock state for which all noisy partitions are built.
        indistinguishability : float
            Source indistinguishability parameter in the interval ``[0, 1]``.
        computation_space : ComputationSpace
            Requested computation space for the SLOS subgraphs. Default is
            ``ComputationSpace.UNBUNCHED``.
        device : str | torch.device | None
            Target device for cached tensors. Default is None.
        dtype : torch.dtype
            Real dtype for internal probability tensors. Default is
            ``torch.float``.

        Raises
        ------
        ValueError
            If ``indistinguishability`` lies outside ``[0, 1]``.
        """

        self.input_state = input_state
        self.indistinguishability = torch.as_tensor(
            indistinguishability, dtype=torch.float64
        )
        self.m = len(input_state)
        self.n_photons = sum(input_state)
        self.computation_space = computation_space
        if (computation_space is not ComputationSpace.FOCK) and (max(input_state) > 1):
            self.computation_space = ComputationSpace.FOCK

        if indistinguishability < 0 or indistinguishability > 1:
            raise ValueError("Indistinguishability must be in range [0, 1].")

        self.device = device
        self.dtype = dtype

        # Weights of good & bad bits respectively
        self.g = torch.sqrt(self.indistinguishability).to(device)
        self.b = (1 - self.g).to(device)

        # Weights associated with each cell in each partition
        self._weights = [
            (self.g ** (self.n_photons - i) * self.b**i).to(device)
            for i in range(self.n_photons + 1)
        ]

        # Partition order i means i photons are treated as distinguishable
        # "bad" bits. Each partition cell contains the remaining indistinguishable
        # state plus one one-hot state per removed photon.
        self._partitions = [
            self._generate_obb_partition(input_state, num_bad_photons, device=device)
            for num_bad_photons in range(0, self.n_photons + 1)
        ]
        # Precompute every unique cell state once; compute_probs can then run
        # each noiseless SLOS subproblem once and reuse it across partitions.
        self._obb_input_states = self._generate_obb_states(
            input_state, self.n_photons, device=device
        )

        # All fock states associated with each photon number n
        self._fock_states_per_n = {
            i: torch.tensor(Combinadics("fock", n=i, m=self.m).enumerate_states()).to(
                device
            )
            for i in range(1, self.n_photons + 1)
        }

    def compute_probs(
        self, unitary: torch.Tensor, slos_graphs: list[SLOSComputeGraph]
    ) -> tuple[list[tuple[int, ...]], torch.Tensor]:
        """Compute noisy probabilities for the cached input state.

        Parameters
        ----------
        unitary : torch.Tensor
            Circuit unitary with shape ``[m, m]`` or batched shape
            ``[batch_size, m, m]``.
        slos_graphs : list[SLOSComputeGraph]
            Shared list of noiseless SLOS graphs owned by
            ``NoisySLOSComputeGraph`` and indexed by photon number ``n-1``.

        Returns
        -------
        tuple[list[tuple[int, ...]], torch.Tensor]
            Output Fock keys and the corresponding probabilities with shape
            ``[batch_size, n_output_states]``.
        """
        if unitary.size(0) == unitary.size(1) and unitary.ndim == 2:
            unitary = unitary.unsqueeze(0)

        probs_per_obb_state = {}
        for state in self._obb_input_states:
            key = tuple(state.tolist())
            n = sum(key)
            _, probs = slos_graphs[n - 1].compute_probs(unitary, state)

            if probs.ndim == 1:
                probs = probs.unsqueeze(0)

            probs_per_obb_state[key] = probs

        self._probs_per_obb_state = probs_per_obb_state

        b = len(unitary)
        output_keys_tensor = self._fock_states_per_n[self.n_photons]
        output_keys = [tuple(row) for row in output_keys_tensor.tolist()]

        output_probs = torch.zeros(
            b, len(output_keys), device=unitary.device, dtype=self.dtype
        )

        for i, partition in enumerate(self._partitions):
            bit_weight = self._weights[i]

            for cell, count in zip(partition[0], partition[1], strict=True):
                cell_distributions = [
                    probs_per_obb_state[tuple(state.tolist())] for state in cell
                ]
                fock_states = [
                    self._fock_states_per_n[int(sum(state))] for state in cell
                ]
                _, convolution = convolve_distributions(
                    fock_states,
                    *cell_distributions,
                )
                output_probs += bit_weight * convolution * count.item()

        # OBB partition weights do not generally sum to one. This normalization
        # assumes output_probs spans the full Fock basis for self.n_photons; it
        # would hide real probability leakage if the output basis were truncated.
        output_probs = output_probs / output_probs.sum(dim=1).unsqueeze(1)
        return output_keys, output_probs

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save the noisy SLOS graph configuration to disk.

        Parameters
        ----------
        path : str | os.PathLike[str]
            Destination path for the serialized graph metadata.
        """
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        metadata = {
            "m": self.m,
            "n_photons": self.n_photons,
            "computation_space": self.computation_space.value,
            "dtype_str": str(self.dtype),
            "indistinguishability": float(self.indistinguishability.item()),
        }

        torch.save({"metadata": metadata}, path)

    @staticmethod
    def _generate_obb_partition(
        input_state: list[int] | tuple[int, ...] | torch.Tensor,
        order: int,
        device: str | torch.device | None = None,
    ) -> list[torch.Tensor]:
        """Generate one Orthogonal Bad Bits partition.

        Parameters
        ----------
        input_state : list[int] | tuple[int, ...] | torch.Tensor
            Input Fock state to partition.
        order : int
            Number of distinguishable, or "bad", photons to extract from the
            input state.
        device : str | torch.device | None
            Device on which the returned tensors are allocated. Default is
            None.

        Returns
        -------
        list[torch.Tensor]
            Two-element list containing the partition cells and their
            multiplicities. The first tensor has shape
            ``[n_cells, cell_size, m]`` and the second tensor stores the count
            for each cell.

        Raises
        ------
        ValueError
            If ``order`` exceeds the total number of photons.
        """
        total_photons = (
            int(torch.sum(input_state).item())
            if isinstance(input_state, Tensor)
            else sum(input_state)
        )
        if order > total_photons:
            raise ValueError("OBB order cannot exceed the number of photons")

        # Convert to tensor if not already
        if not isinstance(input_state, Tensor):
            input_state_tensor = torch.tensor(
                list(input_state),
                dtype=torch.int32,
                device=device,
            )
        else:
            input_state_tensor = input_state.to(dtype=torch.int32)
            if device is not None:
                input_state_tensor = input_state_tensor.to(device)

        tensor_device = input_state_tensor.device

        if order == 0:
            counts = torch.tensor([1], dtype=torch.int64, device=tensor_device)
            return [input_state_tensor.unsqueeze(0).unsqueeze(0), counts]

        # Expand an occupation vector like [2, 0, 1] into photon labels
        # [0, 0, 2]. Combinations over this list enumerate photons, not modes,
        # so two photons in the same mode remain distinct choices.
        positions = torch.arange(
            len(input_state_tensor),
            dtype=torch.long,
            device=tensor_device,
        )
        photon_positions = torch.repeat_interleave(positions, input_state_tensor)

        # Choose the photons assigned to the distinguishable OBB cells for this
        # order. The remaining photons stay in the first, indistinguishable cell.
        remove_indices_list = list(combinations(photon_positions.tolist(), order))
        remove_indices = torch.tensor(
            remove_indices_list,
            dtype=torch.long,
            device=tensor_device,
        )

        n_comb = remove_indices.shape[0]
        input_state_len = input_state_tensor.size(0)

        # Base matrix: original occupation vector repeated once per photon
        # combination, then decremented at the removed photon positions.
        base = input_state_tensor.unsqueeze(0).repeat(n_comb, 1)
        for i, remove_index in enumerate(remove_indices):
            for j in remove_index:
                base[i, j] = base[i, j] - 1  # remove chosen ones

        # Each removed photon becomes a one-hot state. Later, compute_probs
        # convolves the base distribution with these one-hot distributions to
        # reconstruct the full output distribution for that OBB partition.
        missing = torch.zeros(
            (n_comb, order, input_state_len),
            dtype=torch.int32,
            device=tensor_device,
        )
        rows = torch.arange(n_comb, device=tensor_device).unsqueeze(1)
        cols = torch.arange(order, device=tensor_device).unsqueeze(0)
        missing[rows, cols, remove_indices] = 1

        result = torch.cat([base.unsqueeze(1), missing], dim=1)

        # If every photon was removed from the base cell, drop that empty base
        # state so the cell only contains physical one-hot inputs.
        if order == torch.sum(input_state_tensor).item():
            mask = result.any(dim=2)
            result = result[mask]
            result = result.unsqueeze(0)

        # Several photon choices can produce the same cell when photons occupy
        # the same mode. Keep one cell and store its multiplicity separately.
        result, counts = torch.unique(result, return_counts=True, dim=0)
        return [result, counts]

    def _generate_obb_states(
        self,
        input_state: list[int] | tuple[int, ...] | torch.Tensor,
        order: int,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Generate all OBB-derived input states up to a given order.

        Parameters
        ----------
        input_state : list[int] | tuple[int, ...] | torch.Tensor
            Reference input Fock state.
        order : int
            Maximum number of bad photons to include.
        device : str | torch.device | None
            Device on which the returned tensor is allocated. Default is None.

        Returns
        -------
        torch.Tensor
            Tensor of unique OBB states sorted by decreasing photon number.

        Raises
        ------
        ValueError
            If ``order`` exceeds the total number of photons.
        """
        if not isinstance(input_state, Tensor):
            input_state = torch.tensor(
                list(input_state), dtype=torch.int32, device=device
            )
        else:
            input_state = input_state.to(dtype=torch.int32)
            if device is not None:
                input_state = input_state.to(device)

        if order > torch.sum(input_state).item():
            raise ValueError("OBB order cannot exceed the number of photons")

        total_obb_states = input_state.unsqueeze(0)

        for num_bad_photons in range(1, order + 1):
            obb_states = self._generate_obb_partition(
                input_state, num_bad_photons, device=device
            )[0]
            obb_states = obb_states.reshape(-1, obb_states.shape[2])
            total_obb_states = torch.vstack((total_obb_states, obb_states))

        # Remove duplicate rows
        total_obb_states = torch.unique(total_obb_states, dim=0).to(device)

        # Sort by decreasing number of photons
        photon_sums = torch.sum(total_obb_states, dim=1)
        sort_indices = torch.argsort(-photon_sums)
        total_obb_states = total_obb_states[sort_indices]

        return total_obb_states


def convolve_distributions(
    keys: Sequence[Tensor | Sequence[tuple[int, ...]]], *probs: Tensor
) -> tuple[Tensor | Sequence[tuple[int, ...]], Tensor]:
    """Convolve one or more probability distributions over Fock states.

    This helper performs the same mode-merging tensor product used by Perceval
    when combining independent distributions over mode occupations.

    Parameters
    ----------
    keys : list[torch.Tensor | list[tuple[int, ...]]]
        Sequence of state lists matching the input distributions.
    *probs : torch.Tensor
        Input probability distributions. Each tensor is either one-dimensional
        or batched on its leading axis.

    Returns
    -------
    tuple[torch.Tensor | list[tuple[int, ...]], torch.Tensor]
        Combined keys and the corresponding convolved probabilities.

    Raises
    ------
    ValueError
        If the number of key sets does not match the number of probability
        tensors.
    """
    device = probs[0].device
    if len(probs[0].shape) == 1:
        probs = reduce(lambda acc, x: acc + (x.unsqueeze(0),), probs, ())
        batched_input = False
    else:
        batched_input = True

    num_probs = len(probs)
    num_batches = probs[0].size(0)

    if len(keys) != len(probs):
        raise ValueError(
            f"Invalid probability distribution for different length keys "
            f"({len(keys)}) & probs ({len(probs)})"
        )

    if num_probs == 1:
        return keys[0], probs[0]

    def _cartesian_sum(k1, k2):
        k1 = torch.as_tensor(k1, device=device)
        k2 = torch.as_tensor(k2, device=device)
        return (k1.unsqueeze(1) + k2.unsqueeze(0)).reshape(-1, k1.shape[1])

    new_keys = reduce(_cartesian_sum, keys)

    # Cartesian product of every pair of probs
    def _cartesian_product(p1, p2):
        output = p1.unsqueeze(-1) * p2.unsqueeze(-2)
        return output.flatten(start_dim=-2)

    # Unsqueeze each input tensor
    probs = reduce(lambda acc, x: acc + (x.unsqueeze(0),), probs, ())

    new_probs = reduce(_cartesian_product, probs).view(num_batches, -1)

    # Remove duplicated keys & sum corresponding probs
    new_keys, inverse_idx = torch.unique(new_keys, dim=0, return_inverse=True)
    inverse_idx = inverse_idx.unsqueeze(0).expand(num_batches, -1)
    new_probs = torch.zeros(
        num_batches, len(new_keys), dtype=new_probs.dtype, device=device
    ).scatter_add_(dim=1, index=inverse_idx, src=new_probs)

    # Correct the order of the keys & probs
    new_keys = new_keys.flip(0)
    new_probs = new_probs.flip(1)

    if not batched_input:
        new_probs = new_probs.squeeze(0)

    return new_keys, new_probs
