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
Quantum computation processes and factories.
"""

from dataclasses import dataclass
from typing import Literal, overload

import perceval as pcvl
import torch

from ..pcvl_pytorch import CircuitConverter, build_slos_distribution_computegraph
from ..utils.combinadics import Combinadics
from ..utils.deprecations import raise_no_bunching_deprecated
from .base import AbstractComputationProcess
from .computation_space import ComputationSpace

_DEFAULT_SUPERPOSITION_CHUNK_SIZE = 32


@dataclass(frozen=True)
class _SuperpositionSupport:
    """Compact internal representation of active superposition components."""

    basis_indices: list[int]
    coefficients: torch.Tensor
    basis_size: int

    @property
    def batch_size(self) -> int:
        return self.coefficients.shape[0]

    @property
    def device(self) -> torch.device:
        return self.coefficients.device

    @property
    def is_sparse(self) -> bool:
        return True

    @property
    def nnz(self) -> int:
        return len(self.basis_indices)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.batch_size, self.basis_size)


class ComputationProcess(AbstractComputationProcess):
    """Handle quantum circuit computation and state evolution.

    Parameters
    ----------
    circuit : pcvl.Circuit
        Circuit used to build the unitary and simulation graphs.
    input_state : list[int] | torch.Tensor
        Input Fock state or superposition tensor.
    trainable_parameters : list[str]
        Prefixes of trainable circuit parameters.
    input_parameters : list[str]
        Prefixes of input-driven circuit parameters.
    n_photons : int | None
        Number of photons represented by the process.
    dtype : torch.dtype
        Real dtype used for internal tensor conversions.
    device : torch.device | None
        Device on which computation graphs are materialized.
    computation_space : ComputationSpace | None
        Computation space used for basis enumeration.
    no_bunching : bool | None
        Deprecated legacy parameter.
    output_map_func : Any
        Optional output mapping function.
    """

    def __init__(
        self,
        circuit: pcvl.Circuit,
        input_state: list[int] | torch.Tensor,
        trainable_parameters: list[str],
        input_parameters: list[str],
        n_photons: int = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        computation_space: ComputationSpace | None = None,
        memristive_metadata: list[dict] | None = None,
        no_bunching: bool | None = None,
        output_map_func=None,
    ):
        """Initialize a computation process.

        Parameters
        ----------
        circuit : pcvl.Circuit
            Circuit used to build the unitary and simulation graphs.
        input_state : list[int] | torch.Tensor
            Input Fock state or superposition tensor.
        trainable_parameters : list[str]
            Prefixes of trainable circuit parameters.
        input_parameters : list[str]
            Prefixes of input-driven circuit parameters.
        n_photons : int | None
            Number of photons represented by the process.
        dtype : torch.dtype
            Real dtype used for internal tensor conversions.
        device : torch.device | None
            Device on which computation graphs are materialized.
        computation_space : ComputationSpace | None
            Computation space used for basis enumeration.
        memristive_metadata: list[dict] | None
            The memristive phase shifter metadata. If None, it will be stored as an empty list.
        no_bunching : bool | None
            Deprecated legacy parameter.
        output_map_func : Any
            Optional output mapping function.
        """
        self.circuit = circuit
        self.input_state = input_state
        self.n_photons = n_photons
        self.trainable_parameters = trainable_parameters
        self.input_parameters = input_parameters
        self.dtype = dtype
        self.device = device
        self.memristive_metadata = (
            [] if memristive_metadata is None else memristive_metadata
        )

        if no_bunching is not None:
            raise_no_bunching_deprecated(stacklevel=2)

        if computation_space is None:
            computation_space = ComputationSpace.UNBUNCHED

        self.computation_space = computation_space
        self.output_map_func = output_map_func
        self._input_basis_states_cache: list[tuple[int, ...]] | None = None

        # Extract circuit parameters for graph building

        self.m = circuit.m  # Number of modes
        if n_photons is None:
            if type(input_state) is list:
                self.n_photons = sum(input_state)  # Total number of photons
            else:
                raise ValueError("The number of photons should be provided")
        else:
            self.n_photons = n_photons
        # Build computation graphs
        self._setup_computation_graphs()

        # validate initial input state shape when provided as tensor
        if isinstance(self.input_state, torch.Tensor):
            state_tensor: torch.Tensor = self.input_state
            self._validate_superposition_state_shape(state_tensor)

    def _input_basis_states(self) -> list[tuple[int, ...]]:
        """Enumerate the full Fock basis used for superposition inputs."""
        if self._input_basis_states_cache is None:
            self._input_basis_states_cache = Combinadics(
                "fock", self.n_photons, self.m
            ).enumerate_states()
        return self._input_basis_states_cache

    def _setup_computation_graphs(self):
        """Setup unitary and simulation computation graphs."""
        # Determine parameter specs
        parameter_specs = self.trainable_parameters + self.input_parameters

        # Build unitary graph
        self.converter = CircuitConverter(
            self.circuit,
            parameter_specs,
            memristive_metadata=self.memristive_metadata,
            dtype=self.dtype,
            device=self.device,
        )

        # Build simulation graph with correct parameters
        self.simulation_graph = build_slos_distribution_computegraph(
            m=self.m,  # Number of modes
            n_photons=self.n_photons,  # Total number of photons
            computation_space=self.computation_space,
            keep_keys=True,  # Usually want to keep keys for output interpretation
            device=self.device,
            dtype=self.dtype,
        )

    def compute(
        self,
        parameters: list[torch.Tensor],
        memristive_current_state: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute output amplitudes for the configured input state.

        Parameters
        ----------
        parameters : list[torch.Tensor]
            Circuit parameters passed to the converter.
        memristive_current_state : list[torch.Tensor] | None
            The memristive phase shifters current states. Defaults to None
            and will be treated as an empty list.

        Returns
        -------
        torch.Tensor
            Output amplitudes produced by the simulation graph.
        """
        # Generate unitary matrix from parameters

        unitary = self.converter.to_tensor(
            *parameters,
            memristive_current_state=(
                [] if memristive_current_state is None else memristive_current_state
            ),
        )
        self.unitary = unitary
        # Compute output distribution using the input state
        if isinstance(self.input_state, torch.Tensor):
            input_state = [1] * self.n_photons + [0] * (self.m - self.n_photons)
        else:
            input_state = self.input_state

        keys, amplitudes = self.simulation_graph.compute(unitary, input_state)
        return amplitudes

    @overload
    def compute_superposition_state(
        self,
        parameters: list[torch.Tensor],
        *,
        simultaneous_processes: int | None = None,
        return_keys: Literal[True] = True,
        memristive_current_state: list[torch.Tensor] | None = None,
    ) -> tuple[list[tuple[int, ...]], torch.Tensor]: ...

    @overload
    def compute_superposition_state(
        self,
        parameters: list[torch.Tensor],
        *,
        simultaneous_processes: int | None = None,
        return_keys: Literal[False] = False,
        memristive_current_state: list[torch.Tensor] | None = None,
    ) -> torch.Tensor: ...

    def compute_superposition_state(
        self,
        parameters: list[torch.Tensor],
        *,
        simultaneous_processes: int | None = None,
        return_keys: bool = False,
        memristive_current_state: list[torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[list[tuple[int, ...]], torch.Tensor]:
        prepared_state = self._prepare_superposition_support()
        unitary = self.converter.to_tensor(
            *parameters,
            memristive_current_state=(
                [] if memristive_current_state is None else memristive_current_state
            ),
        )
        _keys_out, final_amplitudes = self._compute_chunked_superposition(
            prepared_state,
            unitary if unitary.dim() == 3 else unitary.unsqueeze(0),
            simultaneous_processes=simultaneous_processes,
        )

        if final_amplitudes.shape[0] == 1:
            final_amplitudes = final_amplitudes.squeeze(0)

        if return_keys:
            return _keys_out, final_amplitudes

        return final_amplitudes

    def compute_ebs_simultaneously(
        self,
        parameters: list[torch.Tensor],
        simultaneous_processes: int = 1,
        memristive_current_state: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Evaluate a single circuit parametrisation against all superposed input
        states by chunking them in groups and delegating the heavy work to the
        TorchScript-enabled batch kernel.

        The method converts the trainable parameters into a unitary matrix,
        normalises the input state (if it is not already normalised), filters
        out components with zero amplitude, and then queries the simulation
        graph for batches of Fock states. Each batch feeds
        :meth:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph.compute_batch`, producing a tensor that contains
        the amplitudes of all reachable output states for the selected input
        components. The partial results are accumulated into a preallocated
        tensor and finally weighted by the complex coefficients of
        ``self.input_state`` to produce the global output amplitudes.

        Parameters
        ----------
        parameters : list[torch.Tensor]
            Differentiable parameters that encode the photonic circuit.
        simultaneous_processes : int
            Maximum number of non-zero input components propagated in a single
            call to ``compute_batch``.
        memristive_current_state : list[torch.Tensor] | None
            The memristive phase shifters current states. Defaults to None
            and will be treated as an empty list.

        Returns
        -------
        torch.Tensor
            Superposed output amplitudes with shape
            ``[batch_size, num_output_states]``.

        Raises
        ------
        TypeError
            If ``self.input_state`` is not a ``torch.Tensor``.

        Notes
        -----
            - ``self.input_state`` is normalized in place to avoid an extra
              allocation.They are forwarded to ``self.converter`` to build the unitary matrix used during the
              simulation.
            - Zero-amplitude components are skipped to minimise the number of
              calls to ``compute_batch``.
            - The method is agnostic to the device: tensors remain on the device
              they already occupy, so callers should ensure ``parameters`` and
              ``self.input_state`` live on the same device.
        """

        prepared_state = self._prepare_superposition_support()

        unitary = self.converter.to_tensor(
            *parameters,
            memristive_current_state=(
                [] if memristive_current_state is None else memristive_current_state
            ),
        )
        _keys_out, final_amplitudes = self._compute_chunked_superposition(
            prepared_state,
            unitary if unitary.dim() == 3 else unitary.unsqueeze(0),
            simultaneous_processes=simultaneous_processes,
        )

        if final_amplitudes.shape[0] == 1:
            final_amplitudes = final_amplitudes.squeeze(0)
        if final_amplitudes.ndim == 3 and final_amplitudes.shape[1] == 1:
            final_amplitudes = final_amplitudes.squeeze(1)

        return final_amplitudes

    def compute_with_keys(self, parameters: list[torch.Tensor]):
        """Compute output amplitudes and return them with basis keys.

        Parameters
        ----------
        parameters : list[torch.Tensor]
            Circuit parameters passed to the converter.

        Returns
        -------
        tuple[Any, torch.Tensor]
            Simulation-graph keys and corresponding amplitudes.
        """
        # Generate unitary matrix from parameters
        unitary = self.converter.to_tensor(*parameters)

        # Compute output distribution using the input state
        keys, amplitudes = self.simulation_graph.compute(unitary, self.input_state)

        return keys, amplitudes

    def _validate_superposition_state_shape(self, input_state: torch.Tensor) -> None:
        """Ensure the provided superposition state matches the full Fock basis."""
        if not isinstance(input_state, torch.Tensor):
            raise TypeError("Input state should be a tensor")

        if input_state.dim() == 1:
            state_dim = input_state.shape[0]
        elif input_state.dim() == 2:
            state_dim = input_state.shape[1]
        else:
            raise ValueError(
                f"Superposed input state must be 1D or 2D tensor, got shape {tuple(input_state.shape)}"
            )

        expected = Combinadics("fock", self.n_photons, self.m).compute_space_size()
        if state_dim != expected:
            raise ValueError(
                "Input state dimension mismatch for Fock basis: "
                f"got {state_dim}, expected C(m + n_photons - 1, n_photons) = "
                f"C({self.m + self.n_photons - 1}, {self.n_photons}) = {expected}."
            )

    def _prepare_superposition_tensor(self) -> torch.Tensor:
        """Validate, normalise, and convert the stored superposition state to the correct dtype."""
        if not isinstance(self.input_state, torch.Tensor):
            raise TypeError("Input state should be a tensor")

        tensor = self.input_state

        self._validate_superposition_state_shape(tensor)

        tensor = self._unsqueeze_superposition_tensor(tensor)

        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.complex64)
        elif tensor.dtype == torch.float64:
            tensor = tensor.to(torch.complex128)
        elif tensor.dtype not in (torch.complex64, torch.complex128):
            raise TypeError(
                f"Unsupported dtype for superposition state: {tensor.dtype}"
            )

        tensor = self._normalize_superposition_tensor(tensor)
        self.input_state = tensor
        return tensor

    def _prepare_superposition_support(self) -> _SuperpositionSupport:
        """Return active support for the stored superposition state."""
        tensor = self._prepare_superposition_tensor()
        return self._superposition_support_from_tensor(tensor)

    def _compute_chunked_superposition(
        self,
        prepared_state: _SuperpositionSupport,
        unitary: torch.Tensor,
        *,
        simultaneous_processes: int | None,
    ) -> tuple[list[tuple[int, ...]], torch.Tensor]:
        """Evaluate a superposition by streaming chunked kernel calls into the final tensor."""
        if unitary.dim() != 3:
            raise ValueError(
                "Expected batched unitary tensor for chunked superposition evaluation."
            )

        keys_out = list(self.simulation_graph.mapped_keys)
        if not prepared_state.basis_indices:
            final = torch.zeros(
                (
                    unitary.shape[0],
                    prepared_state.batch_size,
                    len(keys_out),
                ),
                dtype=unitary.dtype,
                device=unitary.device,
            )
            return keys_out, final

        fock_basis_states = self._input_basis_states()
        input_states = [
            (index, fock_basis_states[index]) for index in prepared_state.basis_indices
        ]
        chunk_size = self._resolve_superposition_chunk_size(simultaneous_processes)
        final_amplitudes = torch.zeros(
            (
                unitary.shape[0],
                prepared_state.batch_size,
                len(keys_out),
            ),
            dtype=unitary.dtype,
            device=unitary.device,
        )

        for start in range(0, len(input_states), chunk_size):
            batch = input_states[start : start + chunk_size]
            batch_fock_states = [state for _, state in batch]
            coeffs = prepared_state.coefficients[:, start : start + len(batch)].to(
                device=unitary.device,
                dtype=unitary.dtype,
            )
            _, batch_amplitudes = self.simulation_graph.compute_batch(
                unitary, batch_fock_states
            )
            batch_amplitudes = batch_amplitudes / batch_amplitudes.norm(
                p=2, dim=1, keepdim=True
            ).clamp_min(1e-12)
            final_amplitudes += torch.einsum("se,boe->bso", coeffs, batch_amplitudes)

        return keys_out, final_amplitudes

    @staticmethod
    def _resolve_superposition_chunk_size(simultaneous_processes: int | None) -> int:
        if simultaneous_processes is None:
            return _DEFAULT_SUPERPOSITION_CHUNK_SIZE
        if simultaneous_processes <= 0:
            raise ValueError("simultaneous_processes must be a positive integer.")
        return simultaneous_processes

    @staticmethod
    def _unsqueeze_superposition_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Add a batch dimension while preserving sparse COO storage."""
        if tensor.dim() != 1:
            return tensor
        if not tensor.is_sparse:
            return tensor.unsqueeze(0)

        coalesced = tensor.coalesce()
        nnz = coalesced.values().shape[0]
        batch_indices = torch.zeros((1, nnz), dtype=torch.long, device=coalesced.device)
        indices = torch.cat((batch_indices, coalesced.indices()), dim=0)
        return torch.sparse_coo_tensor(
            indices,
            coalesced.values(),
            (1, tensor.shape[0]),
            dtype=coalesced.dtype,
            device=coalesced.device,
        ).coalesce()

    @staticmethod
    def _normalize_superposition_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize batched superposition tensors without forcing densification."""
        if not tensor.is_sparse:
            norm = tensor.abs().pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
            return tensor / norm

        coalesced = tensor.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        row_indices = indices[0]
        magnitude_sq = values.real.pow(2) + values.imag.pow(2)
        norm_sq = torch.zeros(
            tensor.shape[0], dtype=magnitude_sq.dtype, device=magnitude_sq.device
        )
        norm_sq.scatter_add_(0, row_indices, magnitude_sq)
        norms = norm_sq.sqrt().clamp_min(1e-12)
        scaled_values = values / norms[row_indices]
        return torch.sparse_coo_tensor(
            indices,
            scaled_values,
            coalesced.shape,
            dtype=coalesced.dtype,
            device=coalesced.device,
        ).coalesce()

    @staticmethod
    def _superposition_support_from_tensor(
        tensor: torch.Tensor,
    ) -> _SuperpositionSupport:
        """Extract active basis indices and compact coefficients."""
        if tensor.is_sparse:
            return ComputationProcess._sparse_superposition_support(tensor)
        return ComputationProcess._dense_superposition_support(tensor)

    @staticmethod
    def _dense_superposition_support(tensor: torch.Tensor) -> _SuperpositionSupport:
        """Extract active support from a dense batched superposition tensor."""
        magnitude_sq = tensor.real.pow(2) + tensor.imag.pow(2)
        active_mask = (magnitude_sq >= 1e-13).any(dim=0)
        active_indices_tensor = active_mask.nonzero(as_tuple=False).flatten()
        if active_indices_tensor.numel() == 0:
            coefficients = tensor.new_zeros((tensor.shape[0], 0))
            basis_indices: list[int] = []
        else:
            coefficients = tensor[:, active_indices_tensor]
            basis_indices = [int(idx) for idx in active_indices_tensor.tolist()]
        return _SuperpositionSupport(
            basis_indices=basis_indices,
            coefficients=coefficients,
            basis_size=tensor.shape[-1],
        )

    @staticmethod
    def _sparse_superposition_support(tensor: torch.Tensor) -> _SuperpositionSupport:
        """Extract active support from a sparse COO batched superposition tensor."""
        coalesced = tensor.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        if values.numel() == 0:
            coefficients = values.new_zeros((tensor.shape[0], 0))
            return _SuperpositionSupport(
                basis_indices=[],
                coefficients=coefficients,
                basis_size=tensor.shape[-1],
            )

        magnitude_sq = values.real.pow(2) + values.imag.pow(2)
        active_values_mask = magnitude_sq >= 1e-13
        if not bool(active_values_mask.any().item()):
            coefficients = values.new_zeros((tensor.shape[0], 0))
            return _SuperpositionSupport(
                basis_indices=[],
                coefficients=coefficients,
                basis_size=tensor.shape[-1],
            )

        active_indices = indices[:, active_values_mask]
        active_values = values[active_values_mask]
        rows = active_indices[0]
        cols = active_indices[-1]
        basis_indices_tensor = torch.unique(cols, sorted=True)
        positions = torch.searchsorted(basis_indices_tensor, cols)
        coefficients = values.new_zeros((tensor.shape[0], basis_indices_tensor.numel()))
        coefficients[rows, positions] = active_values
        basis_indices = [int(idx) for idx in basis_indices_tensor.tolist()]
        return _SuperpositionSupport(
            basis_indices=basis_indices,
            coefficients=coefficients,
            basis_size=tensor.shape[-1],
        )


class ComputationProcessFactory:
    """Factory for creating computation processes."""

    @staticmethod
    def create(
        circuit: pcvl.Circuit,
        input_state: list[int] | torch.Tensor,
        trainable_parameters: list[str],
        input_parameters: list[str],
        computation_space: ComputationSpace | None = None,
        memristive_metadata: list[dict] | None = None,
        **kwargs,
    ) -> ComputationProcess:
        """Create a computation process.

        Parameters
        ----------
        circuit : pcvl.Circuit
            Circuit used to build the process.
        input_state : list[int] | torch.Tensor
            Input Fock state or superposition tensor.
        trainable_parameters : list[str]
            Prefixes of trainable circuit parameters.
        input_parameters : list[str]
            Prefixes of input-driven circuit parameters.
        computation_space : ComputationSpace | None
            Computation space used for basis enumeration.
        memristive_metadata: list[dict] | None
            The memristive phase shifter metadata. If None, it will be stored as an empty list.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`ComputationProcess`.

        Returns
        -------
        ComputationProcess
            Created computation process.
        """
        return ComputationProcess(
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=trainable_parameters,
            input_parameters=input_parameters,
            computation_space=computation_space,
            memristive_metadata=(
                [] if memristive_metadata is None else memristive_metadata
            ),
            **kwargs,
        )
