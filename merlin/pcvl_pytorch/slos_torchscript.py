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
This module extends slos_torch.py with TorchScript-optimized computation graphs
for photonic quantum circuit simulations. It separates the graph construction
from the actual computation for improved performance.

The optimized implementation pre-builds the computation graph based on the input state
configuration, which can then be reused for multiple unitary evaluations.
"""

import math
import os
from collections.abc import Callable

import torch
import torch.jit as jit

from merlin.core.computation_space import ComputationSpace
from merlin.utils.deprecations import raise_no_bunching_deprecated
from merlin.utils.dtypes import resolve_float_complex


def _get_complex_dtype_for_float(dtype: torch.dtype) -> torch.dtype:
    """Return the complex dtype corresponding to the provided float dtype.

    This wrapper uses `resolve_float_complex` from `merlin.utils.dtypes` so the
    logic is centralized and automatically picks up optional `complex32` support
    when present in the running PyTorch build.
    """
    try:
        float_dt, complex_dt = resolve_float_complex(dtype)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc
    return complex_dt


def _get_float_dtype_for_complex(dtype: torch.dtype) -> torch.dtype:
    """Return the float dtype corresponding to the provided complex dtype."""
    try:
        float_dt, complex_dt = resolve_float_complex(dtype)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc
    return float_dt


def prepare_vectorized_operations(operations_list, device=None):
    """
    Convert operations list to tensors for vectorized computation.

    Args:
        operations_list: List of operations, each as [src_idx, dest_idx, mode_i]
        device: Optional device to place tensors on (defaults to CPU if None)

    Returns:
        Tuple of tensors: (sources, destinations, modes)
    """
    if not operations_list:
        return (
            torch.tensor([], dtype=torch.long, device=device),
            torch.tensor([], dtype=torch.long, device=device),
            torch.tensor([], dtype=torch.long, device=device),
        )

    # Convert operations to tensor format directly on the specified device
    sources = torch.tensor(
        [op[0] for op in operations_list], dtype=torch.long, device=device
    )
    destinations = torch.tensor(
        [op[1] for op in operations_list], dtype=torch.long, device=device
    )
    modes = torch.tensor(
        [op[2] for op in operations_list], dtype=torch.long, device=device
    )

    return sources, destinations, modes


def layer_compute_vectorized(
    unitary: torch.Tensor,
    prev_amplitudes: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    modes: torch.Tensor,
    p: int,
) -> torch.Tensor:
    """
    Compute amplitudes for a single layer using vectorized operations.

    Args:
        unitary: Batch of unitary matrices [batch_size, m, m]
        prev_amplitudes: Previous layer amplitudes [batch_size, prev_size]
        sources: Source indices for operations [num_ops]
        destinations: Destination indices for operations [num_ops]
        modes: Mode indices for operations [num_ops]
        p: Photon index for this layer

    Returns:
        Next layer amplitudes [batch_size, next_size]
    """

    batch_size = unitary.shape[0]

    # Handle empty operations case
    if sources.shape[0] == 0:
        return prev_amplitudes

    # Determine output size
    next_size = int(destinations.max().item()) + 1
    # Get unitary elements for all operations
    # Shape: [batch_size, num_ops]
    u_elements = unitary[:, modes.to(unitary.device), abs(p)]

    # Get source amplitudes for all operations
    # Shape: [batch_size, num_ops]
    prev_amps = prev_amplitudes[:, sources.to(prev_amplitudes.device)]

    # Compute contributions
    # Shape: [batch_size, num_ops]
    contributions = u_elements.to(prev_amps.device) * prev_amps

    # Create result tensor with same dtype as input
    result = torch.zeros(
        (batch_size, next_size), dtype=prev_amplitudes.dtype, device=destinations.device
    )
    # Now we can use scatter_add_ with a 2D index tensor
    result.scatter_add_(
        1,  # dimension to scatter on (1 for the state indices)
        destinations.repeat(batch_size, 1),  # repeat destinations for each batch
        contributions.to(destinations.device),  # values to add
    )

    return result


def layer_compute_batch(
    unitary: torch.Tensor,
    prev_amplitudes: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    modes: torch.Tensor,
    p: list[int],
) -> torch.Tensor:
    """
    Propagate a layer of the SLOS computation graph while evaluating several
    coherent input components in parallel.

    The pre-computed ``sources``, ``destinations`` and ``modes`` tensors encode
    the sparse transitions that must be applied to go from the amplitudes of the
    previous layer to the amplitudes of the current layer. Each transition picks
    a value from ``prev_amplitudes`` using ``sources`` (the index of the parent
    state), multiplies it by the relevant unitary element ``U[modes, p]`` for
    the photon that is currently being injected, and scatters the contribution
    into ``destinations`` (the index of the child state). When several input
    superposition components need to be evaluated, ``p`` provides the photon
    indices for every component and the computations are vectorised along the
    last axis.

    Args:
        unitary (torch.Tensor): Batch of unitary matrices with shape
            ``[batch_size, m, m]``. The unitary entries are looked up according
            to ``modes`` and the photon indices ``p`` so the tensor can reside
            on either CPU or CUDA as long as it matches the device of
            ``prev_amplitudes``.
        prev_amplitudes (torch.Tensor): Complex amplitudes produced by the
            previous layer with shape ``[batch_size, prev_size, num_inputs]``.
            The third dimension indexes the different coherent input components.
        sources (torch.Tensor): Integer tensor of shape ``[num_ops]`` containing
            the index of the parent state for every sparse transition.
        destinations (torch.Tensor): Integer tensor of shape ``[num_ops]`` with
            the index within the current layer where each contribution must be
            accumulated.
        modes (torch.Tensor): Integer tensor of shape ``[num_ops]`` describing
            which output mode of the unitary matrix is involved in each
            transition.
        p (list[int]): Photon occupation indices for the layer, one entry per
            superposition component. The list length must match the third
            dimension of ``prev_amplitudes``.

    Returns:
        torch.Tensor: Tensor with shape ``[batch_size, next_size, num_inputs]``
        that contains the amplitudes of the current layer after applying all
        transitions. ``next_size`` equals ``destinations.max() + 1`` so the
        method adapts automatically to the sparsity structure.

    Notes:
        * The function is side-effect free: input tensors are never modified in
          place.
        * Zero operations (``len(sources) == 0``) short-circuit to the input in
          order to keep TorchScript graphs simple and avoid unnecessary tensor
          allocations.
    """

    batch_size = unitary.shape[0]
    num_input_states = len(p)

    # Handle empty operations case
    if sources.shape[0] == 0:
        return prev_amplitudes

    # Determine output size
    next_size = int(destinations.max().item()) + 1

    # Convert p to tensor for indexing
    p_tensor = torch.tensor(p, device=unitary.device, dtype=torch.long)

    # Get unitary elements for all operations and input states
    # Shape: [batch_size, num_ops, num_input_states]
    modes_expanded = modes.unsqueeze(-1).expand(-1, num_input_states).to(unitary.device)
    p_expanded = p_tensor.unsqueeze(0).expand(modes.shape[0], -1)
    u_elements = unitary[:, modes_expanded, p_expanded]

    # Get source amplitudes for all operations
    # Shape: [batch_size, num_ops, num_input_states]
    prev_amps = prev_amplitudes[:, sources.to(prev_amplitudes.device), :]

    # Compute contributions
    # Shape: [batch_size, num_ops, num_input_states]
    contributions = u_elements.to(prev_amps.device) * prev_amps

    # Create result tensor with same dtype as input
    result = torch.zeros(
        (batch_size, next_size, num_input_states),
        dtype=prev_amplitudes.dtype,
        device=destinations.device,
    )

    # Scatter add contributions to result
    # Need to expand destinations for all input states
    destinations_expanded = (
        destinations.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, num_input_states)
    )
    result.scatter_add_(
        1,  # dimension to scatter on (1 for the state indices)
        destinations_expanded.to(destinations.device),
        contributions.to(destinations.device),
    )

    return result


def layer_compute_backward(
    unitary: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    modes: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """
    Compute amplitudes for a single layer using vectorized operations.

    Args:
        unitary: Batch of unitary matrices [batch_size, m, m]
        prev_amplitudes: Previous layer amplitudes [batch_size, prev_size]
        sources: Source indices for operations [num_ops]
        destinations: Destination indices for operations [num_ops]
        modes: Mode indices for operations [num_ops]
        p: Photon index for this layer

    Returns:
        Next layer amplitudes [batch_size, next_size]
    """
    inverts = []
    device = unitary.device
    computing_tensors = []
    for p in range(m):
        # Determine output size
        size_sources = int(sources.max().item()) + 1
        size_destinations = int(destinations.max().item()) + 1

        # Get unitary elements for all operations
        u_elements = torch.diag_embed(unitary[:, modes, p])

        destinations_tensor = torch.zeros(
            (1, size_destinations, modes.shape[0]),
            dtype=u_elements.dtype,
            device=device,
        )
        destinations_tensor[:, destinations, torch.arange(destinations.shape[0])] = 1

        sources_tensor = torch.zeros(
            (1, sources.shape[0], size_sources),
            dtype=u_elements.dtype,
            device=device,
        )
        sources_tensor[:, torch.arange(sources.shape[0]), sources] = 1

        computing_tensor = destinations_tensor @ u_elements @ sources_tensor
        computing_tensors.append(computing_tensor)
    batch_tensors = torch.stack(computing_tensors, dim=0)
    inverts = torch.linalg.pinv(batch_tensors)

    return inverts


class SLOSComputeGraph:
    """
    A class that builds and stores the computation graph for SLOS algorithm.

    This separates the graph construction (which depends only on input state, computation_space,
    and output_map_func) from the actual computation using the unitary matrix.
    """

    def __init__(
        self,
        m: int,
        n_photons: int,
        output_map_func: Callable[[tuple[int, ...]], tuple[int, ...] | None] = None,
        computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
        keep_keys: bool = True,
        device=None,  # Optional device parameter
        dtype: torch.dtype = torch.float,  # Optional dtype parameter
        index_photons: list[tuple[int, ...]] = None,
    ):
        """
        Initialize the SLOS computation graph.

        Args:
            m (int): Number of modes in the circuit
            n_photons (int): Number of photons in the input state given to the model during the forward pass
            output_map_func (callable, optional): Function that maps output states
            computation_space (ComputationSpace): Enumeration domain.
            keep_keys (bool): If True, output state keys are returned
            device: Optional device to place tensors on (CPU, CUDA, etc.)
            dtype: Data type precision for floating point calculations (default: torch.float)
                  Use torch.float16 for half precision, torch.float for single precision,
                  or torch.float64 for double precision
            index_photons: List of tuples (first_integer, second_integer). The first_integer is the
                  lowest index layer a photon can take and the second_integer is the highest index

        """
        self.m = m
        self.n_photons = n_photons
        self.output_map_func = output_map_func
        if computation_space is ComputationSpace.DUAL_RAIL:
            if m % 2 != 0:
                raise ValueError("dual_rail compute space requires even m")
            if n_photons != m // 2:
                raise ValueError("dual_rail compute space requires n_photons = m // 2")

        self.computation_space = computation_space
        self.keep_keys = keep_keys
        self.device = device
        self.prev_amplitudes = None
        self.dtype = dtype
        self.ct_inverts = None

        if index_photons is None:
            index_photons = [(0, self.m - 1)] * self.n_photons
            self.reduced_outputs = True
        self.index_photons = index_photons

        # Determine corresponding complex dtype using helper function
        try:
            self.complex_dtype = _get_complex_dtype_for_float(dtype)
        except ValueError:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Must be torch.float16, torch.float, or torch.float64"
            )

        # Check input validity

        # Pre-compute layer structures and operation sequences
        self._build_graph_structure()

        # Create TorchScript function for the core computation
        self._create_torchscript_modules()

    def _build_graph_structure(self):
        """Build the graph structure using dictionary for fast state lookups."""
        list_operations = []  # Operations to perform at each layer
        self.vectorized_operations = []  # the same, vectorized

        # Initial state is all zeros
        last_combinations = {tuple([0] * self.m): (1, 0)}

        # For each photon/layer, compute the state combinations and operations
        for idx in range(self.n_photons):
            combinations = {}
            operations = []  # [src_state_idx, dest_idx, mode_i]

            for state, (norm_factor, src_state_idx) in last_combinations.items():
                nstate = list(state)
                # iterate on the possible values for every photon
                for i in range(
                    self.index_photons[idx][0], self.index_photons[idx][1] + 1
                ):
                    if (
                        self.computation_space
                        in (ComputationSpace.UNBUNCHED, ComputationSpace.DUAL_RAIL)
                        and nstate[i]
                    ):
                        continue
                    if self.computation_space is ComputationSpace.DUAL_RAIL:
                        pair_start = (i // 2) * 2
                        if nstate[pair_start] + nstate[pair_start + 1] >= 1:
                            continue

                    nstate[i] += 1
                    nstate_tuple = tuple(nstate)

                    # consider the state if we don't have output map or we are not at the last layer or
                    # the output map is preserving the state
                    if (
                        not (self.output_map_func)
                        or idx < self.n_photons - 1
                        or self.output_map_func(nstate) is not None
                    ):
                        dest_idx = combinations.get(nstate_tuple, None)
                        if dest_idx is None:
                            dest_idx = combinations[nstate_tuple] = (
                                norm_factor * nstate[i],
                                len(combinations),
                            )
                        # Record the operation: [src_state_idx, dest_idx, mode_i]
                        operations.append([src_state_idx, dest_idx[1], i])

                    nstate[i] -= 1

            list_operations.append(operations)
            last_combinations = combinations

        # For each layer, prepare vectorized operations on the specified device
        for ops in list_operations:
            sources, destinations, modes = prepare_vectorized_operations(
                ops, device=self.device
            )
            self.vectorized_operations.append((sources, destinations, modes))

        # Store only the final layer combinations if needed for output mapping or keys
        self.final_keys = (
            list(last_combinations.keys())
            if self.keep_keys or self.output_map_func
            else None
        )
        self.norm_factor_output = torch.tensor(
            [v[0] for v in last_combinations.values()], dtype=self.dtype
        )
        del last_combinations

        if self.output_map_func is not None:
            self.mapped_keys = []
            mapping_indices = {}  # Maps mapped state to its index
            self.mapped_indices = []  # For each original state, store the mapped index

            for _idx, key in enumerate(self.final_keys):
                mapped_state = self.output_map_func(key)
                # We know mapped_state is not None because we filtered those out during graph construction
                if mapped_state not in mapping_indices:
                    mapping_indices[mapped_state] = len(self.mapped_keys)
                    self.mapped_keys.append(mapped_state)

                mapped_idx = mapping_indices[mapped_state]
                self.mapped_indices.append(mapped_idx)

            self.total_mapped_keys = len(self.mapped_keys)

            self.target_indices = torch.tensor(
                self.mapped_indices, dtype=torch.long, device=self.device
            )

            # Clean up temporary dictionaries
            del mapping_indices
        else:
            self.mapped_keys = self.final_keys
            self.total_mapped_keys = self.keep_keys and len(self.final_keys) or 0

    def _create_torchscript_modules(self):
        """Create TorchScript modules for different parts of the computation."""
        # Create layer computation functions
        self.layer_functions = []

        for _layer_idx, (sources, destinations, modes) in enumerate(
            self.vectorized_operations
        ):
            # Get the photon index for this layer

            # Create a partial function with fixed operation
            def make_layer_fn(s, d, m):
                return lambda u, prev, p_val: layer_compute_vectorized(
                    u,
                    prev,
                    s,
                    d,
                    m,
                    p_val,
                )

            self.layer_functions.append(make_layer_fn(sources, destinations, modes))

        # Create mapping function if needed
        if self.output_map_func is not None:

            @jit.script
            def apply_mapping(
                probabilities: torch.Tensor,
                target_indices: torch.Tensor,
                output_size: int,
            ) -> torch.Tensor:
                """Apply state mapping using optimized index_add_ operation."""
                batch_size = probabilities.shape[0]

                # Create result tensor on the same device as input
                result = torch.zeros(
                    (batch_size, output_size),
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                )

                # Use scatter_add_ in a fully vectorized way
                # Target indices need to be repeated for each batch
                result.scatter_add_(
                    dim=1,  # scatter along the second dimension (output states)
                    index=target_indices.repeat(
                        batch_size, 1
                    ),  # repeat indices for each batch
                    src=probabilities,  # values to add
                )

                # Renormalize
                sum_probs = result.sum(dim=1, keepdim=True)
                safe_sum = torch.where(
                    sum_probs > 0, sum_probs, torch.ones_like(sum_probs)
                )
                normalized_result = result / safe_sum

                return normalized_result

            self.mapping_function = lambda probs: apply_mapping(
                probs, self.target_indices, self.total_mapped_keys
            )
        else:
            self.mapping_function = lambda x: x

    def compute(
        self, unitary: torch.Tensor, input_state: list[int]
    ) -> tuple[list[tuple[int, ...]], torch.Tensor]:
        """
        Compute the amplitudes using the pre-built graph.

        Args:
            unitary (torch.Tensor): Single unitary matrix [m x m] or batch of unitaries [b x m x m].\
                The unitary should be provided in the complex dtype corresponding to the graph's dtype.\
                For example, for torch.float32, use torch.cfloat; for torch.float64, use torch.cdouble.
            input_state (list[int]): Input_state of length self.m with self.n_photons in the input state

        Returns:
            Tuple[List[Tuple[int, ...]], torch.Tensor]:
                - List of tuples representing output Fock state configurations
                - Amplitudes distribution tensor
        """
        if len(unitary.shape) == 2:
            unitary = unitary.unsqueeze(0)  # Add batch dimension [1 x m x m]
        else:
            pass

        if any(n < 0 for n in input_state) or sum(input_state) == 0:
            raise ValueError("Photon numbers cannot be negative or all zeros")

        if self.computation_space is ComputationSpace.UNBUNCHED and not all(
            x in (0, 1) for x in input_state
        ):
            raise ValueError(
                "Input state must be binary (0s and 1s only) in unbunched mode"
            )
        if self.computation_space is ComputationSpace.DUAL_RAIL:
            for k in range(0, self.m, 2):
                if input_state[k] + input_state[k + 1] != 1:
                    raise ValueError(
                        "Input state must contain exactly one photon per pair in dual_rail mode"
                    )

        batch_size, m, m2 = unitary.shape
        if m != m2 or m != self.m:
            raise ValueError(
                f"Unitary matrix must be square with dimension {self.m}x{self.m}"
            )

        # Check dtype - it should match the complex dtype used for the graph building
        if unitary.dtype != self.complex_dtype:
            # Raise an error instead of just warning and converting
            raise ValueError(
                f"Unitary dtype {unitary.dtype} doesn't match the expected complex dtype {self.complex_dtype} "
                f"for the graph built with dtype {self.dtype}. Please provide a unitary with the correct dtype "
                f"or rebuild the graph with a compatible dtype."
            )
        idx_n = []
        self.norm_factor_input = 1
        for i, count in enumerate(input_state):
            for c in range(count):
                self.norm_factor_input *= c + 1
                idx_n.append(i)
                if (i > self.index_photons[len(idx_n) - 1][1]) or (
                    i < self.index_photons[len(idx_n) - 1][0]
                ):
                    raise ValueError(
                        f"Input state photons must be bounded by {self.index_photons}"
                    )

        # Get device from unitary
        device = unitary.device

        # Initial amplitude (batch of 1s on same device as unitary with appropriate dtype)
        amplitudes = torch.ones(
            (batch_size, 1), dtype=self.complex_dtype, device=device
        )

        # Apply each layer
        for layer_idx, layer_fn in enumerate(self.layer_functions):
            p = idx_n[layer_idx]
            amplitudes = layer_fn(
                unitary,
                amplitudes,
                p,
            )

        amplitudes *= torch.sqrt(self.norm_factor_output.to(amplitudes.device))
        amplitudes /= math.sqrt(self.norm_factor_input)
        self.prev_amplitudes = amplitudes  # type: ignore[assignment]

        # Apply output mapping if needed
        if self.output_map_func is not None:
            keys = self.mapped_keys
        else:
            keys = self.final_keys if self.keep_keys else None
        # Remove batch dimension if input was single unitary

        return keys, amplitudes

    def compute_batch(
        self, unitary: torch.Tensor, input_states: list[list[int]]
    ) -> tuple[list[tuple[int, ...]], torch.Tensor]:
        """
        Compute the probability distribution using the pre-built graph.

        Args:
            unitary (torch.Tensor): Single unitary matrix [m x m] or batch of unitaries [b x m x m].\
                The unitary should be provided in the complex dtype corresponding to the graph's dtype.\
                For example, for torch.float32, use torch.cfloat; for torch.float64, use torch.cdouble.
            input_state (list[int]): Input_state of length self.m with self.n_photons in the input state

        Returns:
            Tuple[List[Tuple[int, ...]], torch.Tensor]:
                - List of tuples representing output Fock state configurations
                - Probability distribution tensor
        """
        if len(unitary.shape) == 2:
            unitary = unitary.unsqueeze(0)  # Add batch dimension [1 x m x m]
        else:
            pass

        if any(n < 0 for n in input_states[0]) or sum(input_states[0]) == 0:
            raise ValueError("Photon numbers cannot be negative or all zeros")

        if self.computation_space is ComputationSpace.UNBUNCHED and not all(
            x in (0, 1) for x in input_states[0]
        ):
            raise ValueError(
                "Input state must be binary (0s and 1s only) in unbunched mode"
            )
        if self.computation_space is ComputationSpace.DUAL_RAIL:
            for k in range(0, self.m, 2):
                if input_states[0][k] + input_states[0][k + 1] != 1:
                    raise ValueError(
                        "Input state must contain exactly one photon per pair in dual_rail mode"
                    )

        batch_size, m, m2 = unitary.shape
        if m != m2 or m != self.m:
            raise ValueError(
                f"Unitary matrix must be square with dimension {self.m}x{self.m}"
            )

        # Check dtype - it should match the complex dtype used for the graph building
        if unitary.dtype != self.complex_dtype:
            # Raise an error instead of just warning and converting
            raise ValueError(
                f"Unitary dtype {unitary.dtype} doesn't match the expected complex dtype {self.complex_dtype} "
                f"for the graph built with dtype {self.dtype}. Please provide a unitary with the correct dtype "
                f"or rebuild the graph with a compatible dtype."
            )
        idx_n: list[list[int]] = [[] for _ in range(sum(input_states[0]))]
        norm_factor_input = torch.ones((1, 1, len(input_states)))
        for j, input_state in enumerate(input_states):
            k = 0
            for i, count in enumerate(input_state):
                for c in range(count):
                    norm_factor_input[0, 0, j] *= c + 1
                    idx_n[k].append(i)
                    k += 1
                    if (i > self.index_photons[len(idx_n) - 1][1]) or (
                        i < self.index_photons[len(idx_n) - 1][0]
                    ):
                        raise ValueError(
                            f"Input state photons must be bounded by {self.index_photons}"
                        )

        # Get device from unitary
        device = unitary.device

        # Initial amplitude (batch of 1s on same device as unitary with appropriate dtype)
        amplitudes = torch.ones(
            (batch_size, 1, len(input_states)), dtype=self.complex_dtype, device=device
        )

        # Apply each layer
        for layer_idx, _ in enumerate(self.layer_functions):
            p = idx_n[layer_idx]
            sources, destinations, modes = self.vectorized_operations[layer_idx]
            amplitudes = layer_compute_batch(
                unitary,
                amplitudes,
                sources,
                destinations,
                modes,
                p,
            )

        amplitudes *= torch.sqrt(
            self.norm_factor_output.to(amplitudes.device).unsqueeze(0).unsqueeze(2)
        )
        amplitudes /= torch.sqrt(norm_factor_input.to(amplitudes.device))
        self.prev_amplitudes = amplitudes  # type: ignore[assignment]

        # Apply output mapping if needed
        if self.output_map_func is not None:
            keys = self.mapped_keys
        else:
            keys = self.final_keys if self.keep_keys else None
        # Remove batch dimension if input was single unitary

        return keys, amplitudes

    def compute_probs(self, unitary, input_state):
        """
        Compute the probability distribution using the pre-built graph.

        Args:
            unitary (torch.Tensor): Single unitary matrix [m x m] or batch of unitaries [b x m x m].\
                The unitary should be provided in the complex dtype corresponding to the graph's dtype.\
                For example, for torch.float32, use torch.cfloat; for torch.float64, use torch.cdouble.
            input_state (list[int]): Input_state of length self.m with self.n_photons in the input state

        Returns:
            Tuple[List[Tuple[int, ...]], torch.Tensor]:
                - List of tuples representing output Fock state configurations
                - Probability distribution tensor
        """
        keys, amplitudes = self.compute(unitary, input_state)
        keys, probabilities = self.compute_probs_from_amplitudes(amplitudes)

        if self.keep_keys:
            return keys, probabilities

        return probabilities

    def _prepare_pa_inc(self, unitary):
        self.ct_inverts = []
        for _layer_idx, (sources, destinations, modes) in enumerate(
            self.vectorized_operations
        ):
            self.ct_inverts.append(
                layer_compute_backward(unitary, sources, destinations, modes, self.m)
            )

    def to(self, device: str | torch.device):
        """
        Moves the converter to a specific device.

        :param dtype: The data type to use for the tensors - one can specify either a float or complex dtype.
                      Supported dtypes are torch.float32 or torch.complex64, torch.float64 or torch.complex128.
        :param device: The device to move the converter to.
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(
                f"Expected a string or torch.device, but got {type(device).__name__}"
            )

        if self.output_map_func is not None:
            self.target_indices.to(dtype=dtype, device=self.device)
        for idx, (sources, destinations, modes) in enumerate(
            self.vectorized_operations
        ):
            self.vectorized_operations[idx] = (
                sources.to(device=self.device),
                destinations.to(device=self.device),
                modes.to(device=self.device),
            )
        self._create_torchscript_modules()
        return self

    def compute_pa_inc(
        self,
        unitary: torch.Tensor,
        input_state_prev: list[int],
        input_state: list[int],
        changed_unitary=False,
    ) -> tuple[list[tuple[int, ...]], torch.Tensor]:
        if len(unitary.shape) == 2:
            unitary = unitary.unsqueeze(0)  # Add batch dimension [1 x m x m]
        else:
            pass

        if any(n < 0 for n in input_state) or sum(input_state) == 0:
            raise ValueError("Photon numbers cannot be negative or all zeros")

        if self.computation_space is ComputationSpace.UNBUNCHED and not all(
            x in (0, 1) for x in input_state
        ):
            raise ValueError(
                "Input state must be binary (0s and 1s only) in unbunched mode"
            )
        if self.computation_space is ComputationSpace.DUAL_RAIL:
            for k in range(0, self.m, 2):
                if input_state[k] + input_state[k + 1] != 1:
                    raise ValueError(
                        "Input state must contain exactly one photon per pair in dual_rail mode"
                    )

        batch_size, m, m2 = unitary.shape
        if m != m2 or m != self.m:
            raise ValueError(
                f"Unitary matrix must be square with dimension {self.m}x{self.m}"
            )

        # Check dtype - it should match the complex dtype used for the graph building
        if unitary.dtype != self.complex_dtype:
            # Raise an error instead of just warning and converting
            raise ValueError(
                f"Unitary dtype {unitary.dtype} doesn't match the expected complex dtype {self.complex_dtype} "
                f"for the graph built with dtype {self.dtype}. Please provide a unitary with the correct dtype "
                f"or rebuild the graph with a compatible dtype."
            )

        if self.ct_inverts is None or changed_unitary:
            self._prepare_pa_inc(unitary)

        idx_n_pos = []
        idx_n_neg = []
        self.norm_factor_input = 1
        for i, count in enumerate(input_state):
            for c in range(count):
                self.norm_factor_input *= c + 1
            p = input_state[i] - input_state_prev[i]
            if p > 0:
                idx_n_pos.extend([i] * p)
            elif p < 0:
                idx_n_neg.extend([i] * abs(p))

        amplitudes = self.prev_amplitudes
        if amplitudes is None:
            raise RuntimeError(
                "prev_amplitudes is None - compute must be called before forward"
            )

        num_changes = len(idx_n_pos)

        if num_changes > 0:
            vectorized_operations = self.vectorized_operations[-num_changes:]

            for k in range(num_changes - 1, -1, -1):
                p_neg = idx_n_neg[k]
                invert = self.ct_inverts[k + self.n_photons - num_changes][p_neg]
                amplitudes = amplitudes.unsqueeze(1) @ torch.transpose(invert, -2, -1)
                amplitudes = amplitudes.squeeze(1)

            for layer_idx, (sources, destinations, modes) in enumerate(
                vectorized_operations
            ):
                p_pos = idx_n_pos[layer_idx]
                amplitudes = layer_compute_vectorized(
                    unitary, amplitudes, sources, destinations, modes, p_pos
                )
        amplitudes *= torch.sqrt(self.norm_factor_output.to(amplitudes.device))
        amplitudes /= math.sqrt(self.norm_factor_input)
        self.prev_amplitudes = amplitudes  # type: ignore[assignment]
        # Calculate probabilities
        # probabilities = (amplitudes.abs() ** 2).real

        return amplitudes

    def compute_probs_from_amplitudes(self, amplitudes):
        added_batch_dim = False
        if amplitudes.ndim == 1:
            amplitudes = amplitudes.unsqueeze(0)
            added_batch_dim = True

        is_batched = amplitudes.shape[0] > 1

        probabilities = amplitudes.real**2 + amplitudes.imag**2
        probabilities *= self.norm_factor_output.to(probabilities.device)
        probabilities /= self.norm_factor_input

        # Apply output mapping if needed
        if self.output_map_func is not None:
            probabilities = self.mapping_function(probabilities)
            keys = self.mapped_keys
        else:
            if self.computation_space in (
                ComputationSpace.UNBUNCHED,
                ComputationSpace.DUAL_RAIL,
            ):
                sum_probs = probabilities.sum(dim=1, keepdim=True)
                # Only normalize when sum > 0 to avoid division by zero
                valid_entries = sum_probs > 0
                if valid_entries.any():
                    probabilities = torch.where(
                        valid_entries,
                        probabilities
                        / torch.where(
                            valid_entries, sum_probs, torch.ones_like(sum_probs)
                        ),
                        probabilities,
                    )
            keys = self.final_keys if self.keep_keys else None
        # Remove batch dimension if input was single unitary
        if not is_batched or added_batch_dim:
            probabilities = probabilities.squeeze(0)

        return keys, probabilities


def build_slos_distribution_computegraph(
    m,
    n_photons,
    output_map_func: Callable[[tuple[int, ...]], tuple[int, ...] | None] | None = None,
    computation_space: ComputationSpace | None = None,
    no_bunching: bool | None = None,
    keep_keys: bool = True,
    device=None,
    dtype: torch.dtype = torch.float,
    index_photons: list[tuple[int, ...]] | None = None,
) -> SLOSComputeGraph:
    """Construct a reusable SLOS computation graph.

    Parameters
    ----------
    m : int
        Number of modes in the circuit.
    n_photons : int
        Total number of photons injected in the circuit.
    output_map_func : callable, optional
        Mapping applied to each output Fock state, allowing post-processing.
    computation_space : ComputationSpace, optional
    keep_keys : bool, optional
        Whether to keep the list of mapped Fock states.
    device : torch.device, optional
        Device on which tensors should be allocated.
    dtype : torch.dtype, optional
        Real dtype controlling numerical precision.
    index_photons : list[tuple[int, ...]], optional
        Bounds for each photon placement.

    Returns
    -------
    SLOSComputeGraph
        Pre-built computation graph ready for repeated evaluations.
    """

    if no_bunching is not None:
        raise_no_bunching_deprecated(stacklevel=2)

    if computation_space is None:
        computation_space = ComputationSpace.UNBUNCHED

    compute_graph = SLOSComputeGraph(
        m,
        n_photons,
        output_map_func,
        computation_space,
        keep_keys,
        device,
        dtype,
        index_photons,
    )

    # Add save method to the returned object
    def save(path):
        """
        Save the SLOS computation graph to a file.

        Args:
            path: Path to save the computation graph
        """
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Save metadata
        metadata = {
            "m": compute_graph.m,
            "n_photons": compute_graph.n_photons,
            "computation_space": compute_graph.computation_space.value,
            "keep_keys": compute_graph.keep_keys,
            "dtype_str": str(compute_graph.dtype),
            "has_output_map_func": output_map_func is not None,
        }

        # Save TorchScript layer functions if possible
        # For serializable components only
        torch.save(
            {
                "metadata": metadata,
                "vectorized_operations": compute_graph.vectorized_operations,
                "final_keys": compute_graph.final_keys,
                "mapped_keys": compute_graph.mapped_keys,
                "mapped_indices": compute_graph.mapped_indices
                if hasattr(compute_graph, "mapped_indices")
                else None,
                "total_mapped_keys": compute_graph.total_mapped_keys
                if hasattr(compute_graph, "total_mapped_keys")
                else None,
                "target_indices": compute_graph.target_indices
                if hasattr(compute_graph, "target_indices")
                else None,
            },
            path,
        )

    # Attach the save method to the compute_graph
    compute_graph.save = save  # type: ignore[attr-defined]

    return compute_graph


def load_slos_distribution_computegraph(path):
    """
    Load a previously saved SLOS distribution computation graph.

    Args:
        path: Path to the saved computation graph

    Returns:
        SLOSComputeGraph: Loaded computation graph ready for computations

    Example:
        >>> # Save a computation graph
        >>> graph = build_slos_distribution_computegraph([1, 1])
        >>> graph.save("hom_graph.pt")
        >>>
        >>> # Later, load the saved graph
        >>> loaded_graph = load_slos_distribution_computegraph("hom_graph.pt")
        >>>
        >>> # Use the loaded graph
        >>> unitary = torch.tensor([[0.7071, 0.7071], [0.7071, -0.7071]], dtype=torch.cfloat)
        >>> keys, probs = loaded_graph.compute(unitary)
    """
    # Load saved data
    saved_data = torch.load(path)
    metadata = saved_data["metadata"]

    # Create a minimal graph instance
    m = metadata["m"]
    n_photons = metadata["n_photons"]
    computation_space = ComputationSpace.coerce(metadata.get("computation_space"))
    keep_keys = metadata["keep_keys"]

    # Parse dtype
    dtype_str = metadata.get("dtype_str", "torch.float32")
    if "float16" in dtype_str:
        dtype = torch.float16
    elif "float64" in dtype_str:
        dtype = torch.float64
    else:
        dtype = torch.float32

    # Create basic graph (without output_map_func for now)
    graph = SLOSComputeGraph(
        m, n_photons, None, computation_space, keep_keys, dtype=dtype
    )
    # Restore saved attributes
    graph.vectorized_operations = saved_data["vectorized_operations"]
    graph.final_keys = saved_data["final_keys"]
    graph.mapped_keys = saved_data["mapped_keys"]

    # Restore mapping information if it was used
    if metadata.get("has_output_map_func", False):
        graph.mapped_indices = saved_data["mapped_indices"]
        graph.total_mapped_keys = saved_data["total_mapped_keys"]
        graph.target_indices = saved_data["target_indices"]

        # We need to recreate a dummy output_map_func that uses the saved mapping
        def restored_output_map_func(state):
            # This function just serves as a placeholder to indicate mapping is used
            # The actual mapping is handled by the restored mapped_indices
            return state

        graph.output_map_func = restored_output_map_func

    # Recreate the TorchScript modules
    graph._create_torchscript_modules()

    # Add save method to the loaded graph
    graph.save = lambda p: torch.save(saved_data, p)

    return graph


def compute_slos_distribution(
    unitary: torch.Tensor,
    input_state: list[int],
    output_map_func: Callable[[tuple[int, ...]], tuple[int, ...] | None] | None = None,
    computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
    keep_keys: bool = True,
    index_photons: list[tuple[int, ...]] | None = None,
) -> tuple[list[tuple[int, ...]], torch.Tensor]:
    """
    TorchScript-optimized version of pytorch_slos_output_distribution.

    This function builds the computation graph first, then uses it to compute the probabilities.
    For repeated calculations with the same input configuration but different unitaries,
    it's more efficient to use build_slos_compute_graph() directly.

    Args:
        unitary (torch.Tensor): Single unitary matrix [m x m] or batch of unitaries [b x m x m]
        input_state (List[int]): Number of photons in every mode of the circuit
        output_map_func (callable, optional): Function that maps output states
        computation_space ComputationSpace): Enumeration domain.
        keep_keys (bool): If True, output state keys are returned
        index_photons: List of tuples (first_integer, second_integer). The first_integer is the\
                  lowest index layer a photon can take and the second_integer is the highest index


    Returns:
        Tuple[List[Tuple[int, ...]], torch.Tensor]:
            - List of tuples representing output Fock state configurations
            - Probability distribution tensor
    """
    # Extract device from unitary for graph building
    device = unitary.device if hasattr(unitary, "device") else None

    # Determine appropriate dtype based on unitary's complex dtype
    dtype = _get_float_dtype_for_complex(unitary.dtype)

    # Build graph on the same device as the unitary with matching precision
    graph = build_slos_distribution_computegraph(
        len(input_state),
        sum(input_state),
        output_map_func,
        computation_space,
        no_bunching=None,
        keep_keys=keep_keys,
        device=device,
        dtype=dtype,
        index_photons=index_photons,
    )
    return graph.compute(unitary, input_state)


# Example usage
if __name__ == "__main__":
    import time

    # Test different precisions with explicit dtype specification
    dtypes = [torch.float, torch.float64]
    dtype_names = ["float32", "float64"]

    # Create a test case
    input_state = [1, 1, 0, 0]  # Two photons in first two modes
    m = len(input_state)  # Number of modes derived from input_state
    n_photons = sum(input_state)

    for idx, dtype in enumerate(dtypes):
        print(f"\nTesting with {dtype_names[idx]} precision:")
        # Get corresponding complex dtype using helper function
        complex_dtype = _get_complex_dtype_for_float(dtype)

        # Create random unitary with the appropriate complex dtype
        u = torch.randn(m, m, dtype=complex_dtype)
        q, _ = torch.linalg.qr(u)

        # Method 1: Build graph with specified precision
        start_time = time.time()
        graph = build_slos_distribution_computegraph(m, n_photons, dtype=dtype)
        build_time = time.time() - start_time

        # Compute probabilities
        start_time = time.time()
        keys, probs = graph.compute(q, input_state)
        compute_time = time.time() - start_time

        print("  Method 1 (explicit graph building):")
        print(f"    Graph build time: {build_time:.4f} seconds")
        print(f"    Compute time: {compute_time:.4f} seconds")
        print(f"    Probability sum: {probs.sum().item():.8f}")

        # Method 2: Use compute_slos_distribution with inferred dtype
        start_time = time.time()
        keys2, probs2 = compute_slos_distribution(
            q, input_state
        )  # dtype inferred from unitary
        total_time = time.time() - start_time

        print("  Method 2 (using compute_slos_distribution with inferred dtype):")
        print(f"    Total time: {total_time:.4f} seconds")
        print(f"    Probability sum: {probs2.sum().item():.8f}")
