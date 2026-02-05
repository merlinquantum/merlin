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

import math
from typing import Literal, overload

import perceval as pcvl
import torch

from ..pcvl_pytorch import CircuitConverter, build_slos_distribution_computegraph
from ..utils.combinadics import Combinadics
from ..utils.deprecations import raise_no_bunching_deprecated
from .base import AbstractComputationProcess
from .computation_space import ComputationSpace


class ComputationProcess(AbstractComputationProcess):
    """Handles quantum circuit computation and state evolution."""

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
        no_bunching: bool | None = None,
        output_map_func=None,
    ):
        self.circuit = circuit
        self.input_state = input_state
        self.n_photons = n_photons
        self.trainable_parameters = trainable_parameters
        self.input_parameters = input_parameters
        self.dtype = dtype
        self.device = device

        if no_bunching is not None:
            raise_no_bunching_deprecated(stacklevel=2)

        if computation_space is None:
            computation_space = ComputationSpace.UNBUNCHED

        self.computation_space = computation_space
        self.output_map_func = output_map_func

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

    def _setup_computation_graphs(self):
        """Setup unitary and simulation computation graphs."""
        # Determine parameter specs
        parameter_specs = self.trainable_parameters + self.input_parameters

        # Build unitary graph
        self.converter = CircuitConverter(
            self.circuit, parameter_specs, dtype=self.dtype, device=self.device
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

    def compute(self, parameters: list[torch.Tensor]) -> torch.Tensor:
        """Compute quantum output distribution."""
        # Generate unitary matrix from parameters

        unitary = self.converter.to_tensor(*parameters)
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
        self, parameters: list[torch.Tensor], *, return_keys: Literal[True]
    ) -> tuple[list[tuple[int, ...]], torch.Tensor]: ...

    @overload
    def compute_superposition_state(
        self, parameters: list[torch.Tensor], *, return_keys: Literal[False] = False
    ) -> torch.Tensor: ...

    def compute_superposition_state(
        self, parameters: list[torch.Tensor], *, return_keys: bool = False
    ) -> torch.Tensor | tuple[list[tuple[int, ...]], torch.Tensor]:
        prepared_state = self._prepare_superposition_tensor()
        unitary = self.converter.to_tensor(*parameters)
        changed_unitary = True

        def is_swap_permutation(t1, t2):
            if t1 == t2:
                return False
            diff = [
                (i, i) for i, (x, y) in enumerate(zip(t1, t2, strict=False)) if x != y
            ]
            if len(diff) != 2:
                return False
            i, j = diff[0][0], diff[1][0]

            return t1[i] == t2[j] and t1[j] == t2[i]

        def reorder_swap_chain(lst):
            remaining = lst[:]
            chain = [remaining.pop(0)]
            while remaining:
                for i, candidate in enumerate(remaining):
                    if is_swap_permutation(chain[-1][1], candidate[1]):
                        chain.append(remaining.pop(i))
                        break
                else:
                    chain.append(remaining.pop(0))

            return chain

        mask = (prepared_state.real**2 + prepared_state.imag**2 < 1e-13).all(dim=0)

        masked_input_state = (~mask).int().tolist()

        input_states = [
            (k, self.simulation_graph.mapped_keys[k])
            for k, mask in enumerate(masked_input_state)
            if mask == 1
        ]

        state_list = reorder_swap_chain(input_states)

        prev_state_index, prev_state = state_list.pop(0)

        keys, amplitude = self.simulation_graph.compute(unitary, prev_state)
        amplitudes = torch.zeros(
            (prepared_state.shape[-1], len(self.simulation_graph.mapped_keys)),
            dtype=amplitude.dtype,
            device=prepared_state.device,
        )
        amplitudes[prev_state_index] = amplitude

        for index, fock_state in state_list:
            amplitudes[index] = self.simulation_graph.compute_pa_inc(
                unitary,
                prev_state,
                fock_state,
                changed_unitary=changed_unitary,
            )
            changed_unitary = False
            prev_state = fock_state

        input_state = prepared_state.to(amplitudes.dtype)
        amplitudes = amplitudes / amplitudes.norm(p=2, dim=-1, keepdim=True).clamp_min(
            1e-12
        )

        # The actual sum of amplitudes weighted by input coefficients (for each batch element) is done here
        final_amplitudes = input_state @ amplitudes

        keys_out = list(self.simulation_graph.mapped_keys)

        if return_keys:
            return keys_out, final_amplitudes

        return final_amplitudes

    def compute_ebs_simultaneously(
        self, parameters: list[torch.Tensor], simultaneous_processes: int = 1
    ) -> torch.Tensor:
        """
        Evaluate a single circuit parametrisation against all superposed input
        states by chunking them in groups and delegating the heavy work to the
        TorchScript-enabled batch kernel.

        The method converts the trainable parameters into a unitary matrix,
        normalises the input state (if it is not already normalised), filters
        out components with zero amplitude, and then queries the simulation
        graph for batches of Fock states. Each batch feeds
        :meth:`SLOSComputeGraph.compute_batch`, producing a tensor that contains
        the amplitudes of all reachable output states for the selected input
        components. The partial results are accumulated into a preallocated
        tensor and finally weighted by the complex coefficients of
        ``self.input_state`` to produce the global output amplitudes.

        Args:
            parameters (list[torch.Tensor]): Differentiable parameters that
                encode the photonic circuit. They are forwarded to
                ``self.converter`` to build the unitary matrix used during the
                simulation.
            simultaneous_processes (int): Maximum number of non-zero input
                components that are propagated in a single call to
                ``compute_batch``. Tuning this value allows trading memory
                consumption for wall-clock time on GPU.

        Returns:
            torch.Tensor: The superposed output amplitudes with shape
            ``[batch_size, num_output_states]`` where ``batch_size`` corresponds
            to the number of independent input batches and ``num_output_states``
            is the size of ``self.simulation_graph.mapped_keys``.

        Raises:
            TypeError: If ``self.input_state`` is not a ``torch.Tensor``. The
            simulation graph expects tensor inputs, therefore other sequence
            types (NumPy arrays, lists, etc.) cannot be used here.

        Notes:
            * ``self.input_state`` is normalised in place to avoid an extra
              allocation.
            * Zero-amplitude components are skipped to minimise the number of
              calls to ``compute_batch``.
            * The method is agnostic to the device: tensors remain on the device
              they already occupy, so callers should ensure ``parameters`` and
              ``self.input_state`` live on the same device.
        """

        # input state was validated by _prepare_superposition_tensor, ie: renormalized, typed, and converted from logical basis to fock basis (if shape did not match)
        # we don't want anymore the logical basis but normalization and typing cannot hurt even if it is a small overhead
        prepared_state = self._prepare_superposition_tensor()

        unitary = self.converter.to_tensor(*parameters)
        # Allow classical parameters to be batched: in that case the converter already returns a stack of unitaries.
        batched_parameters = unitary.dim() == 3
        if not batched_parameters:
            unitary = unitary.unsqueeze(0)
        parameter_batch = unitary.shape[0]

        # Find non-zero input states - for efficient processing of only not zero amplitude states
        mask = (prepared_state.real**2 + prepared_state.imag**2 < 1e-13).all(dim=0)
        masked_input_state = (~mask).int().tolist()
        input_states = [
            (k, self.simulation_graph.mapped_keys[k])
            for k, mask in enumerate(masked_input_state)
            if mask == 1
        ]

        # Initialize amplitudes tensor
        amplitudes = torch.zeros(
            (
                parameter_batch,
                prepared_state.shape[-1],
                len(self.simulation_graph.mapped_keys),
            ),
            dtype=unitary.dtype,
            device=prepared_state.device,
        )

        # Process input states in batches
        for i in range(0, len(input_states), simultaneous_processes):
            batch_end = min(i + simultaneous_processes, len(input_states))
            batch_indices = []
            batch_fock_states = []

            for j in range(i, batch_end):
                idx, fock_state = input_states[j]
                batch_indices.append(idx)
                batch_fock_states.append(fock_state)

            # Compute batch amplitudes
            _, batch_amplitudes = self.simulation_graph.compute_batch(
                unitary, batch_fock_states
            )
            # Stack amplitudes for each input state in the batch
            for k, idx in enumerate(batch_indices):
                amplitudes[:, idx, :] = batch_amplitudes[:, :, k]

        # Apply input state coefficients
        input_state = prepared_state.to(amplitudes.dtype)

        amplitudes = amplitudes / amplitudes.norm(p=2, dim=-1, keepdim=True).clamp_min(
            1e-12
        )
        # The actual sum of amplitudes weighted by input coefficients (for each batch element) is done here
        # Combine each prepared input coefficient with the output amplitudes of every propagated Fock component.
        final_amplitudes = torch.einsum("se, beo -> bso", input_state, amplitudes)

        if final_amplitudes.shape[0] == 1:
            final_amplitudes = final_amplitudes.squeeze(0)
        if final_amplitudes.ndim == 3 and final_amplitudes.shape[1] == 1:
            final_amplitudes = final_amplitudes.squeeze(1)

        return final_amplitudes

    def compute_with_keys(self, parameters: list[torch.Tensor]):
        """Compute quantum output distribution and return both keys and probabilities."""
        # Generate unitary matrix from parameters
        unitary = self.converter.to_tensor(*parameters)

        # Compute output distribution using the input state
        keys, amplitudes = self.simulation_graph.compute(unitary, self.input_state)

        return keys, amplitudes

    def _expected_superposition_size(self) -> int:
        """Expected number of Fock states given current computation space."""
        if self.n_photons < 0:
            raise ValueError("Number of photons must be non-negative.")
        if self.computation_space is ComputationSpace.DUAL_RAIL:
            if self.n_photons is None:
                raise ValueError("Dual-rail encoding requires 'n_photons'.")
            if self.m != 2 * self.n_photons:
                raise ValueError(
                    "Dual-rail encoding requires the number of modes to equal 2 * n_photons."
                )
            # Dual-rail limits to 2**n logical states (one photon per rail pair).
            return 2**self.n_photons
        if self.computation_space is ComputationSpace.UNBUNCHED:
            if self.n_photons > self.m:
                raise ValueError(
                    "Invalid configuration: ComputationSpace.UNBUNCHED requires "
                    "n_photons to be less than or equal to the number of modes."
                )
            return math.comb(self.m, self.n_photons)
        return math.comb(self.m + self.n_photons - 1, self.n_photons)

    def _validate_superposition_state_shape(self, input_state: torch.Tensor) -> None:
        """Ensure the provided superposition state matches the configured computation space."""
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

        expected = self._expected_superposition_size()
        if state_dim != expected:
            if (
                self.computation_space is ComputationSpace.DUAL_RAIL
                and state_dim == len(self.simulation_graph.mapped_keys)
            ):
                return
            if self.computation_space is ComputationSpace.DUAL_RAIL:
                explanation = (
                    f"expected 2**n_photons = 2**{self.n_photons} = {expected}"
                )
            elif self.computation_space is ComputationSpace.UNBUNCHED:
                explanation = f"expected C(m, n_photons) = C({self.m}, {self.n_photons}) = {expected}"
            else:
                explanation = (
                    f"expected C(m + n_photons - 1, n_photons) = "
                    f"C({self.m + self.n_photons - 1}, {self.n_photons}) = {expected}"
                )
            raise ValueError(
                "Input state dimension mismatch for computation_space "
                f"'{self.computation_space}': got {state_dim}, {explanation}."
            )

    def _should_defer_state_validation(self, tensor: torch.Tensor) -> bool:
        """Detect amplitude tensors that will be validated after configuring dual-rail space."""
        if tensor.dim() == 1:
            state_dim = tensor.shape[0]
        elif tensor.dim() == 2:
            state_dim = tensor.shape[1]
        else:
            return False

        if self.n_photons is None or self.m is None:
            return False

        return (
            self.computation_space is ComputationSpace.UNBUNCHED
            and self.m == 2 * self.n_photons
            and state_dim == 2**self.n_photons
        )

    def _coerce_superposition_tensor_shape(
        self, tensor: torch.Tensor
    ) -> torch.Tensor | None:
        """Attempt to reconcile tensors encoded in a smaller logical basis."""
        if self.computation_space is not ComputationSpace.FOCK:
            return None

        if self.n_photons is None or self.m is None:
            return None

        if tensor.dim() == 1:
            feature_dim = tensor.shape[0]
        elif tensor.dim() == 2:
            feature_dim = tensor.shape[1]
        else:
            return None

        # Detect tensors encoded in the UNBUNCHED basis and lift them to the Fock basis.
        unbunched_size = math.comb(self.m, self.n_photons)
        if feature_dim != unbunched_size:
            return None

        mapped_keys = [
            tuple(key)
            for key in self.simulation_graph.mapped_keys  # type: ignore[attr-defined]
        ]
        key_to_index = {state: idx for idx, state in enumerate(mapped_keys)}

        try:
            combinator = Combinadics("unbunched", self.n_photons, self.m)
        except ValueError:
            return None

        indices: list[int] = []
        for state in combinator.iter_states():
            index = key_to_index.get(state)
            if index is None:
                return None
            indices.append(index)

        target_dim = len(mapped_keys)
        if tensor.dim() == 1:
            expanded = tensor.new_zeros(target_dim)
            expanded[indices] = tensor
        else:
            expanded = tensor.new_zeros(tensor.shape[0], target_dim)
            expanded[:, indices] = tensor

        return expanded

    def _prepare_superposition_tensor(self) -> torch.Tensor:
        """Validate, normalise, and convert the stored superposition state to the correct dtype."""
        if not isinstance(self.input_state, torch.Tensor):
            raise TypeError("Input state should be a tensor")

        tensor = self.input_state

        coerced = self._coerce_superposition_tensor_shape(tensor)
        if coerced is not None:
            tensor = coerced

        self._validate_superposition_state_shape(tensor)

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.complex64)
        elif tensor.dtype == torch.float64:
            tensor = tensor.to(torch.complex128)
        elif tensor.dtype not in (torch.complex64, torch.complex128):
            raise TypeError(
                f"Unsupported dtype for superposition state: {tensor.dtype}"
            )

        norm = tensor.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
        tensor = tensor / norm
        self.input_state = tensor
        return tensor


class ComputationProcessFactory:
    """Factory for creating computation processes."""

    @staticmethod
    def create(
        circuit: pcvl.Circuit,
        input_state: list[int] | torch.Tensor,
        trainable_parameters: list[str],
        input_parameters: list[str],
        computation_space: ComputationSpace | None = None,
        **kwargs,
    ) -> ComputationProcess:
        """Create a computation process."""
        return ComputationProcess(
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=trainable_parameters,
            input_parameters=input_parameters,
            computation_space=computation_space,
            **kwargs,
        )
