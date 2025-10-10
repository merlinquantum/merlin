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

import perceval as pcvl
import torch

from ..pcvl_pytorch import CircuitConverter, build_slos_distribution_computegraph
from .base import AbstractComputationProcess


class ComputationProcess(AbstractComputationProcess):
    """Handles quantum circuit computation and state evolution."""

    def __init__(
        self,
        circuit: pcvl.Circuit,
        input_state: list[int] | torch.Tensor,
        trainable_parameters: list[str],
        input_parameters: list[str],
        n_photons: int = None,
        reservoir_mode: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        no_bunching: bool = None,
        output_map_func=None,
        index_photons=None,
    ):
        self.circuit = circuit
        self.input_state = input_state
        self.n_photons = n_photons
        self.trainable_parameters = trainable_parameters
        self.input_parameters = input_parameters
        self.reservoir_mode = reservoir_mode
        self.dtype = dtype
        self.device = device
        self.no_bunching = no_bunching
        self.output_map_func = output_map_func
        self.index_photons = index_photons

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
            no_bunching=self.no_bunching,
            keep_keys=True,  # Usually want to keep keys for output interpretation
            device=self.device,
            dtype=self.dtype,
            index_photons=self.index_photons,
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

    def compute_superposition_state(
        self, parameters: list[torch.Tensor]
    ) -> torch.Tensor:
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
            chain = [remaining.pop(0)]  # Commence avec le premier élément
            while remaining:
                for i, candidate in enumerate(remaining):
                    if is_swap_permutation(chain[-1][1], candidate[1]):
                        chain.append(remaining.pop(i))
                        break
                else:
                    chain.append(remaining.pop(0))

            return chain

        if type(self.input_state) is torch.Tensor:
            if len(self.input_state.shape) == 1:
                self.input_state = self.input_state.unsqueeze(0)
            if self.input_state.dtype == torch.float32:
                self.input_state = self.input_state.to(torch.complex64)
            elif self.input_state.dtype == torch.float64:
                self.input_state = self.input_state.to(torch.complex128)

        else:
            raise TypeError("Input state should be a tensor")

        sum_input = self.input_state.abs().pow(2).sum(dim=1).sqrt().unsqueeze(1)
        self.input_state = self.input_state / sum_input

        mask = (self.input_state.real**2 + self.input_state.imag**2 < 1e-13).all(dim=0)

        masked_input_state = (~mask).int().tolist()

        input_states = [
            (k, self.simulation_graph.mapped_keys[k])
            for k, mask in enumerate(masked_input_state)
            if mask == 1
        ]

        state_list = reorder_swap_chain(input_states)

        prev_state_index, prev_state = state_list.pop(0)

        _, amplitude = self.simulation_graph.compute(unitary, prev_state)
        amplitudes = torch.zeros(
            (self.input_state.shape[-1], len(self.simulation_graph.mapped_keys)),
            dtype=amplitude.dtype,
            device=self.input_state.device,
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
        input_state = self.input_state.to(amplitudes.dtype)

        final_amplitudes = input_state @ amplitudes

        return final_amplitudes

    def compute_with_keys(self, parameters: list[torch.Tensor]):
        """Compute quantum output distribution and return both keys and probabilities."""
        # Generate unitary matrix from parameters
        unitary = self.converter.to_tensor(*parameters)

        # Compute output distribution using the input state
        keys, amplitudes = self.simulation_graph.compute(unitary, self.input_state)

        return keys, amplitudes


class ComputationProcessFactory:
    """Factory for creating computation processes."""

    @staticmethod
    def create(
        circuit: pcvl.Circuit,
        input_state: list[int] | torch.Tensor,
        trainable_parameters: list[str],
        input_parameters: list[str],
        reservoir_mode: bool = False,
        no_bunching: bool = None,
        output_map_func=None,
        index_photons=None,
        **kwargs,
    ) -> ComputationProcess:
        """Create a computation process."""
        return ComputationProcess(
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=trainable_parameters,
            input_parameters=input_parameters,
            reservoir_mode=reservoir_mode,
            no_bunching=no_bunching,
            output_map_func=output_map_func,
            index_photons=index_photons,
            **kwargs,
        )
