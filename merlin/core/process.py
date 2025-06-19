
"""
Quantum computation processes and factories.
"""
from typing import List, Optional
import torch
import perceval as pcvl

from .base import AbstractComputationProcess
from merlin.pcvl_pytorch import CircuitConverter, build_slos_distribution_computegraph

class ComputationProcess(AbstractComputationProcess):
    """Handles quantum circuit computation and state evolution."""

    def __init__(self,
                 circuit: pcvl.Circuit,
                 input_state: List[int],
                 trainable_parameters: List[str],
                 input_parameters: List[str],
                 reservoir_mode: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None,
                 no_bunching: bool = None,
                 output_map_func=None,
                 index_photons=None):

        self.circuit = circuit
        self.input_state = input_state
        self.trainable_parameters = trainable_parameters
        self.input_parameters = input_parameters
        self.reservoir_mode = reservoir_mode
        self.dtype = dtype
        self.device = device
        self.no_bunching = no_bunching
        self.output_map_func = output_map_func
        self.index_photons = index_photons

        # Extract circuit parameters for graph building
        if type(input_state) is dict:
            input_state = list(input_state.keys())[0]
        self.m = len(input_state)# Number of modes
        self.n_photons = sum(input_state)  # Total number of photons

        # Build computation graphs
        self._setup_computation_graphs()

    def _setup_computation_graphs(self):
        """Setup unitary and simulation computation graphs."""
        # Determine parameter specs
        if self.reservoir_mode:
            parameter_specs = self.trainable_parameters + self.input_parameters + ["phi_"]
        else:
            parameter_specs = self.trainable_parameters + self.input_parameters

        # Build unitary graph
        self.converter = CircuitConverter(self.circuit,parameter_specs,dtype=self.dtype, device=self.device)

        # Build simulation graph with correct parameters
        self.simulation_graph = build_slos_distribution_computegraph(
            m=self.m,  # Number of modes
            n_photons=self.n_photons,  # Total number of photons
            output_map_func=self.output_map_func,
            no_bunching=self.no_bunching,
            keep_keys=True,  # Usually want to keep keys for output interpretation
            device=self.device,
            dtype=self.dtype,
            index_photons=self.index_photons
        )

    def compute(self, parameters: List[torch.Tensor]) -> torch.Tensor:
        """Compute quantum output distribution."""
        # Generate unitary matrix from parameters
        unitary = self.converter.to_tensor(*parameters)

        # Compute output distribution using the input state
        if type(self.input_state) is dict:
            input_state = list(self.input_state.keys())[0]
        else:
            input_state = self.input_state
        keys, distribution = self.simulation_graph.compute(unitary, input_state)

        return distribution


    def compute_superposition_state(self, parameters: List[torch.Tensor]) -> torch.Tensor:
        unitary = self.converter.to_tensor(*parameters)


        def is_swap_permutation(t1, t2):

            if t1 == t2:
                return False
            diff = [(i, i) for i, (x, y) in enumerate(zip(t1, t2)) if x != y]
            if len(diff) != 2:
                return False
            i, j = diff[0][0], diff[1][0]

            return t1[i] == t2[j] and t1[j] == t2[i]

        def reorder_swap_chain(lst):

            from collections import deque

            remaining = lst[:]
            chain = [remaining.pop(0)]  # Commence avec le premier élément
            while remaining:
                for i, candidate in enumerate(remaining):
                    if is_swap_permutation(chain[-1], candidate):
                        chain.append(remaining.pop(i))
                        break
                else:
                    chain.append(remaining.pop(0))

            return chain

        state_list = reorder_swap_chain(list(self.input_state.keys()))




        prev_state = state_list.pop(0)
        keys, distribution = self.simulation_graph.compute(unitary, prev_state)
        distributions = distribution * self.input_state[prev_state]

        for fock_state in state_list:
            keys, distribution = self.simulation_graph.compute_pa_inc(unitary, prev_state, fock_state)
            distributions += distribution * self.input_state[fock_state]
            prev_state = fock_state

        return distributions

    def compute_with_keys(self, parameters: List[torch.Tensor]):
        """Compute quantum output distribution and return both keys and probabilities."""
        # Generate unitary matrix from parameters
        unitary = self.converter.to_tensor(*parameters)

        # Compute output distribution using the input state
        keys, distribution = self.simulation_graph.compute(unitary, self.input_state)

        return keys, distribution


class ComputationProcessFactory:
    """Factory for creating computation processes."""

    @staticmethod
    def create(circuit: pcvl.Circuit,
               input_state: List[int],
               trainable_parameters: List[str],
               input_parameters: List[str],
               reservoir_mode: bool = False,
               no_bunching: bool = None,
               output_map_func=None,
               index_photons=None,
               **kwargs) -> ComputationProcess:
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
            **kwargs
        )
