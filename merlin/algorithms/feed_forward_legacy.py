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

from itertools import product

import perceval as pcvl
import torch
from perceval.components import BS, PS

from ..core.computation_space import ComputationSpace
from ..core.generators import StateGenerator, StatePattern
from ..measurement.strategies import MeasurementStrategy
from ..utils.deprecations import raise_no_bunching_deprecated
from .layer import QuantumLayer


def create_circuit(M, input_size):
    """Create a quantum photonic circuit with beam splitters and phase shifters.

    Args:
        M (int): Number of modes in the circuit.

    Returns:
        pcvl.Circuit: A quantum photonic circuit with alternating beam splitter layers and phase shifters.
    """
    # TO DO: Use the circuit builder to create this circuit
    circuit = pcvl.Circuit(M)

    def layer_bs(circuit, k, M, j):
        for i in range(k, M - 1, 2):
            theta = pcvl.P(f"phi_{i}_{j}")
            circuit.add(i, BS(theta=theta))

    layer_bs(circuit, 0, M, 0)
    layer_bs(circuit, 1, M, 1)
    layer_bs(circuit, 0, M, 2)
    layer_bs(circuit, 1, M, 3)
    layer_bs(circuit, 0, M, 4)
    for i in range(input_size):
        phi = pcvl.P(f"pl_{i}")
        circuit.add(i, PS(phi))
    layer_bs(circuit, 0, M, 5)
    layer_bs(circuit, 1, M, 6)
    layer_bs(circuit, 0, M, 7)
    layer_bs(circuit, 1, M, 8)
    layer_bs(circuit, 0, M, 9)
    return circuit


def define_layer_no_input(n_modes, n_photons, circuit_type=None):
    """Define a quantum layer for feed-forward processing.

    Args:
        n_modes (int): Number of optical modes.
        n_photons (int): Number of photons in the layer.

    Returns:
        QuantumLayer: A configured quantum layer with trainable parameters.
    """

    circuit = create_circuit(n_modes, 0)
    input_state = StateGenerator.generate_state(n_modes, n_photons, StatePattern.SPACED)

    layer = QuantumLayer(
        input_size=0,
        circuit=circuit,
        n_photons=n_photons,
        input_state=input_state,  # Random Initial quantum state used only for initialization
        measurement_strategy=MeasurementStrategy.amplitudes(ComputationSpace.UNBUNCHED),
        trainable_parameters=["phi"],
    )
    return layer


def define_layer_with_input(M, N, input_size, circuit_type=None):
    """Define the first layers of the feed-forward block, those with an input size > 0.

    Args:
        M (int): Number of modes in the circuit.
        N (int): Number of photons.

    Returns:
        QuantumLayer: The first quantum layer with input parameters.
    """
    # TO DO: The Quantum Layer could be defined with only three variables:
    # (number of modes, number of photons, input size)

    circuit = create_circuit(M, input_size)
    input_state = StateGenerator.generate_state(M, N, StatePattern.SPACED)
    layer = QuantumLayer(
        input_size=input_size,
        circuit=circuit,
        n_photons=N,
        input_state=input_state,  # Random Initial quantum state used only for initialization
        measurement_strategy=MeasurementStrategy.amplitudes(ComputationSpace.UNBUNCHED),
        input_parameters=["pl"],  # Optional: Specify device
        trainable_parameters=["phi"],
    )
    return layer


class FeedForwardBlockLegacy(torch.nn.Module):
    """
    Feed-forward quantum neural network for photonic computation.

    This class models a **conditional feed-forward architecture** used in
    *quantum photonic circuits*. It connects multiple quantum layers in a
    branching tree structure — where each branch corresponds to a sequence
    of photon-detection outcomes on designated conditional modes.

    Each node in this feedforward tree represents a `QuantumLayer` that acts
    on a quantum state conditioned on measurement results of previous layers.

    The recursion continues until a specified depth, allowing the model to
    simulate complex conditional evolution of quantum systems.

    Detector support:
        The current feed-forward implementation expects amplitude access for
        every intermediate layer (``MeasurementStrategy.AMPLITUDES``) and
        therefore assumes ideal PNR detectors. Custom detector transforms or
        Perceval experiments with threshold / hybrid detectors are not yet
        supported inside this block.

    ---
    Args:
        input_size (int):
            Number of classical input features (used for hybrid quantum-classical computation).

        n (int):
            Number of photons in the system.

        m (int):
            Total number of photonic modes.

        depth (int, optional):
            Maximum depth of feed-forward recursion.
            Defaults to `m - 1` if not specified.

        state_injection (bool, optional):
            If True, allows re-injecting quantum states at intermediate steps
            (useful for simulating sources or ancilla modes). Default = False.

        conditional_modes (list[int], optional):
            List of mode indices on which photon detection is performed.
            Determines the branching structure. Defaults to `[0]`.

        layers (list[QuantumLayer], optional):
            Predefined list of quantum layers (if any). If not provided,
            layers are automatically generated.

        circuit_type (str, optional):
            Type of quantum circuit architecture used to build each layer.
            Acts as a “template selector” for circuit structure generation.
    """

    # TO DO: add a "circuit_type" attribute to select quantum circuit template

    def __init__(
        self,
        input_size: int,
        n: int,
        m: int,
        depth: int | None = None,
        state_injection=False,
        conditional_modes: list[int] = None,
        layers: list = None,
        circuit_type=None,
        device=None,
    ):
        super().__init__()

        self.m = m
        self.n_photons = n
        self.input_size = input_size
        self.state_injection = state_injection
        self.device = device or torch.device("cpu")

        self.conditional_modes = conditional_modes or [0]
        self.n_cond = len(self.conditional_modes)
        self.depth = depth if depth is not None else (self.m - 1)

        self.layers = {}
        self.input_segments = {}
        self._output_keys = None

        if layers is None:
            self.define_layers(circuit_type)
        else:
            tuples = self.generate_possible_tuples()
            self.tuples = tuples
            assert len(tuples) == len(layers), (
                "Mismatch between number of tuples and provided layers."
            )
            self.layers = {tuples[k]: layers[k] for k in range(len(layers))}

            start = 0
            for tup in tuples:
                input_size = self.layers[tup].input_size
                self.input_segments[tup] = (start, start + input_size)
                start += input_size
            assert start == self.input_size, f"Input size mismatch: {start}"

        # Move everything to device immediately
        self.to(self.device)

    # =======================================================================
    #  Tuple and Layer Definition Utilities
    # =======================================================================

    def generate_possible_tuples(self):
        """
        Generate all possible conditional outcome tuples.

        Each tuple represents one possible sequence of photon detection results
        across all conditional modes up to a given depth. For example, with
        `n_cond = 2` and `depth = 3`, tuples correspond to binary sequences of
        length `depth * n_cond`.

        Returns:
            list[tuple[int]]:
                List of tuples containing binary measurement outcomes (0/1).
        """
        possible_tuples = []
        for depth in range(self.depth + 1):
            # Each depth adds new outcomes for every conditional mode
            for t in product([0, 1], repeat=depth * self.n_cond):
                if self.state_injection:
                    # Allow all tuples if state re-injection is active
                    possible_tuples.append(t)
                else:
                    # Restrict based on photon conservation constraints
                    n_ones = t.count(1)
                    n_zeros = t.count(0)
                    if n_ones <= self.n_photons - 1 and n_zeros <= (
                        self.m - self.n_photons - 1
                    ):
                        possible_tuples.append(t)
        return possible_tuples

    def define_layers(self, circuit_type):
        """
        Define and instantiate all quantum layers for each measurement outcome path.

        Each tuple (representing a branch of the feedforward tree) is mapped to
        a `QuantumLayer` object. Depending on whether the state injection mode
        is active, the number of modes/photons and the input size differ.

        Args:
            circuit_type (str): Template name or circuit architecture type.

        Raises:
            AssertionError: If total input size does not match after allocation.
        """
        input_size = self.input_size
        tuples = self.generate_possible_tuples()
        self.tuples = tuples
        self.input_segments = {}
        start = 0

        for tup in tuples:
            n = sum(tup)  # number of detected photons (1's)
            m = len(tup)  # number of conditioned modes so far

            # Determine input size allocated to this quantum layer
            if self.state_injection:
                local_input = min(self.m, input_size)
            else:
                local_input = min(self.m - m, input_size)

            # Define quantum layer with or without classical input
            if local_input > 0:
                if self.state_injection:
                    layer = define_layer_with_input(
                        self.m, self.n_photons, local_input, circuit_type=circuit_type
                    )
                else:
                    layer = define_layer_with_input(
                        self.m - m,
                        self.n_photons - n,
                        local_input,
                        circuit_type=circuit_type,
                    )
            else:
                # If no classical input, define a purely quantum layer
                if self.state_injection:
                    layer = define_layer_no_input(self.m, self.n_photons)
                else:
                    layer = define_layer_no_input(self.m - m, self.n_photons - n)

            # Store layer and its input segment boundaries
            self.layers[tup] = layer
            self.input_segments[tup] = (start, start + local_input)
            input_size -= local_input
            start += local_input

        assert input_size == 0, f"Remaining unallocated input size: {input_size}"

    def to(self, device):
        """
        Moves the FeedForwardBlock and all its QuantumLayers to the specified device.

        Args:
            device (str or torch.device): Target device ('cpu', 'cuda', 'mps', etc.)
        """
        device = torch.device(device)
        self.device = device
        super().to(device)

        # Move all quantum layers and their parameters
        for _, layer in self.layers.items():
            if hasattr(layer, "to"):
                layer.to(device)
            elif hasattr(layer, "parameters"):
                for p in layer.parameters():
                    p.data = p.data.to(device)

        return self

    # =======================================================================
    #  Recursive Feedforward Computation
    # =======================================================================

    def parameters(self):
        """Iterate over all trainable parameters from every quantum layer."""
        for layer in self.layers.values():
            yield from layer.parameters()

    def iterate_feedforward(
        self,
        current_tuple,
        remaining_amplitudes,
        keys,
        accumulated_prob,
        intermediary,
        outputs,
        depth=0,
        x=None,
    ):
        """
        Recursive feedforward traversal of the quantum circuit tree.

        At each step:
            1. Evaluate photon detection outcomes (0/1) on conditional modes.
            2. For each possible combination, compute probabilities.
            3. Apply the corresponding quantum layer and recurse deeper.

        Args:
            current_tuple (tuple[int]): Current measurement sequence path.
            remaining_amplitudes (torch.Tensor): Quantum amplitudes of current state.
            keys (list[tuple[int]]): Fock basis keys for amplitudes.
            accumulated_prob (torch.Tensor or float): Product of probabilities so far.
            intermediary (dict): Stores intermediate probabilities.
            outputs (dict): Stores final output probabilities for all branches.
            depth (int): Current recursion depth.
            x (torch.Tensor, optional): Classical input features.
        """
        # Base case: end of tree reached
        if depth >= self.depth:
            fock_probs = remaining_amplitudes.abs().pow(2)
            for i, key in enumerate(keys):
                if key not in outputs:
                    outputs[key] = torch.zeros_like(accumulated_prob)
                outputs[key] += accumulated_prob * fock_probs[:, i]
            return

        # Generate all possible binary measurement outcomes
        outcome_combos = list(product([0, 1], repeat=self.n_cond))
        mode_indices = self._indices_by_values(keys, self.conditional_modes)

        for combo in outcome_combos:
            idx_combo = mode_indices[combo]
            prob_combo = remaining_amplitudes[:, idx_combo].abs().pow(2).sum(dim=1)
            current_key = current_tuple + combo
            intermediary[current_key] = prob_combo

            layer = self.layers.get(current_key, None)
            if layer is not None:
                # Map Fock basis indices to the next layer's key space
                if self.state_injection:
                    match_idx = idx_combo
                    keys_next = keys
                else:
                    keys_next = layer.computation_process.simulation_graph.mapped_keys
                    match_idx = self._match_indices_multi(
                        keys, keys_next, self.conditional_modes, combo
                    )

                # Set input quantum state for the layer
                layer.computation_process.input_state = remaining_amplitudes[
                    :, match_idx
                ]
                start, end = self.input_segments[current_key]

                # Execute layer with or without classical input
                if start != end:
                    amps_next = layer(x[:, start:end])
                else:
                    amps_next = layer()

                # Recurse into next layer
                new_prob = accumulated_prob * prob_combo
                self.iterate_feedforward(
                    current_key,
                    amps_next,
                    keys_next,
                    new_prob,
                    intermediary,
                    outputs,
                    depth + 1,
                    x=x,
                )
            else:
                # Reached an end branch without further layers
                final_tuple = current_key + (0,) * (
                    (self.depth - len(current_tuple)) * self.n_cond
                )
                outputs[final_tuple] = accumulated_prob * prob_combo

    # =======================================================================
    #  Index Management Utilities
    # =======================================================================

    def _indices_by_values(self, keys, modes):
        """
        Compute index masks for all joint outcomes across conditional modes.

        Args:
            keys (torch.Tensor): Tensor of Fock states (basis keys).
            modes (list[int]): Conditional mode indices.

        Returns:
            dict[tuple[int], torch.Tensor]: Mapping from outcome tuple → indices.
        """
        t = torch.tensor(keys)
        combos = list(product([0, 1], repeat=len(modes)))
        out = {}
        for combo in combos:
            mask = torch.ones(len(keys), dtype=torch.bool)
            for j, mode in enumerate(modes):
                mask &= t[:, mode] == combo[j]
            out[combo] = torch.nonzero(mask, as_tuple=True)[0]
        return out

    def _match_indices_multi(self, data, data_out, modes, values):
        """
        Match indices between two Fock bases differing by removed conditional modes.

        Args:
            data (list[tuple[int]]): Original Fock basis.
            data_out (list[tuple[int]]): Reduced Fock basis (after measurement).
            modes (list[int]): Indices of removed modes.
            values (tuple[int]): Measured values (0/1) for removed modes.

        Returns:
            torch.Tensor: Tensor of matching indices.
        """
        out_map = {tuple(row): i for i, row in enumerate(data_out)}
        idx = []
        for tup in data:
            reduced = tuple(v for i, v in enumerate(tup) if i not in modes)
            if reduced in out_map and all(
                tup[m] == values[j] for j, m in enumerate(modes)
            ):
                idx.append(out_map[reduced])
        return torch.tensor(idx)

    # =======================================================================
    #  Forward Pass & Layer Management
    # =======================================================================

    def forward(self, x):
        """
        Perform the full quantum-classical feedforward computation.

        Args:
            x (torch.Tensor): Classical input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Final output tensor containing probabilities for each
                          terminal measurement configuration.
        """
        if x.shape[-1] != self.input_size:
            raise ValueError(f"The input should be of size {self.input_size}")
        intermediary, outputs = {}, {}

        # Run the first quantum layer (root of the tree)
        input_size = min(self.input_size, self.m)
        layer = self.layers[()]
        amplitudes = layer(x[:, :input_size])
        keys = layer.computation_process.simulation_graph.mapped_keys

        # Recursively propagate through all branches
        self.iterate_feedforward(
            (), amplitudes, keys, 1.0, intermediary, outputs, 0, x=x
        )
        self._output_keys = outputs.keys()
        return torch.stack(list(outputs.values()), dim=1)

    def get_output_size(self):
        """Compute the number of output channels (post-measurement outcomes)."""
        x = torch.rand(1, self.input_size)
        return self.forward(x).shape[-1]

    def size_ff_layer(self, k: int):
        """Return number of feed-forward branches at layer depth `k`."""
        tuples_k = [1 for tup in self.tuples if len(tup) == k * self.n_cond]
        return len(tuples_k)

    def define_ff_layer(self, k: int, layers: list):
        """
        Replace quantum layers at a specific depth `k`.

        Args:
            k (int): Feed-forward layer depth index.
            layers (list[QuantumLayer]): List of replacement layers.
        """
        len_layers = self.size_ff_layer(k)
        assert len(layers) == len_layers, f"layers should be of length {len_layers}"
        for i, t in enumerate(product([0, 1], repeat=k)):
            if t in self.layers:
                self.layers[t] = layers[i]
        self._recompute_segments()

    def input_size_ff_layer(self, k: int):
        """Return the list of input sizes for all layers at depth `k`."""
        return [
            self.layers[tup].input_size
            for tup in self.tuples
            if len(tup) == k * self.n_cond
        ]

    @property
    def output_keys(self):
        """Return cached output keys, or compute them via a dummy forward pass."""
        if self._output_keys is None:
            x = torch.rand(1, self.input_size)
            _ = self.forward(x)
        return list(self._output_keys)

    def _recompute_segments(self):
        """
        Recalculate the `input_segments` mapping between the classical input
        vector and each quantum layer, after any structural modification.
        """
        start = 0
        total_input_size = 0
        self.input_segments = {}

        for tup in self.tuples:
            if tup in self.layers:
                input_size = self.layers[tup].input_size
                self.input_segments[tup] = (start, start + input_size)
                start += input_size
                total_input_size += input_size
            else:
                self.input_segments[tup] = (0, 0)

        # Update internal input size
        self.input_size = total_input_size
        print(f"New input size: {self.input_size}")


class PoolingFeedForwardLegacy(torch.nn.Module):
    """
    A quantum-inspired pooling module that aggregates amplitude information
    from an input quantum state representation into a lower-dimensional output space.

    This module computes mappings between input and output Fock states (defined
    by `keys_in` and `keys_out`) based on a specified pooling scheme. It then
    aggregates the amplitudes according to these mappings, normalizing the result
    to preserve probabilistic consistency.

    Parameters
    ----------
    n_modes : int
        Number of input modes in the quantum circuit.
    n_photons : int
        Number of photons used in the quantum simulation.
    n_output_modes : int
        Number of output modes after pooling.
    pooling_modes : list of list of int, optional
        Specifies how input modes are grouped (pooled) into output modes.
        Each sublist contains the indices of input modes to pool together
        for one output mode. If None, an even pooling scheme is automatically generated.
    no_bunching : bool | None, optional (deprecated)
        Deprecated and now removed; use computation_space in MeasurementStrategy instead.

    Attributes
    ----------
    match_indices : torch.Tensor
        Tensor containing the indices mapping input states to output states.
    exclude_indices : torch.Tensor
        Tensor containing indices of input states that have no valid mapping
        to an output state.
    keys_out : list
        List of output Fock state keys (from Perceval simulation graph).
    n_modes : int
        Number of input modes.
    """

    def __init__(
        self,
        n_modes: int,
        n_photons: int,
        n_output_modes: int,
        pooling_modes: list[list[int]] = None,
        no_bunching: bool | None = None,
    ):
        super().__init__()
        if no_bunching is not None:
            raise_no_bunching_deprecated(stacklevel=2)
        if no_bunching is None:
            no_bunching = True
        keys_in = QuantumLayer(
            0,
            circuit=pcvl.Circuit(n_modes),
            n_photons=n_photons,
            computation_space=ComputationSpace.UNBUNCHED
            if no_bunching
            else ComputationSpace.FOCK,
        ).computation_process.simulation_graph.mapped_keys
        keys_out = QuantumLayer(
            0,
            circuit=pcvl.Circuit(n_output_modes),
            n_photons=n_photons,
            computation_space=ComputationSpace.UNBUNCHED
            if no_bunching
            else ComputationSpace.FOCK,
        ).computation_process.simulation_graph.mapped_keys

        # If no pooling structure is provided, construct a balanced one
        if pooling_modes is None:
            num_skips = n_modes // n_output_modes
            first_skips = n_modes % n_output_modes
            index_num_skips = list(range(0, n_modes + 1, num_skips))
            index_first_skips = (
                [0]
                + list(range(1, first_skips + 1))
                + [first_skips] * (n_output_modes - first_skips)
            )
            index_skips = [
                index_first_skip + index_num_skip
                for (index_first_skip, index_num_skip) in zip(
                    index_first_skips, index_num_skips, strict=False
                )
            ]
            pooling_modes = [
                list(range(index_skips[k], index_skips[k + 1]))
                for k in range(n_output_modes)
            ]

        match_indices, exclude_indices = self.match_tuples(
            keys_in, keys_out, pooling_modes
        )
        self.match_indices = torch.tensor(match_indices)
        self.exclude_indices = torch.tensor(exclude_indices)
        self.keys_out = keys_out
        self.n_modes = n_modes

    def forward(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that pools input quantum amplitudes into output modes.

        Parameters
        ----------
        amplitudes : torch.Tensor
            Input tensor of shape `(batch_size, n_input_states)` containing
            the complex amplitudes (or real/imag parts) of quantum states.

        Returns
        -------
        torch.Tensor
            Normalized pooled amplitudes of shape `(batch_size, n_output_states)`.
        """
        batch_size = amplitudes.shape[0]
        device = amplitudes.device
        if device != self.match_indices.device:
            self.match_indices = self.match_indices.to(device)
        output = torch.zeros(
            batch_size,
            len(self.keys_out),
            dtype=amplitudes.dtype,
            device=amplitudes.device,
        )

        # Create a mask to exclude certain indices
        mask = torch.ones(
            amplitudes.shape[1], dtype=torch.bool, device=amplitudes.device
        )
        if self.exclude_indices.numel() != 0:
            mask[self.exclude_indices] = False

        filtered_amplitudes = amplitudes[:, mask]

        # Aggregate amplitudes based on mapping
        output.scatter_add_(
            1,
            self.match_indices.unsqueeze(0).repeat(batch_size, 1),
            filtered_amplitudes,
        )

        # Normalize to preserve total probability
        sum_probs = output.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
        return output / sum_probs

    def match_tuples(
        self, keys_in: list, keys_out: list, pooling_modes: list[list[int]]
    ):
        """
        Matches input and output Fock state tuples based on pooling configuration.

        For each input Fock state (`key_in`), the corresponding pooled output
        state (`key_out`) is computed by summing the photon counts over each
        pooling group. Input states that do not correspond to a valid output
        state are marked for exclusion.

        Parameters
        ----------
        keys_in : list
            List of Fock state tuples representing input configurations.
        keys_out : list
            List of Fock state tuples representing output configurations.
        pooling_modes : list of list of int
            Grouping of input modes into output modes.

        Returns
        -------
        tuple[list[int], list[int]]
            A pair `(indices, exclude_indices)` where:
            - `indices` are the matched indices from input to output keys.
            - `exclude_indices` are input indices with no valid match.
        """
        indices = []
        exclude_indices = []
        for i, key_in in enumerate(keys_in):
            key_out = tuple(
                sum(key_in[i] for i in indices) for indices in pooling_modes
            )
            index = keys_out.index(key_out) if key_out in keys_out else None
            if index is not None:
                indices.append(index)
            else:
                exclude_indices.append(i)

        return indices, exclude_indices
