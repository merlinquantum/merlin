from itertools import product

import perceval as pcvl
import torch
from perceval.components import BS, PS

from merlin import (
    OutputMappingStrategy,
    QuantumLayer,
)


def create_circuit(M, input_size):
    """Create a quantum photonic circuit with beam splitters and phase shifters.

    Args:
        M (int): Number of modes in the circuit.

    Returns:
        pcvl.Circuit: A quantum photonic circuit with alternating beam splitter layers and phase shifters.
    """
    circuit = pcvl.Circuit(M)

    def layer_bs(circuit, k, M, j):
        for i in range(k, M - 1, 2):
            theta = pcvl.P(f"theta_{i}_{j}")
            circuit.add(i, BS(theta=theta))

    layer_bs(circuit, 0, M, 0)
    layer_bs(circuit, 1, M, 1)
    layer_bs(circuit, 0, M, 2)
    layer_bs(circuit, 1, M, 3)
    layer_bs(circuit, 0, M, 4)
    for i in range(input_size):
        phi = pcvl.P(f"phi_{i}")
        circuit.add(i, PS(phi))
    layer_bs(circuit, 0, M, 5)
    layer_bs(circuit, 1, M, 6)
    layer_bs(circuit, 0, M, 7)
    layer_bs(circuit, 1, M, 8)
    layer_bs(circuit, 0, M, 9)
    return circuit


def define_layer_no_input(n_modes, n_photons):
    """Define a quantum layer for feed-forward processing.

    Args:
        n_modes (int): Number of optical modes.
        n_photons (int): Number of photons in the layer.

    Returns:
        QuantumLayer: A configured quantum layer with trainable parameters.
    """
    circuit = create_circuit(n_modes, 0)
    input_state = [1] * n_photons + [0] * (n_modes - n_photons)
    layer = QuantumLayer(
        input_size=0,
        output_size=None,
        circuit=circuit,
        n_photons=n_photons,
        input_state=input_state,  # Random Initial quantum state used only for initialization
        output_mapping_strategy=OutputMappingStrategy.NONE,
        trainable_parameters=["theta"],
        no_bunching=True,
    )
    return layer


def define_layer_with_input(M, N, input_size):
    """Define the first layer of the feed-forward network.

    Args:
        M (int): Number of modes in the circuit.
        N (int): Number of photons.

    Returns:
        QuantumLayer: The first quantum layer with input parameters.
    """
    circuit = create_circuit(M, input_size)
    input_state = [1] * N + [0] * (M - N)
    layer = QuantumLayer(
        input_size=input_size,
        output_size=None,
        circuit=circuit,
        n_photons=N,
        input_state=input_state,  # Random Initial quantum state used only for initialization
        output_mapping_strategy=OutputMappingStrategy.NONE,
        input_parameters=["phi"],  # Optional: Specify device
        trainable_parameters=["theta"],
        no_bunching=True,
    )
    return layer


class FeedForwardBlock(torch.nn.Module):
    """Feed-forward quantum neural network for photonic computation.

    This class implements a feed-forward architecture where quantum layers are
    conditionally activated based on photon detection measurements.

    Args:
        m (int): Total number of modes.
        n_photons (int): Number of photons in the system.
        conditional_mode (int): Mode index used for conditional measurement.
    """

    def __init__(
        self,
        input_size: int,
        n: int,
        m: int,
        depth: int = None,
        state_injection=False,
        conditional_mode: int = 0,
        layers: list[QuantumLayer] = None,
    ):
        super().__init__()
        self.conditional_mode = conditional_mode
        self.m = m
        self.input_size = input_size
        self.n_photons = n
        self.state_injection = state_injection
        self.layers = {}
        if depth is None:
            depth = self.m - 1
        self.depth = depth
        if layers is None:
            self.define_layers()
        else:
            tuples = self.generate_possible_tuples()
            self.tuples = tuples
            assert len(tuples) == len(layers), (
                f"Layers should be a list of Quantum Layers of length {len(tuples)}"
            )
            self.layers = {tuples[k]: layers[k] for k in range(len(layers))}
            start = 0
            self.input_segments = {}
            for _k, tuple in enumerate(tuples):
                input_size = self.layers[tuple].input_size
                self.input_segments[tuple] = (start, start + input_size)
                start += input_size
            assert start == self.input_size, f"Input size can't be higher than {start}"

    def generate_possible_tuples(self):
        """Generate all possible measurement outcome tuples.

        Returns:
            List: Set of tuples representing possible measurement patterns.
        """
        n = self.n_photons
        m = self.m
        possible_tuples = []
        for depth in range(self.depth + 1):
            for t in product([0, 1], repeat=depth):
                if self.state_injection:
                    possible_tuples.append(t)
                elif t.count(1) <= n - 1 and t.count(0) <= (m - n - 1):
                    possible_tuples.append(t)

        return possible_tuples

    def define_layers(self):
        """Define all quantum layers for different measurement outcomes.

        Creates a dictionary mapping measurement tuples to corresponding quantum layers.
        Also creates mapping for input size distribution.
        """
        input_size = self.input_size
        tuples = self.generate_possible_tuples()
        self.tuples = tuples
        self.input_segments = {}  # Track input size for each layer
        start = 0
        for tup in tuples:
            n = sum(tup)
            m = len(tup)
            if self.state_injection:
                input = min(self.m, input_size)
            else:
                input = min(self.m - m, input_size)
            if input > 0:
                if self.state_injection:
                    self.layers[tup] = define_layer_with_input(
                        self.m, self.n_photons, input
                    )
                else:
                    self.layers[tup] = define_layer_with_input(
                        self.m - m, self.n_photons - n, input
                    )
                self.input_segments[tup] = (start, start + input)
            else:
                if self.state_injection:
                    self.layers[tup] = define_layer_no_input(self.m, self.n_photons)
                else:
                    self.layers[tup] = define_layer_no_input(
                        self.m - m, self.n_photons - n
                    )
                self.input_segments[tup] = (0, 0)
            input_size -= input
            start += input
        assert input_size == 0, f"The input size can't be higher than {start}"

    def parameters(self):
        """Return an iterator over all trainable parameters.

        Yields:
            torch.Tensor: Trainable parameters from all quantum layers.
        """
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
        conditional_mode=0,
        x=None,
    ):
        """Recursively process the feed-forward computation.

        Args:
            current_tuple (tuple): Current measurement pattern.
            remaining_amplitudes (torch.Tensor): Quantum state amplitudes.
            keys (list): State basis keys.
            accumulated_prob (torch.Tensor): Accumulated probability.
            intermediary (dict): Intermediate probability values.
            outputs (dict): Final output probabilities mapping fock states to their probabilities.
            depth (int): Current recursion depth.
            conditional_mode (int): Mode index for conditional measurement.
            x (torch.Tensor): Remaining input tensor segments.
        """

        if depth >= self.depth:
            # At required depth,
            # output the fock state probabilities
            # Convert remaining amplitudes to fock state probabilities
            fock_probs = remaining_amplitudes.abs().pow(2)

            for i, key in enumerate(keys):
                if key not in outputs:
                    outputs[key] = torch.zeros_like(accumulated_prob)
                outputs[key] += accumulated_prob * fock_probs[:, i]
            return

        layer_with_photon = self.layers.get(current_tuple + (1,), None)
        layer_without_photon = self.layers.get(current_tuple + (0,), None)
        layer_idx_not, layer_idx = self._indices_by_value(keys, conditional_mode)
        prob_not = remaining_amplitudes[:, layer_idx_not].abs().pow(2).sum(dim=1)
        prob_with = remaining_amplitudes[:, layer_idx].abs().pow(2).sum(dim=1)

        current_key_with = current_tuple + (1,)
        current_key_without = current_tuple + (0,)

        intermediary[current_key_with] = prob_with
        intermediary[current_key_without] = prob_not

        if layer_with_photon is not None:
            m = layer_with_photon.computation_process.m
            conditional_mode = min(self.conditional_mode, m - 1)
            if self.state_injection:
                match_idx_with = layer_idx
                keys_with = keys
            else:
                keys_with = (
                    layer_with_photon.computation_process.simulation_graph.mapped_keys
                )
                match_idx_with = self._match_indices(
                    keys, keys_with, conditional_mode, k_value=1
                )
            layer_with_photon.computation_process.input_state = remaining_amplitudes[
                :, match_idx_with
            ]
            start, end = self.input_segments[current_key_with]

            if start != end:
                probs_with, amplitudes_with = layer_with_photon(
                    x[:, start:end], return_amplitudes=True
                )
            else:
                probs_with, amplitudes_with = layer_with_photon(return_amplitudes=True)

            new_prob_with = accumulated_prob * intermediary[current_key_with]
            self.iterate_feedforward(
                current_key_with,
                amplitudes_with,
                keys_with,
                new_prob_with,
                intermediary,
                outputs,
                depth + 1,
                conditional_mode,
                x,  # Pass the full input tensor
            )
        else:
            final_tuple_with = current_key_with + (0,) * (
                self.depth - len(current_key_with)
            )
            new_prob_with = accumulated_prob * intermediary[current_key_with]
            outputs[final_tuple_with] = new_prob_with

        if layer_without_photon is not None:
            m = layer_without_photon.computation_process.m
            conditional_mode = min(self.conditional_mode, m - 1)
            if self.state_injection:
                match_idx_without = layer_idx_not
                keys_without = keys
            else:
                keys_without = layer_without_photon.computation_process.simulation_graph.mapped_keys
                match_idx_without = self._match_indices(
                    keys, keys_without, conditional_mode, k_value=0
                )
            layer_without_photon.computation_process.input_state = remaining_amplitudes[
                :, match_idx_without
            ]

            # Get input segment for this layer
            start, end = self.input_segments[current_key_without]

            if start != end:
                probs_without, amplitudes_without = layer_without_photon(
                    x[:, start:end], return_amplitudes=True
                )
            else:
                probs_without, amplitudes_without = layer_without_photon(
                    return_amplitudes=True
                )
            new_prob_without = accumulated_prob * intermediary[current_key_without]

            self.iterate_feedforward(
                current_key_without,
                amplitudes_without,
                keys_without,
                new_prob_without,
                intermediary,
                outputs,
                depth + 1,
                conditional_mode,
                x,  # Pass the full input tensor
            )
        else:
            final_tuple_without = current_key_without + (1,) * (
                self.depth - len(current_key_without)
            )
            new_prob_without = accumulated_prob * intermediary[current_key_without]
            outputs[final_tuple_without] = new_prob_without

    def forward(self, x):
        """Forward pass of the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output probabilities for all measurement patterns.
        """
        if x.shape[-1] != self.input_size:
            raise ValueError(f"The input should be of size {self.input_size}")
        intermediary = {}
        outputs = {}
        input_size = min(self.input_size, self.m)
        input = x[:, :input_size]
        layer = self.layers[()]
        probs, amplitudes = layer(input, return_amplitudes=True)
        keys = layer.computation_process.simulation_graph.mapped_keys
        self.iterate_feedforward(
            (),
            amplitudes,
            keys,
            1.0,
            intermediary,
            outputs,
            0,
            self.conditional_mode,
            x=x,
        )
        return torch.stack(list(outputs.values()), dim=1)

    def _indices_by_value(self, keys, k):
        """Find indices where a specific position has value 0 or 1.

        Args:
            keys (list): List of tuples representing quantum states.
            k (int): Position index to check.

        Returns:
            tuple: Indices where value is 0, indices where value is 1.
        """
        # convertir en tenseur PyTorch
        t = torch.tensor(keys)
        # indices où la valeur vaut 0
        idx_0 = torch.nonzero(t[:, k] == 0, as_tuple=True)[0]

        # indices où la valeur vaut 1
        idx_1 = torch.nonzero(t[:, k] == 1, as_tuple=True)[0]

        return idx_0, idx_1

    def _match_indices(self, data, data_out, k, k_value):
        """Match indices between two state representations.

        Args:
            data (list): List of tuples with length n.
            data_out (list): List of tuples with length n-1.
            k (int): Index of the column to remove.
            k_value (int): Value to match at position k (0 or 1).

        Returns:
            torch.Tensor: Indices of matching states.
        """
        # Convert to dict to optimize search
        out_map = {tuple(row): i for i, row in enumerate(data_out)}

        idx = []

        for _i, tup in enumerate(data):
            removed = tup[:k] + tup[k + 1 :]
            if removed in out_map:
                j = out_map[removed]
                if tup[k] == k_value:
                    idx.append(j)

        return torch.tensor(idx)

    def get_output_size(self):
        x = torch.rand(1, self.input_size)
        return self.forward(x).shape[-1]

    def size_ff_layer(self, k: int):
        tuples_k = [1 for tup in self.tuples if len(tup) == k]
        return len(tuples_k)

    def define_ff_layer(self, k: int, layers: list[QuantumLayer]):
        len_layers = self.size_ff_layer(k)
        assert len(layers) == len_layers, f"layers should be of length {len_layers}"
        for i, t in enumerate(product([0, 1], repeat=k)):
            if t in self.layers:
                self.layers[t] = layers[i]
        self._recompute_segments()

    def input_size_ff_layer(self, k: int):
        return [self.layers[tup].input_size for tup in self.tuples if len(tup) == k]

    def _recompute_segments(self):
        """Recompute input segments based on current layer configuration.

        This method recalculates the input_segments mapping and updates input_size
        based on the current layers, similar to the computation in define_layers.
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

        # Update input_size and print new value
        self.input_size = total_input_size
        print(f"New input size: {self.input_size}")


if __name__ == "__main__":
    from itertools import chain

    import perceval as pcvl
    from perceval.components import BS, PS

    L = torch.nn.Linear(20, 20)
    feed_forward = FeedForwardBlock(
        20, 2, 6, depth=3, conditional_mode=5, state_injection=True
    )
    layers = list(feed_forward.layers.values())
    feed_forward = FeedForwardBlock(
        20, 2, 6, depth=3, state_injection=True, conditional_mode=5, layers=layers
    )
    params = chain(L.parameters(), feed_forward.parameters())
    optimizer = torch.optim.Adam(params)
    print(feed_forward.get_output_size())
    print(feed_forward.input_size_ff_layer(1))
    print(feed_forward.size_ff_layer(1))
    feed_forward.define_ff_layer(1, layers[1:3])
    x = torch.rand(1, 20)
    for _ in range(10):
        res = feed_forward(L(x))
        result = feed_forward(L(x)).pow(2).sum()
        print(result)
        result.backward()
        optimizer.step()
        optimizer.zero_grad()
