import math
import time
import warnings
from itertools import chain, product

import exqalibur as xqlbr
import perceval as pcvl  # Import Perceval only in worker process
import torch
import torch.nn as nn
from perceval.utils import allstate_iterator

from merlin import OutputMappingStrategy, QuantumLayer


class BosonSampler:
    """
    Basic (trainable) boson sampler.

    Attributes:
    -----------
    dims : tuple
        Dimensions of the vector we will pass through the model
    m : int
        Number of modes in the circuit.
    n : int
        Number of photons in the circuit.
    circuit : pcvl.Circuit
        The variational circuit.
    parameters : list
        List of parameters for the variational circuit.
    n_params : int
        Number of parameters in the variational circuit.

    Methods:
    --------
    init_phases():
        Initializes the phases of the parameters to pi.
    set_parameters(params):
        Sets the parameters of the variational circuit.
    compute(input_states):
        Computes the output distribution of the circuit.
    load_from_checkpoint(checkpoint):
        Loads the parameters from the checkpoint.
    save_checkpoint(checkpoint_path):
        Saves the parameters from the checkpoint.

    """

    def __init__(
        self,
        dims: list | tuple,
        eps: float = 1e-5,
        circuit: None | pcvl.Circuit = None,
        trainable_parameters: list = None,
    ) -> None:
        """
        Constructs all the necessary attributes for the VariationalCircuit object.

        Parameters:
        -----------
        dims : tuple
            Dimensions of the vector we will pass through the model
        eps : float
            Epsilon value added to avoid zero division when backpropagating.
        circuit (optional): pcvl.Circuit
            Circuit to implement
        trainable_parameters : list
            List of trainable parameters for the variational circuit. By default, all parameters are trainable
        """
        self.dims = dims[1:]
        self.batch_size = dims[0]
        self.m = sum(self.dims) + 2
        self.n = len(self.dims) + 1
        self.eps = eps

        if circuit is None:
            self.circuit = pcvl.components.GenericInterferometer(
                self.m,
                pcvl.components.catalog["mzi phase last"].generate,
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            if trainable_parameters is not None:
                raise AttributeError(
                    "Trainable parameters were given without a corresponding circuit."
                )
        else:
            self.circuit = circuit

        index_photons = []
        photon_range = (0, -1)
        for dim in self.dims:
            photon_range = (photon_range[1] + 1, photon_range[1] + dim)
            index_photons.append(photon_range)
        index_photons.append((photon_range[1] + 1, photon_range[1] + 2))

        self.parameters = self.circuit.get_parameters()
        self.n_params = len(self.parameters)
        input_state = [self.n] + [0] * (self.m - 1)

        if trainable_parameters is None:
            if circuit is not None:
                warnings.warn(
                    "No trainable parameters were given. Setting them to ['phi'] ",
                    stacklevel=2,
                )
            trainable_parameters = ["phi"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start = time.time()
        self.model = QuantumLayer(
            input_size=0,
            output_size=None,
            circuit=self.circuit,
            n_photons=self.n,
            trainable_parameters=trainable_parameters,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            device=self.device,
            dtype=torch.float32,
            shots=1000,
            sampling_method="multinomial",
            no_bunching=True,
            index_photons=index_photons,
        )
        self.get_post_selected_space()
        print("time to load model", time.time() - start)

        self.comb = math.comb(self.m + self.n - 1, self.n)

        # precompute the normalization factors
        input_state_xqlbr = xqlbr.FockState(input_state)
        self.norm = torch.ones((self.comb,), dtype=torch.complex64)
        for i, state in enumerate(allstate_iterator(input_state_xqlbr)):
            self.norm[i] = state.prodnfact()
        self.norm = self.norm.sqrt()

    def get_post_selected_space(self) -> None:
        """Get the post-selected space using the updated QuantumLayer API."""
        ranges = [range(dim) for dim in self.dims] + [range(2)]
        self.postselected_states = []
        self.postselected_states_idx_pos = []
        self.postselected_states_idx = []
        index = 0

        # FIXED: Access simulation graph through computation_process
        # Get the mapped keys from the computation process
        if hasattr(self.model, "computation_process") and hasattr(
            self.model.computation_process, "simulation_graph"
        ):
            simulation_graph = self.model.computation_process.simulation_graph

            # Get the keys - they might be in final_keys or mapped_keys depending on mapping strategy
            if (
                hasattr(simulation_graph, "mapped_keys")
                and simulation_graph.mapped_keys is not None
            ):
                all_outputs_keys = simulation_graph.mapped_keys
            elif (
                hasattr(simulation_graph, "final_keys")
                and simulation_graph.final_keys is not None
            ):
                all_outputs_keys = simulation_graph.final_keys
            else:
                # Fallback: compute a dummy output to get the keys
                dummy_params = self.model._create_dummy_parameters()
                keys, _ = simulation_graph.compute(
                    self.model.computation_process.converter.to_tensor(*dummy_params),
                    self.model.input_state,
                )
                all_outputs_keys = keys
        else:
            # Alternative: Get keys by running a dummy computation
            dummy_input = (
                torch.zeros(0)
                if self.model.input_size == 0
                else torch.zeros(1, self.model.input_size)
            )

            # Access the computation process to get the distribution and keys
            params = self.model.prepare_parameters(
                [dummy_input] if self.model.input_size > 0 else []
            )
            unitary = self.model.computation_process.converter.to_tensor(*params)
            keys, _ = self.model.computation_process.simulation_graph.compute(
                unitary, self.model.input_state
            )
            all_outputs_keys = keys

        # Create mapping from state tuples to indices
        all_outputs_idx_map = {state: i for i, state in enumerate(all_outputs_keys)}

        self.value_indices = []
        self.value_labels = []

        for state in product(
            *ranges
        ):  # (range(self.dims[0], ..., range(self.dims[-1], range(2))
            fock_state = []
            for dim_index in range(len(self.dims)):
                fock_state += [
                    int(ii == state[dim_index]) for ii in range(self.dims[dim_index])
                ]  # += [0, 0, ... , 1, ..., 0]
            fock_state += [int(x == state[-1]) for x in range(2)]

            self.postselected_states += [fock_state]
            self.postselected_states_idx += [fock_state]

            if state[-1] == 0:
                fock_state_tuple = tuple(fock_state)
                if fock_state_tuple in all_outputs_idx_map:
                    self.postselected_states_idx_pos += [
                        all_outputs_idx_map[fock_state_tuple]
                    ]
                    self.value_indices += [index]
                    self.value_labels += [state[:-1]]
                else:
                    print(
                        f"Warning: State {fock_state_tuple} not found in output space"
                    )
            index += 1

    def set_parameters(self, params) -> None:
        """
        Sets the parameters of the variational circuit.

        Parameters:
        -----------
        params : dict
            Dictionary of parameter names and their values.
        """
        for p in self.parameters:
            p.set_value(params[p.name])

    def _boson_forward(
        self, apply_sampling: bool = False, shots: int = None, unitaries=None
    ) -> None:
        len_ps = len(self.postselected_states)
        self.bs_output_pos = torch.zeros(len_ps // 2, self.model.output_size).to(
            self.device
        )
        self.bs_output_neg = torch.zeros(len_ps // 2, self.model.output_size).to(
            self.device
        )

        if unitaries is None:
            input_parameters = ()
            params = self.model.prepare_parameters(input_parameters)
            # FIXED: Access unitary_graph through computation_process
            unitaries = self.model.computation_process.converter.to_tensor(*params)

        for index in range(len_ps):
            fock_state = self.postselected_states[index]

            # FIXED: Use the new forward method signature
            if index % 2 == 0:
                # For the new QuantumLayer API, we need to provide input parameters even if empty
                if self.model.input_size == 0:
                    self.bs_output_pos[index // 2, :] = self._forward_with_custom_state(
                        fock_state, unitaries, apply_sampling, shots
                    )
                else:
                    # dummy_input = torch.zeros(self.model.input_size)
                    self.bs_output_pos[index // 2, :] = self._forward_with_custom_state(
                        fock_state, unitaries, apply_sampling, shots
                    )
            else:
                if self.model.input_size == 0:
                    self.bs_output_neg[index // 2, :] = self._forward_with_custom_state(
                        fock_state, unitaries, apply_sampling, shots
                    )
                else:
                    # dummy_input = torch.zeros(self.model.input_size)
                    self.bs_output_neg[index // 2, :] = self._forward_with_custom_state(
                        fock_state, unitaries, apply_sampling, shots
                    )

    def _forward_with_custom_state(
        self, input_state, unitaries, apply_sampling=False, shots=None
    ):
        """
        Helper method to compute forward pass with a custom input state.
        This works around the new QuantumLayer API structure.
        """
        # Store original input state
        original_input_state = self.model.input_state

        try:
            # Temporarily set the new input state
            self.model.input_state = input_state
            self.model.computation_process.input_state = input_state

            # Compute distribution directly using the computation process
            keys, distribution = (
                self.model.computation_process.simulation_graph.compute(
                    unitaries, input_state
                )
            )

            # Apply sampling if requested
            if apply_sampling and shots and shots > 0:
                distribution = self.model.autodiff_process.sampling_noise.pcvl_sampler(
                    distribution, shots
                )

            # Apply output mapping
            return self.model.output_mapping(distribution)

        finally:
            # Restore original input state
            self.model.input_state = original_input_state
            self.model.computation_process.input_state = original_input_state

    def _compute(
        self,
        data: torch.Tensor,
        apply_sampling: bool = False,
        shots: int = None,
        unitaries=None,
    ) -> torch.Tensor:
        batch_size = data.shape[0]
        data_flat = data.flatten(start_dim=1)
        self.eps = 1e-5
        # start = time.time()
        batch_dim = tuple([-1] + [1 for _ in range(data_flat.dim() - 1)])
        min_val = data_flat.min(dim=1).values.reshape(batch_dim) - self.eps
        max_val = data_flat.max(dim=1).values.reshape(batch_dim) + self.eps
        normalized_data = (data_flat - min_val) / (max_val - min_val)

        positive_amplitudes = torch.sqrt(normalized_data) / math.sqrt(
            data_flat.shape[1]
        )
        negative_amplitudes = torch.sqrt(1 - normalized_data) / math.sqrt(
            data_flat.shape[1]
        )
        self._boson_forward(apply_sampling, shots, unitaries)

        negative_output = negative_amplitudes @ self.bs_output_neg
        positive_output = positive_amplitudes @ self.bs_output_pos
        upa = positive_output + negative_output

        upa = torch.abs(upa) ** 2
        ps_idx_pos = torch.tensor(self.postselected_states_idx_pos)
        ps_idx_neg = ps_idx_pos + 1

        output = upa[:, ps_idx_pos] / (upa[:, ps_idx_pos] + upa[:, ps_idx_neg])
        output_reshaped = output.reshape((batch_size,) + self.dims)

        return output_reshaped

    def compute(
        self,
        data: torch.Tensor,
        apply_sampling: bool = False,
        shots: int = None,
        unitaries=None,
    ) -> torch.Tensor:
        """
        This function computes the output of the variational circuit for 2*data.shape different states.
        The output probabilities given by the variational circuit are used to obtain the desired tensor.

        Parameters:
        -----------
        data: torch.Tensor
            Vector to pass through the variational circuit.
        apply_sampling: bool
            Parameter used in the quantum layer to decided whether to apply sampling or not. If not, the model outputs the probability corresponding to every output state
        shots: int
            Number of shots to generate when the sampling is enabled
        Returns:
        --------
        torch.Tensor shape: data.shape
        """
        return self._compute(data, apply_sampling, shots)

    def load_from_checkpoint(self, checkpoint_path: str):
        """
        Load the weights of the boson sampler
        Parameters:
        ---------
        checkpoint_path: str
            path to the checkpoint file
        """
        ckpt = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(ckpt)

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save the weights of the boson sampler
        Parameters:
        -------------
        checkpoint_path: str
            Path to save the weights of the boson sampler.
        """
        torch.save(self.model.state_dict(), checkpoint_path)


# Test code
if __name__ == "__main__":
    dims = (16, 4, 4, 2)
    a = torch.rand(dims)

    L = nn.Linear(32, 32)
    bs = BosonSampler(dims)

    combined_params = chain(L.parameters(), bs.model.parameters())

    # Example: Use combined parameters in an optimizer
    optimizer = torch.optim.AdamW(combined_params, lr=0.01)

    for i in range(100):
        result = bs.compute(L(a.view(16, 32)).view(dims))
        loss = result.square().sum()  # dummy example to check if the backward propagation and the optimization step work
        loss.backward(retain_graph=True)
        print(f"Iteration {i + 1}, loss: {loss.item():.6f}")
        optimizer.step()
        optimizer.zero_grad()
