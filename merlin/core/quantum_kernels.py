import torch
import perceval as pcvl
import numpy as np

from .utils import generate_all_fock_states
from ..sampling.autodiff import AutoDiffProcess
from ..pcvl_pytorch.locirc_to_tensor import CircuitConverter
from ..pcvl_pytorch.slos_torchscript import build_slos_distribution_computegraph as build_slos_graph

from typing import Union
from torch import Tensor

dtype_to_torch = {
    'float': torch.float64,
    'complex': torch.complex128,
    'float64': torch.float64,
    'float32': torch.float32,
    'complex128': torch.complex128,
    'complex64': torch.complex64,
    torch.float64: torch.float64,
    torch.float32: torch.float32,
    torch.complex128: torch.complex128,
    torch.complex64: torch.complex64,
    np.float64: torch.float64,
    np.float32: torch.float32,
    np.complex128: torch.complex128,
    np.complex64: torch.complex64,
}

class FeatureMap:
    """
    Quantum Feature Map

    FeatureMap embeds a datapoint within a quantum circuit and      
    computes the associated unitary for quantum kernel methods.

    :param circuit: Circuit with data-embedding parameters.
    :param input_parameters: Parameters which encode each datapoint.
    :param dtype: Data type for generated unitary.
    :param device: Device on which to calculate the unitary.
    """

    def __init__(
        self,
        circuit: pcvl.Circuit,
        input_size: int,
        input_parameters: Union[str, list[str]],
        *,
        trainable_parameters: list[str] = None,
        dtype: str = torch.float32,
        device=None
    ):
        self.circuit = circuit
        self.input_size = input_size
        self.trainable_parameters = trainable_parameters or []
        self.dtype = dtype_to_torch.get(dtype, torch.float32)
        self.device = device or torch.device('cpu')
        self.is_trainable = bool(trainable_parameters)

        if isinstance(input_parameters, list):            
            if len(input_parameters) > 1:
                raise ValueError('Only a single input parameter is allowed.')
            
            self.input_parameters = input_parameters[0]
        else:
            self.input_parameters = input_parameters

        self._circuit_graph = CircuitConverter(
            circuit,
            [self.input_parameters]+self.trainable_parameters,
            dtype=self.dtype,
            device=device
        )
        # Set training parameters as torch parameters
        self._training_dict = {}
        for param_name in self.trainable_parameters:
            param_length = len(self._circuit_graph.spec_mappings[param_name])

            p = torch.rand(param_length, requires_grad=True)
            self._training_dict[param_name] = torch.nn.Parameter(p)

    def compute_unitary(self, x: Union[Tensor, np.ndarray, float], *training_parameters: Tensor) -> Tensor:
        """
        Computes the unitary associated with the feature map and given 
        datapoint and training parameters.

        :param x: Input datapoint or dataset.
        :param training_parameters: If specified, the unitary for a
            specific set of training parameters is given. If not,
            internal parameters are used instead.
        """
        if not isinstance(x, torch.Tensor):
            x = [x] if isinstance(x, (float, int)) else x
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        else:
            x = x.to(dtype=self.dtype, device=self.device)

        if not self.is_trainable:
            return self._circuit_graph.to_tensor(x)

        if not training_parameters:
            training_parameters = self._training_dict.values()

        return self._circuit_graph.to_tensor(x, *training_parameters)

    def is_datapoint(self, x: Union[Tensor, np.ndarray, float, int]) -> bool:
        """Checks whether an input data is a singular datapoint or dataset."""
        if self.input_size == 1 and (isinstance(x, (float, int)) or x.ndim == 0):
            return True
        
        error_msg = f'Given value shape {tuple(x.shape)} does not match data shape {self.input_size}.'
        num_elements = x.numel() if isinstance(x, Tensor) else x.size
        
        if num_elements % self.input_size or x.ndim > 2:
            raise ValueError(error_msg)
            
        if self.input_size == 1:
            if num_elements == 1:
                return True
            elif x.ndim == 1:
                return False
            elif x.ndim == 2 and 1 in x.shape:
                return False
        else:
            if x.ndim == 1 and x.shape[0] == self.input_size:
                return True
            elif x.ndim == 2:
                return 1 in x.shape and self.input_size in x.shape
        
        raise ValueError(error_msg)


class FidelityKernel(torch.nn.Module):
    r"""
    Fidelity Quantum Kernel 

    For a given input Fock state, :math:`|s \rangle` and feature map,
    :math:`U`, the fidelity quantum kernel estimates the following inner 
    product using SLOS:
    .. math:: 
        |\langle s | U^{\dagger}(x_2) U(x_1) | s \rangle|^{2}

    Transition probabilities are computed in parallel for each pair of 
    datapoints in the input datasets.

    :param feature_map: Feature map object that encodes a given 
        datapoint within its circuit.
    :param input_state: Input state into circuit. 
    :param shots: Number of circuit shots. If `None`, the exact 
        transition probabilities are returned. Default: `None`.
    :param sampling_method: Probability distributions are post-
        processed with some pseudo-sampling method: 'multinomial', 
        'binomial' or 'gaussian'.
    :param no_bunching: Whether or not to post-select out results with 
        bunching.
    :param force_psd: Projects training kernel matrix to closest 
    positive semi-definite. Default: `True`.
    :param device: Device on which to perform SLOS
    :param dtype: Datatype with which to perform SLOS

    Examples
    --------
    For a given training and test datasets, one can construct the 
    training and test kernel matrices in the following structure:
    .. code-block:: python
        >>> circuit = Circuit(2) // PS(P("X0") // BS() // PS(P("X1") // BS()
        >>> feature_map = FeatureMap(circuit, ["X"])
        >>>
        >>> quantum_kernel = FidelityKernel(
        >>>     feature_map,
        >>>     input_state=[0, 4],
        >>>     no_bunching=False,
        >>> )
        >>> # Construct the training & test kernel matrices
        >>> K_train = quantum_kernel(X_train)
        >>> K_test = quantum_kernel(X_test, X_train)

    Use with scikit-learn for kernel-based machine learning:.
    .. code-block:: python
        >>> from sklearn import SVC
        >>> 
        >>> # For a support vector classification problem
        >>> svc = SVC(kernel='precomputed')
        >>> svc.fit(K_train, y_train)
        >>> y_pred = svc.predict(K_test)
    """

    def __init__(
        self,
        feature_map: Union[FeatureMap, pcvl.Circuit],
        input_state: list,
        *,
        shots: int = None,
        sampling_method: str = 'multinomial',
        no_bunching=True,
        force_psd=True,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.feature_map = feature_map
        self.input_state = input_state
        self.shots = shots or 0
        self.sampling_method = sampling_method
        self.no_bunching = no_bunching
        self.force_psd = force_psd
        self.device = device or feature_map.device
        self.dtype = dtype or feature_map.dtype
        self.input_size = self.feature_map.input_size
        
        if self.feature_map.circuit.m != len(input_state):
            raise ValueError('Input state length does not match circuit size.')
        
        self.is_trainable = feature_map.is_trainable
        if self.is_trainable:
            for param_name, param in feature_map._training_dict.items():
                self.register_parameter(param_name, param)

        if max(input_state) > 1 and no_bunching:
            raise ValueError(
                f"Bunching must be enabled for an input state with"
                f"{max(input_state)} in one mode.")
        elif all(x == 1 for x in input_state) and no_bunching:
            raise ValueError(
                "For `no_bunching = True`, the kernel value will always be 1"
                " for an input state with a photon in all modes.")

        m, n = len(input_state), sum(input_state)

        self._slos_graph = build_slos_graph(
            m=m,
            n_photons=n,
            no_bunching=no_bunching,
            keep_keys=False,
            device=device,
            dtype=self.dtype
        )
        # Find index of input state in output distribution
        all_fock_states = list(generate_all_fock_states(m, n, no_bunching=no_bunching))
        self._input_state_index = all_fock_states.index(tuple(input_state))

        # For sampling
        self._autodiff_process = AutoDiffProcess()


    def forward(self, x1: Union[float, np.ndarray, Tensor], x2=None):
        """
        Calculate the quantum kernel for input data `x1` and `x2.` If `x1` 
        and `x2` are datapoints, a scalar value is returned. For input 
        datasets the kernel matrix is computed.

        :param x1: Input datapoint or dataset.
        :param x2: Input datapoint or dataset. If `None`, the kernel matrix 
            is assumed to be symmetric with input datasets, x1, x1 and only 
            the upper triangular is calculated. Default: `None`.

        If you would like the diagonal and lower triangular to be explicitly 
        calculated for identical inputs, please specify an argument `x2`.
        """
        if x2 is not None and type(x1) is not type(x2):
            raise TypeError(
                'x2 should be of the same type as x1, if x2 is not None.')
        
        # Return scalar value for input datapoints
        if self.feature_map.is_datapoint(x1):
            if x2 is None:
                raise ValueError(
                    'For input datapoints, please specify an x2 argument.')
            return self._return_kernel_scalar(x1, x2)
        
        x1 = x1.reshape(-1, self.input_size)
        x2 = x2.reshape(-1, self.input_size) if x2 is not None else None
        
        # Check if we are constructing training matrix
        equal_inputs = self._check_equal_inputs(x1, x2)

        U_forward = torch.stack(
            [self.feature_map.compute_unitary(x) for x in x1])
        
        len_x1 = len(x1)
        if x2 is not None:
            U_adjoint = torch.stack([
                self.feature_map.compute_unitary(x).transpose(0, 1).conj()
                for x in x2])
            
            # Calculate circuit unitary for every pair of datapoints
            all_circuits = U_forward.unsqueeze(1) @ U_adjoint.unsqueeze(0)
            all_circuits = all_circuits.view(-1, *all_circuits.shape[2:])
        else:
            U_adjoint = U_forward.conj().transpose(1, 2)

            # Take circuit unitaries for upper diagonal of kernel matrix only
            upper_idx = torch.triu_indices(
                len_x1, len_x1,
                offset=1,
                dtype=self.feature_map.dtype,
                device=self.feature_map.device,
            )
            all_circuits = U_forward[upper_idx[0]] @ U_adjoint[upper_idx[1]]

        # Distribution for every evaluated circuit
        all_probs = self._slos_graph.compute(
            all_circuits, self.input_state)[1]

        if self.shots > 0:
            all_probs = self._autodiff_process.sampling_noise.pcvl_sampler(
                all_probs, self.shots, self.sampling_method
            )

        transition_probs = all_probs[:, self._input_state_index]

        if x2 is None:
            # Copy transition probs to upper & lower diagonal
            kernel_matrix = torch.zeros(
                len_x1, len_x1, dtype=self.dtype, device=self.device)
            
            upper_idx = upper_idx.to(self.device)
            kernel_matrix[upper_idx[0], upper_idx[1]] = transition_probs
            kernel_matrix[upper_idx[1], upper_idx[0]] = transition_probs
            kernel_matrix.fill_diagonal_(1)

            if self.force_psd:
                kernel_matrix = self._project_psd(kernel_matrix)

        else:
            kernel_matrix = transition_probs.reshape(len_x1, len(x2))

            if self.force_psd and equal_inputs:
                # Symmetrize the matrix
                kernel_matrix = 0.5 * (kernel_matrix + kernel_matrix.transpose(0, 1))
                kernel_matrix = self._project_psd(kernel_matrix)

        if isinstance(x1, np.ndarray):
            kernel_matrix = kernel_matrix.detach().numpy()

        return kernel_matrix

    def _return_kernel_scalar(self, x1, x2):
        """Returns scalar kernel value for input datapoints"""
        if isinstance(x1, float):
            x1, x2 = np.array(x1), np.array(x2)
            
        x1, x2 = x1.reshape(self.input_size), x2.reshape(self.input_size)
        
        U = self.feature_map.compute_unitary(x1)
        U_adjoint = self.feature_map.compute_unitary(x2)
        U_adjoint = U_adjoint.conj().transpose(0, 1)

        probs = self._slos_graph.compute(U @ U_adjoint, self.input_state)[1]

        if self.shots > 0:
            probs = self._autodiff_process.sampling_noise.pcvl_sampler(
                probs, self.shots, self.sampling_method
            )
        return probs[self._input_state_index].item()

    @staticmethod
    def _project_psd(matrix: Tensor) -> Tensor:
        """Projects a symmetric matrix to closest positive semi-definite"""
        # Perform spectral decomposition and set negative eigenvalues to 0
        eigenvals, eigenvecs = torch.linalg.eigh(matrix)
        eigenvals = torch.diag(torch.where(eigenvals > 0, eigenvals, 0))

        matrix_psd = eigenvecs @ eigenvals @ eigenvecs.transpose(0, 1)
        matrix_psd.fill_diagonal_(1)
        
        return matrix_psd
    
    @staticmethod
    def _check_equal_inputs(x1, x2) -> bool:
        """Checks whether x1 and x2 are equal."""
        if x2 is None:
            return True
        elif x1.shape != x2.shape:
            return False
        elif isinstance(x1, Tensor):
            return torch.allclose(x1, x2)
        elif isinstance(x1, np.ndarray):
            return np.allclose(x1, x2)
        return False

