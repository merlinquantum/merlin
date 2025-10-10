import itertools
from collections.abc import Callable
from typing import cast

import numpy as np
import perceval as pcvl
import torch
from torch import Tensor

from ..builder.circuit_builder import ANGLE_ENCODING_MODE_ERROR, CircuitBuilder
from ..pcvl_pytorch.locirc_to_tensor import CircuitConverter
from ..pcvl_pytorch.slos_torchscript import (
    build_slos_distribution_computegraph as build_slos_graph,
)
from ..sampling.autodiff import AutoDiffProcess

dtype_to_torch: dict[object, torch.dtype] = {
    "float": torch.float64,
    "complex": torch.complex128,
    "float64": torch.float64,
    "float32": torch.float32,
    "complex128": torch.complex128,
    "complex64": torch.complex64,
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

    Args:
        circuit: Circuit or builder containing the data-encoding gates.
        input_parameters: Parameter prefix(es) that host the classical data.
        dtype: Torch dtype used when constructing the unitary.
        device: Torch device on which unitaries are evaluated.
    """

    def __init__(
        self,
        circuit: pcvl.Circuit | CircuitBuilder,
        input_size: int,
        input_parameters: str | list[str] | None,
        *,
        trainable_parameters: list[str] | None = None,
        dtype: str | torch.dtype = torch.float32,
        device: torch.device | None = None,
        encoder: Callable[[Tensor], Tensor] | None = None,  # was: callable | None
    ):
        builder_trainable: list[str] = []
        builder_input: list[str] = []

        self._angle_encoding_specs: dict[str, dict[str, object]] = {}

        if isinstance(circuit, CircuitBuilder):
            builder_trainable = circuit.trainable_parameter_prefixes
            builder_input = circuit.input_parameter_prefixes
            self._angle_encoding_specs = circuit.angle_encoding_specs
            circuit = circuit.to_pcvl_circuit(pcvl)

        self.circuit = circuit
        self.input_size = input_size
        if trainable_parameters is None:
            trainable_parameters = builder_trainable
        self.trainable_parameters = trainable_parameters or []
        self.dtype = dtype_to_torch.get(dtype, torch.float32)
        self.device = device or torch.device("cpu")
        self.is_trainable = bool(trainable_parameters)
        self._encoder = encoder  # NEW

        if input_parameters is None:
            if builder_input:
                input_parameters = builder_input[0]
            else:
                raise ValueError(
                    "input_parameters must be provided when no input layer is defined in the builder."
                )

        if isinstance(input_parameters, list):
            if len(input_parameters) > 1:
                raise ValueError("Only a single input parameter is allowed.")

            self.input_parameters = input_parameters[0]
        else:
            self.input_parameters = input_parameters

        self._circuit_graph = CircuitConverter(
            circuit,
            [self.input_parameters] + self.trainable_parameters,
            dtype=self.dtype,
            device=device,
        )
        # Set training parameters as torch parameters
        self._training_dict: dict[str, torch.nn.Parameter] = {}
        for param_name in self.trainable_parameters:
            param_length = len(self._circuit_graph.spec_mappings[param_name])

            p = torch.rand(param_length, requires_grad=True)
            self._training_dict[param_name] = torch.nn.Parameter(p)

    def _px_len(self) -> int:
        """Return how many angle-encoding slots the underlying circuit expects."""
        return len(self._circuit_graph.spec_mappings.get(self.input_parameters, []))

    def _subset_sum_expand(self, x: Tensor, k: int) -> Tensor:
        """
        Deterministic series-style expansion: non-empty subset sums of x in
        increasing subset-size order, truncated/padded to length k.

        Args:
            x: Input feature tensor expected to be one-dimensional.
            k: Desired number of encoded features to return.

        Returns:
            Tensor: Encoded tensor of length ``k`` on the configured device/dtype.
        """
        x = x.to(dtype=self.dtype, device=self.device).reshape(-1)
        d = x.shape[0]
        vals: list[Tensor] = []
        # generate sums for subset sizes 1..d
        for r in range(1, d + 1):
            for idxs in itertools.combinations(range(d), r):
                vals.append(x[list(idxs)].sum())
                if len(vals) == k:
                    return torch.stack(vals, dim=0)
        # if fewer than k (shouldn't happen for k <= 2^d-1), pad with zeros
        if len(vals) == 0:
            return torch.zeros(k, dtype=self.dtype, device=self.device)
        pad = k - len(vals)
        return torch.cat(
            [
                torch.stack(vals, dim=0),
                torch.zeros(pad, dtype=self.dtype, device=self.device),
            ],
            dim=0,
        )

    def _encode_x(self, x: Tensor) -> Tensor:
        """Map raw features to the circuit's required parameter shape.

        Preference order:
        1. Builder-provided combination metadata (from :class:`CircuitBuilder`).
        2. A user-supplied encoder callable.
        3. The deterministic subset-sum expansion used by legacy feature maps.

        Args:
            x: Input feature tensor to be embedded.

        Returns:
            Tensor: Encoded tensor matching the circuit's expected parameter length.
        """
        x = x.to(dtype=self.dtype, device=self.device).reshape(-1)
        px_len = self._px_len()

        spec = self._angle_encoding_specs.get(self.input_parameters)

        if spec:
            encoded = self._encode_with_specs(x, spec)
            if encoded.numel() != px_len:
                raise ValueError(
                    f"Angle encoding produced {encoded.numel()} parameters but circuit expects {px_len}"
                )
            return encoded

        if x.numel() == px_len:
            return x
        if x.numel() < px_len:
            # Try provided encoder if available
            if callable(self._encoder):
                try:
                    encoded = self._encoder(x)
                    # Allow numpy/torch outputs and ensure correct shape/device/dtype
                    if isinstance(encoded, np.ndarray):
                        encoded = torch.from_numpy(encoded)
                    encoded = torch.as_tensor(
                        encoded, dtype=self.dtype, device=self.device
                    ).reshape(-1)
                    if encoded.numel() != px_len:
                        # Fall back if encoder does not match spec
                        return self._subset_sum_expand(x, px_len)
                    return encoded
                except Exception:
                    # Encoder failed; use deterministic subset-sum expansion
                    return self._subset_sum_expand(x, px_len)
            # No encoder provided; series-style expansion
            return self._subset_sum_expand(x, px_len)
        # x longer than needed; truncate
        return x[:px_len]

    def _encode_with_specs(self, x: Tensor, spec: dict[str, object]) -> Tensor:
        """Encode input vector using builder-provided angle encoding metadata.

        Args:
            x: Flattened input feature tensor.
            spec: Metadata describing combinations and scales produced by the builder.

        Returns:
            Tensor: Encoded tensor obeying the combination rules.
        """
        combos = spec.get("combinations", [])
        scales = spec.get("scales", {})

        if not isinstance(combos, list) or not all(
            isinstance(c, tuple) for c in combos
        ):
            raise ValueError(
                "Invalid angle encoding metadata: 'combinations' must be a list of tuples"
            )

        if not isinstance(scales, dict):
            raise ValueError("Invalid angle encoding metadata: 'scales' must be a dict")

        x_flat = x.to(dtype=self.dtype, device=self.device).reshape(-1)
        encoded_vals: list[Tensor] = []
        feature_dim = x_flat.shape[0]

        for combo in combos:  # type: ignore[assignment]
            indices = list(combo)
            if any(idx >= feature_dim for idx in indices):
                raise ValueError(
                    f"Input feature dimension {feature_dim} insufficient for angle encoding combination {combo}"
                )

            selected = x_flat[indices]
            scale_tensor = torch.tensor(
                [float(scales.get(idx, 1.0)) for idx in indices],
                dtype=self.dtype,
                device=self.device,
            )
            encoded_vals.append((selected * scale_tensor).sum())

        if not encoded_vals:
            return torch.zeros(0, dtype=self.dtype, device=self.device)

        return torch.stack(encoded_vals, dim=0)

    def compute_unitary(
        self, x: Tensor | np.ndarray | float, *training_parameters: Tensor
    ) -> Tensor:
        """Generate the circuit unitary after encoding ``x`` and applying trainables.

        Args:
            x: Single datapoint to embed; accepts scalars, numpy arrays, or tensors.
            *training_parameters: Optional overriding trainable tensors.

        Returns:
            Tensor: Complex unitary matrix representing the prepared circuit.
        """
        # Normalize input to tensor on correct device/dtype
        if isinstance(x, torch.Tensor):
            x = x.to(dtype=self.dtype, device=self.device)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device=self.device, dtype=self.dtype)
        elif isinstance(x, (float, int)):
            # scalar datapoint: only valid if input_size == 1
            x = torch.tensor([x], dtype=self.dtype, device=self.device)
        else:
            raise TypeError(f"Unsupported input type: {type(x)!r}")

        # Encode x to match the circuit's input parameter spec
        x_encoded = self._encode_x(x)

        if not self.is_trainable:
            return self._circuit_graph.to_tensor(x_encoded)

        # Use provided training parameters or fall back to internal ones
        if training_parameters:
            params_to_use: tuple[Tensor, ...] = training_parameters
        else:
            # Cast to a Tensor tuple for mypy; Parameter is a Tensor subtype
            params_to_use = cast(
                tuple[Tensor, ...], tuple(self._training_dict.values())
            )
        return self._circuit_graph.to_tensor(x_encoded, *params_to_use)

    def is_datapoint(self, x: Tensor | np.ndarray | float | int) -> bool:
        """Determine if ``x`` describes one sample or a batch.

        Args:
            x: Candidate input data.

        Returns:
            bool: ``True`` when ``x`` corresponds to a single datapoint.
        """
        if isinstance(x, (float, int)):
            if self.input_size == 1:
                return True
            raise ValueError(
                f"Given value shape () does not match data shape {self.input_size}."
            )

        # x is array-like (Tensor or ndarray)
        if isinstance(x, Tensor):
            ndim = x.ndim
            shape = tuple(x.shape)
            num_elements = x.numel()
        else:
            ndim = x.ndim
            shape = tuple(x.shape)
            num_elements = x.size

        error_msg = (
            f"Given value shape {shape} does not match data shape {self.input_size}."
        )
        if num_elements % self.input_size or ndim > 2:
            raise ValueError(error_msg)

        if self.input_size == 1:
            if num_elements == 1 and ndim == 1:
                return True
            if num_elements == 1 and ndim == 2:
                return False
            if ndim > 1:
                return False
        else:
            if ndim == 1 and shape[0] == self.input_size:
                return True
            if ndim == 2:
                return False
        raise ValueError(error_msg)

    @classmethod
    def simple(
        cls,
        input_size: int,
        n_modes: int,
        n_photons: int | None = None,
        *,
        dtype: str | torch.dtype = torch.float32,
        device: torch.device | None = None,
        angle_encoding_scale: float = 1.0,
        trainable: bool = True,
        trainable_prefix: str = "phi",
    ) -> "FeatureMap":
        """
        Simple factory method to create a FeatureMap with minimal configuration.

        Args:
            input_size: Classical feature dimension.
            n_modes: Number of photonic modes used by the helper circuit.
            n_photons: Optional photon count (defaults to ``input_size``).
            dtype: Target dtype for internal tensors.
            device: Optional torch device handle.
            angle_encoding_scale: Global scaling applied to angle encoding features.
            trainable: Whether to expose a trainable rotation layer.
            trainable_prefix: Prefix used for the generated trainable parameter names.

        Returns:
            FeatureMap: Configured feature-map instance.
        """
        if n_photons is None:
            n_photons = input_size

        if input_size > n_modes:
            raise ValueError(ANGLE_ENCODING_MODE_ERROR)

        builder = CircuitBuilder(n_modes=n_modes)

        builder.add_entangling_layer(depth=1)
        input_modes = list(range(input_size))

        builder.add_angle_encoding(
            modes=input_modes,
            name="input",
            scale=angle_encoding_scale,
        )

        trainable_parameters: list[str] | None
        if trainable:
            builder.add_rotation_layer(trainable=True, name=trainable_prefix)
            trainable_parameters = [trainable_prefix]
        else:
            trainable_parameters = None

        builder.add_entangling_layer(depth=1)

        return cls(
            circuit=builder,
            input_size=input_size,
            input_parameters=None,
            trainable_parameters=trainable_parameters,
            dtype=dtype,
            device=device,
        )


class KernelCircuitBuilder:
    """
    Builder class for creating quantum kernel circuits with photonic backends.

    This class provides a fluent interface for building quantum kernel circuits
    with various configurations, inspired by the core.layer architecture.
    """

    def __init__(self):
        self._input_size: int | None = None
        self._n_modes: int | None = None
        self._n_photons: int | None = None
        self._dtype: str | torch.dtype = torch.float32
        self._device: torch.device | None = None
        self._use_bandwidth_tuning: bool = False
        self._angle_encoding_scale: float = 1.0
        self._trainable: bool = True
        self._trainable_prefix: str = "phi"

    def input_size(self, size: int) -> "KernelCircuitBuilder":
        """Set the input dimensionality."""
        self._input_size = size
        return self

    def n_modes(self, modes: int) -> "KernelCircuitBuilder":
        """Set the number of modes in the circuit."""
        self._n_modes = modes
        return self

    def n_photons(self, photons: int) -> "KernelCircuitBuilder":
        """Set the number of photons."""
        self._n_photons = photons
        return self

    def trainable(
        self,
        enabled: bool = True,
        *,
        prefix: str = "phi",
    ) -> "KernelCircuitBuilder":
        """Enable or disable trainable rotations generated by the helper."""
        self._trainable = enabled
        if enabled:
            self._trainable_prefix = prefix
        return self

    def dtype(self, dtype: str | torch.dtype) -> "KernelCircuitBuilder":
        """Set the data type for computations."""
        self._dtype = dtype
        return self

    def device(self, device: torch.device) -> "KernelCircuitBuilder":
        """Set the computation device."""
        self._device = device
        return self

    def bandwidth_tuning(self, enabled: bool = True) -> "KernelCircuitBuilder":
        """Enable or disable bandwidth tuning."""
        self._use_bandwidth_tuning = enabled
        return self

    def angle_encoding(
        self,
        *,
        scale: float = 1.0,
    ) -> "KernelCircuitBuilder":
        """Configure the angle encoding scale."""
        self._angle_encoding_scale = scale
        return self

    def build_feature_map(self) -> FeatureMap:
        """
        Build and return a FeatureMap instance.

        :return: Configured FeatureMap
        :raises ValueError: If required parameters are missing
        """
        if self._input_size is None:
            raise ValueError("Input size must be specified")

        n_modes = self._n_modes or max(self._input_size + 1, 4)

        trainable_params: list[str] | None
        if self._trainable:
            trainable_params = [self._trainable_prefix]
        else:
            trainable_params = None

        builder = CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer(depth=1)

        if self._input_size > n_modes:
            raise ValueError(ANGLE_ENCODING_MODE_ERROR)

        input_modes = list(range(self._input_size))

        builder.add_angle_encoding(
            modes=input_modes,
            name="input",
            scale=self._angle_encoding_scale,
        )

        if self._trainable:
            builder.add_rotation_layer(trainable=True, name=self._trainable_prefix)

        builder.add_entangling_layer(depth=1)

        return FeatureMap(
            circuit=builder,
            input_size=self._input_size,
            input_parameters=None,
            trainable_parameters=trainable_params,
            dtype=self._dtype,
            device=self._device,
        )

    def build_fidelity_kernel(
        self,
        input_state: list[int] | None = None,
        *,
        shots: int = 0,
        sampling_method: str = "multinomial",
        no_bunching: bool = False,
        force_psd: bool = True,
    ) -> "FidelityKernel":
        """
        Build and return a FidelityKernel instance.

        :param input_state: Input Fock state. If None, automatically generated
        :param shots: Number of sampling shots
        :param sampling_method: Sampling method for shots
        :param no_bunching: Whether to exclude bunched states
        :param force_psd: Whether to project to positive semi-definite
        :return: Configured FidelityKernel
        """
        feature_map = self.build_feature_map()

        # Generate default input state if not provided
        if input_state is None:
            n_modes = self._n_modes or max(self._input_size or 2, 4)
            n_photons = self._n_photons or (self._input_size or 2)
            input_state = [1] * n_photons + [0] * (n_modes - n_photons)

        return FidelityKernel(
            feature_map=feature_map,
            input_state=input_state,
            shots=shots,
            sampling_method=sampling_method,
            no_bunching=no_bunching,
            force_psd=force_psd,
            device=self._device,
            dtype=self._dtype,
        )


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
        datapoint within its circuit
    :param input_state: Input state into circuit.
    :param shots: Number of circuit shots. If `None`, the exact
        transition probabilities are returned. Default: `None`.
    :param sampling_method: Probability distributions are post-
        processed with some pseudo-sampling method: 'multinomial',
        'binomial' or 'gaussian'.
    :param no_bunching: Whether or not to post-select out results with
        bunching. Default: `False`.
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
        >>> # For a support vector classification problem
        >>> svc = SVC(kernel='precomputed')
        >>> svc.fit(K_train, y_train)
        >>> y_pred = svc.predict(K_test)
    """

    def __init__(
        self,
        feature_map: FeatureMap,
        input_state: list[int],
        *,
        shots: int | None = None,
        sampling_method: str = "multinomial",
        no_bunching: bool = False,
        force_psd: bool = True,
        device: torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ):
        super().__init__()
        self.feature_map = feature_map
        self.input_state = input_state
        self.shots = shots or 0
        self.sampling_method = sampling_method
        self.no_bunching = no_bunching
        self.force_psd = force_psd
        self.device = device or feature_map.device
        # Normalize to a torch.dtype
        if dtype is None:
            self.dtype = feature_map.dtype
        else:
            mapped = dtype_to_torch.get(dtype)
            self.dtype = mapped if mapped is not None else feature_map.dtype
        self.input_size = self.feature_map.input_size

        if self.feature_map.circuit.m != len(input_state):
            raise ValueError("Input state length does not match circuit size.")

        self.is_trainable = feature_map.is_trainable
        if self.is_trainable:
            for param_name, param in feature_map._training_dict.items():
                self.register_parameter(param_name, param)

        if max(input_state) > 1 and no_bunching:
            raise ValueError(
                f"Bunching must be enabled for an input state with"
                f"{max(input_state)} in one mode."
            )
        elif all(x == 1 for x in input_state) and no_bunching:
            raise ValueError(
                "For `no_bunching = True`, the kernel value will always be 1"
                " for an input state with a photon in all modes."
            )

        m, n = len(input_state), sum(input_state)

        self._slos_graph = build_slos_graph(
            m=m,
            n_photons=n,
            no_bunching=no_bunching,
            keep_keys=True,
            device=device,
            dtype=self.dtype,
        )
        # Find index of input state in output distribution
        keys = self._slos_graph.final_keys
        self._input_state_index = keys.index(tuple(input_state))
        # For sampling
        self._autodiff_process = AutoDiffProcess()

    def forward(
        self,
        x1: float | np.ndarray | Tensor,
        x2: float | np.ndarray | Tensor | None = None,
    ):
        """
        Calculate the quantum kernel for input data `x1` and `x2.` If
        `x1` and `x2` are datapoints, a scalar value is returned. For
        input datasets the kernel matrix is computed.
        """
        # Convert inputs to tensors and ensure they are on the correct device
        if not isinstance(x1, torch.Tensor):
            x1 = torch.as_tensor(x1, dtype=self.dtype)

        if x2 is not None:
            if isinstance(x2, np.ndarray):
                x2 = torch.from_numpy(x2).to(device=x1.device, dtype=self.dtype)
            elif isinstance(x2, torch.Tensor):
                x2 = x2.to(device=x1.device, dtype=self.dtype)

        # Return scalar value for input datapoints
        if self.feature_map.is_datapoint(x1):
            if x2 is None:
                raise ValueError("For input datapoints, please specify an x2 argument.")
            return self._return_kernel_scalar(x1, x2)

        # Ensure tensors before reshaping (satisfies mypy)
        if x2 is not None and not isinstance(x2, torch.Tensor):
            x2 = torch.as_tensor(x2, dtype=self.dtype, device=self.device)

        x1 = x1.reshape(-1, self.input_size)
        x2 = x2.reshape(-1, self.input_size) if x2 is not None else None

        # Check if we are constructing training matrix
        equal_inputs = self._check_equal_inputs(x1, x2)
        U_forward = torch.stack([
            self.feature_map.compute_unitary(x).to(x1.device) for x in x1
        ])

        len_x1 = len(x1)
        if x2 is not None:
            U_adjoint = torch.stack([
                self.feature_map.compute_unitary(x).transpose(0, 1).conj().to(x1.device)
                for x in x2
            ])

            # Calculate circuit unitary for every pair of datapoints
            all_circuits = U_forward.unsqueeze(1) @ U_adjoint.unsqueeze(0)
            all_circuits = all_circuits.view(-1, *all_circuits.shape[2:])
        else:
            U_adjoint = U_forward.conj().transpose(1, 2)

            # Take circuit unitaries for upper diagonal of kernel matrix only
            upper_idx = torch.triu_indices(
                len_x1,
                len_x1,
                offset=1,
                device=x1.device,
            )
            all_circuits = U_forward[upper_idx[0]] @ U_adjoint[upper_idx[1]]

        # Distribution for every evaluated circuit
        all_probs = self._slos_graph.compute(all_circuits, self.input_state)[1]

        if self.shots > 0:
            # Convert complex amplitudes to real probabilities for multinomial sampling
            real_probs = torch.abs(all_probs).square()
            all_probs = self._autodiff_process.sampling_noise.pcvl_sampler(
                real_probs, self.shots, self.sampling_method
            )

        transition_probs = torch.abs(all_probs[:, self._input_state_index])

        if x2 is None:
            # Copy transition probs to upper & lower diagonal
            kernel_matrix = torch.zeros(
                len_x1, len_x1, dtype=self.dtype, device=x1.device
            )

            upper_idx = upper_idx.to(x1.device)
            transition_probs = transition_probs.to(dtype=self.dtype, device=x1.device)
            kernel_matrix[upper_idx[0], upper_idx[1]] = transition_probs
            kernel_matrix[upper_idx[1], upper_idx[0]] = transition_probs
            kernel_matrix.fill_diagonal_(1)

            if self.force_psd:
                kernel_matrix = self._project_psd(kernel_matrix)

        else:
            transition_probs = transition_probs.to(dtype=self.dtype, device=x1.device)
            kernel_matrix = transition_probs.reshape(len_x1, len(x2))

            if self.force_psd and equal_inputs:
                # Symmetrize the matrix
                kernel_matrix = 0.5 * (kernel_matrix + kernel_matrix.T)
                kernel_matrix = self._project_psd(kernel_matrix)

        return kernel_matrix

    def _return_kernel_scalar(
        self,
        x1: Tensor | np.ndarray | float | int,
        x2: Tensor | np.ndarray | float | int,
    ) -> float:
        """Returns scalar kernel value for input datapoints"""
        # Normalize to torch.Tensor on correct device/dtype
        if isinstance(x1, np.ndarray):
            x1_t = torch.from_numpy(x1)
        elif isinstance(x1, (float, int)):
            x1_t = torch.tensor([x1])
        else:
            x1_t = x1
        if isinstance(x2, np.ndarray):
            x2_t = torch.from_numpy(x2)
        elif isinstance(x2, (float, int)):
            x2_t = torch.tensor([x2])
        else:
            x2_t = x2

        x1_t = torch.as_tensor(x1_t, dtype=self.dtype, device=self.device).reshape(
            self.input_size
        )
        x2_t = torch.as_tensor(x2_t, dtype=self.dtype, device=self.device).reshape(
            self.input_size
        )

        U = self.feature_map.compute_unitary(x1_t)
        U_adjoint = self.feature_map.compute_unitary(x2_t)
        U_adjoint = U_adjoint.conj().T

        probs = self._slos_graph.compute(U @ U_adjoint, self.input_state)[1]

        if self.shots > 0:
            # Convert complex amplitudes to real probabilities for multinomial sampling
            real_probs = torch.abs(probs).square()
            probs = self._autodiff_process.sampling_noise.pcvl_sampler(
                real_probs, self.shots, self.sampling_method
            )
        return torch.abs(probs[0, self._input_state_index]).item()

    @classmethod
    def simple(
        cls,
        input_size: int,
        n_modes: int,
        n_photons: int | None = None,
        input_state: list[int] | None = None,
        *,
        shots: int = 0,
        sampling_method: str = "multinomial",
        no_bunching: bool = False,
        force_psd: bool = True,
        trainable: bool = True,
        dtype: str | torch.dtype = torch.float32,
        device: torch.device | None = None,
        angle_encoding_scale: float = 1.0,
    ) -> "FidelityKernel":
        """
        Simple factory method to create a FidelityKernel with minimal configuration.
        """
        if n_photons is None:
            n_photons = input_size
        feature_map = FeatureMap.simple(
            input_size=input_size,
            n_modes=n_modes,
            n_photons=n_photons,
            trainable=trainable,
            dtype=dtype,
            device=device,
            angle_encoding_scale=angle_encoding_scale,
        )

        if input_state is None:
            input_state = [1] * n_photons + [0] * (n_modes - n_photons)

        return cls(
            feature_map=feature_map,
            input_state=input_state,
            shots=shots,
            sampling_method=sampling_method,
            no_bunching=no_bunching,
            force_psd=force_psd,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _project_psd(matrix: Tensor) -> Tensor:
        """Projects a symmetric matrix to closest positive semi-definite"""
        # Perform spectral decomposition and set negative eigenvalues to 0
        eigenvals, eigenvecs = torch.linalg.eigh(matrix)
        eigenvals = torch.diag(torch.where(eigenvals > 0, eigenvals, 0))

        matrix_psd = eigenvecs @ eigenvals @ eigenvecs.T

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
