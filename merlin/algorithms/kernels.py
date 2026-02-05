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

import itertools
import warnings
from collections.abc import Callable
from typing import cast

import numpy as np
import perceval as pcvl
import torch
from torch import Tensor

from ..builder.circuit_builder import ANGLE_ENCODING_MODE_ERROR, CircuitBuilder
from ..core.computation_space import ComputationSpace
from ..core.generators import StateGenerator, StatePattern
from ..measurement.autodiff import AutoDiffProcess
from ..measurement.detectors import DetectorTransform, resolve_detectors
from ..measurement.photon_loss import PhotonLossTransform, resolve_photon_loss
from ..pcvl_pytorch.locirc_to_tensor import CircuitConverter
from ..pcvl_pytorch.slos_torchscript import (
    build_slos_distribution_computegraph as build_slos_graph,
)
from ..utils.deprecations import sanitize_parameters
from ..utils.dtypes import to_torch_dtype
from .module import MerlinModule


class FeatureMap:
    """
    Quantum Feature Map

    FeatureMap embeds a datapoint within a quantum circuit and
    computes the associated unitary for quantum kernel methods.

    Args:
        circuit: Pre-compiled :class:`pcvl.Circuit` to encode features.
        input_size: Dimension of incoming classical data (required).
        builder: Optional :class:`CircuitBuilder` to compile into a circuit.
        experiment: Optional :class:`pcvl.Experiment` providing both the circuit and detector configuration.
                   Exactly one of ``circuit``, ``builder``, or ``experiment`` must be supplied.
        input_parameters: Parameter prefix(es) that host the classical data.
        dtype: Torch dtype used when constructing the unitary.
        device: Torch device on which unitaries are evaluated.
    """

    def __init__(
        self,
        circuit: pcvl.Circuit | None = None,
        input_size: int | None = None,
        *,
        builder: CircuitBuilder | None = None,
        experiment: pcvl.Experiment | None = None,
        input_parameters: str | list[str] | None,
        trainable_parameters: list[str] | None = None,
        dtype: str | torch.dtype = torch.float32,
        device: torch.device | None = None,
        encoder: Callable[[Tensor], Tensor] | None = None,  # was: callable | None
    ):
        builder_trainable: list[str] = []
        builder_input: list[str] = []

        self._angle_encoding_specs: dict[str, dict[str, object]] = {}
        self.experiment: pcvl.Experiment | None = None

        # The feature map can be defined from exactly one artefact among circuit, builder, or experiment.
        if sum(x is not None for x in (circuit, builder, experiment)) != 1:
            raise ValueError(
                "Provide exactly one of 'circuit', 'builder', or 'experiment'."
            )

        resolved_circuit: pcvl.Circuit | None = None

        if builder is not None:
            builder_trainable = builder.trainable_parameter_prefixes
            builder_input = builder.input_parameter_prefixes
            self._angle_encoding_specs = builder.angle_encoding_specs
            resolved_circuit = builder.to_pcvl_circuit(pcvl)
            self.experiment = pcvl.Experiment(resolved_circuit)
        elif circuit is not None:
            resolved_circuit = circuit
            self.experiment = pcvl.Experiment(resolved_circuit)
        elif experiment is not None:
            if (
                not experiment.is_unitary
                or experiment.post_select_fn is not None
                or experiment.heralds
            ):
                raise ValueError(
                    "The provided experiment must be unitary, and must not have post-selection or heralding."
                )
            if experiment.min_photons_filter:
                raise ValueError(
                    "The provided experiment must not have a minimum photons filter."
                )
            self.experiment = experiment
            resolved_circuit = experiment.unitary_circuit()
        else:  # pragma: no cover - defensive guard
            raise RuntimeError("Resolved circuit could not be determined.")

        self.circuit = resolved_circuit
        if input_size is None:
            raise TypeError("FeatureMap requires 'input_size' to be specified.")
        self.input_size = input_size
        if trainable_parameters is None:
            trainable_parameters = builder_trainable
        self.trainable_parameters = list(trainable_parameters or [])
        self.dtype = to_torch_dtype(dtype)
        self.device = device or torch.device("cpu")
        self.is_trainable = bool(self.trainable_parameters)
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
            self.circuit,
            [self.input_parameters] + self.trainable_parameters,
            dtype=self.dtype,
            device=self.device,
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
    @sanitize_parameters
    def simple(
        cls,
        input_size: int,
        *,
        dtype: str | torch.dtype = torch.float32,
        device: torch.device | None = None,
        angle_encoding_scale: float = 1.0,
        n_modes: int = None,
    ) -> "FeatureMap":
        """
        Simple factory method to create a FeatureMap with minimal configuration.

        Args:
            input_size: Classical feature dimension. Maximum is 20.
            n_photons: Optional photon count (defaults to ``input_size``).
            dtype: Target dtype for internal tensors.
            device: Optional torch device handle.
            angle_encoding_scale: Global scaling applied to angle encoding features.
            n_modes: Number of photonic modes used by the helper circuit. If it is not defined: n_modes=input_size. Maximum is 20.

        Returns:
            FeatureMap: Configured feature-map instance.
        """
        if n_modes is None:
            n_modes = input_size
        if input_size > 20 or n_modes > 20:
            raise ValueError(
                "Input size too large for the simple layer construction. For large inputs (with larger size than 20), please use the CircuitBuilder. Here is a quick tutorial on how to use it: https://merlinquantum.ai/quickstart/first_quantum_layer.html#circuitbuilder-walkthrough"
            )
        if input_size < 1:
            raise ValueError(f"input_size must be at least 1, got {input_size}")

        if input_size > n_modes:
            raise ValueError(ANGLE_ENCODING_MODE_ERROR)

        if n_modes == 1:
            builder = CircuitBuilder(n_modes=2)

            # Trainable entangling layer before encoding
            builder.add_entangling_layer(trainable=True, name="LI_simple")

            # Angle encoding
            builder.add_angle_encoding(
                modes=[1], name="input", subset_combinations=False
            )

            # Trainable entangling layer after encoding
            builder.add_entangling_layer(trainable=True, name="RI_simple")

        else:
            builder = CircuitBuilder(n_modes=n_modes)

            # Trainable entangling layer before encoding
            builder.add_entangling_layer(
                trainable=True,
                name="LI_simple",
            )

            # Angle encoding
            builder.add_angle_encoding(
                modes=list(range(int(input_size))),
                name="input",
                subset_combinations=False,
                scale=angle_encoding_scale,
            )

            # Trainable entangling layer after encoding
            builder.add_entangling_layer(trainable=True, name="RI_simple")

        return cls(
            builder=builder,
            input_size=input_size,
            input_parameters=None,
            trainable_parameters=[
                "LI_simple",
                "RI_simple",
            ],
            dtype=dtype,
            device=device,
        )


class KernelCircuitBuilder:
    """
    Builder class for creating quantum kernel circuits with photonic backends.

    This class provides a fluent interface for building quantum kernel circuits
    with various configurations, inspired by the core.layer architecture.
    """

    def __init__(self) -> None:
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
        builder.add_superpositions(depth=1)

        if self._input_size > n_modes:
            raise ValueError(ANGLE_ENCODING_MODE_ERROR)

        input_modes = list(range(self._input_size))

        builder.add_angle_encoding(
            modes=input_modes,
            name="input",
            scale=self._angle_encoding_scale,
        )

        if self._trainable:
            builder.add_rotations(trainable=True, name=self._trainable_prefix)

        builder.add_superpositions(depth=1)

        return FeatureMap(
            builder=builder,
            input_size=self._input_size,
            input_parameters=None,
            trainable_parameters=trainable_params,
            dtype=self._dtype,
            device=self._device,
        )

    @sanitize_parameters
    def build_fidelity_kernel(
        self,
        input_state: list[int] | None = None,
        *,
        shots: int = 0,
        sampling_method: str = "multinomial",
        computation_space: ComputationSpace | str | None = None,
        force_psd: bool = True,
    ) -> "FidelityKernel":
        """
        Build and return a FidelityKernel instance.

        :param input_state: Input Fock state. If None, automatically generated
        :param shots: Number of sampling shots
        :param sampling_method: Sampling method for shots
        :param computation_space: Logical computation subspace; one of {"fock", "unbunched", "dual_rail"}.
        :param force_psd: Whether to project to positive semi-definite
        :return: Configured FidelityKernel
        """
        feature_map = self.build_feature_map()

        # Generate default input state if not provided
        if input_state is None:
            n_modes = self._n_modes or max(self._input_size or 2, 4)
            n_photons = self._n_photons or (self._input_size or 2)
            input_state = StateGenerator.generate_state(
                n_modes, n_photons, StatePattern.SPACED
            )

        return FidelityKernel(
            feature_map=feature_map,
            input_state=input_state,
            shots=shots,
            sampling_method=sampling_method,
            computation_space=computation_space,
            force_psd=force_psd,
            device=self._device,
            dtype=self._dtype,
        )


class FidelityKernel(MerlinModule):
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
    :param computation_space: Logical computation subspace; one of
        ``{"fock", "unbunched", "dual_rail"}``. Default: ``FOCK``.
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

    @sanitize_parameters
    def __init__(
        self,
        feature_map: FeatureMap,
        input_state: list[int],
        *,
        shots: int | None = None,
        sampling_method: str = "multinomial",
        computation_space: ComputationSpace | str | None = None,
        force_psd: bool = True,
        device: torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ):
        super().__init__()
        if computation_space is None:
            computation_space = ComputationSpace.FOCK
        else:
            computation_space = ComputationSpace.coerce(computation_space)
        self.computation_space = computation_space
        self.feature_map = feature_map
        self.input_state = input_state
        self.shots = shots or 0
        self.sampling_method = sampling_method
        self.no_bunching = self.computation_space is not ComputationSpace.FOCK
        self.force_psd = force_psd
        base_device = device if device is not None else feature_map.device
        self.device = (
            torch.device(base_device)
            if base_device is not None
            else torch.device("cpu")
        )
        # Normalize to a torch.dtype
        if dtype is None:
            self.dtype = feature_map.dtype
        else:
            self.dtype = to_torch_dtype(dtype, default=feature_map.dtype)
        self.input_size = self.feature_map.input_size

        if self.feature_map.circuit.m != len(input_state):
            raise ValueError("Input state length does not match circuit size.")

        self.is_trainable = feature_map.is_trainable
        if self.is_trainable:
            for param_name, param in feature_map._training_dict.items():
                self.register_parameter(param_name, param)

        experiment = getattr(self.feature_map, "experiment", None)
        if experiment is None:
            experiment = pcvl.Experiment(self.feature_map.circuit)
            self.feature_map.experiment = experiment

        self._validate_experiment(experiment)
        self.experiment = experiment
        experiment_circuit = self.experiment.unitary_circuit()
        if experiment_circuit.m != self.feature_map.circuit.m:
            raise ValueError(
                "Experiment circuit must have the same number of modes as the feature map circuit."
            )

        if max(input_state) > 1 and self.computation_space is not ComputationSpace.FOCK:
            raise ValueError(
                f"Bunching must be enabled for an input state with"
                f"{max(input_state)} in one mode."
            )
        elif (
            all(x == 1 for x in input_state)
            and self.computation_space is not ComputationSpace.FOCK
        ):
            raise ValueError(
                "For non-FOCK computation_space, the kernel value will always be 1 "
                "for an input state with a photon in all modes."
            )

        m, n = len(input_state), sum(input_state)
        self._detectors, self._empty_detectors = resolve_detectors(self.experiment, m)

        # Verify that no Detector was defined in experiment if using non-FOCK space:
        if (
            not self._empty_detectors
            and self.computation_space is not ComputationSpace.FOCK
        ):
            raise RuntimeError(
                "computation_space must be FOCK if Experiment contains at least one Detector."
            )

        self._slos_graph = build_slos_graph(
            m=m,
            n_photons=n,
            computation_space=self.computation_space,
            keep_keys=True,
            device=device,
            dtype=self.dtype,
        )
        # Resolve raw simulation keys and photon loss transform
        raw_keys = [tuple(int(v) for v in key) for key in self._slos_graph.final_keys]
        self._raw_output_keys = raw_keys
        try:
            self._input_state_index = raw_keys.index(tuple(input_state))
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(
                "Input state is not present in the simulation basis produced by the circuit."
            ) from exc

        self._photon_survival_probs, empty_noise_model = resolve_photon_loss(
            self.experiment, m
        )
        self.has_custom_noise_model = not empty_noise_model

        self._photon_loss_transform = PhotonLossTransform(
            raw_keys,
            self._photon_survival_probs,
            dtype=self.dtype,
            device=self.device,
        )
        self._photon_loss_is_identity = self._photon_loss_transform.is_identity
        self._photon_loss_keys = self._photon_loss_transform.output_keys
        try:
            self._photon_loss_input_index = self._photon_loss_keys.index(
                tuple(input_state)
            )
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(
                "Photon loss transform did not preserve the original input Fock state."
            ) from exc

        self._detector_transform = DetectorTransform(
            self._photon_loss_keys,
            self._detectors,
            dtype=self.dtype,
            device=self.device,
        )
        self._detector_is_identity = self._detector_transform.is_identity
        weight_device = self.device or torch.device("cpu")
        one_hot = torch.zeros(
            len(self._raw_output_keys), dtype=self.dtype, device=weight_device
        )
        one_hot[self._input_state_index] = 1.0
        loss_vector = self._apply_photon_loss_transform(one_hot)
        if loss_vector.ndim > 1:
            loss_vector = loss_vector.squeeze(0)
        detection_vector = (
            loss_vector
            if self._detector_is_identity
            else self._detector_transform(loss_vector)
        )
        if detection_vector.ndim > 1:
            detection_vector = detection_vector.squeeze(0)
        detection_vector = detection_vector.to(dtype=self.dtype, device=weight_device)
        nonzero = torch.nonzero(detection_vector > 1e-8, as_tuple=True)[0]
        self._input_detection_index = None
        if nonzero.numel() == 1 and torch.isclose(
            detection_vector[nonzero[0]],
            torch.tensor(
                1.0, dtype=detection_vector.dtype, device=detection_vector.device
            ),
            atol=1e-6,
        ):
            self._input_detection_index = int(nonzero[0].item())
        self.register_buffer("_input_detection_weights", detection_vector)
        # For sampling
        self._autodiff_process = AutoDiffProcess()

    def _apply_photon_loss_transform(self, distribution: Tensor) -> Tensor:
        """Apply photon loss transform when a noise model is defined."""
        if self._photon_loss_is_identity:
            return distribution
        return self._photon_loss_transform(distribution)

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

        if isinstance(x2, torch.Tensor) or x2 is None:
            x1 = x1.reshape(-1, self.input_size)
            x2 = x2.reshape(-1, self.input_size) if x2 is not None else None
        else:
            raise (TypeError("x2 is not None nor torch.Tensor"))

        # Check if we are constructing training matrix
        equal_inputs = self._check_equal_inputs(x1, x2)
        U_forward = torch.stack([
            self.feature_map.compute_unitary(x).to(x1.device) for x in x1
        ])

        len_x1 = len(x1)
        if x2 is not None:
            x2_tensor = (
                x2
                if isinstance(x2, torch.Tensor)
                else torch.as_tensor(x2, dtype=self.dtype, device=self.device)
            )
            U_adjoint = torch.stack([
                self.feature_map.compute_unitary(x).transpose(0, 1).conj().to(x1.device)
                for x in x2_tensor
            ])
            if isinstance(x2, torch.Tensor):
                U_adjoint = torch.stack([
                    self.feature_map
                    .compute_unitary(x)
                    .transpose(0, 1)
                    .conj()
                    .to(x1.device)
                    for x in x2
                ])
            else:
                raise (TypeError("x2 is not None nor torch.Tensor"))

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
        _, probabilities = self._slos_graph.compute_probs(
            all_circuits, self.input_state
        )

        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(0)
        probabilities = probabilities.to(dtype=self.dtype)
        loss_probs = self._apply_photon_loss_transform(probabilities)
        detection_probs = self._detector_transform(loss_probs)

        if self.shots > 0:
            detection_probs = self._autodiff_process.sampling_noise.pcvl_sampler(
                detection_probs, self.shots, self.sampling_method
            )

        if self._input_detection_index is not None:
            transition_probs = detection_probs[:, self._input_detection_index]
        else:
            weights = self._input_detection_weights.to(
                dtype=detection_probs.dtype, device=detection_probs.device
            )
            transition_probs = detection_probs @ weights

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
            x2_tensor = (
                x2
                if isinstance(x2, torch.Tensor)
                else torch.as_tensor(x2, dtype=self.dtype, device=self.device)
            )
            transition_probs = transition_probs.to(dtype=self.dtype, device=x1.device)
            kernel_matrix = transition_probs.reshape(len_x1, len(x2_tensor))
            if isinstance(x2, torch.Tensor):
                transition_probs = transition_probs.to(
                    dtype=self.dtype, device=x1.device
                )
                kernel_matrix = transition_probs.reshape(len_x1, len(x2))
            else:
                raise (TypeError("x2 is not None nor torch.Tensor"))

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

        kernel_unitary = U @ U_adjoint
        _, probabilities = self._slos_graph.compute_probs(
            kernel_unitary, self.input_state
        )
        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(0)
        probabilities = probabilities.to(dtype=self.dtype, device=self.device)

        loss_probs = self._apply_photon_loss_transform(probabilities)
        detection_probs = self._detector_transform(loss_probs)

        if self.shots > 0:
            detection_probs = self._autodiff_process.sampling_noise.pcvl_sampler(
                detection_probs, self.shots, self.sampling_method
            )

        if self._input_detection_index is not None:
            value = detection_probs[0, self._input_detection_index]
        else:
            weights = self._input_detection_weights.to(
                dtype=detection_probs.dtype, device=detection_probs.device
            )
            value = (detection_probs @ weights)[0]

        return value.item()

    @classmethod
    @sanitize_parameters
    def simple(
        cls,
        input_size: int,
        *,
        shots: int = 0,
        sampling_method: str = "multinomial",
        computation_space: ComputationSpace | str | None = None,
        force_psd: bool = True,
        dtype: str | torch.dtype = torch.float32,
        device: torch.device | None = None,
        angle_encoding_scale: float = 1.0,
        n_modes: int = None,
    ) -> "FidelityKernel":
        """
        Simple factory method to create a FidelityKernel with minimal configuration.
        """
        feature_map = FeatureMap.simple(
            input_size=input_size,
            n_modes=n_modes,
            dtype=dtype,
            device=device,
            angle_encoding_scale=angle_encoding_scale,
        )

        if n_modes is None:
            state_size = input_size
        else:
            state_size = max(n_modes, input_size)

        if state_size == 1:
            input_state = [0, 1]
        else:
            input_state = state_size * [0]
            for i in range(state_size):
                if i % 2 == 1:
                    input_state[i] = 1

        return cls(
            feature_map=feature_map,
            input_state=input_state,
            shots=shots,
            sampling_method=sampling_method,
            computation_space=computation_space,
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

    @staticmethod
    def _validate_experiment(experiment: pcvl.Experiment) -> None:
        """Validate that the provided experiment is compatible with fidelity kernels."""
        if (
            not experiment.is_unitary
            or experiment.post_select_fn is not None
            or experiment.heralds
            or experiment.in_heralds
        ):
            raise ValueError(
                "The provided experiment must be unitary, and must not have post-selection or heralding."
            )
        if experiment.min_photons_filter:
            warnings.warn(
                "The 'min_photons_filter' from the experiment is currently ignored.",
                UserWarning,
                stacklevel=2,
            )
