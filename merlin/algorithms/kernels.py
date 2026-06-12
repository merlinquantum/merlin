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
from ..core.state import StatePattern, generate_state
from ..core.state_vector import StateVector
from ..measurement.autodiff import AutoDiffProcess
from ..measurement.detectors import resolve_detectors
from ..measurement.photon_loss import resolve_photon_loss
from ..measurement.strategies import MeasurementStrategy
from ..pcvl_pytorch.locirc_to_tensor import CircuitConverter
from ..utils.deprecations import sanitize_parameters
from ..utils.dtypes import to_torch_dtype
from .layer import QuantumLayer
from .layer_utils import _build_simple_circuit
from .module import MerlinModule


def _require_angle_encoding_spec(
    angle_encoding_specs: dict[str, dict[str, object]],
    prefix: str | None,
) -> dict[str, object]:
    """Return validated angle-encoding metadata for a builder-backed prefix.

    Parameters
    ----------
    angle_encoding_specs : dict[str, dict[str, object]]
        Builder-derived angle-encoding metadata keyed by input prefix.
    prefix : str | None
        Input parameter prefix expected to have angle-encoding metadata.

    Returns
    -------
    dict[str, object]
        Metadata for ``prefix``.

    Raises
    ------
    RuntimeError
        If the metadata or its scale entries are missing or malformed.
    """
    if prefix is None:
        raise RuntimeError(
            "Builder-backed FeatureMap is missing an input parameter prefix for angle encoding."
        )

    spec = angle_encoding_specs.get(prefix)
    if spec is None:
        raise RuntimeError(
            "Builder-backed FeatureMap is missing angle_encoding_specs for input "
            f"prefix {prefix!r}. This metadata should be produced by "
            "CircuitBuilder.add_angle_encoding(...)."
        )

    combos = spec.get("combinations")
    if (
        not isinstance(combos, list)
        or not combos
        or not all(isinstance(combo, tuple) for combo in combos)
    ):
        raise RuntimeError(
            "Builder-backed FeatureMap has invalid angle_encoding_specs for input "
            f"prefix {prefix!r}: 'combinations' must be a non-empty list of tuples."
        )

    scales = spec.get("scales")
    if not isinstance(scales, dict):
        raise RuntimeError(
            "Builder-backed FeatureMap has invalid angle_encoding_specs for input "
            f"prefix {prefix!r}: 'scales' must be a dictionary."
        )

    feature_indices = sorted({idx for combo in combos for idx in combo})
    missing_scale_indices = [idx for idx in feature_indices if idx not in scales]
    if missing_scale_indices:
        raise RuntimeError(
            "Builder-backed FeatureMap is missing angle-encoding scale entries "
            f"for input prefix {prefix!r}: {missing_scale_indices}."
        )

    return spec


def _project_psd_matrix(matrix: Tensor) -> Tensor:
    """Project a symmetric matrix to the closest positive semi-definite matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        Symmetric matrix to project.

    Returns
    -------
    torch.Tensor
        Positive semi-definite projection of ``matrix``.
    """
    eigenvals, eigenvecs = torch.linalg.eigh(matrix)
    eigenvals = torch.diag(torch.where(eigenvals > 0, eigenvals, 0))
    return eigenvecs @ eigenvals @ eigenvecs.T


def _inputs_are_equal(x1: Tensor, x2: Tensor | None) -> bool:
    """Check whether two tensor batches represent the same inputs.

    Parameters
    ----------
    x1 : torch.Tensor
        First input batch.
    x2 : torch.Tensor | None
        Second input batch, if provided.

    Returns
    -------
    bool
        ``True`` when ``x2`` is omitted or both tensors have equal shape and
        values.
    """
    if x2 is None:
        return True
    if x1.shape != x2.shape:
        return False
    return torch.allclose(x1, x2)


class FeatureMap:
    """Quantum feature map.

    FeatureMap describes how classical data is embedded in a photonic circuit
    for quantum kernel methods.

    ``FidelityKernel`` treats this object as a descriptor. It passes the stored
    experiment, parameter prefixes, input size, dtype, and device to
    ``_CCInvQuantumLayer``, the internal adapter over the
    :class:`~merlin.algorithms.layer.QuantumLayer`
    backend. Legacy unitary-building state remains on ``FeatureMap`` only to
    support deprecated direct calls to :meth:`compute_unitary`.

    Parameters
    ----------
    circuit : pcvl.Circuit | None
        Pre-compiled Perceval circuit used to encode features.
    input_size : int | None
        Dimension of incoming classical data. Required.
    builder : CircuitBuilder | None
        Optional builder used to compile a circuit declaratively.
    experiment : pcvl.Experiment | None
        Optional experiment providing both the circuit and detector
        configuration. Exactly one of ``circuit``, ``builder``, or
        ``experiment`` must be supplied.
    input_parameters : str | list[str] | None
        Parameter prefix(es) that host the classical data.
    trainable_parameters : list[str] | None
        Optional trainable parameter prefixes.
    dtype : str | torch.dtype
        Torch dtype used when constructing the unitary.
    device : torch.device | None
        Torch device on which unitaries are evaluated.
    encoder : Callable[[torch.Tensor], torch.Tensor] | None
        Optional custom encoder used only by the deprecated
        :meth:`compute_unitary` path. ``FidelityKernel`` supports this argument
        for compatibility with a deprecation warning.
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
        encoder: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize a feature-map descriptor.

        Parameters
        ----------
        circuit : pcvl.Circuit | None
            Pre-compiled Perceval circuit used to encode features.
        input_size : int | None
            Dimension of incoming classical data. Required.
        builder : CircuitBuilder | None
            Optional builder used to compile a circuit declaratively.
        experiment : pcvl.Experiment | None
            Optional experiment providing both the circuit and detector
            configuration. Exactly one of ``circuit``, ``builder``, or
            ``experiment`` must be supplied.
        input_parameters : str | list[str] | None
            Parameter prefix or single-item prefix list hosting the classical
            data.
        trainable_parameters : list[str] | None
            Optional trainable parameter prefixes.
        dtype : str | torch.dtype
            Torch dtype used when constructing the unitary. Default is
            ``torch.float32``.
        device : torch.device | None
            Torch device on which unitaries are evaluated. If omitted, CPU is
            used.
        encoder : Callable[[torch.Tensor], torch.Tensor] | None
            Optional custom encoder used by the deprecated
            :meth:`compute_unitary` path when the raw input shape does not
            match the circuit parameter layout. ``FidelityKernel`` supports
            this argument for compatibility with a deprecation warning.

        Raises
        ------
        TypeError
            If ``input_size`` is omitted.
        ValueError
            If the circuit source, experiment configuration, or input parameter
            declaration is invalid.
        """
        builder_trainable: list[str] = []
        builder_input: list[str] = []

        self._angle_encoding_specs: dict[str, dict[str, object]] = {}
        self._requires_angle_encoding_specs = False
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
            self._requires_angle_encoding_specs = bool(self._angle_encoding_specs)
            resolved_circuit = builder.to_pcvl_circuit(pcvl)
            self.experiment = pcvl.Experiment(resolved_circuit)
        elif circuit is not None:
            resolved_circuit = circuit
            self.experiment = pcvl.Experiment(resolved_circuit)
        elif experiment is not None:
            _post_select_fn = experiment.post_select_fn
            _has_non_trivial_post_select = (
                _post_select_fn is not None and _post_select_fn != pcvl.PostSelect()
            )
            if (
                not experiment.is_unitary
                or _has_non_trivial_post_select
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
        # TODO: In release 0.5.x, remove FeatureMap.encoder support.
        self._encoder = encoder

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
        # Legacy compiler state kept only for deprecated compute_unitary.
        # FidelityKernel does not read this converter or training dictionary.
        self._training_dict: dict[str, torch.nn.Parameter] = {}
        for param_name in self.trainable_parameters:
            param_length = len(self._circuit_graph.spec_mappings[param_name])

            p = torch.rand(param_length, requires_grad=True)
            self._training_dict[param_name] = torch.nn.Parameter(p)

    def _px_len(self) -> int:
        """Return how many angle-encoding slots the deprecated converter expects.

        .. warning:: *Deprecated since version 0.4:*
            This helper belongs to the legacy ``FeatureMap.compute_unitary``
            path. ``FidelityKernel`` uses ``_CCInvQuantumLayer`` over the
            :class:`~merlin.algorithms.layer.QuantumLayer` backend and does
            not rely on this method.

        Returns
        -------
        int
            Number of angle-encoding slots expected by the legacy converter.
        """
        # TODO: In release 0.5.x, remove this legacy compute_unitary helper.
        return len(self._circuit_graph.spec_mappings.get(self.input_parameters, []))

    def _subset_sum_expand(self, x: Tensor, k: int) -> Tensor:
        """Expand an input vector into deterministic subset sums.

        .. warning:: *Deprecated since version 0.4:*
            This helper belongs to the legacy ``FeatureMap.compute_unitary``
            path. ``FidelityKernel`` uses ``_CCInvQuantumLayer`` over the
            :class:`~merlin.algorithms.layer.QuantumLayer` backend and does
            not rely on this method.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor expected to be one-dimensional.
        k : int
            Desired number of encoded features to return.

        Returns
        -------
        torch.Tensor
            Encoded tensor of length ``k`` on the configured device and dtype.
        """
        # TODO: In release 0.5.x, remove legacy subset expansion support.
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
        """Map raw features to the deprecated converter's parameter shape.

        .. warning:: *Deprecated since version 0.4:*
            This helper belongs to the legacy ``FeatureMap.compute_unitary``
            path. ``FidelityKernel`` uses ``_CCInvQuantumLayer`` over the
            :class:`~merlin.algorithms.layer.QuantumLayer` backend and does
            not rely on this method.

        Preference order:

        1. Builder-provided combination metadata (from :class:`CircuitBuilder`).
        2. A user-supplied encoder callable.
        3. The deterministic subset-sum expansion used by legacy feature maps.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor to be embedded.

        Returns
        -------
        torch.Tensor
            Encoded tensor matching the circuit's expected parameter length.
        """
        # TODO: In release 0.5.x, remove legacy encoder/subset/truncation support.
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

        .. warning:: *Deprecated since version 0.4:*
            This helper belongs to the legacy ``FeatureMap.compute_unitary``
            path. ``FidelityKernel`` uses ``_CCInvQuantumLayer`` over the
            :class:`~merlin.algorithms.layer.QuantumLayer` backend and does
            not rely on this method.

        Parameters
        ----------
        x : torch.Tensor
            Flattened input feature tensor.
        spec : dict[str, object]
            Metadata describing combinations and scales produced by the builder.

        Returns
        -------
        torch.Tensor
            Encoded tensor obeying the combination rules.
        """
        # TODO: In release 0.5.x, remove this legacy compute_unitary helper.
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

    @sanitize_parameters
    def compute_unitary(
        self,
        x: torch.Tensor | np.ndarray | float | int,
        *training_parameters: torch.Tensor,
    ) -> torch.Tensor:
        """Generate the circuit unitary after encoding `x` and applying trainables.

        .. warning:: *Deprecated since version 0.4:*
            ``compute_unitary`` is deprecated and will be removed in a future release.
            It uses legacy compiler state stored on ``FeatureMap``. Use
            :class:`FidelityKernel` for kernel computations; ``FidelityKernel``
            uses ``_CCInvQuantumLayer`` over the
            :class:`~merlin.algorithms.layer.QuantumLayer` backend
            and treats ``FeatureMap`` as a descriptor without relying on this
            method.

        Parameters
        ----------
        x : torch.Tensor | numpy.ndarray | float | int
            Single datapoint to embed; accepts scalars, NumPy arrays, or
            tensors.
        training_parameters : torch.Tensor
            Optional overriding trainable tensors.

        Returns
        -------
        torch.Tensor
            Complex unitary matrix representing the prepared circuit.

        Raises
        ------
        TypeError
            If ``x`` has an unsupported type.
        ValueError
            If angle encoding metadata is inconsistent with ``x``.
        """
        # TODO: In release 0.5.x, remove deprecated compute_unitary support.
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

    def is_datapoint(self, x: torch.Tensor | np.ndarray | float | int) -> bool:
        """Determine if ``x`` describes one sample or a batch.

        Parameters
        ----------
        x : torch.Tensor | numpy.ndarray | float | int
            Candidate input data.

        Returns
        -------
        bool
            ``True`` when ``x`` corresponds to a single datapoint.

        Raises
        ------
        ValueError
            If ``x`` cannot be reshaped into samples of size
            ``input_size``.
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
        # TODO: In release 0.5.x, remove the n_modes parameter.
        n_modes: int | None = None,
    ) -> "FeatureMap":
        """Simple factory method to create a FeatureMap with minimal configuration.

        The circuit uses ``n_modes = input_size + 1`` by default.

        Parameters
        ----------
        input_size : int
            Classical feature dimension. Maximum is 19.
        dtype : str | torch.dtype
            Target dtype for internal tensors.
        device : torch.device | None
            Optional torch device handle.
        angle_encoding_scale : float
            Global scaling applied to angle encoding features. Default is ``1.0``.
        n_modes : int | None
            .. warning:: *Deprecated since version 0.4:*
                Passing ``n_modes`` is deprecated and will be removed in
                release 0.5. The value is still honoured in 0.4, but in
                0.5 the mode count will be fixed to ``input_size + 1``
                and this parameter will be removed. Use
                :class:`~merlin.builder.circuit_builder.CircuitBuilder`
                directly if you need a different mode count.

        Returns
        -------
        FeatureMap
            Configured feature-map instance.

        Raises
        ------
        ValueError
            If ``input_size`` is outside the supported range.
        """
        # TODO: In release 0.5.x, remove n_modes handling and always use input_size + 1.
        if n_modes is None:
            n_modes = input_size + 1

        if n_modes < 2:
            raise ValueError(f"The number of modes must be at least 2, got {n_modes}")
        if input_size > 19 or n_modes > 20:
            raise ValueError(
                "Input size too large for the simple layer construction. For large inputs (with larger size than 20), please use the CircuitBuilder. Here is a quick tutorial on how to use it: https://merlinquantum.ai/quickstart/first_quantum_layer.html#circuitbuilder-walkthrough"
            )
        if input_size < 1:
            raise ValueError(f"input_size must be at least 1, got {input_size}")

        if input_size > n_modes:
            raise ValueError(ANGLE_ENCODING_MODE_ERROR)

        builder = _build_simple_circuit(input_size, n_modes, angle_encoding_scale)
        # input_parameters=None indicates that the builder's input layer is inferred by FeatureMap
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
    """Builder for creating photonic quantum kernel circuits.

    .. warning:: *Deprecated since version 0.4:*
        ``KernelCircuitBuilder`` is deprecated and will be removed in release 0.5.
        Use :class:`~merlin.builder.circuit_builder.CircuitBuilder` with
        :class:`FeatureMap` and :class:`FidelityKernel` directly instead::

            builder = CircuitBuilder(n_modes=3)
            builder.add_entangling_layer(name="U1")
            builder.add_angle_encoding(modes=[0, 1], name="input")
            builder.add_entangling_layer(name="U2")
            feature_map = FeatureMap(builder=builder, input_size=2)
            kernel = FidelityKernel(feature_map=feature_map, input_state=[1, 0, 1])

    This class provides a fluent interface for building quantum kernel circuits
    with various configurations, inspired by the core.layer architecture.
    """

    # TODO: In release 0.5.x, remove KernelCircuitBuilder.
    @sanitize_parameters
    def __init__(self) -> None:
        self._input_size: int | None = None
        self._n_modes: int | None = None
        self._n_photons: int | None = None
        self._dtype: str | torch.dtype = torch.float32
        self._device: torch.device | None = None
        self._angle_encoding_scale: float = 1.0
        self._trainable: bool = True
        self._trainable_prefix: str = "phi"

    def input_size(self, size: int) -> "KernelCircuitBuilder":
        """Set the input dimensionality.

        Parameters
        ----------
        size : int
            Number of classical features encoded by the kernel circuit.

        Returns
        -------
        KernelCircuitBuilder
            Builder instance for method chaining.
        """
        self._input_size = size
        return self

    def n_modes(self, modes: int) -> "KernelCircuitBuilder":
        """Set the number of modes in the circuit.

        Parameters
        ----------
        modes : int
            Number of photonic modes in the generated circuit.

        Returns
        -------
        KernelCircuitBuilder
            Builder instance for method chaining.
        """
        self._n_modes = modes
        return self

    def n_photons(self, photons: int) -> "KernelCircuitBuilder":
        """Set the number of photons.

        Parameters
        ----------
        photons : int
            Number of photons used when generating a default input state.

        Returns
        -------
        KernelCircuitBuilder
            Builder instance for method chaining.
        """
        self._n_photons = photons
        return self

    def trainable(
        self,
        enabled: bool = True,
        *,
        prefix: str = "phi",
    ) -> "KernelCircuitBuilder":
        """Enable or disable trainable rotations generated by the helper.

        Parameters
        ----------
        enabled : bool
            Whether trainable rotations are added to generated feature maps.
            Default is ``True``.
        prefix : str
            Parameter prefix used for trainable rotations when ``enabled`` is
            ``True``. Default is ``"phi"``.

        Returns
        -------
        KernelCircuitBuilder
            Builder instance for method chaining.
        """
        self._trainable = enabled
        if enabled:
            self._trainable_prefix = prefix
        return self

    def dtype(self, dtype: str | torch.dtype) -> "KernelCircuitBuilder":
        """Set the data type for computations.

        Parameters
        ----------
        dtype : str | torch.dtype
            Real dtype used by generated feature maps and kernels.

        Returns
        -------
        KernelCircuitBuilder
            Builder instance for method chaining.
        """
        self._dtype = dtype
        return self

    def device(self, device: torch.device) -> "KernelCircuitBuilder":
        """Set the computation device.

        Parameters
        ----------
        device : torch.device
            Device on which generated kernels evaluate tensors.

        Returns
        -------
        KernelCircuitBuilder
            Builder instance for method chaining.
        """
        self._device = device
        return self

    def angle_encoding(
        self,
        *,
        scale: float = 1.0,
    ) -> "KernelCircuitBuilder":
        """Configure the angle encoding scale.

        Parameters
        ----------
        scale : float
            Multiplicative scale applied to angle encoding features. Default is
            ``1.0``.

        Returns
        -------
        KernelCircuitBuilder
            Builder instance for method chaining.
        """
        self._angle_encoding_scale = scale
        return self

    # TODO: In release 0.5.x, remove KernelCircuitBuilder.build_feature_map.
    @sanitize_parameters
    def build_feature_map(self) -> FeatureMap:
        """Build and return a :class:`FeatureMap` instance.

        .. warning:: *Deprecated since version 0.4:*
            Use :class:`~merlin.builder.circuit_builder.CircuitBuilder` with
            :class:`FeatureMap` directly instead.

        Returns
        -------
        FeatureMap
            Configured feature map.

        Raises
        ------
        ValueError
            If required parameters are missing.
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

    # TODO: In release 0.5.x, remove KernelCircuitBuilder.build_fidelity_kernel.
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
        """Build and return a :class:`~merlin.algorithms.kernels.FidelityKernel` instance.

        .. warning:: *Deprecated since version 0.4:*
            Use :class:`~merlin.builder.circuit_builder.CircuitBuilder` with
            :class:`FeatureMap` and :class:`FidelityKernel` directly instead.

        Parameters
        ----------
        input_state : list[int] | None
            Input Fock state. If ``None``, it is generated automatically.
        shots : int
            Number of sampling shots. Default is ``0``.
        sampling_method : str
            Sampling method for pseudo-sampling. Default is
            ``"multinomial"``.
        computation_space : ComputationSpace | str | None
            Logical computation subspace; one of ``{"fock", "unbunched",
            "dual_rail"}``.
        force_psd : bool
            Whether to project to the nearest positive semi-definite matrix.
            Default is ``True``.

        Returns
        -------
        FidelityKernel
            Configured fidelity kernel.
        """
        feature_map = self.build_feature_map()

        # TODO: In release 0.5.x, remove KernelCircuitBuilder default
        # input_state generation and let FidelityKernel infer it directly.
        if input_state is None:
            n_modes = feature_map.circuit.m
            n_photons = self._n_photons or (self._input_size or 2)
            input_state = list(generate_state(n_modes, n_photons, StatePattern.SPACED))

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


class _CCInvQuantumLayer(QuantumLayer):
    """Internal ``QuantumLayer`` subclass used as the computation backend for ``FidelityKernel``.

    The name ``CCInv`` reflects two aspects of the computation: "CC" for
    ``CircuitConverter``, the component used to build the unitary tensor from
    the Perceval circuit; and "Inv" for the conjugate transpose (inverse)
    applied to the second operand when forming the kernel unitary
    ``U(x1) @ U†(x2)``.

    This layer owns encoding, unitary computation, SLOS simulation, photon
    loss, and detector transforms for the fidelity quantum kernel.
    ``FidelityKernel`` constructs one instance and delegates all circuit-level
    computation to it.

    Parameters
    ----------
    experiment : pcvl.Experiment
        Unitary Perceval experiment produced by the FeatureMap descriptor.
    input_state : list[int]
        Input Fock state occupation list.
    input_size : int
        Number of encoded circuit input parameters passed to
        :class:`~merlin.algorithms.layer.QuantumLayer`.
    raw_input_size : int
        Dimension of the raw classical input vector accepted by
        :class:`FidelityKernel`.
    input_parameters : list[str]
        Perceval parameter prefix(es) used for angle encoding.
    trainable_parameters : list[str]
        Perceval parameter prefixes exposed as trainable ``nn.Parameter``s.
    computation_space : ComputationSpace
        Computation space used for basis enumeration.
    dtype : torch.dtype
        Real dtype for internal tensors.
    device : torch.device
        Device on which computation graphs are placed.
    force_psd : bool
        Whether to project symmetric kernel matrices to the nearest positive
        semi-definite matrix.
    angle_encoding_specs : dict[str, dict[str, object]] | None
        Builder-derived angle-encoding metadata used to map raw feature
        tensors to encoded circuit parameters.
    requires_angle_encoding_specs : bool
        Whether missing builder angle-encoding metadata is an invalid internal
        state instead of a direct-circuit compatibility case.
    encoder : Callable[[torch.Tensor], torch.Tensor] | None
        Deprecated compatibility encoder copied from :class:`FeatureMap`.

    Notes
    -----
    ``_compute_unitary`` calls ``computation_process.converter.to_tensor``
    directly. This is safe for single-threaded use but will race under
    ``DataLoader(num_workers>0)`` because the converter holds shared mutable
    state. Use ``num_workers=0`` when the kernel module is in a DataLoader worker.
    """

    def __init__(
        self,
        experiment: pcvl.Experiment,
        input_state: list[int],
        input_size: int,
        raw_input_size: int,
        input_parameters: list[str],
        trainable_parameters: list[str],
        computation_space: ComputationSpace,
        dtype: torch.dtype,
        device: torch.device,
        force_psd: bool = True,
        angle_encoding_specs: dict[str, dict[str, object]] | None = None,
        requires_angle_encoding_specs: bool = False,
        encoder: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(
            experiment=experiment,
            input_state=input_state,
            input_size=input_size,
            input_parameters=input_parameters,
            trainable_parameters=trainable_parameters,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=computation_space,
            ),
            dtype=dtype,
            device=device,
        )
        self._raw_input_size = raw_input_size
        self._kernel_input_state = list(input_state)
        self.force_psd = force_psd
        self.angle_encoding_specs = angle_encoding_specs or {}
        self._requires_angle_encoding_specs = requires_angle_encoding_specs
        # TODO: In release 0.5.x, remove FeatureMap.encoder compatibility.
        self._encoder = encoder

        raw_keys = self._raw_output_keys
        try:
            self._input_state_index: int = raw_keys.index(tuple(input_state))
        except ValueError as exc:
            raise ValueError(
                "Input state is not present in the simulation basis produced by the circuit."
            ) from exc

        # Build the detection weight vector: track which output bin the input
        # state maps to after photon loss and detector transforms.
        weight_device = self.device or torch.device("cpu")
        one_hot = torch.zeros(len(raw_keys), dtype=self.dtype, device=weight_device)
        one_hot[self._input_state_index] = 1.0
        detection_vector = self._apply_detection_pipeline(one_hot)
        detection_vector = detection_vector.to(dtype=self.dtype, device=weight_device)

        nonzero = torch.nonzero(detection_vector > 1e-8, as_tuple=True)[0]
        self._input_detection_index: int | None = None
        if nonzero.numel() == 1 and torch.isclose(
            detection_vector[nonzero[0]],
            torch.tensor(
                1.0, dtype=detection_vector.dtype, device=detection_vector.device
            ),
            atol=1e-6,
        ):
            self._input_detection_index = int(nonzero[0].item())
        self.register_buffer("_input_detection_weights", detection_vector)
        self._kernel_autodiff_process = AutoDiffProcess()

    def _encode_single(self, x: Tensor) -> Tensor:
        """Encode one datapoint to the circuit's input parameter shape.

        Parameters
        ----------
        x : torch.Tensor
            Flat feature tensor of length ``input_size``.

        Returns
        -------
        torch.Tensor
            Encoded tensor matching the number of circuit input parameters.

        Raises
        ------
        ValueError
            If a compatibility encoder produces the wrong number of circuit
            input parameters.
        """
        x = x.to(dtype=self.dtype, device=self.device).reshape(-1)
        prefix = self.input_parameters[0] if self.input_parameters else None

        # Angle encoding specs (from CircuitBuilder) take priority.
        if self._requires_angle_encoding_specs:
            _require_angle_encoding_spec(self.angle_encoding_specs, prefix)
            return self._prepare_input_encoding(x, prefix)

        if prefix and self.angle_encoding_specs:
            spec = self.angle_encoding_specs.get(prefix)
            if spec:
                return self._prepare_input_encoding(x, prefix)

        # No CircuitBuilder metadata available: the circuit was constructed
        # directly with pcvl.Circuit or pcvl.Experiment. Preserve the legacy
        # FeatureMap encoding behavior for non-major releases.
        # TODO: In release 0.5.x, remove encoder/subset/truncation compatibility.
        spec_mappings = self.computation_process.converter.spec_mappings
        px_len = len(spec_mappings.get(prefix, [])) if prefix else x.numel()

        if x.numel() == px_len:
            return x
        if px_len == 0:
            return x[:0]
        if x.numel() < px_len:
            if self._encoder is not None:
                encoded = self._encoder(x)
                if isinstance(encoded, np.ndarray):
                    encoded = torch.from_numpy(encoded)
                encoded_tensor = torch.as_tensor(
                    encoded, dtype=self.dtype, device=self.device
                ).reshape(-1)
                if encoded_tensor.numel() != px_len:
                    raise ValueError(
                        "FeatureMap.encoder produced "
                        f"{encoded_tensor.numel()} parameters, but the circuit "
                        f"expects {px_len} parameters for prefix {prefix!r}."
                    )
                return encoded_tensor
            return self._subset_sum_expand(x, px_len)
        return x[:px_len]

    @staticmethod
    def _subset_sum_expand(x: Tensor, k: int) -> Tensor:
        """Expand a vector into deterministic subset sums up to length ``k``.

        Parameters
        ----------
        x : torch.Tensor
            One-dimensional input tensor.
        k : int
            Target output length.

        Returns
        -------
        torch.Tensor
            Tensor of length ``k`` containing subset sums, padded with zeros
            when fewer than ``k`` sums are available.
        """
        # TODO: In release 0.5.x, remove legacy subset expansion support.
        d = x.shape[0]
        vals: list[Tensor] = []
        for r in range(1, d + 1):
            for idxs in itertools.combinations(range(d), r):
                vals.append(x[list(idxs)].sum())
                if len(vals) == k:
                    return torch.stack(vals, dim=0)
        if len(vals) == 0:
            return torch.zeros(k, dtype=x.dtype, device=x.device)
        pad = k - len(vals)
        return torch.cat(
            [
                torch.stack(vals, dim=0),
                torch.zeros(pad, dtype=x.dtype, device=x.device),
            ],
            dim=0,
        )

    def _apply_detection_pipeline(self, distribution: Tensor) -> Tensor:
        """Apply photon-loss and detector transforms in sequence.

        This is the single canonical place where the two-step detection
        pipeline is applied. Both the detection-weight initialisation in
        ``__init__`` and the per-batch path in ``_compute_transition_probs``
        call this method so that adding a future step only requires one change.

        Parameters
        ----------
        distribution : torch.Tensor
            Probability distribution tensor; either 1-D (single output
            vector) or 2-D ``(batch, bins)``.

        Returns
        -------
        torch.Tensor
            Distribution after photon-loss and detector transforms, with
            any trailing batch dimension squeezed away if the input was 1-D.
        """
        result = self._apply_photon_loss_transform(distribution)
        if result.ndim > 1 and distribution.ndim == 1:
            result = result.squeeze(0)
        result = self._apply_detector_transform(result)
        if result.ndim > 1 and distribution.ndim == 1:
            result = result.squeeze(0)
        return result

    def _compute_unitary(self, x_enc: Tensor) -> Tensor:
        """Evaluate the circuit unitary for an already-encoded input.

        Parameters
        ----------
        x_enc : torch.Tensor
            Encoded input tensor produced by :meth:`_encode_single`.

        Returns
        -------
        torch.Tensor
            Complex unitary matrix of shape ``(m, m)``.
        """
        return self.computation_process.converter.to_tensor(*self.thetas, x_enc)

    def _compute_kernel_unitary(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Compute the combined kernel unitary ``U(x1) @ U†(x2)``.

        Parameters
        ----------
        x1 : torch.Tensor
            First raw feature tensor.
        x2 : torch.Tensor
            Second raw feature tensor.

        Returns
        -------
        torch.Tensor
            Combined kernel unitary of shape ``(m, m)``.
        """
        U1 = self._compute_unitary(self._encode_single(x1))
        U2 = self._compute_unitary(self._encode_single(x2))
        return U1 @ U2.conj().mT

    def _compute_unitary_batch(self, x_batch: Tensor) -> Tensor:
        """Compute a batch of circuit unitaries.

        Parameters
        ----------
        x_batch : torch.Tensor
            Batch of feature tensors with shape ``(N, input_size)``.

        Returns
        -------
        torch.Tensor
            Stacked unitary tensor of shape ``(N, m, m)``.
        """
        # Serial loop is intentional: CircuitConverter.to_tensor holds shared
        # mutable state and is not safe to call concurrently.
        return torch.stack([
            self._compute_unitary(self._encode_single(x)) for x in x_batch
        ])

    def _compute_transition_probs(
        self,
        all_circuits: Tensor,
        input_state: list[int],
        shots: int,
        sampling_method: str,
    ) -> Tensor:
        """Run SLOS and transforms on a batch of combined kernel unitaries.

        Parameters
        ----------
        all_circuits : torch.Tensor
            Batch of combined kernel unitaries with shape ``(P, m, m)``.
        input_state : list[int]
            Input Fock occupation list.
        shots : int
            Number of pseudo-sampling shots; 0 for exact probabilities.
        sampling_method : str
            Sampling method; one of ``"multinomial"``, ``"binomial"``,
            ``"gaussian"``.

        Returns
        -------
        torch.Tensor
            Transition probability for each circuit, shape ``(P,)``.
        """
        _, probabilities = self.computation_process.simulation_graph.compute_probs(
            all_circuits, input_state
        )
        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(0)
        probabilities = probabilities.to(dtype=self.dtype)
        detection_probs = self._apply_detection_pipeline(probabilities)

        if shots > 0:
            detection_probs = self._kernel_autodiff_process.sampling_noise.pcvl_sampler(
                detection_probs, shots, sampling_method
            )

        if self._input_detection_index is not None:
            return detection_probs[:, self._input_detection_index]
        input_detection_weights = cast(Tensor, self._input_detection_weights)
        weights = input_detection_weights.to(
            dtype=detection_probs.dtype, device=detection_probs.device
        )
        return detection_probs @ weights

    def forward(
        self,
        *input_parameters: Tensor | StateVector,
        shots: int | None = None,
        sampling_method: str | None = None,
        simultaneous_processes: int | None = None,
    ) -> Tensor:
        """Compute a fidelity-kernel matrix from raw feature batches.

        Parameters
        ----------
        input_parameters : torch.Tensor | merlin.core.state_vector.StateVector
            One or two raw feature batches. The first tensor has shape
            ``(N, raw_input_size)``. The optional second tensor has shape
            ``(M, raw_input_size)``. If omitted, a symmetric training Gram
            matrix for the first tensor is returned.
        shots : int | None
            Number of pseudo-sampling shots. If omitted, exact probabilities
            are used.
        sampling_method : str | None
            Sampling method used when ``shots`` is positive. If omitted,
            ``"multinomial"`` is used.
        simultaneous_processes : int | None
            Present for signature compatibility with
            :class:`~merlin.algorithms.layer.QuantumLayer`; currently ignored.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape ``(N, N)`` when ``x2`` is omitted, otherwise
            shape ``(N, M)``.

        Raises
        ------
        TypeError
            If inputs are not tensors.
        ValueError
            If zero or more than two input tensors are provided.
        """
        if not 1 <= len(input_parameters) <= 2:
            raise ValueError(
                "_CCInvQuantumLayer.forward expects one or two tensor inputs."
            )

        x1 = input_parameters[0]
        if not isinstance(x1, Tensor):
            raise TypeError("_CCInvQuantumLayer.forward expects tensor inputs.")

        x2: Tensor | None = None
        if len(input_parameters) == 2:
            x2_candidate = input_parameters[1]
            if not isinstance(x2_candidate, Tensor):
                raise TypeError("_CCInvQuantumLayer.forward expects tensor inputs.")
            x2 = x2_candidate

        effective_shots = 0 if shots is None else shots
        effective_sampling_method = sampling_method or "multinomial"
        equal_inputs = _inputs_are_equal(x1, x2)
        U_forward = self._compute_unitary_batch(x1).to(x1.device)

        len_x1 = len(x1)
        if x2 is not None:
            U_adjoint = (
                self._compute_unitary_batch(x2).conj().transpose(1, 2).to(x1.device)
            )
            all_circuits = U_forward.unsqueeze(1) @ U_adjoint.unsqueeze(0)
            all_circuits = all_circuits.view(-1, *all_circuits.shape[2:])
        else:
            if len_x1 < 2:
                return torch.ones(
                    len_x1,
                    len_x1,
                    dtype=self.dtype,
                    device=x1.device,
                )

            U_adjoint = U_forward.conj().transpose(1, 2)
            upper_idx = torch.triu_indices(
                len_x1,
                len_x1,
                offset=1,
                device=x1.device,
            )
            all_circuits = U_forward[upper_idx[0]] @ U_adjoint[upper_idx[1]]

        transition_probs = self._compute_transition_probs(
            all_circuits,
            self._kernel_input_state,
            effective_shots,
            effective_sampling_method,
        )

        if x2 is None:
            kernel_matrix = torch.zeros(
                len_x1, len_x1, dtype=self.dtype, device=x1.device
            )
            upper_idx = upper_idx.to(x1.device)
            transition_probs = transition_probs.to(dtype=self.dtype, device=x1.device)
            kernel_matrix[upper_idx[0], upper_idx[1]] = transition_probs
            kernel_matrix[upper_idx[1], upper_idx[0]] = transition_probs
            kernel_matrix.fill_diagonal_(1)

            if self.force_psd:
                kernel_matrix = _project_psd_matrix(kernel_matrix)
            return kernel_matrix

        transition_probs = transition_probs.to(dtype=self.dtype, device=x1.device)
        kernel_matrix = transition_probs.reshape(len_x1, len(x2))

        if self.force_psd and equal_inputs:
            kernel_matrix = 0.5 * (kernel_matrix + kernel_matrix.T)
            kernel_matrix = _project_psd_matrix(kernel_matrix)

        return kernel_matrix


class FidelityKernel(MerlinModule):
    r"""
    Fidelity Quantum Kernel

    Next-release deprecations for the legacy ``FeatureMap`` unitary path:

    - ``FeatureMap.compute_unitary``
    - ``FeatureMap._px_len``
    - ``FeatureMap._encode_x``
    - ``FeatureMap._encode_with_specs``
    - ``FeatureMap._subset_sum_expand``

    ``FidelityKernel`` does not use these methods. It delegates kernel
    computation to ``_CCInvQuantumLayer``, which uses the
    :class:`~merlin.algorithms.layer.QuantumLayer` backend.

    For a given input Fock state, :math:`|s \rangle` and feature map,
    :math:`U`, the fidelity quantum kernel estimates the following inner
    product using SLOS:

    .. math::
        |\langle s | U^{\dagger}(x_2) U(x_1) | s \rangle|^{2}

    Transition probabilities are computed in parallel for each pair of
    datapoints in the input datasets.

    Parameters
    ----------
    feature_map : FeatureMap
        Feature map object that encodes a given datapoint within its circuit.
    input_state : list[int] | None
        Input Fock state occupation list. If ``None``, the state is derived
        from ``n_photons`` when given, otherwise defaults to an alternating
        single-photon state ``[1, 0, 1, 0, ...]`` of length
        ``feature_map.circuit.m``.
    n_photons : int | None
        Number of photons to place in the input state when ``input_state`` is
        ``None``. If ``n_photons <= ceil(m / 2)`` (where ``m`` is the number of
        circuit modes), photons are spread in the alternating pattern
        ``[1, 0, 1, 0, ...]``; otherwise all alternating positions are filled
        first and then remaining positions are filled left to right
        (e.g. 4 photons in 6 modes → ``[1, 1, 1, 0, 1, 0]``), and a
        ``UserWarning`` is emitted.  If ``input_state`` is also provided,
        ``sum(input_state)`` must equal ``n_photons``, otherwise a
        ``ValueError`` is raised.  Default: ``None``.
    shots : int | None
        Number of circuit shots. If ``None``, the exact transition
        probabilities are returned. Default: ``None``.
    sampling_method : str
        Probability distributions are post-processed with a pseudo-sampling
        method: ``"multinomial"``, ``"binomial"``, or ``"gaussian"``.
        Default is ``"multinomial"``.
    computation_space : ComputationSpace | str | None
        Logical computation subspace; one of
        ``{"fock", "unbunched", "dual_rail"}``. Default: ``FOCK``.
    force_psd : bool
        Projects the training kernel matrix to the closest positive
        semi-definite matrix. Default is ``True``.
    device : torch.device | None
        Device on which to perform SLOS.
    dtype : str | torch.dtype | None
        Datatype with which to perform SLOS.

    Examples
    --------
    For a given training and test datasets, one can construct the training and
    test kernel matrices with the following structure:

    .. code-block:: python

        circuit = Circuit(2) // PS(P("X0")) // BS() // PS(P("X1")) // BS()
        feature_map = FeatureMap(circuit, ["X"])

        quantum_kernel = FidelityKernel(
            feature_map,
            input_state=[0, 4],
        )
        K_train = quantum_kernel(X_train)
        K_test = quantum_kernel(X_test, X_train)

    Use with scikit-learn for kernel-based machine learning:

    .. code-block:: python

        from sklearn.svm import SVC

        svc = SVC(kernel="precomputed")
        svc.fit(K_train, y_train)
        y_pred = svc.predict(K_test)

    .. warning::
        ``FidelityKernel`` is **not thread-safe**. The internal
        ``CircuitConverter`` holds shared mutable state. Using this module
        inside a ``DataLoader`` with ``num_workers > 0`` will produce a data
        race. Always set ``num_workers=0`` when the kernel is used as part of
        a DataLoader worker.
    """

    @sanitize_parameters
    def __init__(
        self,
        feature_map: FeatureMap,
        input_state: list[int] | None = None,
        *,
        n_photons: int | None = None,
        shots: int | None = None,
        sampling_method: str = "multinomial",
        computation_space: ComputationSpace | str | None = None,
        force_psd: bool = True,
        device: torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> None:
        """Initialize a fidelity quantum kernel.

        Parameters
        ----------
        feature_map : FeatureMap
            Feature-map descriptor that provides the circuit or experiment,
            parameter prefixes, input size, dtype, and device.
        input_state : list[int] | None
            Input Fock state occupation list. If ``None``, the state is derived
            from ``n_photons`` when given, otherwise defaults to an alternating
            single-photon state ``[1, 0, 1, 0, ...]`` of length
            ``feature_map.circuit.m``.
        n_photons : int | None
            Number of photons used to derive ``input_state`` when
            ``input_state`` is ``None``.  Must satisfy
            ``1 <= n_photons <= feature_map.circuit.m``. If
            ``n_photons <= ceil(m / 2)``, an alternating state is produced;
            otherwise all alternating positions are filled first, then the
            remaining positions are filled left to right, and a ``UserWarning``
            is emitted.
            If ``input_state`` is also provided, its photon count must equal
            ``n_photons``.  Default is ``None``.
        shots : int | None
            Number of pseudo-sampling shots. If omitted or ``None``, exact
            probabilities are used. Default is ``None``.
        sampling_method : str
            Pseudo-sampling method used when ``shots`` is positive. Default is
            ``"multinomial"``.
        computation_space : ComputationSpace | str | None
            Logical computation subspace. If omitted, ``ComputationSpace.FOCK``
            is used.
        force_psd : bool
            Whether training kernel matrices are projected to the nearest
            positive semi-definite matrix. Default is ``True``.
        device : torch.device | None
            Device on which the kernel backend evaluates tensors. If omitted,
            the feature map device is used.
        dtype : str | torch.dtype | None
            Real dtype used by the kernel backend. If omitted, the feature map
            dtype is used.

        Raises
        ------
        ValueError
            If the input state, experiment, circuit size, or computation space
            is incompatible with fidelity-kernel evaluation.
        RuntimeError
            If detector transforms are combined with a non-FOCK computation
            space.
        """
        super().__init__()
        if computation_space is None:
            computation_space = ComputationSpace.FOCK
        else:
            computation_space = ComputationSpace.coerce(computation_space)
        self.computation_space = computation_space
        self.feature_map = feature_map
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
        backend_input_size = self._resolve_backend_input_size()

        m = self.feature_map.circuit.m
        # Validate that the provided input state and n_photons are compatible if both are given
        if input_state is not None and n_photons is not None:
            if sum(input_state) != n_photons:
                raise ValueError(
                    f"n_photons={n_photons} does not match the photon count "
                    f"{sum(input_state)} of the provided input_state."
                )
        # We infer the input states if not given
        elif input_state is None:
            # If n_photons is not given, default to all alternating positions.
            if n_photons is None:
                input_state = [1 if i % 2 == 0 else 0 for i in range(m)]
            # Otherwise, we generate the input state based on n_photons and m
            else:
                # Validation of n_photons
                if n_photons <= 0 or n_photons > m:
                    raise ValueError(
                        f"n_photons must be between 1 and {m} (the number of "
                        f"circuit modes), got {n_photons}."
                    )
                alternating_slot_count = (m + 1) // 2
                if n_photons <= alternating_slot_count:
                    state = [0] * m
                    for i in range(n_photons):
                        state[2 * i] = 1
                    input_state = state
                # More photons than alternating positions: fill them first,
                # then continue filling remaining positions left to right.
                else:
                    warnings.warn(
                        f"n_photons={n_photons} exceeds the {alternating_slot_count} "
                        "available alternating positions. Alternating positions "
                        "are filled first, then remaining positions are filled "
                        "left to right, which may not correspond to a physically "
                        "realistic hardware configuration.",
                        UserWarning,
                        stacklevel=2,
                    )
                    state = [1 if i % 2 == 0 else 0 for i in range(m)]
                    remaining = n_photons - sum(state)
                    for i in range(1, m, 2):
                        if remaining <= 0:
                            break
                        state[i] = 1
                        remaining -= 1
                    input_state = state

        if self.feature_map.circuit.m != len(input_state):
            raise ValueError("Input state length does not match circuit size.")

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

        m = len(input_state)
        _detectors, _empty_detectors = resolve_detectors(self.experiment, m)

        # Verify that no Detector was defined in experiment if using non-FOCK space.
        if not _empty_detectors and self.computation_space is not ComputationSpace.FOCK:
            raise RuntimeError(
                "computation_space must be FOCK if Experiment contains at least one Detector."
            )

        # Resolve noise model presence from the experiment before building the backend.
        _, empty_noise_model = resolve_photon_loss(self.experiment, m)
        self.has_custom_noise_model = not empty_noise_model

        self._quantum_layer = _CCInvQuantumLayer(
            experiment=self.experiment,
            input_state=input_state,
            input_size=backend_input_size,
            raw_input_size=self.input_size,
            input_parameters=[self.feature_map.input_parameters],
            trainable_parameters=self.feature_map.trainable_parameters,
            computation_space=self.computation_space,
            dtype=self.dtype,
            device=self.device,
            force_psd=self.force_psd,
            angle_encoding_specs=self.feature_map._angle_encoding_specs,
            requires_angle_encoding_specs=(
                self.feature_map._requires_angle_encoding_specs
            ),
            encoder=self.feature_map._encoder,
        )

        self.is_trainable = feature_map.is_trainable

    @property
    def input_state(self) -> list[int]:
        """Input Fock state occupation list used for kernel evaluation.

        Returns
        -------
        list[int]
            Copy of the input Fock state used by the kernel backend.
        """
        return list(self._quantum_layer._kernel_input_state)

    def forward(
        self,
        x1: float | np.ndarray | torch.Tensor,
        x2: float | np.ndarray | torch.Tensor | None = None,
    ) -> float | Tensor:
        """Calculate the quantum kernel for input data ``x1`` and ``x2``.

        If ``x1`` and ``x2`` are datapoints, a scalar value is returned. For
        input datasets the kernel matrix is computed.

        Parameters
        ----------
        x1 : float | numpy.ndarray | torch.Tensor
            First input datapoint or dataset.
        x2 : float | numpy.ndarray | torch.Tensor | None
            Second input datapoint or dataset. If omitted, the training kernel
            matrix for ``x1`` is computed.

        Returns
        -------
        float | torch.Tensor
            Scalar kernel value for datapoints, or a kernel matrix for
            datasets.

        Raises
        ------
        TypeError
            If ``x2`` cannot be converted to a tensor when provided.
        ValueError
            If scalar datapoints are passed without ``x2`` or if input shapes
            are incompatible with the feature-map input size.
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
            if not isinstance(x2, torch.Tensor):
                x2 = torch.as_tensor(x2, dtype=self.dtype, device=self.device)

            x1_batch = torch.as_tensor(
                x1, dtype=self.dtype, device=self.device
            ).reshape(1, self.input_size)
            x2_batch = torch.as_tensor(
                x2, dtype=self.dtype, device=self.device
            ).reshape(1, self.input_size)
            self._quantum_layer.force_psd = self.force_psd
            kernel_matrix = self._quantum_layer(
                x1_batch,
                x2_batch,
                shots=self.shots,
                sampling_method=self.sampling_method,
            )
            return float(kernel_matrix.reshape(-1)[0].item())

        # Ensure tensors before reshaping (satisfies mypy)
        if x2 is not None and not isinstance(x2, torch.Tensor):
            x2 = torch.as_tensor(x2, dtype=self.dtype, device=self.device)

        if isinstance(x2, torch.Tensor) or x2 is None:
            x1 = x1.reshape(-1, self.input_size)
            x2 = x2.reshape(-1, self.input_size) if x2 is not None else None
        else:
            raise TypeError("x2 is not None nor torch.Tensor")

        self._quantum_layer.force_psd = self.force_psd
        if x2 is None:
            return self._quantum_layer(
                x1,
                shots=self.shots,
                sampling_method=self.sampling_method,
            )

        return self._quantum_layer(
            x1,
            x2,
            shots=self.shots,
            sampling_method=self.sampling_method,
        )

    # TODO: In release 0.5.x, remove FidelityKernel.simple.
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
        # TODO: In release 0.5.x, remove the n_modes parameter.
        n_modes: int | None = None,
    ) -> "FidelityKernel":
        """Create a simple fidelity kernel with minimal configuration.

        .. warning:: *Deprecated since version 0.4:*
            This factory method is deprecated and will be removed in release 0.5.
            Build a feature map with :meth:`FeatureMap.simple` and pass it
            directly to :class:`FidelityKernel`.

        Parameters
        ----------
        input_size : int
            Classical feature dimension.
        shots : int
            Number of pseudo-sampling shots. Default is ``0``.
        sampling_method : str
            Sampling method used when ``shots`` is positive. Default is
            ``"multinomial"``.
        computation_space : ComputationSpace | str | None
            Logical computation subspace.
        force_psd : bool
            Whether to project the training kernel matrix to the nearest
            positive semi-definite matrix. Default is ``True``.
        dtype : str | torch.dtype
            Target dtype for internal tensors. Default is ``torch.float32``.
        device : torch.device | None
            Device on which to execute computations.
        angle_encoding_scale : float
            Global scaling applied to angle encoding features. Default is ``1.0``.
        n_modes : int | None
            .. warning:: *Deprecated since version 0.4:*
                Passing ``n_modes`` is deprecated and will be removed in
                release 0.5. The value is still honoured in 0.4, but in
                0.5 the mode count will be fixed to ``input_size + 1``
                and this parameter will be removed. Use
                :class:`~merlin.builder.circuit_builder.CircuitBuilder`
                directly if you need a different mode count.

        Returns
        -------
        FidelityKernel
            Configured fidelity kernel.

        Raises
        ------
        ValueError
            If the generated feature map or input state is incompatible with
            fidelity-kernel evaluation.
        RuntimeError
            If the generated experiment configuration is incompatible with the
            requested computation space.
        """
        # TODO: In release 0.5.x, remove n_modes handling; always use input_size + 1.
        state_size = n_modes if n_modes is not None else input_size + 1

        if n_modes is None:
            feature_map = FeatureMap.simple(
                input_size=input_size,
                dtype=dtype,
                device=device,
                angle_encoding_scale=angle_encoding_scale,
            )
        else:
            feature_map = FeatureMap.simple(
                input_size=input_size,
                n_modes=n_modes,
                dtype=dtype,
                device=device,
                angle_encoding_scale=angle_encoding_scale,
            )

        input_state = state_size * [0]
        for i in range(state_size):
            if i % 2 == 0:
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
        return _project_psd_matrix(matrix)

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

    def _resolve_backend_input_size(self) -> int:
        """Resolve the encoded input size required by the kernel backend.

        Returns
        -------
        int
            Number of encoded circuit input parameters expected by the
            kernel backend.

        Warns
        -----
        DeprecationWarning
            If ``FeatureMap.encoder`` or direct-circuit subset/truncation
            compatibility is required for the kernel path.
        """
        # TODO: In release 0.5.x, remove legacy kernel encoding compatibility.
        if self.feature_map._requires_angle_encoding_specs:
            _require_angle_encoding_spec(
                self.feature_map._angle_encoding_specs,
                self.feature_map.input_parameters,
            )

        spec_mappings = self.feature_map._circuit_graph.spec_mappings
        input_parameter_count = len(
            spec_mappings.get(self.feature_map.input_parameters, [])
        )

        has_compatibility_encoder = self.feature_map._encoder is not None
        if has_compatibility_encoder:
            warnings.warn(
                "FeatureMap.encoder support inside FidelityKernel is deprecated "
                "and will be removed in a future release. Migrate by either "
                "expressing the encoding with CircuitBuilder.add_angle_encoding(...) "
                "or pre-encoding the data before the kernel call, then construct "
                "FeatureMap with input_size equal to the encoded circuit-parameter "
                "count.",
                DeprecationWarning,
                stacklevel=3,
            )

        if self.feature_map._angle_encoding_specs:
            return input_parameter_count

        if self.input_size == input_parameter_count:
            return input_parameter_count
        if has_compatibility_encoder:
            return input_parameter_count

        warnings.warn(
            "FidelityKernel support for direct pcvl.Circuit or pcvl.Experiment "
            "feature maps whose input_size differs from the circuit input "
            "parameter count is deprecated and will be removed in a future "
            "release. Received input_size="
            f"{self.input_size}, but input prefix "
            f"{self.feature_map.input_parameters!r} maps to "
            f"{input_parameter_count} circuit input parameters. Migrate by either "
            "expressing the raw-feature encoding with "
            "CircuitBuilder.add_angle_encoding(...) or pre-encoding the data "
            "before the kernel call, then construct FeatureMap with input_size "
            "equal to the encoded circuit-parameter count.",
            DeprecationWarning,
            stacklevel=3,
        )
        return input_parameter_count

    @staticmethod
    def _validate_experiment(experiment: pcvl.Experiment) -> None:
        """Validate that the provided experiment is compatible with fidelity kernels."""
        post_select_fn = experiment.post_select_fn
        has_non_trivial_post_select = (
            post_select_fn is not None and post_select_fn != pcvl.PostSelect()
        )
        if (
            not experiment.is_unitary
            or has_non_trivial_post_select
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
