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

# Acknowledgements
# ----------------
# This implementation is inspired by the quantum optical reservoir computing
# framework introduced in:
# Sakurai, Akitada, et al. "Quantum optical reservoir computing powered by
# boson sampling." Optica Quantum 3.3 (2025): 238-245.


"""Reservoir classifier built on top of a frozen Merlin QuantumLayer."""

from __future__ import annotations

import hashlib
import warnings
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from sklearn.base import clone
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import TensorDataset

from ..algorithms.layer import QuantumLayer
from ..algorithms.module import MerlinModule
from ..measurement.strategies import MeasurementStrategy


class _ReservoirLayerProxy:
    """Public facade that delegates to the current QuantumLayer instance.

    Assigning selected attributes triggers a rebuild of the underlying
    reservoir layer so the external API stays simple:
    ``reservoir.layer.n_modes = ...``,
    ``reservoir.layer.measurement_strategy = ...``,
    ``reservoir.layer.noise = ...``.
    """

    def __init__(self, parent: ReservoirClassifier) -> None:
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name: str):
        return getattr(self._parent._quantum_layer, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_parent":
            object.__setattr__(self, name, value)
            return
        if name == "n_modes":
            self._parent._set_layer_n_modes(value)
            return
        if name == "measurement_strategy":
            self._parent._set_layer_measurement_strategy(value)
            return
        if name == "noise":
            self._parent._set_layer_noise(value)
            return
        setattr(self._parent._quantum_layer, name, value)

    def __call__(self, *args: Any, **kwargs: Any):
        return self._parent._quantum_layer(*args, **kwargs)

    @property
    def n_modes(self) -> int:
        return self._parent._quantum_layer.circuit.m

    @property
    def measurement_strategy(self) -> Any:
        return self._parent._quantum_layer.measurement_strategy

    @property
    def noise(self) -> Any | None:
        return self._parent._noise


class ReservoirClassifier(MerlinModule):
    """Frozen photonic reservoir with a trainable linear readout.

    The quantum reservoir is initialized once from a Haar-random interferometer
    and kept frozen afterwards. Training is therefore split in two stages:
    :meth:`fit_reservoir` fits the reservoir preprocessing, and
    :meth:`transform_reservoir` computes standardized quantum embeddings.
    Downstream optimization only updates the classical readout returned by
    :meth:`parameters`.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        n_photons: int,
        reduction: Any | None = None,
        concatenate: bool = True,
        cache: bool = True,
        seed: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize a frozen photonic reservoir classifier.

        Parameters
        ----------
        in_features : int
            Number of classical input features expected per sample before any
            optional dimensionality reduction.
        out_features : int
            Number of output logits produced by the linear readout.
        n_photons : int
            Number of photons injected in the default reservoir input state.
        reduction : object|None
            Optional scikit-learn decomposition estimator used to compress the
            input before encoding it in the reservoir. The estimator must expose
            ``fit`` and ``transform`` and define ``n_components`` at
            construction time.
        concatenate : bool
            If ``True``, concatenate the raw classical inputs to the normalized
            reservoir features before the readout. If ``False``, the readout
            only sees reservoir features.
        cache : bool
            If ``True``, cache the fitted training-set reservoir features so
            repeated access to the same data avoids recomputing the frozen
            quantum layer. If ``False``, :meth:`fit_reservoir` skips the
            quantum pass and reservoir embeddings are recomputed on demand.
        seed : int|None
            Optional random seed used for the Haar-random unitary and the lazy
            readout initialization.
        device : torch.device|None
            Torch device hosting the readout parameters and the reservoir
            feature tensors.
        dtype : torch.dtype|None
            Floating-point dtype used for the readout parameters and the
            reservoir feature tensors.

        Raises
        ------
        ValueError
            If ``in_features``, ``out_features``, or ``n_photons`` is not a
            strictly positive integer, or if ``reduction.n_components`` is not
            a strictly positive integer.
        TypeError
            If ``reduction`` is not ``None`` and does not behave like a
            scikit-learn decomposition estimator.
        """
        super().__init__()

        self._validate_positive_integer(in_features, "in_features")
        self._validate_positive_integer(out_features, "out_features")
        self._validate_positive_integer(n_photons, "n_photons")

        resolved_device, resolved_dtype, _ = MerlinModule.setup_device_and_dtype(
            device, dtype
        )
        # Casting to the correct types early to ensure consistent behavior and error handling
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_photons = int(n_photons)
        self.concatenate = bool(concatenate)
        self.cache = bool(cache)
        self.seed = None if seed is None else int(seed)
        self.device = resolved_device or torch.device("cpu")
        self.dtype = resolved_dtype
        # The reduction needs to be validated and stored before inferring
        # the logical encoded feature width.
        self._reduction_template = self._validate_reduction(reduction)
        self.reduction = (
            clone(self._reduction_template)
            if self._reduction_template is not None
            else None
        )
        self.encoded_input_features = self._infer_encoded_input_features()
        self.quantum_input_features = self.encoded_input_features + 1
        self._warn_for_configuration()

        self._unitary_matrix = self._draw_unitary(
            self.quantum_input_features,
            self.seed,
        )
        self._measurement_strategy = MeasurementStrategy.probs()
        self._noise = None
        self._quantum_layer = self._build_layer(
            unitary_matrix=self._unitary_matrix,
            encoded_input_size=self.encoded_input_features,
            n_modes=self.quantum_input_features,
            measurement_strategy=self._measurement_strategy,
            noise=self._noise,
        )
        self.layer = _ReservoirLayerProxy(self)
        self.readout: nn.LazyLinear | nn.Linear
        self.readout = nn.LazyLinear(out_features, dtype=self.dtype)
        self.readout.to(device=self.device, dtype=self.dtype)

        self._is_fitted = False
        self._input_min: float | None = None
        self._input_max: float | None = None
        self._quantum_mean: torch.Tensor | None = None
        self._quantum_std: torch.Tensor | None = None
        self._fit_fingerprint: str | None = None
        self._fit_quantum_cache: torch.Tensor | None = None

    def to(self, *args: Any, **kwargs: Any) -> ReservoirClassifier:
        """Move the classifier and quantum-layer runtime state.

        Parameters
        ----------
        *args : tuple[Any, ...]
            Positional arguments forwarded to :meth:`torch.nn.Module.to` and
            the :class:`merlin.algorithms.layer.QuantumLayer`'s to method.
        **kwargs : dict[str, Any]
            Keyword arguments forwarded to :meth:`torch.nn.Module.to` and
            the :class:`merlin.algorithms.layer.QuantumLayer`'s to method.

        Returns
        -------
        ReservoirClassifier
            Updated classifier instance.

        Raises
        ------
        ValueError
            If the requested dtype is not supported by Merlin modules.
        """
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")

        if args:
            first_arg = args[0]
            if isinstance(first_arg, torch.dtype):
                dtype = first_arg if dtype is None else dtype
            elif isinstance(first_arg, (torch.device, str)):
                device = first_arg if device is None else device
            elif isinstance(first_arg, torch.Tensor):
                if device is None:
                    device = first_arg.device
                if dtype is None:
                    dtype = first_arg.dtype

        if len(args) > 1 and isinstance(args[1], torch.dtype) and dtype is None:
            dtype = args[1]

        if dtype is not None:
            MerlinModule.setup_device_and_dtype(None, dtype)

        super().to(*args, **kwargs)
        self._quantum_layer.to(*args, **kwargs)

        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            _, self.dtype, _ = MerlinModule.setup_device_and_dtype(None, dtype)

        return self

    @staticmethod
    def _validate_positive_integer(value: Any, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def _validate_reduction(reduction: Any | None) -> Any | None:
        # the reduction technique needs to be a scikit-learn style decomposition estimator
        # exposing fit() and transform() methods, or None for no reduction
        if reduction is None:
            return None
        if not hasattr(reduction, "fit") or not hasattr(reduction, "transform"):
            raise TypeError(
                "reduction must expose scikit-learn style fit() and transform() methods."
            )
        if not hasattr(reduction, "n_components"):
            raise TypeError(
                "reduction must expose an n_components attribute "
                "(e.g. a scikit-learn decomposition estimator such as PCA or FastICA)."
            )
        return reduction

    def _infer_encoded_input_features(self) -> int:
        # From the reduction technique given, infer how many classical features
        # reach the reservoir encoder. If no reduction is given, we encode the
        # raw input directly.
        if self._reduction_template is None:
            return self.in_features

        n_components = getattr(self._reduction_template, "n_components", None)
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError(
                "reduction must expose a positive integer n_components at construction time."
            )
        return int(n_components)

    def _warn_for_configuration(self) -> None:
        # If no reduction is given and the resulting circuit width is large, the
        # direct reservoir simulation can become expensive.
        if self._reduction_template is None and self.quantum_input_features > 19:
            warnings.warn(
                "ReservoirClassifier without reduction uses "
                f"{self.quantum_input_features} modes, which can become very expensive.",
                UserWarning,
                stacklevel=3,
            )
        # if the number of photons is above 10, there is a warning that the user should be careful because it can become computationally expensive.
        if self.n_photons > 10:
            warnings.warn(
                "ReservoirClassifier configured with "
                f"{self.n_photons} photons and {self.quantum_input_features} modes. "
                "Recommended photon counts stay at or below 10 for practical runs.",
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def _draw_unitary(n_modes: int, seed: int | None) -> np.ndarray:
        # the unitary is built from a Haar-random matrix drawn with perceval, and converted to a numpy array
        # The seed is used to ensure reproducibility of the random unitary,
        # and is applied in a way that does not affect global randomness outside of this function.
        if seed is None:
            return np.array(pcvl.Matrix.random_unitary(n_modes), dtype=np.complex128)

        np_state = np.random.get_state()
        try:
            np.random.seed(seed)
            return np.array(pcvl.Matrix.random_unitary(n_modes), dtype=np.complex128)
        finally:
            np.random.set_state(np_state)

    def _build_layer(
        self,
        *,
        unitary_matrix: np.ndarray,
        encoded_input_size: int,
        n_modes: int,
        measurement_strategy: Any,
        noise: Any | None = None,
    ) -> QuantumLayer:
        # CIRCUIT DESIGN ###
        # The reservoir is made of a:
        # - Haar-random interferometer,
        # - an encoding stage of input-dependent phase shifts,
        # - a symmetric output interferometer.
        matrix = pcvl.Matrix(unitary_matrix)
        interferometer_left = pcvl.Unitary(matrix)
        interferometer_right = pcvl.Unitary(matrix.copy())
        encoder = pcvl.Circuit(n_modes)
        for idx in range(encoded_input_size):
            encoder.add(idx, pcvl.PS(pcvl.P(f"px{idx + 1}")))

        circuit = interferometer_left // encoder // interferometer_right

        # INPUT STATE DESIGN ###
        # The input state is designed to have n_photons distributed one mode out of two from the first mode.
        # For instance, for n_photons=3, the input state is |1,0,1,0,1,...> .
        if self.n_photons > n_modes:
            raise ValueError(
                f"n_photons={self.n_photons} cannot exceed n_modes={n_modes} for the default reservoir input state."
            )
        input_state = [0] * n_modes
        step = (n_modes - 1) / (self.n_photons - 1) if self.n_photons > 1 else 0
        for photon_idx in range(self.n_photons):
            input_state[int(round(photon_idx * step))] = 1

        layer_kwargs = {
            "input_size": encoded_input_size,
            "trainable_parameters": [],
            "input_parameters": ["px"],
            "input_state": input_state,
            "measurement_strategy": measurement_strategy,
            "device": self.device,
            "dtype": self.dtype,
        }
        if noise is None:
            layer = QuantumLayer(
                circuit=circuit,
                **layer_kwargs,
            )
        else:
            # TODO: broaden the supported noise surface as SLOS support
            # expands. For now we simply rebuild the layer through a Perceval
            # Experiment so QuantumLayer can resolve the currently supported
            # photon-loss noise path.
            experiment = pcvl.Experiment()
            experiment.add(0, circuit)
            experiment.noise = noise
            if not hasattr(experiment, "in_heralds"):
                experiment.in_heralds = {}
            if not hasattr(experiment, "heralds"):
                experiment.heralds = {}
            layer = QuantumLayer(
                experiment=experiment,
                **layer_kwargs,
            )
        # The layer is frozen, so we disable gradients and set it to eval mode.
        # This also ensures that any internal buffers are properly registered.
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
        layer.eval()
        return layer

    def _reset_readout(self) -> None:
        self.readout = nn.LazyLinear(self.out_features, dtype=self.dtype)
        self.readout.to(device=self.device, dtype=self.dtype)

    def _invalidate_fit_state(self) -> None:
        self._is_fitted = False
        self._input_min = None
        self._input_max = None
        self._quantum_mean = None
        self._quantum_std = None
        self._fit_fingerprint = None
        self._fit_quantum_cache = None
        self.reduction = (
            clone(self._reduction_template)
            if self._reduction_template is not None
            else None
        )
        self._reset_readout()

    def _rebuild_quantum_layer(self) -> None:
        processor = getattr(self._quantum_layer, "processor", None)
        self._quantum_layer = self._build_layer(
            unitary_matrix=self._unitary_matrix,
            encoded_input_size=self.encoded_input_features,
            n_modes=self.quantum_input_features,
            measurement_strategy=self._measurement_strategy,
            noise=self._noise,
        )
        self._quantum_layer.processor = processor
        self._invalidate_fit_state()

    def _set_layer_measurement_strategy(self, measurement_strategy: Any) -> None:
        self._measurement_strategy = measurement_strategy
        self._rebuild_quantum_layer()

    def _set_layer_noise(self, noise: Any | None) -> None:
        self._noise = noise
        self._rebuild_quantum_layer()

    def _set_layer_n_modes(self, n_modes: int) -> None:
        n_modes = int(n_modes)
        if n_modes <= 0:
            raise ValueError("n_modes must be a positive integer.")
        minimum_n_modes = self.encoded_input_features + 1
        if n_modes < minimum_n_modes:
            raise ValueError(
                "n_modes cannot be smaller than the number of encoded input "
                f"features plus one ({minimum_n_modes})."
            )

        self.quantum_input_features = n_modes
        self._unitary_matrix = self._draw_unitary(n_modes, self.seed)
        self._rebuild_quantum_layer()

    @staticmethod
    def _ensure_2d_numpy(X: np.ndarray | torch.Tensor | list[Any]) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            array = X.detach().cpu().numpy()
        else:
            array = np.asarray(X)

        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise RuntimeError("ReservoirClassifier expects a 2D feature matrix.")
        return np.asarray(array, dtype=np.float32)

    def _coerce_input(self, X: np.ndarray | torch.Tensor | list[Any]) -> np.ndarray:
        # verify that the input can be coerced to a 2D numpy array of the correct shape, and raise an informative error if not.
        array = self._ensure_2d_numpy(X)
        if array.shape[1] != self.in_features:
            raise RuntimeError(
                f"Expected input with {self.in_features} features, got {array.shape[1]}."
            )
        return array

    @staticmethod
    def _coerce_targets(y: np.ndarray | torch.Tensor | list[Any]) -> torch.Tensor:
        # the targets are coerced to a 1D torch tensor of dtype long, which is the expected format for classification targets in PyTorch.
        if isinstance(y, torch.Tensor):
            targets = y.detach().cpu()
        else:
            targets = torch.as_tensor(y)
        return targets.to(dtype=torch.long)

    @staticmethod
    def _normalize_min_max(
        data: np.ndarray, data_min: float, data_max: float
    ) -> np.ndarray:
        epsilon = 1e-8
        return (data - data_min) / (data_max - data_min + epsilon)

    @staticmethod
    def _normalize_standard(
        data: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        epsilon = 1e-8
        return (data - mean) / (std + epsilon)

    def _transform_reduction(self, X: np.ndarray) -> np.ndarray:
        # Possible not to have any reduction applied, in which case we just return the input as-is.
        # Otherwise, we apply the fitted reduction to the input data and return the reduced features.
        if self.reduction is None:
            return X
        return np.asarray(self.reduction.transform(X), dtype=np.float32)

    def _transform_and_normalize_input(self, X: np.ndarray) -> np.ndarray:
        reduced = self._transform_reduction(X)
        return self._normalize_min_max(
            reduced,
            self._input_min,
            self._input_max,
        )

    def _reservoir_feature_width(self) -> int:
        grouping = getattr(self._measurement_strategy, "grouping", None)
        if grouping is not None:
            return int(grouping.output_size)
        return int(self.layer.output_size)

    def _materialize_readout(self, input_width: int) -> None:
        # LazyLinear would otherwise initialize on the first real forward pass,
        # using the ambient torch RNG state at that moment. We materialize it
        # here, right after fit_reservoir() determines the final feature width,
        # so identical model seeds also produce identical readout parameters.
        if not isinstance(self.readout.weight, UninitializedParameter):
            return

        dummy = torch.zeros((1, input_width), dtype=self.dtype, device=self.device)
        if self.seed is None:
            with torch.no_grad():
                _ = self.readout(dummy)
            return

        cpu_state = torch.random.get_rng_state()
        cuda_state = None
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()

        try:
            torch.manual_seed(self.seed)
            with torch.no_grad():
                _ = self.readout(dummy)
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    def _encode_quantum(
        self,
        X_reduced_normalized: np.ndarray,
        processor: Any | None = None,
    ) -> torch.Tensor:
        # Forward pass through the reservoir quantum layer, encoding the input features as phase shifts.
        # Cast the input to a torch tensor on the correct device and dtype, and ensure no gradients are tracked since the reservoir is frozen.
        inputs = torch.as_tensor(
            X_reduced_normalized,
            dtype=self.dtype,
            device=self.device,
        )
        with torch.no_grad():
            self.layer.eval()
            effective_processor = self._resolve_processor(processor)
            if effective_processor is None:
                outputs = self._quantum_layer(inputs)
            else:
                outputs = effective_processor.forward(self._quantum_layer, inputs)
        return outputs.to(dtype=self.dtype).detach()

    def _resolve_processor(self, processor: Any | None) -> Any | None:
        layer_processor = getattr(self._quantum_layer, "processor", None)
        if processor is not None:
            if layer_processor is not None:
                warnings.warn(
                    "Both processor and reservoir.layer.processor are set; "
                    "using the processor argument for this call.",
                    UserWarning,
                    stacklevel=3,
                )
            return processor
        return layer_processor

    @staticmethod
    def _fingerprint(X: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(X)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(f"{contiguous.shape}:{contiguous.dtype}".encode())
        digest.update(contiguous.tobytes())
        return digest.hexdigest()

    def fit_reservoir(
        self,
        X: np.ndarray | torch.Tensor | list[Any],
        *,
        processor: Any | None = None,
    ) -> ReservoirClassifier:
        """Fit the frozen reservoir preprocessing on an input feature matrix.

        Parameters
        ----------
        X : numpy.ndarray|torch.Tensor|list[Any]
            Two-dimensional feature matrix used to fit the optional reduction,
            learn the min-max input scaling, and initialize the readout input
            width. When ``cache=True``, this method also computes and stores
            the fitted training-set reservoir features and their
            standardization statistics. When ``cache=False``, the quantum pass
            is deferred until the fitted training data is transformed.
        processor : merlin.core.merlin_processor.MerlinProcessor|None
            Optional processor used to evaluate the frozen quantum reservoir
            through Merlin's local or remote execution path when
            ``cache=True``. If omitted, ``reservoir.layer.processor`` is used
            when configured. If both are set, this argument wins and emits a
            ``UserWarning``. If neither is set, the quantum layer is executed
            locally. This argument is not used when ``cache=False``.

        Returns
        -------
        ReservoirClassifier
            The fitted classifier instance.

        Raises
        ------
        RuntimeError
            If ``X`` cannot be interpreted as a two-dimensional feature matrix
            with ``in_features`` columns.
        """
        # This method fits the reservoir to the input data X by performing the following steps:
        # 1. Coerce the input to a 2D numpy array and validate its
        X_np = self._coerce_input(X)
        # 2. Fit the dimensionality reduction technique (if any) and transform the input data accordingly.
        self.reduction = (
            clone(self._reduction_template)
            if self._reduction_template is not None
            else None
        )
        if self.reduction is None:
            reduced = X_np
        else:
            self.reduction.fit(X_np)
            reduced = np.asarray(self.reduction.transform(X_np), dtype=np.float32)

        # 3. Normalize the reduced features to a [0, 1] range based on the min and max of the reduced data.
        self._input_min = float(np.min(reduced))
        self._input_max = float(np.max(reduced))
        reduced_normalized = self._normalize_min_max(
            reduced,
            self._input_min,
            self._input_max,
        )
        self._fit_fingerprint = self._fingerprint(X_np)
        if self.cache:
            # 4. Encode the normalized features into quantum states using the reservoir's quantum layer.
            quantum = self._encode_quantum(reduced_normalized, processor=processor)
            self._quantum_mean = quantum.mean(dim=0).detach().cpu()
            self._quantum_std = quantum.std(dim=0, unbiased=False).detach().cpu()
            self._fit_quantum_cache = quantum.detach().cpu()
            reservoir_feature_width = int(quantum.shape[1])
        else:
            self._quantum_mean = None
            self._quantum_std = None
            self._fit_quantum_cache = None
            reservoir_feature_width = self._reservoir_feature_width()

        # 5. Cache the quantum features of the training data if caching is enabled, and store a fingerprint of the input data for cache validation during prediction.
        feature_width = reservoir_feature_width + (
            self.in_features if self.concatenate else 0
        )
        self._materialize_readout(feature_width)
        self._is_fitted = True
        return self

    #: Alias for :meth:`fit_reservoir` provided for API consistency with
    #: scikit-learn pipeline conventions (``estimator.sample(X)`` ↔
    #: ``estimator.fit_reservoir(X)``).
    sample = fit_reservoir

    def _require_fitted(self) -> None:
        # The output of the reservoir are extracted before the readout layer is trained.
        if not self._is_fitted:
            raise RuntimeError(
                "ReservoirClassifier must be fitted with fit_reservoir() before use."
            )

    def _compute_quantum_features(
        self,
        X_np: np.ndarray,
        *,
        processor: Any | None = None,
    ) -> torch.Tensor:
        # Centralize the "cached training set vs fresh encoding" decision:
        # reuse the stored reservoir features when X matches the fitted data,
        # otherwise run the reservoir on the new inputs, then apply the
        # normalization statistics learned during fit_reservoir().
        self._require_fitted()

        fingerprint = self._fingerprint(X_np)
        if (
            self._quantum_mean is None or self._quantum_std is None
        ) and fingerprint != self._fit_fingerprint:
            raise RuntimeError(
                "ReservoirClassifier with cache=False must initialize quantum "
                "normalization on the fitted training data before transforming "
                "new inputs."
            )

        if self.cache and self._fit_quantum_cache is not None:
            # Cache hit path: if X matches the fitted training set, reuse the
            # stored quantum embeddings instead of running the reservoir again.
            if fingerprint == self._fit_fingerprint:
                quantum = self._fit_quantum_cache.clone()
            else:
                # Cache miss path: transform and encode the new inputs.
                reduced_normalized = self._transform_and_normalize_input(X_np)
                quantum = self._encode_quantum(reduced_normalized, processor=processor)
        else:
            # No cache available: always go through reduction + normalization +
            # reservoir encoding.
            reduced_normalized = self._transform_and_normalize_input(X_np)
            quantum = self._encode_quantum(reduced_normalized, processor=processor)

        mean = self._quantum_mean
        std = self._quantum_std
        if mean is None or std is None:
            mean = quantum.mean(dim=0).detach().cpu()
            std = quantum.std(dim=0, unbiased=False).detach().cpu()
            self._quantum_mean = mean
            self._quantum_std = std
            if self.cache:
                self._fit_quantum_cache = quantum.detach().cpu()
        # Standardize the quantum features with the statistics learned on the
        # training set so train and inference use the same feature scale.
        return self._normalize_standard(
            quantum.to(self.device),
            mean.to(device=self.device, dtype=self.dtype),
            std.to(device=self.device, dtype=self.dtype),
        )

    def _make_feature_tensor(
        self,
        X: np.ndarray | torch.Tensor | list[Any],
        *,
        processor: Any | None = None,
    ) -> torch.Tensor:
        X_np = self._coerce_input(X)
        quantum_features = self._compute_quantum_features(X_np, processor=processor)
        if not self.concatenate:
            return quantum_features

        raw_tensor = torch.as_tensor(X_np, dtype=self.dtype, device=self.device)
        return torch.cat((raw_tensor, quantum_features), dim=1)

    def transform_reservoir(
        self,
        X: np.ndarray | torch.Tensor | list[Any],
        *,
        processor: Any | None = None,
    ) -> torch.Tensor:
        """Transform raw inputs into standardized reservoir embeddings.

        Parameters
        ----------
        X : numpy.ndarray|torch.Tensor|list[Any]
            Two-dimensional feature matrix to transform through the fitted
            reservoir preprocessing and frozen quantum layer.
        processor : merlin.core.merlin_processor.MerlinProcessor|None
            Optional processor used to evaluate the frozen quantum reservoir.
            If omitted, ``reservoir.layer.processor`` is used when configured.
            If both are set, this argument wins and emits a ``UserWarning``. If
            neither is set, the quantum layer is executed locally.

        Returns
        -------
        torch.Tensor
            Standardized quantum reservoir embeddings on CPU. The returned
            tensor does not include raw input features, even when
            ``concatenate=True``.

        Raises
        ------
        RuntimeError
            If :meth:`fit_reservoir` was not called first, if ``X`` does not
            have the expected shape, or if ``cache=False`` and quantum
            normalization has not yet been initialized from the fitted training
            data.
        """
        X_np = self._coerce_input(X)
        return self._compute_quantum_features(X_np, processor=processor).detach().cpu()

    def make_dataset(
        self,
        X: np.ndarray | torch.Tensor | list[Any],
        y: np.ndarray | torch.Tensor | list[Any],
        *,
        processor: Any | None = None,
    ) -> TensorDataset:
        """Build a readout-ready dataset from raw inputs and classification targets.

        Parameters
        ----------
        X : numpy.ndarray|torch.Tensor|list[Any]
            Two-dimensional feature matrix to transform into readout features.
        y : numpy.ndarray|torch.Tensor|list[Any]
            One-dimensional classification targets aligned with ``X``.
        processor : merlin.core.merlin_processor.MerlinProcessor|None
            Optional processor used to evaluate the frozen quantum reservoir.
            If omitted, ``reservoir.layer.processor`` is used when configured.
            If both are set, this argument wins and emits a ``UserWarning``. If
            neither is set, the quantum layer is executed locally.

        Returns
        -------
        torch.utils.data.TensorDataset
            Dataset containing the transformed readout features on CPU and the
            integer classification targets.

        Raises
        ------
        RuntimeError
            If :meth:`fit_reservoir` was not called first, if ``X`` does not
            have the expected shape, or if ``X`` and ``y`` do not contain the
            same number of samples.
        """
        features = self._make_feature_tensor(X, processor=processor).detach().cpu()
        targets = self._coerce_targets(y)
        if features.shape[0] != targets.shape[0]:
            raise RuntimeError("X and y must contain the same number of samples.")
        return TensorDataset(features, targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the trainable linear readout to precomputed feature vectors.

        Parameters
        ----------
        x : torch.Tensor
            Batch of readout input features. This tensor must already contain
            the reservoir features, with optional raw-feature concatenation
            applied if needed.

        Returns
        -------
        torch.Tensor
            Output logits produced by the linear readout.
        """
        # only the readout is concerned here as the quantum features are already computed.
        return self.readout(x.to(device=self.device, dtype=self.dtype))

    def predict(
        self,
        X: np.ndarray | torch.Tensor | list[Any],
        *,
        processor: Any | None = None,
    ) -> torch.Tensor:
        """Run the frozen reservoir and readout on a batch of raw inputs.

        Parameters
        ----------
        X : numpy.ndarray|torch.Tensor|list[Any]
            Two-dimensional feature matrix to encode through the reservoir.
        processor : merlin.core.merlin_processor.MerlinProcessor|None
            Optional processor used to evaluate the frozen quantum reservoir.
            If omitted, ``reservoir.layer.processor`` is used when configured.
            If both are set, this argument wins and emits a ``UserWarning``. If
            neither is set, the quantum layer is executed locally.

        Returns
        -------
        torch.Tensor
            Output logits on CPU for each input sample.

        Raises
        ------
        RuntimeError
            If :meth:`fit_reservoir` was not called first or if ``X`` does not
            have the expected shape.
        """
        # The predict method:
        # 1. computes the quantum features for the input data X,
        # 2. concatenates them with the raw features if configured to do so,
        # 3. applies the readout layer to produce the final output logits.
        # The quantum feature computation is centralized in _compute_quantum_features() which handles caching and normalization consistently between training and inference.
        features = self._make_feature_tensor(X, processor=processor)
        self.eval()
        with torch.no_grad():
            logits = self.forward(features)
        return logits.detach().cpu()

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        """Yield the trainable parameters of the classifier.

        Parameters
        ----------
        recurse : bool
            Forwarded to :meth:`torch.nn.Module.parameters`.

        Returns
        -------
        collections.abc.Iterator[torch.nn.Parameter]
            Iterator over the readout parameters only. The reservoir parameters
            are frozen and therefore excluded.
        """
        yield from self.readout.parameters(recurse=recurse)

    def named_parameters(  # type: ignore[override]
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        """Yield named trainable parameters of the classifier.

        Parameters
        ----------
        prefix : str
            Optional prefix prepended to the yielded parameter names.
        recurse : bool
            Forwarded to :meth:`torch.nn.Module.named_parameters`.
        remove_duplicate : bool
            Forwarded to :meth:`torch.nn.Module.named_parameters`.

        Returns
        -------
        collections.abc.Iterator[tuple[str, torch.nn.Parameter]]
            Iterator over the named readout parameters only.
        """
        readout_prefix = f"{prefix}.readout" if prefix else "readout"
        yield from self.readout.named_parameters(
            prefix=readout_prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def save(self, path: str | Path) -> None:
        """Serialize the classifier state to disk.

        Parameters
        ----------
        path : str|pathlib.Path
            Destination file used to store the configuration, frozen reservoir
            state, preprocessing statistics, cache, and readout parameters.

        Returns
        -------
        None
            This method writes the checkpoint to disk in-place.
        """
        reduction_fitted = (
            self.reduction if self._is_fitted else self._reduction_template
        )
        readout_initialized = not isinstance(
            self.readout.weight, UninitializedParameter
        )
        payload = {
            "config": {
                "in_features": self.in_features,
                "out_features": self.out_features,
                "n_photons": self.n_photons,
                "reduction": self._reduction_template,
                "concatenate": self.concatenate,
                "cache": self.cache,
                "seed": self.seed,
                "dtype": self.dtype,
            },
            "layer_state": {
                "n_modes": self.quantum_input_features,
                "measurement_strategy": self._measurement_strategy,
                "noise": self._noise,
            },
            "unitary_matrix": self._unitary_matrix,
            "reduction_fitted": reduction_fitted,
            "is_fitted": self._is_fitted,
            "input_min": self._input_min,
            "input_max": self._input_max,
            "quantum_mean": self._quantum_mean,
            "quantum_std": self._quantum_std,
            "fit_fingerprint": self._fit_fingerprint,
            "fit_quantum_cache": self._fit_quantum_cache,
            "readout_initialized": readout_initialized,
            "readout_in_features": (
                int(self.readout.in_features) if readout_initialized else None
            ),
            "readout_state_dict": self.readout.state_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        device: str | torch.device | None = None,
    ) -> ReservoirClassifier:
        """Restore a serialized classifier from disk.

        Parameters
        ----------
        path : str|pathlib.Path
            Checkpoint file previously generated with :meth:`save`.
        map_location : str|torch.device|None
            Optional map-location argument forwarded to :func:`torch.load`.
        device : str|torch.device|None
            Optional device override applied to the restored classifier after
            loading the checkpoint configuration.

        Returns
        -------
        ReservoirClassifier
            Restored classifier with the frozen reservoir state, preprocessing
            statistics, and readout parameters loaded from ``path``.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        RuntimeError
            If the checkpoint cannot be deserialized by :func:`torch.load`.

        Notes
        -----
        This method calls :func:`torch.load` with ``weights_only=False``, which
        allows arbitrary Python objects to be unpickled. Only load checkpoints
        from trusted sources. Loading a malicious file can execute arbitrary
        code on your machine.
        """
        try:
            if map_location is None:
                payload = torch.load(Path(path), weights_only=False)
            else:
                payload = torch.load(
                    Path(path),
                    map_location=map_location,
                    weights_only=False,
                )
        except TypeError:
            if map_location is None:
                payload = torch.load(Path(path))
            else:
                payload = torch.load(Path(path), map_location=map_location)

        config = dict(payload["config"])
        layer_state = dict(payload.get("layer_state", {}))
        # Backward compatibility for checkpoints saved before layer-only state
        # was split out of the public constructor config.
        saved_n_modes = layer_state.get("n_modes", config.pop("n_modes", None))
        saved_measurement_strategy = layer_state.get(
            "measurement_strategy",
            config.pop("measurement_strategy", None),
        )
        missing = object()
        saved_noise = layer_state.get("noise", missing)
        if saved_noise is missing:
            saved_noise = layer_state.get("noise_model", missing)
        if saved_noise is missing:
            saved_noise = config.pop("noise", missing)
        if saved_noise is missing:
            saved_noise = config.pop("noise_model", None)
        if device is not None:
            config["device"] = torch.device(device)
        model = cls(**config)
        model._unitary_matrix = np.asarray(
            payload["unitary_matrix"], dtype=np.complex128
        )
        if saved_n_modes is not None:
            model.quantum_input_features = int(saved_n_modes)
        if saved_measurement_strategy is not None:
            model._measurement_strategy = saved_measurement_strategy
        model._noise = saved_noise
        model._quantum_layer = model._build_layer(
            unitary_matrix=model._unitary_matrix,
            encoded_input_size=model.encoded_input_features,
            n_modes=model.quantum_input_features,
            measurement_strategy=model._measurement_strategy,
            noise=model._noise,
        )

        if payload["readout_initialized"]:
            model.readout = nn.Linear(
                payload["readout_in_features"],
                model.out_features,
                dtype=model.dtype,
                device=model.device,
            )

        model.readout.load_state_dict(payload["readout_state_dict"])
        model.reduction = payload["reduction_fitted"]
        model._is_fitted = bool(payload["is_fitted"])
        model._input_min = payload["input_min"]
        model._input_max = payload["input_max"]
        model._quantum_mean = payload["quantum_mean"]
        model._quantum_std = payload["quantum_std"]
        model._fit_fingerprint = payload["fit_fingerprint"]
        model._fit_quantum_cache = payload["fit_quantum_cache"]
        return model
