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

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Sequence
from typing import cast

import perceval as pcvl
import torch

'''def _is_sequence(value: object) -> bool:
    """Return True if the value behaves like a sequence (and is not a string)."""
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, Sequence)


def _normalize_noise_parameter(
    param: object, n_modes: int, label: str
) -> list[float]:
    """
    Broadcast a noise parameter (brightness or transmittance) to a per-mode list.

    Args:
        param: Raw parameter value—scalar, tensor, or sequence.
        n_modes: Number of optical modes.
        label: Parameter name for error reporting.

    Returns:
        list[float]: Per-mode parameter values.
    """
    if param is None:
        return [1.0] * n_modes

    if isinstance(param, torch.Tensor):
        values = param.detach().cpu().tolist()
    elif _is_sequence(param):
        values = [float(v) for v in cast(Sequence[float], param)]
    else:
        try:
            scalar = float(param)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Unsupported {label} value: {param!r}") from exc
        return [scalar] * n_modes

    if not values:
        raise ValueError(f"{label} sequence cannot be empty.")

    if len(values) == 1:
        values = values * n_modes
    elif len(values) != n_modes:
        raise ValueError(
            f"{label} must provide either one value or exactly {n_modes} values; received {len(values)}."
        )

    return [float(v) for v in values]'''


class PhotonLossTransform(torch.nn.Module):
    """
    Linear map applying per-mode photon loss to a Fock probability vector.

    Parameters
    ----------
    simulation_keys : Sequence[Sequence[int]] | torch.Tensor
        Raw Fock states produced by the simulator.
    survival_probs : Sequence[float]
        One survival probability per optical mode.
    dtype : torch.dtype | None
        Torch dtype for the transform matrix. Defaults to ``torch.float32``.
    device : torch.device | str | None
        Device used to stage the transform matrix.
    """

    def __init__(
        self,
        simulation_keys: Sequence[Sequence[int]] | torch.Tensor,
        survival_probs: Sequence[float],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the photon-loss transform.

        Parameters
        ----------
        simulation_keys : Sequence[Sequence[int]] | torch.Tensor
            Raw Fock states produced by the simulator.
        survival_probs : Sequence[float]
            One survival probability per optical mode.
        dtype : torch.dtype | None
            Torch dtype for the transform matrix. Defaults to ``torch.float32``.
        device : torch.device | str | None
            Device used to stage the transform matrix.
        """
        super().__init__()

        if simulation_keys is None or len(simulation_keys) == 0:
            raise ValueError(
                "simulation_keys must contain at least one mapped Fock key."
            )

        self._dtype = dtype or torch.float32
        self._device = torch.device(device) if device is not None else None

        self._simulation_keys = self._normalize_keys(simulation_keys)
        self._n_modes = len(self._simulation_keys[0])

        if any(len(key) != self._n_modes for key in self._simulation_keys):
            raise ValueError("All simulation keys must have the same number of modes.")

        if len(survival_probs) != self._n_modes:
            raise ValueError(
                f"Expected {self._n_modes} survival probabilities, received {len(survival_probs)}."
            )
        self._survival_probs = self._validate_survival_probs(survival_probs)

        matrix, loss_keys, is_identity = self._build_transform()

        self._loss_keys: list[tuple[int, ...]] = loss_keys
        self._is_identity = is_identity

        if is_identity:
            buffer_kwargs = {}
            if self._device is not None:
                buffer_kwargs["device"] = self._device
            placeholder = torch.empty(
                (0, 0),
                dtype=self._dtype,
                device=buffer_kwargs.get("device", None),
            )
            self.register_buffer("_matrix", placeholder, persistent=False)
        else:
            self.register_buffer("_matrix", matrix)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_keys(
        keys: Sequence[Sequence[int]] | torch.Tensor,
    ) -> list[tuple[int, ...]]:
        """Convert raw simulator keys into canonical tuples."""
        if isinstance(keys, torch.Tensor):
            if keys.ndim != 2:
                raise ValueError("simulation_keys tensor must have shape (N, M).")
            return [tuple(int(v) for v in row.tolist()) for row in keys]

        normalized: list[tuple[int, ...]] = []
        for key in keys:
            normalized.append(tuple(int(v) for v in key))
        return normalized

    @staticmethod
    def _validate_survival_probs(survival_probs: Sequence[float]) -> list[float]:
        """Ensure survival probabilities lie in [0, 1]."""
        validated: list[float] = []
        for idx, prob in enumerate(survival_probs):
            try:
                value = float(prob)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Survival probability at index {idx} is not numeric: {prob!r}"
                ) from exc
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Survival probability at index {idx} must be within [0, 1]; received {value}."
                )
            validated.append(value)
        return validated

    @staticmethod
    def _mode_loss_distribution(
        photon_count: int, survival_prob: float
    ) -> list[tuple[int, float]]:
        """Enumerate photon survival outcomes for a single mode."""
        failure_prob = 1.0 - survival_prob
        outcomes: list[tuple[int, float]] = []
        for survivors in range(photon_count, -1, -1):
            probability = (
                math.comb(photon_count, survivors)
                * (survival_prob**survivors)
                * (failure_prob ** (photon_count - survivors))
            )
            if probability == 0.0:
                continue
            outcomes.append((survivors, probability))
        return outcomes or [(0, 1.0)]

    def _build_transform(
        self,
    ) -> tuple[torch.Tensor | None, list[tuple[int, ...]], bool]:
        """
        Construct the photon loss transform matrix and associated Fock keys.
        """
        identity = all(
            math.isclose(prob, 1.0, rel_tol=1e-12, abs_tol=1e-12)
            for prob in self._survival_probs
        )
        if identity:
            return (
                None,
                [tuple(int(v) for v in key) for key in self._simulation_keys],
                True,
            )

        key_to_index: dict[tuple[int, ...], int] = {}
        loss_keys: list[tuple[int, ...]] = []
        row_entries: list[dict[int, float]] = []

        for sim_key in self._simulation_keys:
            per_mode = [
                self._mode_loss_distribution(count, self._survival_probs[mode])
                for mode, count in enumerate(sim_key)
            ]

            combined: dict[int, float] = {}
            for outcomes in itertools.product(*per_mode):
                loss_key = tuple(outcome[0] for outcome in outcomes)
                probability = 1.0
                for _, partial_prob in outcomes:
                    probability *= partial_prob

                if probability == 0.0:
                    continue

                column_index = key_to_index.get(loss_key)
                if column_index is None:
                    column_index = len(loss_keys)
                    key_to_index[loss_key] = column_index
                    loss_keys.append(loss_key)
                combined[column_index] = combined.get(column_index, 0.0) + probability

            row_entries.append(combined)

        rows = len(self._simulation_keys)
        cols = len(loss_keys)
        device_kwargs = {}
        if self._device is not None:
            device_kwargs["device"] = self._device

        matrix = torch.zeros(
            (rows, cols),
            dtype=self._dtype,
            device=device_kwargs.get("device", None),
        )
        for row_idx, entries in enumerate(row_entries):
            for col_idx, prob in entries.items():
                matrix[row_idx, col_idx] = prob

        return matrix, loss_keys, False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def output_keys(self) -> list[tuple[int, ...]]:
        """Classical Fock keys after photon loss."""
        return self._loss_keys

    @property
    def output_size(self) -> int:
        """int: Number of classical outcomes after photon loss."""
        return len(self._loss_keys)

    @property
    def is_identity(self) -> bool:
        """Whether the transform corresponds to perfect transmission."""
        return self._is_identity

    def forward(self, distribution: torch.Tensor) -> torch.Tensor:
        """Apply the photon-loss transform to a Fock probability vector.

        Parameters
        ----------
        distribution : torch.Tensor
            Fock probability vector.

        Returns
        -------
        torch.Tensor
            Probability vector after photon loss as a 1D torch tensor.
        """
        if self._is_identity:
            return distribution

        matrix: torch.Tensor = cast(torch.Tensor, self._matrix)  # type: ignore[attr-defined]
        return distribution @ matrix

    def to(self, *args, **kwargs):
        """Move the transform to a new device or dtype.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :meth:`torch.nn.Module.to`.
        **kwargs
            Keyword arguments forwarded to :meth:`torch.nn.Module.to`.

        Returns
        -------
        PhotonLossTransform
            Updated transform instance.
        """
        result = super().to(*args, **kwargs)

        dtype = kwargs.get("dtype")
        device = kwargs.get("device")

        if dtype is None and len(args) > 0 and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if device is None and len(args) > 0 and isinstance(args[0], torch.device):
            device = args[0]

        if dtype is not None:
            self._dtype = dtype
        if device is not None:
            self._device = device

        return result


def resolve_photon_loss(
    experiment: pcvl.Experiment, n_modes: int
) -> tuple[list[float], bool]:
    """Resolve photon loss from the experiment's noise model.

    Parameters
    ----------
    experiment : pcvl.Experiment
        Quantum experiment carrying the noise model.
    n_modes : int
        Number of photonic modes to cover.

    Returns
    -------
    tuple[list[float], bool]
        Per-mode survival probabilities and a flag indicating whether an
        effective noise model was provided.
    """
    survival_probs = [1.0] * n_modes  # Default: no loss
    empty_noise_model = True

    noise_model = getattr(experiment, "noise", None)
    if noise_model is None:
        return survival_probs, empty_noise_model

    brightness = cast(float, getattr(noise_model, "brightness", None))
    transmittance = cast(float, getattr(noise_model, "transmittance", None))

    if brightness is None and transmittance is None:
        survival_probs = [1.0] * n_modes
        empty_noise_model = True
    elif brightness is None:
        warnings.warn(
            "Brightness not specified in noise model; assuming 1.0.", stacklevel=2
        )
        survival_probs = [transmittance] * n_modes
        empty_noise_model = False
    elif transmittance is None:
        warnings.warn(
            "Transmittance not specified in noise model; assuming 1.0.", stacklevel=2
        )
        survival_probs = [brightness] * n_modes
        empty_noise_model = False
    else:
        survival_probs = [float(brightness) * float(transmittance)] * n_modes
        empty_noise_model = False

    if not all(0.0 <= prob <= 1.0 for prob in survival_probs):
        raise ValueError("Photon survival probabilities must be within [0, 1].")
    if empty_noise_model and not all(
        math.isclose(prob, 1.0, rel_tol=1e-12, abs_tol=1e-12) for prob in survival_probs
    ):
        raise ValueError(
            "Inconsistent noise model: marked as empty but contains non-trivial loss parameters."
        )

    return survival_probs, empty_noise_model
