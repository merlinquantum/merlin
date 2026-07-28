# MIT License
#
# Copyright (c) 2026 Quandela
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

import copy
import warnings

import numpy as np
import torch
from multipledispatch import dispatch
from perceval.components import (
    BS,
    PERM,
    PS,
    AComponent,
    Barrier,
    BSConvention,
    Circuit,
    Unitary,
)
from perceval.utils import Matrix
from perceval.utils.algorithms.circuit_optimizer import CircuitOptimizer

from ..utils.dtypes import resolve_float_complex

SUPPORTED_COMPONENTS = (PS, BS, PERM, Unitary, Barrier)

_DECOMPOSITION_CACHE: dict[bytes, Circuit] = {}
"""Cache mapping target-matrix bytes to its alpha-corrected Clements mesh.

Keys are ``ndarray.tobytes()`` of the complex128 target matrix (so identical
unitaries — same bytes — always hit the same entry). Values are the completed
(fixed-parameter, alpha-corrected) ``Circuit`` objects from the first
successful ``CircuitOptimizer().optimize_rectangle`` call for that matrix.

Consequences:

* Each target matrix is decomposed at most once regardless of how many
  ``CircuitConverter`` instances are built for circuits that contain it.
* The global NumPy RNG state is perturbed at most once per unique matrix,
  so repeated or rebuild operations are deterministic from the caller's
  perspective and do not accumulate RNG drift.
* ``phase_imprecision`` quantization, being a discontinuous function of the
  mesh phases, now also produces consistent results across rebuilds because
  the mesh phases are identical every time.

Thread safety: the cache is populated lazily without a lock. Concurrent
writes for the same key both produce valid (equivalent) results, so the
only observable consequence is a redundant optimizer call during a race —
not incorrect output.
"""

_DEFAULT_OPTIMIZE_RECTANGLE = CircuitOptimizer.optimize_rectangle
"""Optimizer implementation used to create entries in the decomposition cache."""
"""Tuple of quantum components supported by CircuitConverter.

Components:
    PS: Phase shifter with single phi parameter
    BS: Beam splitter with theta and four phi parameters
    PERM: Mode permutation (no parameters)
    Unitary: Generic unitary matrix (no parameters)
    Barrier: Synchronization barrier (removed during compilation)
"""

_FIT_TOLERANCE = 1e-3
"""Tolerance for unitary decomposition fit quality.

After fitting a unitary to an MZI mesh, the overlap magnitude
    abs(trace(target^H @ fitted) / dimension)
should be close to 1.0. If the overlap falls below (1 - tolerance),
the fit is considered to have failed and a RuntimeError is raised.
With default CircuitOptimizer settings (fidelity error ~1e-6),
the overlap should be > 0.999 in all normal cases.
"""


def _phase_shifter_max_error(component: AComponent) -> float:
    """Return the local Perceval phase-error half-width for a phase shifter.

    Parameters
    ----------
    component : AComponent
        Perceval component to inspect.

    Returns
    -------
    float
        Positive stochastic phase-error half-width stored on the component,
        or 0.0 when the component has no local phase-error setting.
    """
    if not isinstance(component, PS):
        return 0.0
    return float(getattr(component, "_max_error", 0.0))


def _circuit_has_local_phase_error(circuit: Circuit) -> bool:
    """Return whether a circuit contains phase shifters with local error.

    Parameters
    ----------
    circuit : Circuit
        Perceval circuit to inspect.

    Returns
    -------
    bool
        True when at least one phase shifter has ``max_error > 0``.
    """
    return any(_phase_shifter_max_error(component) > 0.0 for _, component in circuit)


def _decompose_unitaries(
    circuit: Circuit,
    phase_imprecision: float = 0.0,
    phase_error: float = 0.0,
) -> Circuit:
    """Replace black-box ``Unitary`` blocks with Clements MZI meshes.

    Called when circuit phase noise is configured (``phase_imprecision > 0``
    or ``phase_error > 0``). Each non-``PERM`` ``Unitary`` component is fitted
    to a rectangular (Clements) mesh of MZIs with all parameters fixed as
    constants. This makes the unitary's structure explicit, though the fixed
    mesh parameters themselves are not tunable and receive no phase noise.
    ``PERM`` components (waveguide crossings with no programmable phases) are
    left untouched.
    A 1-mode ``Unitary`` (a bare global phase) is replaced by a single ``PS``
    without running the optimizer.

    The fit reproduces each target unitary within the optimizer threshold
    (fidelity error ~1e-6, matrix entries within ~1e-3). Its arbitrary global
    phase is folded into the mesh's output phase layer so that blocks acting
    on a subset of modes keep their relative phase with the rest of the
    circuit.

    Decomposition results are memoized in ``_DECOMPOSITION_CACHE`` keyed on
    the matrix bytes. After the first construction for a given matrix, every
    subsequent ``CircuitConverter`` build reuses the cached mesh (deep-copied
    for independence), so: (a) the global NumPy RNG is perturbed at most once
    per unique matrix; (b) rebuilding or reloading a model always produces the
    same mesh phases; and (c) ``phase_imprecision`` quantization, which is a
    discontinuous function of the individual mesh phases, gives consistent
    results across builds.

    .. note::
        Decomposition complexity is ``O(m^3)`` in the optimizer iterations and
        typically takes ~0.2 s for 10 modes. For ``m >= 16`` the first build
        may take several seconds; consider building the converter once and
        reusing it.

    When phase noise is configured, a warning is emitted if the decomposition
    fit error is comparable to or exceeds the configured noise scale. In this
    case, the fit error becomes an uncontrolled artifact that can dominate
    over the modeled physical noise.

    Parameters
    ----------
    circuit : Circuit
        Perceval circuit to transform.
    phase_imprecision : float
        Deterministic phase quantization step in radians. Used to estimate
        the physical noise scale for warning purposes.
    phase_error : float
        Stochastic phase perturbation half-width in radians. Used to estimate
        the physical noise scale for warning purposes.

    Returns
    -------
    Circuit
        The original circuit object when it contains no non-``PERM``
        ``Unitary`` component, otherwise a new circuit with each such
        component replaced by an equivalent MZI mesh.

    Raises
    ------
    ValueError
        If a ``Unitary`` component uses polarization.
    RuntimeError
        If the Clements fit does not converge for a component.
    """
    if not any(
        isinstance(component, Unitary) and not isinstance(component, PERM)
        for _, component in circuit
    ):
        return circuit

    new_circuit = Circuit(circuit.m)
    for r, component in circuit:
        if not isinstance(component, Unitary) or isinstance(component, PERM):
            new_circuit.add(r, component)
            continue
        if component.requires_polarization:
            raise ValueError(
                f"Circuit phase noise cannot be applied to polarized unitary "
                f"'{component.name}' on modes {tuple(r)}: decomposition into "
                f"an MZI mesh is not supported for polarized components."
            )

        target = np.asarray(component.compute_unitary(), dtype=complex)

        # A 1-mode Unitary is a bare global phase exp(i*phi) with no beam
        # splitters to place. The MZI optimizer has no defined behaviour for
        # a 1x1 matrix, so handle it directly by emitting a single PS.
        if component.m == 1:
            ps_phase = float(np.angle(target[0, 0]))
            new_circuit.add(r[0], PS(ps_phase))
            continue

        # Warn for large mode counts where the optimizer cost is superlinear
        # and can take several seconds on the first build.
        if component.m >= 16:
            warnings.warn(
                f"Decomposing a {component.m}-mode unitary '{component.name}' "
                f"into a Clements mesh. The optimizer cost is O(m^3) in "
                f"iteration count; for m >= 16 this may take several seconds. "
                f"The result is cached after the first build for each unique matrix.",
                UserWarning,
                stacklevel=3,
            )

        # Check cache before running the optimizer. The cache key is the
        # bytes representation of the complex128 target matrix, which
        # uniquely identifies the unitary up to floating-point equality.
        cache_key = target.tobytes()
        # Do not use entries created by the default optimizer when the method
        # has been replaced. Besides keeping cache contents tied to the
        # optimizer that created them, this ensures validation is performed
        # against the active implementation.
        optimizer_is_default = (
            CircuitOptimizer.optimize_rectangle is _DEFAULT_OPTIMIZE_RECTANGLE
        )
        if cache_key in _DECOMPOSITION_CACHE and optimizer_is_default:
            # Reuse a deep copy of the cached mesh so each CircuitConverter
            # gets an independent set of components.
            cached_mesh = copy.deepcopy(_DECOMPOSITION_CACHE[cache_key])
            cached_fitted = np.asarray(cached_mesh.compute_unitary(), dtype=complex)
            if not np.allclose(
                cached_fitted.conj().T @ cached_fitted,
                np.eye(component.m),
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"Cached Clements decomposition is not unitary for "
                    f"unitary '{component.name}' on modes {tuple(r)}."
                )
            cached_overlap = np.trace(target.conj().T @ cached_fitted) / component.m
            if abs(cached_overlap) < 1.0 - _FIT_TOLERANCE:
                raise RuntimeError(
                    f"Cached Clements decomposition failed the fit quality check "
                    f"for unitary '{component.name}' on modes {tuple(r)}: "
                    f"overlap magnitude = {abs(cached_overlap):.6e}."
                )
            cached_parameters = cached_mesh.get_parameters(all_params=False)
            cached_output_phases = [
                parameter
                for parameter in cached_parameters
                if parameter.name.startswith("phL")
            ]
            if len(cached_output_phases) == component.m:
                new_circuit.add(r[0], cached_mesh, merge=True)
                continue

            # Entries created by an older optimizer/template are not valid
            # for the current output-phase contract. Treat them as misses.
            del _DECOMPOSITION_CACHE[cache_key]

        try:
            mesh = CircuitOptimizer().optimize_rectangle(Matrix(target))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Clements decomposition did not converge for unitary "
                f"'{component.name}' on modes {tuple(r)}: {exc}"
            ) from exc

        # The fit reproduces the target only up to a global phase e^{i*alpha}.
        # On a block covering a subset of modes that phase is physical (it is
        # relative to the untouched modes), so fold -alpha into the mesh's
        # output phase layer, which has exactly one PS per mode.
        fitted = np.asarray(mesh.compute_unitary(), dtype=complex)
        overlap = np.trace(target.conj().T @ fitted) / component.m

        # Verify fit quality: if overlap magnitude is too small, the fit failed
        # and alpha becomes meaningless (trace near zero makes phase arbitrary).
        # So failures are raised explicitly and not hidden.
        if abs(overlap) < 1.0 - _FIT_TOLERANCE:
            raise RuntimeError(
                f"Clements decomposition fit quality check failed for unitary "
                f"'{component.name}' on modes {tuple(r)}: overlap magnitude "
                f"|trace(target^H @ fitted) / m| = {abs(overlap):.6e} < "
                f"{1.0 - _FIT_TOLERANCE:.6e}. The fitted mesh does not adequately "
                f"reproduce the target unitary. This may indicate a bug in "
                f"CircuitOptimizer or insufficient optimization iterations."
            )

        alpha = float(np.angle(overlap))

        # Warn if decomposition fit error is comparable to configured phase noise.
        # For unitary matrices, the fit quality is measured by the overlap magnitude:
        # overlap = trace(target^H @ fitted) / m, which is bounded in [0, 1].
        # The fit error is 1 - |overlap|. With default CircuitOptimizer settings
        # (fidelity error ~1e-6), we expect |overlap| > 0.999, i.e., fit_error < 0.001.
        #
        # For comparison, a phase error of delta_phi rad induces a unitary entry
        # change ~delta_phi, so the noise scale is approximately
        # max(phase_imprecision, phase_error). If fit_error > 0.1 * noise_scale,
        # the fit error likely dominates the physical noise being modeled.
        if phase_imprecision > 0.0 or phase_error > 0.0:
            noise_scale = max(phase_imprecision, phase_error)
            # Remove the arbitrary global phase before measuring the matrix
            # residual. This detects entry-wise fit errors even when the
            # trace overlap happens to cancel them for a particular target.
            phase_aligned_fitted = fitted * np.exp(-1j * np.angle(overlap))
            fit_error = float(
                np.linalg.norm(target - phase_aligned_fitted, ord="fro")
                / np.sqrt(component.m)
            )
            if fit_error > 0.1 * noise_scale:
                warnings.warn(
                    f"Unitary decomposition phase-aligned fit residual "
                    f"({fit_error:.3e}) is "
                    f"comparable to or exceeds the configured phase noise scale "
                    f"(max(phase_imprecision, phase_error)={noise_scale:.3e}). "
                    f"For component '{component.name}' on modes {tuple(r)}, the "
                    f"decomposition artifact may dominate over the modeled physical noise. "
                    f"Consider using smaller phase noise values or an exact decomposition method.",
                    UserWarning,
                    stacklevel=3,
                )

        mesh_parameters = mesh.get_parameters(all_params=False)
        output_phases = [p for p in mesh_parameters if p.name.startswith("phL")]
        if len(output_phases) != component.m:
            raise RuntimeError(
                f"Unexpected mesh structure from CircuitOptimizer for unitary "
                f"'{component.name}': expected {component.m} output phases "
                f"('phL*'), found {len(output_phases)}."
            )
        for parameter in mesh_parameters:
            value = float(parameter)
            if parameter.name.startswith("phL"):
                value -= alpha
            parameter.fix_value(value)

        # Store the completed mesh in the cache before adding to the circuit,
        # so that future builds for the same matrix skip the optimizer entirely.
        new_circuit.add(r[0], mesh, merge=True)
        _DECOMPOSITION_CACHE[cache_key] = copy.deepcopy(mesh)
    return new_circuit


class CircuitConverter:
    """Convert a parameterized Perceval circuit into a differentiable PyTorch unitary matrix.

    This class converts Perceval quantum circuits into PyTorch tensors that can be used
    in neural network training with automatic differentiation. It supports batch processing
    for efficient training and handles various quantum components like beam splitters,
    phase shifters, and unitary operations.

    Parameters
    ----------
    circuit : pcvl.Circuit
        Perceval circuit to convert.
    input_specs : list[str] | None
        Parameter name prefixes used to group parameters into input tensors.
    dtype : torch.dtype
        Target tensor dtype.
    device : torch.device
        Device used for tensor operations.
    phase_imprecision : float
        Deterministic quantization step applied to every phase shifter before
        building the unitary. This models finite phase-setting resolution: a
        commanded phase ``phi`` is mapped to
        ``round(phi / phase_imprecision) * phase_imprecision`` with a
        straight-through estimator, so the forward pass uses the quantized
        value while the backward pass keeps the identity gradient through the
        commanded phase. This is nearest-grid rounding, not truncation. Exact
        half-step ties follow ``torch.round`` behavior. For example,
        ``phi = pi / 8`` with ``phase_imprecision = pi / 4`` quantizes to
        ``0`` because ``round(0.5) == 0``. Default value is 0.0.
    phase_error : float
        Stochastic uniform perturbation half-width in radians. This models
        random phase noise around the quantized phase after any
        ``phase_imprecision`` step. When active, the effective sampled phase is
        ``round(phi / phase_imprecision) * phase_imprecision + epsilon`` with
        ``epsilon ~ Uniform(-phase_error, phase_error)``. If
        ``phase_imprecision`` is inactive, the sampled phase is
        ``phi + epsilon``. Fresh samples are drawn only when :meth:`to_tensor`
        is called with ``apply_phase_error=True``; otherwise the converter
        remains deterministic. Default value is 0.0.

    Notes
    -----
    Supported Components:
        - PS (Phase Shifter)
        - BS (Beam Splitter)
        - PERM (Permutation)
        - Unitary (Generic unitary matrix)
        - Barrier (no-op, removed during compilation)

    Phase Noise Parameter Flow:
        Phase noise parameters (`phase_imprecision` and `phase_error`) are
        configured at converter initialization and automatically applied during
        unitary generation. The flow is:

        1. **Initialization**: User passes `phase_imprecision` and/or
           `phase_error` to CircuitConverter via QuantumLayer
           (Step 4: through InitializationContext → ComputationProcessFactory
           → ComputationProcess → CircuitConverter).

        2. **Compilation**: During `_compile_circuit()`, constant phase shifters
           are marked as dynamic if `phase_error > 0.0`, ensuring fresh
           perturbations on each call. Quantization-only noise allows
           precomputation since it is deterministic.

        3. **Conversion**: Each call to `to_tensor(*params, apply_phase_error=bool)`
           applies both quantization (always, if configured) and perturbations
           (only if `apply_phase_error=True`). Monte Carlo sampling is done by
           calling `to_tensor()` multiple times with `apply_phase_error=True`
           and averaging the resulting probability distributions.

        4. **Gradient Flow**: Phase quantization uses straight-through estimators
           to preserve gradients to the commanded phase. Perturbations use
           `torch.empty_like(phase)` to ensure proper device/dtype handling
           and do NOT require gradients (they are stochastic noise, not
           learnable parameters).

        5. **Effective Phase**: For a phase shifter commanded to phase ``phi``,
           the forward phase is:

           - ``phi`` when both phase noises are inactive;
           - ``round(phi / phase_imprecision) * phase_imprecision`` when only
             ``phase_imprecision`` is active;
           - ``phi + epsilon`` when only ``phase_error`` is active;
           - ``round(phi / phase_imprecision) * phase_imprecision + epsilon``
             when both are active.

           The quantization uses nearest-grid rounding through
           :func:`torch.round`; it is not floor or truncation.

    Black-Box Unitaries Under Circuit Noise:
        When ``phase_imprecision > 0`` or ``phase_error > 0``, every
        non-``PERM`` ``Unitary`` component is automatically decomposed at
        construction into a rectangular (Clements) mesh of MZIs with fixed
        numeric phases plus an output PS layer. Phase noise is then applied
        to the PS components of the mesh (fidelity error ~1e-6, global phase
        compensated), matching Perceval's convention that ``phase_imprecision``
        targets phase shifters only. Without circuit noise the fast path is
        kept: ``Unitary`` components stay precomputed constant tensors.
        ``PERM`` components (waveguide crossings) are never decomposed.

    Example:
        Basic usage with a single phase shifter:

        >>> import torch
        >>> import perceval as pcvl
        >>> from merlin.pcvl_pytorch.locirc_to_tensor import CircuitConverter
        >>>
        >>> # Create a simple circuit with one phase shifter
        >>> circuit = pcvl.Circuit(1) // pcvl.PS(pcvl.P("phi"))
        >>>
        >>> # Convert to PyTorch with gradient tracking
        >>> converter = CircuitConverter(circuit, input_specs=["phi"])
        >>> phi_params = torch.tensor([0.5], requires_grad=True)
        >>> unitary = converter.to_tensor(phi_params)
        >>> print(unitary.shape)  # torch.Size([1, 1])

        Multiple parameters with grouping:

        >>> # Circuit with multiple phase shifters
        >>> circuit = (pcvl.Circuit(2)
        ...            // pcvl.PS(pcvl.P("theta1"))
        ...            // (1, pcvl.PS(pcvl.P("theta2"))))
        >>>
        >>> converter = CircuitConverter(circuit, input_specs=["theta"])
        >>> theta_params = torch.tensor([0.1, 0.2], requires_grad=True)
        >>> unitary = converter.to_tensor(theta_params)
        >>> print(unitary.shape)  # torch.Size([2, 2])

        Batch processing for training:

        >>> # Batch of parameter values
        >>> batch_params = torch.tensor([[0.1], [0.2], [0.3]], requires_grad=True)
        >>> converter = CircuitConverter(circuit, input_specs=["phi"])
        >>> batch_unitary = converter.to_tensor(batch_params)
        >>> print(batch_unitary.shape)  # torch.Size([3, 1, 1])

        Training integration:

        >>> # Training loop with beam splitter
        >>> circuit = pcvl.Circuit(2) // pcvl.BS.Rx(pcvl.P("theta"))
        >>> converter = CircuitConverter(circuit, ["theta"])
        >>> theta = torch.tensor([0.5], requires_grad=True)
        >>> optimizer = torch.optim.Adam([theta], lr=0.01)
        >>>
        >>> for step in range(10):
        ...     optimizer.zero_grad()
        ...     unitary = converter.to_tensor(theta)
        ...     loss = some_loss_function(unitary)
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        circuit: Circuit,
        input_specs: list[str] = None,
        memristive_metadata: list[dict] | None = None,
        dtype: torch.dtype = torch.complex64,
        device: torch.device = torch.device("cpu"),
        phase_imprecision: float = 0.0,
        phase_error: float = 0.0,
    ):
        """Initialize the CircuitConverter with a Perceval circuit.

        Parameters
        ----------
        circuit : pcvl.Circuit
            Parameterized Perceval circuit to convert.
        input_specs : list[str] | None
            Parameter name prefixes used to group parameters into separate
            tensors. If ``None``, all parameters go into a single tensor.
        memristive_metadata: list[dict] | None
            The memristive phase shifter metadata. If None, it will be stored as an empty list.
        dtype : torch.dtype
            Tensor dtype.
        device : torch.device
            PyTorch device for tensor operations.
        phase_imprecision : float
            Deterministic quantization step applied to every phase shifter
            before building the unitary. This models finite phase-setting
            resolution: a commanded phase ``phi`` is mapped to
            ``round(phi / phase_imprecision) * phase_imprecision`` with a
            straight-through estimator, so the forward pass uses the quantized
            value while the backward pass keeps the identity gradient through
            the commanded phase. This is nearest-grid rounding, not
            truncation. If omitted, no phase quantization is applied. Default
            value is 0.0.
        phase_error : float
            Stochastic uniform perturbation half-width in radians. This models
            random phase noise around the configured phase after any
            ``phase_imprecision`` quantization. The sampled perturbation is
            added after quantization, so both noises compose as
            ``round(phi / phase_imprecision) * phase_imprecision + epsilon``.
            Fresh samples are drawn only when :meth:`to_tensor` is called with
            ``apply_phase_error=True``; otherwise the converter remains
            deterministic. If omitted, no stochastic phase perturbation is
            configured. Default value is 0.0.

        Raises
        ------
        ValueError
            If ``input_specs`` do not match circuit parameters, or if a phase
            noise value is negative.
        TypeError
            If ``circuit`` is not a Perceval circuit.
        """

        # device is the device where the tensors will be allocated, default is set with torch.device('xxx')
        # in pytorch module, there is no discovery of the device from parameters, so it is the user's responsibility to
        # set the device, with .to() before calling the generation function
        self.device = device
        self.memristive_metadata = (
            [] if memristive_metadata is None else memristive_metadata
        )
        self.memristive_metadata_name_to_index = {
            memristive_metadata[i]["name"]: i
            for i in range(len(self.memristive_metadata))
        }
        self.memristive_current_state: list[torch.Tensor] = []
        self.input_params = None
        self.batch_size = 1

        self.set_dtype(dtype)
        self._phase_imprecision = float(phase_imprecision)
        self._phase_error = float(phase_error)
        self._apply_phase_error = False

        # Validate that phase noise parameters are non-negative
        if self._phase_imprecision < 0.0:
            raise ValueError("phase_imprecision must be non-negative.")
        if self._phase_error < 0.0:
            raise ValueError("phase_error must be non-negative.")

        assert isinstance(circuit, Circuit), (
            f"Expected a Perceval LO circuit, but got {type(circuit).__name__}"
        )
        # Circuit phase noise must reach the phases inside black-box Unitary
        # blocks, so decompose them into MZI meshes before building the
        # parameter mapping. Without noise the fast path (precomputed constant
        # tensor per Unitary) is kept.
        if self._phase_imprecision > 0.0 or self._phase_error > 0.0:
            circuit = _decompose_unitaries(
                circuit,
                phase_imprecision=self._phase_imprecision,
                phase_error=self._phase_error,
            )
        self.circuit = circuit
        if self._phase_error > 0.0 and _circuit_has_local_phase_error(self.circuit):
            warnings.warn(
                "Circuit contains pcvl.PS(max_error > 0) while phase_error is "
                "configured. Local PS max_error overrides phase_error for those "
                "phase shifters.",
                UserWarning,
                stacklevel=2,
            )

        # Create parameter mapping - it will map parameter names to their index in the input tensors
        self.param_mapping = {}
        self.spec_mappings = {}  # Track the mapping of input specs to parameter names

        self.nb_input_tensor = input_specs and len(input_specs) or 0
        param_names = [p.name for p in circuit.get_parameters()]

        if input_specs is None:
            self.param_mapping = {
                p.name: (0, idx) for idx, p in enumerate(self.circuit.get_parameters())
            }
        else:
            # Now create the mappings for parameters
            for i, spec in enumerate(input_specs):
                matching_params = [p for p in param_names if p.startswith(spec)]
                self.spec_mappings[spec] = matching_params

                if not matching_params:
                    raise ValueError(
                        f"No parameters found matching the input spec '{spec}'."
                    )
                for j, param in enumerate(matching_params):
                    self.param_mapping[param] = (i, j)

            # Check if all parameters are covered
            for param in param_names:
                if param not in self.param_mapping:
                    if not (
                        (param in self.memristive_metadata_name_to_index.keys())
                        and (len(self.memristive_metadata) > 0)
                    ):
                        raise ValueError(
                            f"Parameter '{param}' not covered by any input spec"
                        )

        self.list_rct = self._compile_circuit()

    def set_dtype(self, dtype: torch.dtype):
        """Set the tensor data types for float and complex operations.

        Parameters
        ----------
        dtype : torch.dtype
            Target dtype (float32/complex64 or float64/complex128).

        Raises
        ------
        TypeError
            If ``dtype`` is not supported.
        """
        float_dtype, complex_dtype = resolve_float_complex(dtype)
        self.tensor_fdtype = float_dtype
        self.tensor_cdtype = complex_dtype

    def to(self, dtype: torch.dtype, device: str | torch.device):
        """Move the converter to a specific device and dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Target tensor dtype (float32/complex64 or float64/complex128).
        device : str | torch.device
            Target device (string or torch.device).

        Returns
        -------
        CircuitConverter
            ``self`` for method chaining.

        Raises
        ------
        TypeError
            If ``device`` type is not supported.
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(
                f"Expected a string or torch.device, but got {type(device).__name__}"
            )
        self.set_dtype(dtype)

        for idx, (r, c) in enumerate(self.list_rct):
            if isinstance(c, torch.Tensor):
                self.list_rct[idx] = (
                    r,
                    c.to(dtype=self.tensor_cdtype, device=self.device),
                )
        # Memristive current state
        for state in range(len(self.memristive_current_state)):
            self.memristive_current_state[state] = self.memristive_current_state[
                state
            ].to(dtype=self.tensor_fdtype, device=self.device)

        return self

    def _compile_circuit(self):
        """Precompile the circuit to optimize performance.

        This method:
        1. Removes barrier components (no-ops)
        2. Precomputes tensors for components without parameters
        3. Merges adjacent non-parameterized components to reduce computation

        Returns
        -------
        list[tuple[range | object, torch.Tensor | AComponent]]
            List of (mode_range, component_or_tensor) tuples for the compiled circuit

        Raises
        ------
        TypeError
            If the circuit contains unsupported component types.
        """

        # we are building a list of components or precompiled tensors or dimension (1, m, m)
        list_rct = []
        for r, c in self.circuit:
            if not isinstance(c, SUPPORTED_COMPONENTS):
                raise TypeError(
                    f"{c} type not supported for conversion to PyTorch tensor."
                )
            if isinstance(c, Barrier):
                continue
            # A component must remain dynamic (not precomputed as a tensor) when
            # it carries phases that receive stochastic perturbations, because
            # fresh samples must be drawn on every call to to_tensor().
            # Deterministic phase_imprecision-only components can still be
            # precomputed: _compute_tensor applies quantization at that point,
            # and the result is the same on every call.
            #
            # PS: also sensitive when the component carries a per-component
            #     max_error (local phase error override).
            is_phase_error_sensitive = isinstance(c, PS) and (
                self._phase_error > 0.0 or _phase_shifter_max_error(c) > 0.0
            )
            if not c.get_parameters(all_params=False) and not is_phase_error_sensitive:
                # we can already compute the tensor for this component
                curr_comp_tensor = self._compute_tensor(c)
                list_rct.append((r, curr_comp_tensor))
            else:
                list_rct.append((r, c))

        # in second pass, we will be fusing the adjacent numeric components together
        for idx, (r, ct) in enumerate(list_rct):
            if ct is None:
                # this component has been merged with a previous one, skip it
                continue
            if isinstance(ct, torch.Tensor):
                # let us check all the following components that could be merged with this one
                merge_group = [(r, ct)]
                min_group = r[0]
                max_group = r[-1]
                blocked_modes = set()
                for j in range(idx + 1, len(list_rct)):
                    r2, c2 = list_rct[j]
                    if c2 is None:
                        continue
                    if not isinstance(c2, torch.Tensor) or any(
                        mode in blocked_modes for mode in r2
                    ):
                        for ir in r2:
                            blocked_modes.add(ir)
                        if len(blocked_modes) == self.circuit.m:
                            # all modes are blocked, we cannot merge anymore
                            break
                    else:
                        # we can merge this component with the previous one
                        merge_group.append((r2, c2))
                        if r2[0] < min_group:
                            min_group = r2[0]
                        if r2[-1] > max_group:
                            max_group = r2[-1]
                        # remove the component from the list
                        list_rct[j] = (r2, None)  # noqa: B909
                if len(merge_group) > 1:
                    # we have a group of components that can be merged
                    # we will compute the tensor for the whole group
                    merged_tensor = torch.eye(
                        max_group - min_group + 1,
                        dtype=self.tensor_cdtype,
                        device=self.device,
                    )
                    for r, c in merge_group:
                        c = c.to(self.device)
                        merged_tensor[r[0] - min_group : (r[-1] - min_group + 1), :] = (
                            c
                            @ merged_tensor[
                                r[0] - min_group : (r[-1] - min_group + 1), :
                            ]
                        )
                    list_rct[idx] = (range(min_group, max_group + 1), merged_tensor)

        # Remove None entries from the list
        return [item for item in list_rct if item[1] is not None]

    def to_tensor(
        self,
        *input_params: torch.Tensor,
        batch_size: int | None = None,
        apply_phase_error: bool = False,
        memristive_current_state: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        r"""Convert the parameterized circuit to a PyTorch unitary tensor.

        Phase Noise Processing:
            This method applies configured phase noise to all phase shifters during
            unitary generation. The noise is applied in two stages:

            1. **phase_imprecision (deterministic, always applied)**:
               If configured, every phase is quantized to the nearest multiple of
               ``phase_imprecision`` using a straight-through estimator. This uses
               ``torch.round(phase / phase_imprecision) * phase_imprecision``:
               it is nearest-grid rounding, not truncation. Exact half-step ties
               follow ``torch.round`` behavior, so ``pi / 8`` with a ``pi / 4``
               step quantizes to ``0``. Gradients flow through the commanded phase,
               while the forward pass uses the quantized value. This is always
               active and does not require ``apply_phase_error=True``.

            2. **phase_error (stochastic, controlled by apply_phase_error flag)**:
               If configured and `apply_phase_error=True`, fresh samples from
               Uniform(-phase_error, phase_error) are drawn and added to each phase
               after quantization. The samples respect the phase tensor's device and
               dtype via `torch.empty_like()`. Each call with `apply_phase_error=True`
               produces a different unitary. For Monte Carlo averaging of probabilistic
               outputs, call this method multiple times with `apply_phase_error=True`,
               collect the resulting probability distributions, and average them.

            Parameter Flow (see class Notes for full context):
            - layer_utils.classify_noise() → extracts phase settings to NoiseGroups
            - ComputationProcess.__init__() → stores phase settings from NoiseGroups
            - ComputationProcess._setup_computation_graphs() → passes to CircuitConverter
            - CircuitConverter.to_tensor() ← receives apply_phase_error flag each call

        Parameters
        ----------
        input_params : torch.Tensor
            Variable number of parameter tensors. Each tensor has shape
            ``(num_params,)`` or ``(batch_size, num_params)`` in the order of
            ``input_specs``.
        batch_size : int | None
            Explicit batch size. If ``None``, it is inferred from the input
            tensors.
        memristive_current_state : list[torch.Tensor] | None
            The memristive phase shifters current states. Defaults to None
            and will be treated as an empty list.
        apply_phase_error : bool
            Whether to draw fresh stochastic perturbations for configured
            ``phase_error`` values during this conversion. This flag does not
            affect deterministic ``phase_imprecision`` quantization, which is
            applied whenever ``phase_imprecision`` is positive. The perturbation
            is added after quantization. Default value is False.

        Returns
        -------
        torch.Tensor
            Complex unitary tensor of shape ``(circuit.m, circuit.m)`` for a
            single sample or ``(batch_size, circuit.m, circuit.m)`` for batched
            inputs.

        Raises
        ------
        ValueError
            If the wrong number of input tensors is provided.
        TypeError
            If ``input_params`` is not a list or tuple.
        """
        if len(input_params) == 1 and isinstance(input_params[0], list):
            input_params = input_params[0]  # type: ignore[assignment]
        if len(input_params) != self.nb_input_tensor:
            raise ValueError(
                f"Expected {self.nb_input_tensor} input tensors, but got {len(input_params)}."
            )
        if not isinstance(input_params, list) and not isinstance(input_params, tuple):
            raise TypeError(
                f"Expected a list of input tensors, but got {type(input_params).__name__}."
            )

        self.torch_params = input_params
        self.memristive_current_state = (
            [] if memristive_current_state is None else memristive_current_state
        )
        if len(self.memristive_current_state) < len(self.memristive_metadata):
            raise ValueError(
                "Expected at least "
                f"{len(self.memristive_metadata)} memristive current state value(s) "
                f"for the configured memristive metadata, but got "
                f"{len(self.memristive_current_state)}."
            )

        if batch_size is None:
            if input_params and input_params[0].dim() > 1:
                has_batch = True
                batch_size = input_params[0].shape[0]
            else:
                has_batch = False
                batch_size = 1
        else:
            has_batch = True
        self.batch_size = batch_size

        previous_apply_phase_error = self._apply_phase_error
        self._apply_phase_error = apply_phase_error
        try:
            converted_tensor = (
                torch
                .eye(self.circuit.m, dtype=self.tensor_cdtype, device=self.device)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
            # Build unitary tensor by composing component unitaries
            for r, c in self.list_rct:
                if isinstance(c, torch.Tensor):
                    # If the component is already a tensor, use it directly, just move it to the correct device and dtype
                    # and expand it to the batch size
                    curr_comp_tensor = c.to(
                        dtype=self.tensor_cdtype, device=self.device
                    ).expand(batch_size, -1, -1)
                else:
                    curr_comp_tensor = self._compute_tensor(c)

                # Compose unitaries
                contribution = converted_tensor[..., r[0] : (r[-1] + 1), :].clone()
                converted_tensor[..., r[0] : (r[-1] + 1), :] = (
                    curr_comp_tensor @ contribution.to(curr_comp_tensor.device)
                )
        finally:
            self._apply_phase_error = previous_apply_phase_error

        if not has_batch:
            # If no batch dimension was provided, remove the batch dimension
            converted_tensor = converted_tensor.squeeze(0)

        return converted_tensor

    def _apply_phase_noise(
        self,
        phase: torch.Tensor,
        phase_error_half_width: float,
    ) -> torch.Tensor:
        """Apply configured phase noise to a phase tensor in-place (returns new tensor).

        Applies, in order:

        1. Deterministic nearest-grid quantization (STE) when
           ``self._phase_imprecision > 0``.
        2. Stochastic uniform perturbation when ``self._apply_phase_error`` is
           ``True`` and ``phase_error_half_width > 0``.

        This helper is used by the :meth:`_compute_tensor` overload for ``PS``.
        Phase noise is not applied to ``BS`` parameters, matching Perceval's
        ``phase_imprecision`` noise model which targets only phase shifters.

        Parameters
        ----------
        phase : torch.Tensor
            Phase value(s) as a real-valued tensor. The tensor may be scalar,
            1-D ``(batch_size,)``, or any shape that is compatible with the
            calling code.
        phase_error_half_width : float
            Half-width of the ``Uniform(-w, w)`` perturbation to apply when
            ``self._apply_phase_error`` is ``True``. The caller selects the
            per-component ``max_error`` or falls back to ``self._phase_error``.

        Returns
        -------
        torch.Tensor
            Phase tensor with quantization and/or perturbation applied. The
            returned tensor shares the device and dtype of the input.
        """
        if self._phase_imprecision > 0.0:
            step = phase.new_tensor(self._phase_imprecision)
            phase_quantized = torch.round(phase / step) * step
            # Straight-through estimator: autograd sees d phase / d commanded = 1.
            phase = phase + (phase_quantized - phase).detach()

        if self._apply_phase_error and phase_error_half_width > 0.0:
            noise = torch.empty_like(phase).uniform_(
                -phase_error_half_width, phase_error_half_width
            )
            phase = phase + noise

        return phase

    @dispatch((Unitary, PERM))
    def _compute_tensor(self, comp: AComponent) -> torch.Tensor:
        """Compute tensor for Unitary and Permutation components.

        Args:
            comp: Unitary or PERM component (no parameters)

        Returns:
            Batched unitary tensor of shape (batch_size, comp_size, comp_size)
        """
        return (
            torch
            .tensor(
                comp.compute_unitary(), dtype=self.tensor_cdtype, device=self.device
            )
            .unsqueeze(0)
            .expand(self.batch_size, -1, -1)
        )

    @dispatch(BS)
    def _compute_tensor(self, comp: AComponent) -> torch.Tensor:  # type: ignore[no-redef]
        """Compute tensor for Beam Splitter component.

        Handles different BS conventions (Rx, Ry, H) and processes 5 parameters:
        theta, phi_tl, phi_bl, phi_tr, phi_br.

        Phase noise (``phase_imprecision`` and ``phase_error``) is not applied to
        BS parameters. Noise is applied exclusively to PS (phase shifter)
        components, matching Perceval's ``phase_imprecision`` noise model.

        Parameters
        ----------
        comp : AComponent
            ``BS`` component with parameters.

        Returns
        -------
        torch.Tensor
            Batched 2×2 unitary tensor of shape ``(batch_size, 2, 2)``.

        Raises
        ------
        NotImplementedError
            If the BS convention is not ``Rx``, ``Ry``, or ``H``.
        """
        param_values = []

        for _index, param in enumerate(comp.get_parameters(all_params=True)):
            if param.is_variable:
                tensor_id, idx_in_tensor = self.param_mapping[param.name]
                raw = self.torch_params[tensor_id][..., idx_in_tensor]
            else:
                raw = torch.tensor(
                    float(param), dtype=self.tensor_fdtype, device=self.device
                )
            param_values.append(raw)

        cos_theta = torch.cos(param_values[0] / 2)
        sin_theta = torch.sin(param_values[0] / 2)
        phi_tl_tr = param_values[1] + param_values[3]  # phi_tl_val + phi_tr_val
        u00_mul = torch.cos(phi_tl_tr) + 1j * torch.sin(phi_tl_tr)

        phi_tr_bl = param_values[3] + param_values[2]  # phi_tr_val + phi_bl_val
        u01_mul = torch.cos(phi_tr_bl) + 1j * torch.sin(phi_tr_bl)

        phi_tl_br = param_values[1] + param_values[4]  # phi_tl_val + phi_br_val
        u10_mul = torch.cos(phi_tl_br) + 1j * torch.sin(phi_tl_br)

        phi_bl_br = param_values[2] + param_values[4]  # phi_bl_val + phi_br_val
        u11_mul = torch.cos(phi_bl_br) + 1j * torch.sin(phi_bl_br)

        bs_convention = comp._convention
        if bs_convention == BSConvention.Rx:
            unitary_tensor = torch.tensor(
                [[1, 1j], [1j, 1]], dtype=self.tensor_cdtype, device=self.device
            )
        elif bs_convention == BSConvention.Ry:
            unitary_tensor = torch.tensor(
                [[1, -1], [1, 1]], dtype=self.tensor_cdtype, device=self.device
            )
        elif bs_convention == BSConvention.H:
            unitary_tensor = torch.tensor(
                [[1, 1], [1, -1]], dtype=self.tensor_cdtype, device=self.device
            )
        else:
            raise NotImplementedError(
                f"BS convention : {comp._convention.name} not supported."
            )

        unitary_tensor = (
            unitary_tensor
            .unsqueeze(0)
            .repeat(self.batch_size, 1, 1)
            .to(cos_theta.device)
        )
        unitary_tensor[..., 0, 0] *= u00_mul.to(self.device) * cos_theta
        unitary_tensor[..., 0, 1] *= u01_mul.to(self.device) * sin_theta
        unitary_tensor[..., 1, 1] *= u11_mul.to(self.device) * cos_theta
        unitary_tensor[..., 1, 0] *= u10_mul.to(self.device) * sin_theta
        return unitary_tensor

    @dispatch(PS)
    def _compute_tensor(self, comp: AComponent) -> torch.Tensor:  # type: ignore[no-redef]
        """Compute tensor for Phase Shifter component.

        Applies phase noise to the phase value before constructing the phase
        unitary exp(1j * phase). This method is called by to_tensor() for each
        phase shifter in the circuit.

        Phase Noise Processing (in order):
            1. **Read Phase**: Retrieve the phase value from the PS component.
               This can be a constant, a trainable parameter, or an input-driven
               parameter. The value is converted to real dtype for noise application.

            2. **Quantization (phase_imprecision)**:
               If self._phase_imprecision > 0.0, apply deterministic STE
               quantization. The commanded phase is rounded to the nearest
               multiple of ``phase_imprecision`` with
               ``torch.round(phase / phase_imprecision) * phase_imprecision``,
               while gradients pass through the original commanded phase
               unchanged. This models finite phase resolution in hardware.
               It is not truncation. For example, ``pi / 8`` with a
               ``pi / 4`` imprecision is exactly half a step, and
               ``torch.round(0.5)`` sends it to ``0``.

            3. **Perturbation (phase_error)**:
               If self._apply_phase_error and self._phase_error > 0.0, draw
               fresh ``Uniform(-phase_error, phase_error)`` samples and add them
               to the quantized phase. If quantization is inactive, samples are
               added to the commanded phase. Samples are drawn using
               ``torch.empty_like(phase)`` to respect the phase tensor's device,
               dtype, and batch shape. Perturbations do NOT require gradients;
               they are stochastic noise, not learnable. Optimization updates
               the commanded phase, not the noise.

            4. **Complex Conversion**: Convert the noisy phase to the complex
               phase unitary exp(1j * phase).

        Gradient Flow:
            - Quantization: STE ensures gradients bypass the quantization step
            - Perturbations: .detach() on phase_error ensures no gradients flow
              through noise samples, only through the commanded phase
            - Result: dL/d(phase_commanded) is well-defined and updates the
              circuit parameters during backprop

        Device and Dtype Safety:
            - Perturbations use torch.empty_like(phase) to match device/dtype
            - Batch handling is automatic via broadcasting
            - Results are in the converter's configured complex_dtype

        Args:
            comp: PS component with phi parameter

        Returns:
            Batched 1x1 phase tensor of shape (batch_size, 1, 1) in complex dtype
        """
        if comp.param("phi").is_variable:
            param_name = comp.param("phi").name
            if len(self.memristive_metadata) > 0:
                if param_name in self.memristive_metadata_name_to_index.keys():
                    index = self.memristive_metadata_name_to_index[param_name]
                    phase = self.memristive_current_state[index]
                else:
                    tensor_id, idx_in_tensor = self.param_mapping[param_name]
                    phase = self.torch_params[tensor_id][..., idx_in_tensor]
            else:
                tensor_id, idx_in_tensor = self.param_mapping[param_name]
                phase = self.torch_params[tensor_id][..., idx_in_tensor]

        else:
            phase = torch.tensor(
                comp.param("phi")._value, dtype=self.tensor_fdtype, device=self.device
            )

        if phase.ndim == 0 and self.batch_size > 1:
            phase = phase.expand(self.batch_size)

        # Compute phase error half-width: prefer component-specific max error,
        # then fall back to the global phase_error
        component_phase_error = _phase_shifter_max_error(comp)
        phase_error_half_width = (
            component_phase_error if component_phase_error > 0.0 else self._phase_error
        )

        # Apply quantization and perturbation via the common helper
        phase = self._apply_phase_noise(phase, phase_error_half_width)

        unitary_tensor = torch.exp(1j * phase.to(self.tensor_cdtype)).reshape(
            -1, 1
        )  # reshape so that in any case, we have 2 dim
        return unitary_tensor.unsqueeze(-1)  # to change shape of tensor to (b, 1, 1)
