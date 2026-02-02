from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from ..core.computation_space import ComputationSpace

if TYPE_CHECKING:
    from ..measurement.strategies import (
        MeasurementStrategyLike,
    )

_MEASUREMENT_STRATEGY_ENUM_MIGRATIONS = {
    "NONE": "amplitudes()",
    "PROBABILITIES": "probs(computation_space)",
    "MODE_EXPECTATIONS": "mode_expectations(computation_space)",
    "AMPLITUDES": "amplitudes()",
}


def _convert_no_bunching_init(
    method_qualname: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Converter for QuantumLayer.__init__ deprecated `no_bunching`.
    Removes `no_bunching`, sets/validates `computation_space`.
    """
    no_bunching = kwargs.pop("no_bunching", None)
    comp_space_in = kwargs.get("computation_space", None)

    if comp_space_in is None:
        if no_bunching is None:
            comp_value = ComputationSpace.UNBUNCHED
        else:
            comp_value = ComputationSpace.default(no_bunching=bool(no_bunching))
    else:
        comp_value = ComputationSpace.coerce(comp_space_in)
        if no_bunching is not None:
            derived_nb = comp_value is ComputationSpace.UNBUNCHED
            if bool(no_bunching) != derived_nb:
                raise ValueError(
                    "Incompatible 'no_bunching' value with selected 'computation_space'. "
                )

    kwargs["computation_space"] = comp_value
    return kwargs


# ---------------------------------------------------------------------------
# Deprecation registry (parameter-based)
# ---------------------------------------------------------------------------


def _remove_QuantumLayer_simple_n_params(
    method_qualname: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Removes the n_params parameter from QuantumLayer.simple()"""
    _ = kwargs.pop("n_params", None)
    return kwargs


def _remove_FeatureMap_simple_n_photons(
    method_qualname: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Removes the n_photons parameter from FeatureMap.simple()"""
    _ = kwargs.pop("n_photons", None)
    return kwargs


def _remove_FeatureMap_simple_trainable(
    method_qualname: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Removes the trainable parameter from FeatureMap.simple()"""
    _ = kwargs.pop("trainable", None)
    return kwargs


def _remove_FidelityKernel_simple_n_photons(
    method_qualname: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Removes the n_photons parameter from FidelityKernel.simple()"""
    _ = kwargs.pop("n_photons", None)
    return kwargs


def _remove_FidelityKernel_simple_trainable(
    method_qualname: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Removes the trainable parameter from FidelityKernel.simple()"""
    _ = kwargs.pop("trainable", None)
    return kwargs


def _remove_FidelityKernel_input_state(
    method_qualname: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Removes the input_state parameter from FidelityKernel.simple()"""
    _ = kwargs.pop("input_state", None)
    return kwargs


# Global deprecation registry: keys are "ClassName.method_name.param_name"
# Values are tuples: (message, severity, converter)
# - message: str | None → the text to emit; None means no emission
# - severity: True | False | None → True=error, False=warning, None=silent
# - converter: optional callable to sanitize kwargs when the param is present
DEPRECATION_REGISTRY: dict[
    str,
    tuple[
        str | None,
        bool | None,
        Callable[[str, dict[str, Any]], dict[str, Any]] | None,
    ],
] = {
    # QuantumLayer.__init__ deprecations
    "QuantumLayer.__init__.ansatz": (
        "Use 'circuit' or 'CircuitBuilder' to define the quantum circuit.",
        True,
        None,
    ),
    "QuantumLayer.__init__.no_bunching": (
        "The 'no_bunching' keyword is deprecated; prefer selecting the computation_space instead.",
        False,
        _convert_no_bunching_init,
    ),
    # QuantumLayer.simple deprecations
    "QuantumLayer.simple.no_bunching": (
        "The 'no_bunching' keyword is deprecated; prefer selecting the computation_space instead.",
        False,
        _convert_no_bunching_init,
    ),
    "QuantumLayer.simple.computation_space": (
        "The 'computation_space' keyword is deprecated; move it into MeasurementStrategy.probs(computation_space).",
        False,
        None,
    ),
    "QuantumLayer.simple.n_params": (
        "Since merlin >= 0.3, input parameter allocation is automatically inferred from input dimensionality, following Gan et al. (2022) on Fock-space expressivity. Manual control of input/trainable parameters is deprecated.",
        False,
        _remove_QuantumLayer_simple_n_params,
    ),
    "QuantumLayer.simple.reservoir_mode": (
        "The 'reservoir_mode' argument is no longer supported in the 'simple' method. Use torch tooling to freeze weights when needed, e.g., call layer.requires_grad_(False).",
        True,
        None,
    ),
    # QuantumLayer.set_sampling_config method-level deprecation (fatal)
    "QuantumLayer.set_sampling_config": (
        "QuantumLayer.set_sampling_config() is deprecated. Provide 'shots' and 'sampling_method' directly to 'forward()'.",
        True,
        None,
    ),
    # FeatureMap.simple deprecations
    "FeatureMap.simple.n_photons": (
        "Since merlin >= 0.3, the number of photons is automatically inferred from input dimensionality. Manual control of photons is deprecated.",
        False,
        _remove_FeatureMap_simple_n_photons,
    ),
    "FeatureMap.simple.trainable": (
        "Since merlin >= 0.3, input parameter allocation is automatically inferred from input dimensionality, following Gan et al. (2022) on Fock-space expressivity. Manual control of input/trainable parameters is deprecated.",
        False,
        _remove_FeatureMap_simple_trainable,
    ),
    # FidelityKernel.simple deprecations
    "FidelityKernel.simple.n_photons": (
        "Since merlin >= 0.3, the number of photons is automatically inferred from input dimensionality. Manual control of photons is deprecated.",
        False,
        _remove_FidelityKernel_simple_n_photons,
    ),
    "FidelityKernel.simple.trainable": (
        "Since merlin >= 0.3, input parameter allocation is automatically inferred from input dimensionality, following Gan et al. (2022) on Fock-space expressivity. Manual control of input/trainable parameters is deprecated.",
        False,
        _remove_FidelityKernel_simple_trainable,
    ),
    "FidelityKernel.simple.input_state": (
        "Since merlin >= 0.3, The input state is alway going to be a [0,1,0,1,...] state depending on input size.",
        False,
        _remove_FidelityKernel_input_state,
    ),
}

# ---------------------------------------------------------------------------
# Deprecation helpers (registry + converters)
# ---------------------------------------------------------------------------


def _collect_deprecations_and_converters(
    method_qualname: str, raw_kwargs: dict[str, Any]
) -> tuple[
    list[str],
    list[str],
    list[Callable[[str, dict[str, Any]], dict[str, Any]]],
]:
    """Inspect kwargs against the global deprecation registry and return:
    - warn messages (non-fatal deprecations),
    - raise messages (fatal deprecations),
    - converters to apply (callables) for present deprecated params.
    """
    warn_msgs: list[str] = []
    raise_msgs: list[str] = []
    converters: list[Callable[[str, dict[str, Any]], dict[str, Any]]] = []

    # Method-level deprecation without a specific parameter
    if method_qualname in DEPRECATION_REGISTRY:
        msg, severity, converter = DEPRECATION_REGISTRY[method_qualname]
        if msg is not None and severity is not None:
            base = msg
            if severity is True:
                raise_msgs.append(base)
            elif severity is False:
                warn_msgs.append(base)
        if converter is not None:
            converters.append(converter)

    for key in sorted(raw_kwargs.keys()):
        full_name = f"{method_qualname}.{key}"
        if full_name in DEPRECATION_REGISTRY:
            msg, severity, converter = DEPRECATION_REGISTRY[full_name]
            if msg is not None and severity is not None:
                base = f"Parameter '{key}' is deprecated. {msg}"
                if severity is True:
                    raise_msgs.append(base)
                elif severity is False:
                    warn_msgs.append(base)
            if converter is not None:
                converters.append(converter)

    return warn_msgs, raise_msgs, converters


# ---------------------------------------------------------------------------
# MeasurementStrategy normalization + deprecations
# ---------------------------------------------------------------------------


def normalize_measurement_strategy(
    measurement_strategy: MeasurementStrategyLike | str | None,
    computation_space: ComputationSpace | str | None,
) -> tuple[MeasurementStrategyLike, ComputationSpace]:
    """Normalize measurement strategy + computation space with deprecation warnings.

    Enforces the v0.3 requirement that computation_space must live inside MeasurementStrategy
    when using the new factory methods (probs, mode_expectations, partial).

    Rules:
    1. If MeasurementStrategy instance (new API) + constructor computation_space provided
       → ERROR: user must move computation_space into the factory method
    2. If legacy enum (PROBABILITIES, etc) + constructor computation_space
       → OK with deprecation warning (backward compat)
    3. If MeasurementStrategy instance only → use its computation_space
    4. If legacy enum only → wrap with computation_space param
    """
    from ..measurement.strategies import (
        MeasurementKind,
        MeasurementStrategy,
        _LegacyMeasurementStrategy,
    )

    # Track whether computation_space was explicitly provided by user
    computation_space_provided = computation_space is not None

    if measurement_strategy is None:
        if computation_space is None:
            computation_space = ComputationSpace.UNBUNCHED
        else:
            computation_space = ComputationSpace.coerce(computation_space)
        measurement_strategy = MeasurementStrategy.probs(computation_space)
        return measurement_strategy, computation_space

    if isinstance(measurement_strategy, str):
        warnings.warn(
            "Passing measurement_strategy as a string is deprecated. "
            "Use MeasurementStrategy.probs(...) instead. Will be removed in v0.4.",
            DeprecationWarning,
            stacklevel=2,
        )
        normalized = measurement_strategy.upper()
        try:
            measurement_strategy = MeasurementKind[normalized]
        except KeyError as exc:
            raise TypeError(
                f"Unknown measurement_strategy: {measurement_strategy}"
            ) from exc

    if isinstance(measurement_strategy, MeasurementStrategy):
        # NEW API: MeasurementStrategy instance (e.g., from .probs(), .partial(), etc)
        strategy_space = measurement_strategy.computation_space
        if strategy_space is None:
            raise ValueError(
                "MeasurementStrategy must define computation_space. "
                "Use MeasurementStrategy.probs(computation_space) instead."
            )

        # CONFLICT CHECK: Constructor computation_space + new factory method
        if computation_space_provided:
            raise TypeError(
                "Cannot specify 'computation_space' in QuantumLayer constructor "
                "when using MeasurementStrategy.probs(), .mode_expectations(), or .partial(). "
                "Move 'computation_space' into the factory method instead. "
                "For example: MeasurementStrategy.probs(computation_space=ComputationSpace.FOCK) "
                "instead of QuantumLayer(..., computation_space=..., measurement_strategy=...)."
            )

        return measurement_strategy, strategy_space

    # Only set default if not explicitly provided
    if computation_space is None:
        computation_space = ComputationSpace.UNBUNCHED
    else:
        computation_space = ComputationSpace.coerce(computation_space)

    if isinstance(measurement_strategy, _LegacyMeasurementStrategy):
        # LEGACY API: Enum-style access (PROBABILITIES, MODE_EXPECTATIONS, AMPLITUDES)
        # These are allowed with constructor computation_space for backward compat
        if computation_space_provided:
            warnings.warn(
                "Passing 'computation_space' as a separate argument with legacy "
                "MeasurementStrategy enum values is deprecated. "
                "Use MeasurementStrategy.probs(computation_space=...) instead. "
                "Will be required in v0.4.",
                DeprecationWarning,
                stacklevel=2,
            )

        if measurement_strategy == _LegacyMeasurementStrategy.PROBABILITIES:
            measurement_strategy = MeasurementStrategy.probs(computation_space)
        elif measurement_strategy == _LegacyMeasurementStrategy.MODE_EXPECTATIONS:
            measurement_strategy = MeasurementStrategy.mode_expectations(
                computation_space
            )
        elif measurement_strategy == _LegacyMeasurementStrategy.AMPLITUDES:
            measurement_strategy = MeasurementStrategy.amplitudes()
        elif measurement_strategy == _LegacyMeasurementStrategy.NONE:
            measurement_strategy = MeasurementStrategy.amplitudes()

    return measurement_strategy, computation_space


def warn_deprecated_enum_access(owner: str, name: str) -> bool:
    """Warn on deprecated enum-style attribute access and return True if handled."""
    if owner == "MeasurementStrategy" and name in _MEASUREMENT_STRATEGY_ENUM_MIGRATIONS:
        replacement = _MEASUREMENT_STRATEGY_ENUM_MIGRATIONS[name]
        warnings.warn(
            f"{owner}.{name} is deprecated. Use {owner}.{replacement} instead. "
            "Will be removed in v0.4.",
            DeprecationWarning,
            stacklevel=2,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Decorator API
# ---------------------------------------------------------------------------

# (converter defined above and referenced inline in the registry)


F = TypeVar("F", bound=Callable[..., Any])


@overload
def sanitize_parameters(func: F) -> F:  # bare decorator usage
    ...


@overload
def sanitize_parameters(
    *processors: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> Callable[[F], F]:  # factory usage with processors
    ...


def sanitize_parameters(*args: Any, **_kw: Any) -> Any:
    """Decorator to centralize parameter sanitization for method calls.

    Usage:
    - As a plain decorator: `@sanitize_parameters` (no parentheses)
    - As a factory with processors: `@sanitize_parameters(proc1, proc2, ...)`

    Behavior:
    - Emits standardized warnings/errors based on the global deprecation registry.
    - Applies converter functions registered for present deprecated params.
    - Applies any additional `processors(qual, kwargs)` provided, sequentially, each receiving and returning kwargs.
    """

    def _build_decorator(
        processors: Sequence[Callable[[str, dict[str, Any]], dict[str, Any]]],
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*f_args: Any, **kwargs: Any) -> Any:
                if not f_args:
                    # Defensive: methods should always receive `self` as first arg.
                    return func(*f_args, **kwargs)

                # Use __qualname__ to capture Class.method.
                qual = func.__qualname__

                # 1) Collect deprecation messages and converters based on present kwargs
                warn_msgs, raise_msgs, converters = (
                    _collect_deprecations_and_converters(qual, kwargs)
                )
                if raise_msgs:
                    raise ValueError(" ".join(raise_msgs))
                if warn_msgs:
                    warnings.warn(" ".join(warn_msgs), DeprecationWarning, stacklevel=2)

                # 2) Apply converters for deprecated params
                for conv in converters:
                    kwargs = conv(qual, dict(kwargs))

                # 2b) Apply optional processors
                for proc in processors:
                    kwargs = proc(qual, dict(kwargs))

                # 3) Rely on Python's own signature checking to reject unknown kwargs.

                return func(*f_args, **kwargs)

            return wrapper

        return decorator

    # Bare decorator usage: @sanitize_parameters
    if len(args) == 1 and callable(args[0]) and hasattr(args[0], "__qualname__"):
        func = cast(Callable[..., Any], args[0])
        params = list(inspect.signature(func).parameters.values())
        if params and params[0].name in {"self", "cls"}:
            return _build_decorator([])(func)

    # Factory usage: @sanitize_parameters(proc1, proc2, ...)
    processors = cast(Sequence[Callable[[str, dict[str, Any]], dict[str, Any]]], args)
    return _build_decorator(processors)
