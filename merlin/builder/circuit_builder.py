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

"""
Circuit builder for constructing quantum circuits declaratively.
"""

import inspect
import math
import numbers
from collections.abc import Callable
from itertools import combinations
from typing import Any

import torch

from ..core.circuit import Circuit
from ..core.components import (
    BeamSplitter,
    EntanglingBlock,
    GenericInterferometer,
    ParameterRole,
    Rotation,
)

ANGLE_ENCODING_MODE_ERROR = "You cannot encore more features than mode with Builder, try making your own circuit by building your Circuit with Perceval"


class ModuleGroup:
    """Helper class for a group of circuit modes.

    Parameters
    ----------
    modes : list[int]
        Modes spanned by the grouped module.
    """

    def __init__(self, modes: list[int]):
        """Store the list of modes spanned by the grouped module.

        Parameters
        ----------
        modes : list[int]
            Modes spanned by the grouped module.
        """
        self.modes = modes


class CircuitBuilder:
    """Builder for quantum circuits using a declarative API.

    Parameters
    ----------
    n_modes : int
        Number of photonic modes available in the circuit.
    """

    def __init__(self, n_modes: int):
        """Initialize bookkeeping structures for a circuit with ``n_modes`` modes.

        Parameters
        ----------
        n_modes : int
            Number of photonic modes available in the circuit.
        """
        self.n_modes = n_modes
        self.circuit = Circuit(n_modes)

        # Track component counts for naming - these should NEVER reset
        self._layer_counter = 0
        self._trainable_counter = 0
        self._input_counter = 0
        self._entangling_layer_counter = 0
        self._superposition_counter = 0
        self._entangling_counter = 0
        self._memristor_counter = 0

        self._trainable_prefixes: list[str] = []
        self._trainable_prefix_set: set[str] = set()
        self._input_prefixes: list[str] = []
        self._input_prefix_set: set[str] = set()
        self._angle_encoding_specs: dict[str, list[tuple[int, ...]]] = {}
        self._angle_encoding_scales: dict[str, dict[int, float]] = {}
        self._angle_encoding_counts: dict[str, int] = {}
        self.memristive_specs: list[dict] = []
        self._memristor_prefixes: list[str] = []
        self._memristor_prefix_set: set[str] = set()
        self._trainable_name_counts: dict[str, int] = {}
        self._used_trainable_names: set[str] = set()

    @staticmethod
    def _deduce_prefix(name: str | None) -> str | None:
        """Strip numeric suffixes so we can reuse the textual stem as a prefix.

        Args:
            name: Full parameter name provided by the user or generator.

        Returns:
            Optional[str]: The textual stem without trailing digits or underscores.
        """
        if not name:
            return None

        base = name
        while True:
            trimmed = base.rstrip("0123456789")
            trimmed = trimmed.rstrip("_")
            if trimmed == base:
                break
            base = trimmed

        return base or name

    def _register_trainable_prefix(self, name: str | None):
        """Record the stem of a trainable parameter for later discovery calls.

        Args:
            name: Newly created trainable parameter name whose stem should be tracked.
        """
        prefix = self._deduce_prefix(name)
        if prefix and prefix not in self._trainable_prefix_set:
            self._trainable_prefix_set.add(prefix)
            self._trainable_prefixes.append(prefix)

    def _register_input_prefix(self, name: str | None):
        """Track stems used for data-driven parameters (angle encodings).

        Args:
            name: Input parameter name emitted while constructing an encoding layer.
        """
        prefix = self._deduce_prefix(name)
        if prefix and prefix not in self._input_prefix_set:
            self._input_prefix_set.add(prefix)
            self._input_prefixes.append(prefix)

    def _register_memristor_prefix(self, name: str | None):
        """Track stems used for memristor parameters.

        Args:
            name: Input parameter name emitted while adding a memristive phase-shifter.
        """
        prefix = self._deduce_prefix(name)
        if prefix and prefix not in self._input_prefix_set:
            self._memristor_prefix_set.add(prefix)
            self._memristor_prefixes.append(prefix)

    def _unique_trainable_name(self, base: str) -> str:
        """Return a unique trainable identifier derived from ``base``.

        Args:
            base: Desired stem for the parameter name (may collide with earlier ones).

        Returns:
            str: Collision-free parameter name derived from ``base``.
        """
        count = self._trainable_name_counts.get(base, 0)
        candidate = base if count == 0 else f"{base}_{count}"

        # Retry with incremented suffix while the candidate is already bound
        while candidate in self._used_trainable_names:
            count += 1
            candidate = f"{base}_{count}"

        # Next request for the same base will continue from the updated count
        self._trainable_name_counts[base] = count + 1 if candidate != base else 1
        self._used_trainable_names.add(candidate)
        return candidate

    def add_rotations(
        self,
        modes: int | list[int] | ModuleGroup | None = None,
        *,
        axis: str = "z",
        trainable: bool = False,
        as_input: bool = False,
        angle: float | None = None,
        value: float | None = None,
        name: str | None = None,
        role: str | ParameterRole | None = None,
    ) -> "CircuitBuilder":
        """Add one or multiple rotations across the provided modes.

        Parameters
        ----------
        modes : int | list[int] | ModuleGroup | None
            Single mode, list of modes, module group, or ``None`` to target
            all modes.
        axis : str
            Axis of rotation for each inserted phase shifter.
        trainable : bool
            Promote the rotations to trainable parameters.
        as_input : bool
            Mark the rotations as input-driven parameters.
        angle : float | None
            Optional fixed value for the rotations. Alias of ``value``.
        value : float | None
            Optional fixed value for the rotations. Alias of ``angle``.
        name : str | None
            Optional stem used for generated parameter names.
        role : str | ParameterRole | None
            Explicit :class:`~merlin.core.components.ParameterRole` taking
            precedence over other flags.

        Returns
        -------
        CircuitBuilder
            ``self`` for fluent chaining.
        """
        if isinstance(modes, ModuleGroup):
            target_modes = list(modes.modes)
        elif modes is None:
            target_modes = list(range(self.n_modes))
        elif isinstance(modes, int):
            target_modes = [modes]
        else:
            target_modes = list(modes)

        if not target_modes:
            return self

        resolved_role: ParameterRole
        if role is not None:
            if isinstance(role, str):
                role_map = {
                    "fixed": ParameterRole.FIXED,
                    "input": ParameterRole.INPUT,
                    "trainable": ParameterRole.TRAINABLE,
                    "memristor": ParameterRole.MEMRISTOR,
                }
                resolved_role = role_map.get(role.lower(), ParameterRole.FIXED)
            else:
                resolved_role = role
        elif as_input:
            resolved_role = ParameterRole.INPUT
        elif trainable:
            resolved_role = ParameterRole.TRAINABLE
        else:
            resolved_role = ParameterRole.FIXED

        final_value = (
            angle if angle is not None else (value if value is not None else 0.0)
        )

        for current_mode in target_modes:
            if current_mode < 0 or current_mode >= self.n_modes:
                continue

            if name is not None:
                if resolved_role == ParameterRole.INPUT:
                    custom_name = f"{name}{self._input_counter + 1}"
                    self._input_counter += 1
                elif resolved_role == ParameterRole.TRAINABLE:
                    base_name = (
                        f"{name}_{current_mode}" if len(target_modes) > 1 else name
                    )
                    custom_name = self._unique_trainable_name(base_name)
                elif resolved_role == ParameterRole.MEMRISTOR:
                    custom_name = f"{name}{self._memristor_counter + 1}"
                    self._memristor_counter += 1
                else:
                    custom_name = (
                        f"{name}_{current_mode}" if len(target_modes) > 1 else name
                    )

            elif resolved_role == ParameterRole.INPUT:
                custom_name = f"px{self._input_counter + 1}"
                self._input_counter += 1
            elif resolved_role == ParameterRole.TRAINABLE:
                base_name = f"theta_{self._trainable_counter}_{current_mode}"
                self._trainable_counter += 1
                custom_name = self._unique_trainable_name(base_name)
            elif resolved_role == ParameterRole.MEMRISTOR:
                custom_name = f"mem{self._memristor_counter + 1}"
                self._memristor_counter += 1
            else:
                custom_name = None

            rotation = Rotation(
                target=current_mode,
                role=resolved_role,
                value=final_value,
                axis=axis,
                custom_name=custom_name,
            )
            self.circuit.add(rotation)

            if resolved_role == ParameterRole.TRAINABLE:
                self._register_trainable_prefix(rotation.custom_name or name)
            elif resolved_role == ParameterRole.INPUT:
                self._register_input_prefix(rotation.custom_name or name)
            elif resolved_role == ParameterRole.MEMRISTOR:
                self._register_memristor_prefix(rotation.custom_name or name)

        self._layer_counter += 1
        return self

    def add_angle_encoding(
        self,
        modes: list[int] | None = None,
        name: str | None = None,
        *,
        scale: float = 1.0,
        subset_combinations: bool = False,
        max_order: int | None = None,
    ) -> "CircuitBuilder":
        """Convenience method for angle-based input encoding.

        Parameters
        ----------
        modes : list[int] | None
            Optional list of circuit modes to target. Defaults to all modes.
        name : str | None
            Prefix used for generated input parameters. Defaults to ``"px"``.
        scale : float
            Global scaling factor applied before angle mapping.
        subset_combinations : bool
            When ``True``, generate higher-order feature combinations up to
            ``max_order``.
        max_order : int | None
            Optional cap on the size of feature combinations when
            ``subset_combinations`` is enabled. ``None`` uses all orders.

        Returns
        -------
        CircuitBuilder
            ``self`` for fluent chaining.
        """
        if name is None:
            name = "px"

        if modes is None:
            target_modes = list(range(self.n_modes))
        elif isinstance(modes, ModuleGroup):
            target_modes = modes.modes
        else:
            target_modes = list(modes)

        if not target_modes:
            return self

        invalid_modes = [
            mode for mode in target_modes if mode < 0 or mode >= self.n_modes
        ]
        if invalid_modes:
            raise ValueError(ANGLE_ENCODING_MODE_ERROR)

        # Assign contiguous logical feature indices so downstream encoders do not rely on physical modes
        start_idx = self._angle_encoding_counts.get(name, 0)
        feature_indices = list(range(start_idx, start_idx + len(target_modes)))
        self._angle_encoding_counts[name] = start_idx + len(target_modes)

        scale_map = self._normalize_angle_scale(scale, feature_indices)

        combos: list[tuple[int, ...]] = []
        # When subset_combinations is enabled we enumerate every logical feature subset
        # (up to max_order); each subset translates into a dedicated rotation parameter.
        if subset_combinations and feature_indices:
            max_subset_order = len(feature_indices) if max_order is None else max_order
            max_subset_order = max(1, min(max_subset_order, len(feature_indices)))

            for order in range(1, max_subset_order + 1):
                for combo in combinations(feature_indices, order):
                    combos.append(combo)
        else:
            combos = [(idx,) for idx in feature_indices]

        if not combos:
            combos = [(idx,) for idx in feature_indices]

        required_rotations = len(combos)
        emitted = 0
        while emitted < required_rotations:
            span = min(len(target_modes), required_rotations - emitted)
            chunk_modes = [
                target_modes[(emitted + offset) % len(target_modes)]
                for offset in range(span)
            ]
            self.add_rotations(modes=chunk_modes, role=ParameterRole.INPUT, name=name)
            emitted += span

        spec_list = self._angle_encoding_specs.setdefault(name, [])
        # Persist the logical feature groupings so downstream encoders know which raw
        # features to aggregate when reconstructing this angle encoding.
        spec_list.extend(combos)

        stored_scale = self._angle_encoding_scales.setdefault(name, {})
        for idx, value in scale_map.items():
            if idx in stored_scale and not math.isclose(
                stored_scale[idx], value, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(
                    f"Conflicting scale for feature index {idx} in angle encoding '{name}': "
                    f"{stored_scale[idx]} vs {value}"
                )
            stored_scale[idx] = value

        return self

    @staticmethod
    def _normalize_angle_scale(
        scale: float, feature_indices: list[int]
    ) -> dict[int, float]:
        """Normalize scale specification to a per-feature mapping.

        Args:
            scale: Global scaling factor supplied by the caller.
            feature_indices: Logical feature indices requiring a per-feature scale.

        Returns:
            Dict[int, float]: Mapping from logical feature index to scale factor.
        """
        if not isinstance(scale, numbers.Real):
            raise TypeError("scale must be a real number")

        factor = float(scale)
        return dict.fromkeys(feature_indices, factor)

    @staticmethod
    def _validate_memristive_update_rule(update_rule: Callable) -> None:
        """Validates that update rule has exactly 2 positional parameters

        Raises
        ------
        TypeError
            If signature does not match or annotations are missing/incorrect.
        """

        try:
            sig = inspect.signature(update_rule)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"update_rule must be callable with inspectable signature, "
                f"but got {type(update_rule).__name__}: {e}"
            ) from e

        params = list(sig.parameters.values())
        positional_params = [
            param
            for param in params
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(positional_params) != 2:
            raise TypeError(
                f"update_rule must accept exactly 2 positional arguments, got {len(positional_params)}. "
                f"Expected: update_rule(state: torch.Tensor, "
                f"output: torch.Tensor | StateVector | ProbabilityDistribution | PartialMeasurement) "
                f"-> torch.Tensor"
            )

    def add_memristive_ps(
        self,
        mode: int,
        update_rule: Callable,
        initial_state: float,
        name: str | None = None,
        detach_at_each_forward: bool = True,
    ) -> "CircuitBuilder":
        """Add a memristive phase shifter that will update in regards to the update rule after each forward pass.

        Parameters
        ----------
        mode : int
           Circuit mode to target.
        update_rule : ~collections.abc.Callable
           Callable with signature ``update_rule(state: torch.Tensor, output: torch.Tensor | StateVector |
           ProbabilityDistribution | PartialMeasurement) -> torch.Tensor``. The update rule receives the current
           memristive state and the layer output, and must return a new state tensor of shape ``[batch_size]``.
        initial_state : float
           The initial value of the phase shifter. This will be the value used after each :meth:`~merlin.algorithms.layer.QuantumLayer.reset` call.
        name : str | None
            Prefix used for the generated memristive phase shifter parameter. Defaults to ``"mem"``.
        detach_at_each_forward : bool
            Controls gradient flow through the memristive recurrence (state → new_state):

            - ``True`` (default): The new state is detached at each forward, preventing gradients from flowing
              through the recurrence chain. Earlier inputs receive zero gradients from the memristive state.
            - ``False``: The new state retains gradients, allowing full gradient flow through the entire
              recurrence history. All inputs can receive gradients through the accumulated state chain.

        Returns
        -------
        CircuitBuilder
            ``self`` for fluent chaining.
        """

        if name is None:
            name = "mem"

        # Verifying the parameters
        if not isinstance(mode, int):
            raise ValueError(f"The mode parameter must be an int, got {type(mode)}")

        if mode < 0 or mode >= self.n_modes:
            raise ValueError(
                f"The assigned mode must be between 0 and CircuitBuilder.n_modes ({self.n_modes} here). Got {mode}."
            )

        scalar_initial_state = initial_state
        if isinstance(initial_state, torch.Tensor):
            is_valid_tensor_scalar = (
                initial_state.ndim == 0
                and not initial_state.is_complex()
                and (
                    torch.is_floating_point(initial_state)
                    or initial_state.dtype
                    in {
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                        torch.uint8,
                    }
                )
            )
            if not is_valid_tensor_scalar:
                raise ValueError(
                    f"The initial_state parameter must be an float, got {type(initial_state)}"
                )
            scalar_initial_state = initial_state.item()
        elif not isinstance(initial_state, numbers.Real):
            raise ValueError(
                f"The initial_state parameter must be an float, got {type(initial_state)}"
            )

        # Validate update_rule signature before accepting it
        self._validate_memristive_update_rule(update_rule)

        # Assign contiguous logical feature indices so downstream encoders do not rely on physical modes

        self.add_rotations(
            modes=mode,
            role=ParameterRole.MEMRISTOR,
            name=name,
            value=scalar_initial_state,
        )

        self.memristive_specs.append({
            "target_mode": mode,
            "name": f"{name}{self._memristor_counter}",
            "update_rule": update_rule,
            "initial_state": scalar_initial_state,
            "detach_at_each_forward": detach_at_each_forward,
        })
        return self

    def add_entangling_layer(
        self,
        modes: list[int] | None = None,
        *,
        trainable: bool = True,
        model: str = "mzi",
        name: str | None = None,
        trainable_inner: bool | None = None,
        trainable_outer: bool | None = None,
    ) -> "CircuitBuilder":
        """Add an entangling layer spanning a range of modes.

        Parameters
        ----------
        modes : list[int] | None
            Optional list describing the span. ``None`` targets all modes; one
            element targets ``modes[0]`` through the final mode; two elements
            target the inclusive range ``[modes[0], modes[1]]``.
        trainable : bool
            Whether internal phase shifters should be trainable.
        model : str
            ``"mzi"`` or ``\"bell\"`` to select the internal interferometer
            template.
        name : str | None
            Optional prefix used for generated parameter names.
        trainable_inner : bool | None
            Override for the internal phase shifters.
        trainable_outer : bool | None
            Override for the output phase shifters.

        Returns
        -------
        CircuitBuilder
            ``self`` for fluent chaining.

        Raises
        ------
        ValueError
            If the provided modes are invalid or span fewer than two modes.
        """
        if modes is None:
            start = 0
            end = self.n_modes - 1
        else:
            if len(modes) == 0:
                return self
            if len(modes) == 1:
                start = modes[0]
                end = self.n_modes - 1
            elif len(modes) == 2:
                start, end = modes
            else:
                raise ValueError(
                    "`modes` must be None, a single index, or a two-element range for entangling layers."
                )
        if not isinstance(model, str):
            raise TypeError("model must be a string")
        normalized_model = model.lower()
        if normalized_model not in {"mzi", "bell"}:
            raise ValueError("model must be either 'mzi' or 'bell'")

        if start > end:
            start, end = end, start

        if start < 0 or end >= self.n_modes:
            raise ValueError("Entangling layer span exceeds available modes")

        span = end - start + 1
        if span < 2:
            raise ValueError("Entangling layer requires at least two modes")

        base_prefix = name or f"el_{self._entangling_layer_counter}"
        prefix = self._unique_trainable_name(base_prefix) if trainable else base_prefix

        component = GenericInterferometer(
            start_mode=start,
            span=span,
            trainable=trainable,
            name_prefix=prefix,
            model=normalized_model,
            trainable_inner=trainable_inner,
            trainable_outer=trainable_outer,
        )

        self.circuit.add(component)

        if component.trainable:
            self._register_trainable_prefix(prefix)

        self._entangling_layer_counter += 1
        self._layer_counter += 1
        return self

    def add_superpositions(
        self,
        targets: tuple[int, int] | list[tuple[int, int]] | None = None,
        *,
        depth: int = 1,
        theta: float = 0.785398,
        phi: float = 0.0,
        trainable: bool | None = None,
        trainable_theta: bool | None = None,
        trainable_phi: bool | None = None,
        modes: list[int] | ModuleGroup | None = None,
        name: str | None = None,
    ) -> "CircuitBuilder":
        """Add one or more superposition (beam splitter) components.

        Parameters
        ----------
        targets : tuple[int, int] | list[tuple[int, int]] | None
            Explicit mode pairs. When omitted, nearest neighbours over
            ``modes`` or all modes are used.
        depth : int
            Number of sequential passes to apply.
        theta : float
            Baseline mixing angle for fixed beam splitters.
        phi : float
            Baseline relative phase for fixed beam splitters.
        trainable : bool | None
            Convenience flag to mark both ``theta`` and ``phi`` trainable.
        trainable_theta : bool | None
            Whether the mixing angle should be trainable.
        trainable_phi : bool | None
            Whether the relative phase should be trainable.
        modes : list[int] | ModuleGroup | None
            Optional mode list or module group used when ``targets`` is
            omitted.
        name : str | None
            Optional stem used for generated parameter names.

        Returns
        -------
        CircuitBuilder
            ``self`` for fluent chaining.
        """
        if depth < 1:
            raise ValueError("depth must be at least 1")

        if targets is not None:
            if isinstance(targets, tuple):
                pair_list = [targets]
            else:
                pair_list = list(targets)

            resolved_pairs: list[tuple[int, int]] = []
            for pair in pair_list:
                if len(pair) != 2:
                    raise ValueError("Each target must be a pair of mode indices")
                left, right = pair
                if left == right:
                    continue
                if (
                    left < 0
                    or right < 0
                    or left >= self.n_modes
                    or right >= self.n_modes
                ):
                    raise ValueError("Beam splitter targets must refer to valid modes")
                resolved_pairs.append((left, right))

            if not resolved_pairs:
                return self

            theta_flag = trainable if trainable is not None else False
            phi_flag = trainable if trainable is not None else False

            if trainable_theta is not None:
                theta_flag = trainable_theta
            if trainable_phi is not None:
                phi_flag = trainable_phi

            theta_role = ParameterRole.TRAINABLE if theta_flag else ParameterRole.FIXED
            phi_role = ParameterRole.TRAINABLE if phi_flag else ParameterRole.FIXED

            total_components = depth * len(resolved_pairs)
            single_component = total_components == 1

            if name is None:
                # Keep legacy-friendly prefixes so downstream tooling can match "theta"/"phi".
                theta_base = "theta_bs"
                phi_base = "phi_bs"
                self._superposition_counter += 1
            else:
                theta_base = f"{name}_theta"
                phi_base = f"{name}_phi"

            component_index = 0
            for _depth_idx in range(depth):
                for _pair_idx, pair in enumerate(resolved_pairs):
                    suffix = "" if single_component else f"_{component_index}"
                    theta_name = None
                    phi_name = None

                    if theta_role == ParameterRole.TRAINABLE:
                        base_name = (
                            theta_base if not suffix else f"{theta_base}{suffix}"
                        )
                        theta_name = self._unique_trainable_name(base_name)
                        self._register_trainable_prefix(theta_name)

                    if phi_role == ParameterRole.TRAINABLE:
                        base_name = phi_base if not suffix else f"{phi_base}{suffix}"
                        phi_name = self._unique_trainable_name(base_name)
                        self._register_trainable_prefix(phi_name)

                    bs = BeamSplitter(
                        targets=pair,
                        theta_value=theta,
                        phi_value=phi,
                        theta_role=theta_role,
                        phi_role=phi_role,
                        theta_name=theta_name,
                        phi_name=phi_name,
                    )
                    self.circuit.add(bs)

                    component_index += 1

            return self

        # Fallback: behave like an entangling block across a span of modes
        if isinstance(modes, ModuleGroup):
            mode_list = list(modes.modes)
        elif modes is None:
            mode_list = list(range(self.n_modes))
        else:
            mode_list = list(modes)

        if len(mode_list) < 2:
            return self

        ent_trainable = (
            trainable
            if trainable is not None
            else bool(trainable_theta or trainable_phi)
        )

        block = EntanglingBlock(
            targets=mode_list if modes is not None else "all",
            depth=depth,
            trainable=ent_trainable,
            name_prefix=name,
        )

        if ent_trainable:
            base = name or f"eb{self._entangling_counter}"
            self._entangling_counter += 1
            prefix = self._unique_trainable_name(base)
            block.name_prefix = prefix
            self._register_trainable_prefix(prefix)

        self.circuit.add(block)
        return self

    def build(self) -> Circuit:
        """Build and return the circuit.

        Returns
        -------
        merlin.core.circuit.Circuit
            Circuit instance populated with components.
        """
        return self.circuit

    def to_pcvl_circuit(self, pcvl_module=None):
        """Convert the constructed circuit into a Perceval circuit.

        Parameters
        ----------
        pcvl_module : Any
            Optional Perceval module. If ``None``, attempts to import
            ``perceval``.

        Returns
        -------
        Any
            A ``pcvl.Circuit`` instance mirroring the components tracked by
            this builder.

        Raises
        ------
        ImportError
            If ``perceval`` is not installed and no module is provided.
        """
        if pcvl_module is None:
            try:
                import perceval as pcvl_module  # type: ignore
            except (
                ImportError
            ) as exc:  # pragma: no cover - exercised when dependency missing
                raise ImportError(
                    "perceval is required to convert a circuit to a Perceval representation. "
                    "Install perceval-quandela or provide a custom module via 'pcvl_module'."
                ) from exc

        circuit = self.build()
        pcvl_circuit = pcvl_module.Circuit(circuit.n_modes)

        for idx, component in enumerate(circuit.components):
            if isinstance(component, Rotation):
                if component.role == ParameterRole.FIXED:
                    phi = component.value
                else:
                    custom_name = (
                        component.custom_name or f"phi_{component.target}_{idx}"
                    )
                    phi = pcvl_module.P(custom_name)
                pcvl_circuit.add(component.target, pcvl_module.PS(phi))

            elif isinstance(component, BeamSplitter):
                if component.theta_role == ParameterRole.FIXED:
                    theta = component.theta_value
                else:
                    theta_name = component.theta_name or f"theta_bs_{idx}"
                    theta = pcvl_module.P(theta_name)

                if component.phi_role == ParameterRole.FIXED:
                    phi_tr = component.phi_value
                else:
                    phi_name = component.phi_name or f"phi_bs_{idx}"
                    phi_tr = pcvl_module.P(phi_name)

                pcvl_circuit.add(
                    component.targets, pcvl_module.BS(theta=theta, phi_tr=phi_tr)
                )

            elif isinstance(component, EntanglingBlock):
                if component.targets == "all":
                    mode_list = list(range(circuit.n_modes))
                else:
                    mode_list = list(component.targets)

                if len(mode_list) < 2:
                    continue

                prefix = component.name_prefix or f"eb_{idx}"
                pair_index = 0

                for _ in range(component.depth):
                    for left, right in zip(mode_list[:-1], mode_list[1:], strict=False):
                        if component.trainable:
                            theta_name = f"{prefix}_theta_{pair_index}"
                            phi_name = f"{prefix}_phi_{pair_index}"
                            theta = pcvl_module.P(theta_name)
                            phi_tr = pcvl_module.P(phi_name)
                            pair_index += 1
                            pcvl_circuit.add(
                                (left, right),
                                pcvl_module.BS(theta=theta, phi_tr=phi_tr),
                            )
                        else:
                            pcvl_circuit.add((left, right), pcvl_module.BS())

            elif isinstance(component, GenericInterferometer):
                if component.span < 2:
                    continue

                prefix = component.name_prefix or f"gi_{idx}"
                model = getattr(component, "model", "mzi")

                if model == "mzi":

                    def _mzi_factory(
                        i: int,
                        *,
                        inner_trainable: bool = getattr(
                            component, "trainable_inner", component.trainable
                        ),
                        outer_trainable: bool = getattr(
                            component, "trainable_outer", component.trainable
                        ),
                        base: str = prefix,
                    ):
                        """Build a Mach-Zehnder interferometer optionally parameterised per index."""
                        if inner_trainable:
                            phi_inner = pcvl_module.P(f"{base}_li{i}")
                        else:
                            phi_inner = 0.0
                        if outer_trainable:
                            phi_outer = pcvl_module.P(f"{base}_lo{i}")
                        else:
                            phi_outer = 0.0
                        return (
                            pcvl_module.BS()
                            // pcvl_module.PS(phi_inner)
                            // pcvl_module.BS()
                            // pcvl_module.PS(phi_outer)
                        )

                    gi = pcvl_module.GenericInterferometer(
                        component.span,
                        lambda i, factory=_mzi_factory: factory(i),
                        shape=pcvl_module.InterferometerShape.RECTANGLE,
                    )
                    pcvl_circuit.add(component.start_mode, gi)

                elif model == "bell":

                    def _bell_factory(
                        i: int,
                        *,
                        inner_trainable: bool = getattr(
                            component, "trainable_inner", component.trainable
                        ),
                        outer_trainable: bool = getattr(
                            component, "trainable_outer", component.trainable
                        ),
                        base: str = prefix,
                    ):
                        """Build a Mach-Zehnder interferometer optionally parameterised per index."""
                        if inner_trainable:
                            phi_inner = pcvl_module.P(f"{base}_li{i}")
                        else:
                            phi_inner = 0.0
                        if outer_trainable:
                            phi_outer = pcvl_module.P(f"{base}_lo{i}")
                        else:
                            phi_outer = 0.0

                        circuit = pcvl_module.Circuit(2)
                        circuit.add(0, pcvl_module.BS())
                        circuit.add(0, pcvl_module.PS(phi_inner))
                        circuit.add(1, pcvl_module.PS(phi_outer))
                        circuit.add(0, pcvl_module.BS())

                        return circuit

                    gi = pcvl_module.GenericInterferometer(
                        component.span,
                        lambda i, factory=_bell_factory: factory(i),
                        shape=pcvl_module.InterferometerShape.RECTANGLE,
                    )
                    pcvl_circuit.add(component.start_mode, gi)
                else:
                    raise ValueError(f"{model} not implemented yet")

            else:
                # Components like Measurement are metadata only and do not map to a pcvl operation
                continue

        return pcvl_circuit

    @classmethod
    def from_circuit(cls, circuit: Circuit) -> "CircuitBuilder":
        """Create a builder from an existing circuit.

        Parameters
        ----------
        circuit : merlin.core.circuit.Circuit
            Circuit object whose components should seed the builder.

        Returns
        -------
        CircuitBuilder
            A new builder instance wrapping the provided circuit.
        """
        builder = cls(circuit.n_modes)
        builder.circuit = circuit
        return builder

    @property
    def trainable_parameter_prefixes(self) -> list[str]:
        """Expose the unique set of trainable prefixes in insertion order.

        Returns
        -------
        list[str]
            Trainable parameter stems discovered so far.
        """
        return list(self._trainable_prefixes)

    @property
    def input_parameter_prefixes(self) -> list[str]:
        """Expose the order-preserving set of input prefixes.

        Returns
        -------
        list[str]
            Input parameter stems emitted during encoding.
        """
        return list(self._input_prefixes)

    @property
    def angle_encoding_specs(self) -> dict[str, dict[str, Any]]:
        """Return metadata describing configured angle encodings.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping from encoding prefix to combination metadata.
        """
        return {
            prefix: {
                "combinations": list(combos),
                "scales": dict(self._angle_encoding_scales.get(prefix, {})),
            }
            for prefix, combos in self._angle_encoding_specs.items()
        }
