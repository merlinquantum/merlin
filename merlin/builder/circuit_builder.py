"""
Circuit builder for constructing quantum circuits declaratively.
"""

import math
import numbers
import warnings
from copy import deepcopy
from itertools import combinations
from typing import Any

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
    """Helper class for grouping modules."""

    def __init__(self, modes: list[int]):
        """Store the list of modes spanned by the grouped module."""
        self.modes = modes


class CircuitBuilder:
    """
    Builder for quantum circuits using a declarative API.
    """

    def __init__(self, n_modes: int):
        """Initialise bookkeeping structures for a circuit with ``n_modes`` modes.

        Args:
            n_modes: Number of photonic modes available in the circuit.
        """
        self.n_modes = n_modes
        self.circuit = Circuit(n_modes)

        # Track component counts for naming - these should NEVER reset
        self._layer_counter = 0
        self._trainable_counter = 0
        self._input_counter = 0
        self._copy_counter = 0
        self._generic_counter = 0
        self._entangling_counter = 0

        # Section tracking for adjoint support
        self._section_markers: list[Any] = []
        self._current_section: dict[str, Any] | None = None

        # Track components before any sections for "_all_" reference
        self._pre_section_end_idx = 0

        self._trainable_prefixes: list[str] = []
        self._trainable_prefix_set: set[str] = set()
        self._input_prefixes: list[str] = []
        self._input_prefix_set: set[str] = set()
        self._angle_encoding_specs: dict[str, list[tuple[int, ...]]] = {}
        self._angle_encoding_scales: dict[str, dict[int, float]] = {}
        self._angle_encoding_counts: dict[str, int] = {}

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

    def add_rotation(
        self,
        target: int,
        angle: float = 0.0,
        trainable: bool = False,
        name: str | None = None,
    ) -> "CircuitBuilder":
        """Add a single rotation.

        The builder keeps rotations in an abstract form by appending a
        :class:`~merlin.core.components.Rotation` entry to its internal
        :class:`~merlin.core.circuit.Circuit`. The actual photonic primitive is
        created later in :meth:`to_pcvl_circuit`, where each rotation is
        materialised as a Perceval phase shifter ``pcvl.PS``. Fixed rotations use
        the numeric ``angle`` provided here, while trainable ones are assigned a
        symbolic ``pcvl.P`` that Perceval treats as an optimisable parameter.

        Args:
            target: Circuit mode index receiving the phase shifter.
            angle: Initial numeric value used when the rotation is fixed.
            trainable: Whether to expose the rotation angle as a learnable parameter.
            name: Optional custom stem for the underlying parameter name.

        Returns:
            CircuitBuilder: ``self`` to allow method chaining.
        """
        role = ParameterRole.TRAINABLE if trainable else ParameterRole.FIXED

        if role == ParameterRole.TRAINABLE:
            if name is None:
                base_name = f"theta_{self._trainable_counter}"
                self._trainable_counter += 1
            else:
                base_name = name
            custom_name = self._unique_trainable_name(base_name)
        else:
            custom_name = name

        rotation = Rotation(
            target=target, role=role, value=angle, custom_name=custom_name
        )

        self.circuit.add(rotation)

        if role == ParameterRole.TRAINABLE:
            self._register_trainable_prefix(custom_name)
        elif role == ParameterRole.INPUT:
            self._register_input_prefix(custom_name or name)
        return self

    def add_rotation_layer(
        self,
        modes: list[int] | ModuleGroup | None = None,
        *,
        axis: str = "z",
        trainable: bool = False,
        as_input: bool = False,
        value: float | None = None,
        name: str | None = None,
        role: str | ParameterRole | None = None,
    ) -> "CircuitBuilder":
        """Add a rotation layer across a set of modes.

        Args:
            modes: Modes (or module group) receiving the rotations; defaults to all modes.
            axis: Axis of rotation to apply on each mode.
            trainable: Promote every rotation in the layer to trainable parameters.
            as_input: Mark the rotations as data-driven inputs (legacy convenience flag).
            value: Default fixed value assigned when the layer is not trainable/input.
            name: Optional stem used when generating parameter names per mode.
            role: Explicit :class:`ParameterRole` taking precedence over other flags.

        Returns:
            CircuitBuilder: ``self`` to facilitate fluent chaining.
        """
        if modes is None:
            target_modes = list(range(self.n_modes))
        elif isinstance(modes, ModuleGroup):
            target_modes = modes.modes
        else:
            target_modes = modes

        # Determine role (new interface takes precedence)
        if role is not None:
            if isinstance(role, str):
                role_map = {
                    "fixed": ParameterRole.FIXED,
                    "input": ParameterRole.INPUT,
                    "trainable": ParameterRole.TRAINABLE,
                }
                final_role = role_map.get(role, ParameterRole.FIXED)
            else:
                final_role = role
        elif as_input:
            final_role = ParameterRole.INPUT
        elif trainable:
            final_role = ParameterRole.TRAINABLE
        else:
            final_role = ParameterRole.FIXED

        # Determine value
        final_value = value if value is not None else 0.0

        for _idx, mode in enumerate(target_modes):
            if mode >= self.n_modes:
                continue

            # Generate appropriate name based on role
            if name is not None:
                # For input parameters with custom name, create simple indexed names
                if final_role == ParameterRole.INPUT:
                    # Use global input counter for unique naming
                    custom_name = f"{name}{self._input_counter + 1}"
                    self._input_counter += 1
                elif final_role == ParameterRole.TRAINABLE:
                    base_name = f"{name}_{mode}" if len(target_modes) > 1 else name
                    custom_name = self._unique_trainable_name(base_name)
                else:
                    # Keep existing naming for non-input params
                    custom_name = f"{name}_{mode}" if len(target_modes) > 1 else name
            elif final_role == ParameterRole.INPUT:
                # Default input naming: px1, px2, etc. - using global counter
                custom_name = f"px{self._input_counter + 1}"
                self._input_counter += 1
            elif final_role == ParameterRole.TRAINABLE:
                base_name = f"theta_{self._trainable_counter}_{mode}"
                self._trainable_counter += 1
                custom_name = self._unique_trainable_name(base_name)
            else:
                custom_name = None

            rotation = Rotation(
                target=mode,
                role=final_role,
                value=final_value,
                axis=axis,
                custom_name=custom_name,
            )
            self.circuit.add(rotation)

            if final_role == ParameterRole.TRAINABLE:
                self._register_trainable_prefix(rotation.custom_name or name)
            elif final_role == ParameterRole.INPUT:
                self._register_input_prefix(rotation.custom_name or name)

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

        Args:
            modes: Optional list of circuit modes to target. Defaults to all modes.
            name: Prefix used for generated input parameters. Defaults to ``"px"``.
            scale: Global scaling factor applied before angle mapping.
            subset_combinations: When ``True``, generate higher-order feature
                combinations (up to ``max_order``) similar to the legacy
                ``FeatureEncoder``.
            max_order: Optional cap on the size of feature combinations when
                ``subset_combinations`` is enabled. ``None`` uses all orders.

        Returns:
            CircuitBuilder: ``self`` for fluent chaining.
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
            self.add_rotation_layer(
                modes=chunk_modes, role=ParameterRole.INPUT, name=name
            )
            emitted += span

        spec_list = self._angle_encoding_specs.setdefault(name, [])
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

    def add_generic_interferometer(
        self,
        modes: list[int] | None = None,
        *,
        trainable: bool = True,
        name: str | None = None,
    ) -> "CircuitBuilder":
        """Add a generic interferometer spanning a range of modes.

        Args:
            modes: Optional list describing the span. ``None`` targets all modes;
                one element targets ``modes[0]`` through the final mode; two elements
                target the inclusive range ``[modes[0], modes[1]]``.
            trainable: Whether internal phase shifters should be trainable.
            name: Optional prefix used for generated parameter names.

        Raises:
            ValueError: If the provided modes are invalid or span fewer than two modes.

        Returns:
            CircuitBuilder: ``self`` for fluent chaining.
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
                    "`modes` must be None, a single index, or a two-element range for generic interferometers."
                )

        if start > end:
            start, end = end, start

        if start < 0 or end >= self.n_modes:
            raise ValueError("Generic interferometer span exceeds available modes")

        span = end - start + 1
        if span < 2:
            raise ValueError("Generic interferometer requires at least two modes")

        if name is None:
            prefix = f"gi_{self._generic_counter}"
        else:
            prefix = name

        component = GenericInterferometer(
            start_mode=start,
            span=span,
            trainable=trainable,
            name_prefix=prefix,
        )

        self.circuit.add(component)

        if trainable:
            self._register_trainable_prefix(prefix)

        self._generic_counter += 1
        self._layer_counter += 1
        return self

    def add_superposition(
        self,
        targets: tuple[int, int],
        theta: float = 0.785398,
        phi: float = 0.0,
        trainable_theta: bool = False,
        trainable_phi: bool = False,
        name: str | None = None,
    ) -> "CircuitBuilder":
        """Add a beam splitter (superposition component).

        Args:
            targets: Pair of mode indices connected by the beam splitter.
            theta: Fixed mixing angle used when ``trainable_theta`` is ``False``.
            phi: Fixed relative phase applied when ``trainable_phi`` is ``False``.
            trainable_theta: Promote the mixing angle to a trainable parameter.
            trainable_phi: Promote the relative phase to a trainable parameter.
            name: Optional stem used to generate parameter names for this component.

        Returns:
            CircuitBuilder: ``self`` to allow chaining additional builder calls.
        """
        theta_role = ParameterRole.TRAINABLE if trainable_theta else ParameterRole.FIXED
        phi_role = ParameterRole.TRAINABLE if trainable_phi else ParameterRole.FIXED

        theta_name = f"{name}_theta" if name else None
        phi_name = f"{name}_phi" if name else None

        bs = BeamSplitter(
            targets=targets,
            theta_value=theta,
            phi_value=phi,
            theta_role=theta_role,
            phi_role=phi_role,
            theta_name=theta_name,
            phi_name=phi_name,
        )

        self.circuit.add(bs)

        if theta_role == ParameterRole.TRAINABLE:
            self._register_trainable_prefix(theta_name or "theta")
        elif theta_role == ParameterRole.INPUT:
            self._register_input_prefix(theta_name or "theta")

        if phi_role == ParameterRole.TRAINABLE:
            self._register_trainable_prefix(phi_name or "phi")
        elif phi_role == ParameterRole.INPUT:
            self._register_input_prefix(phi_name or "phi")
        return self

    def add_entangling_layer(
        self, depth: int = 1, trainable: bool = False, name: str | None = None
    ) -> "CircuitBuilder":
        """Add entangling layer(s).

        When ``trainable`` is ``True`` the block is converted into parameterized beam
        splitters during ``to_pcvl_circuit`` so that interferometric mixing can be
        optimised. Generated parameters share a common prefix derived from ``name``.

        Args:
            depth: Number of successive nearest-neighbour passes to add.
            trainable: Whether to expose the internal beam splitters as parameters.
            name: Optional stem used for generated parameter names when trainable.

        Returns:
            CircuitBuilder: ``self`` for chaining additional builder calls.
        """
        block = EntanglingBlock(depth=depth, trainable=trainable, name_prefix=name)

        if trainable:
            base = name or f"eb{self._entangling_counter}"
            self._entangling_counter += 1
            prefix = self._unique_trainable_name(base)
            block.name_prefix = prefix
            self._register_trainable_prefix(prefix)

        self.circuit.add(block)
        return self

    def begin_section(
        self,
        name: str,
        compute_adjoint: bool = False,
        reference: str | None = None,
        share_trainable: bool = True,
        share_input: bool = False,
    ) -> "CircuitBuilder":
        """
        Mark the beginning of a circuit section.

        Args:
            name: Name of the section
            compute_adjoint: Whether to compute the adjoint of this section
            reference: Name of section to copy structure from (or "_all_" for all preceding)
            share_trainable: Whether to share trainable parameters from reference
            share_input: Whether to share input parameters from reference

        Returns:
            CircuitBuilder: ``self`` so builder calls can be chained.
        """
        if self._current_section is not None:
            warnings.warn(
                f"Section '{self._current_section['name']}' was not closed. Closing it now.",
                stacklevel=2,
            )
            self.end_section()

        # Update pre-section end index if this is the first section
        if not self._section_markers and reference != "_all_":
            self._pre_section_end_idx = len(self.circuit.components)

        self._current_section = {
            "name": name,
            "compute_adjoint": compute_adjoint,
            "reference": reference,
            "share_trainable": share_trainable,
            "share_input": share_input,
            "start_idx": len(self.circuit.components),
        }

        # If referencing, copy components now
        if reference:
            self._copy_from_reference(reference)
            # End section immediately after copying
            self._current_section["end_idx"] = len(self.circuit.components)
            self._section_markers.append(self._current_section)
            self._current_section = None

        return self

    def add_adjoint_section(
        self,
        name: str,
        reference: str,
        share_trainable: bool = True,
        share_input: bool = True,
    ) -> "CircuitBuilder":
        """Convenience method for adding adjoint of an existing section.

        Args:
            name: Name assigned to the new adjoint section.
            reference: Existing section to mirror.
            share_trainable: Whether to reuse the referenced trainable parameters.
            share_input: Whether to reuse the referenced input parameters.

        Returns:
            CircuitBuilder: ``self`` for fluent chaining.
        """
        return self.begin_section(
            name=name,
            compute_adjoint=True,
            reference=reference,
            share_trainable=share_trainable,
            share_input=share_input,
        )

    def _copy_from_reference(self, ref_name: str):
        """Copy components from referenced section with parameter sharing rules.

        Args:
            ref_name: Name of the section (or ``"_all_"`` for the pre-section content) to copy.
        """
        if ref_name == "_all_":
            # Copy everything before sections started
            start_idx = 0
            end_idx = self._pre_section_end_idx
        else:
            # Find the referenced section
            ref_section = None
            for section in self._section_markers:
                if section["name"] == ref_name:
                    ref_section = section
                    break

            if not ref_section:
                raise ValueError(f"Section '{ref_name}' not found")

            start_idx = ref_section["start_idx"]
            end_idx = ref_section["end_idx"]

        # Copy components with parameter transformation
        for idx in range(start_idx, end_idx):
            comp = self.circuit.components[idx]
            new_comp = self._transform_component(
                comp,
                self._current_section["share_trainable"],
                self._current_section["share_input"],
            )
            self.circuit.add(new_comp)

    def _transform_component(self, comp, share_trainable, share_input):
        """Transform a copied component according to sharing rules.

        Args:
            comp: Original component instance to duplicate.
            share_trainable: Whether trainable parameters should be reused.
            share_input: Whether input-parameter names should be reused.

        Returns:
            Any: Deep-copied component with adjusted parameter naming.
        """
        new_comp = deepcopy(comp)

        if isinstance(comp, Rotation):
            if comp.role == ParameterRole.TRAINABLE:
                if not share_trainable:
                    # Generate new trainable parameter name
                    if comp.custom_name:
                        new_comp.custom_name = (
                            f"{comp.custom_name}_copy{self._copy_counter}"
                        )
                    else:
                        new_comp.custom_name = f"theta_copy_{self._copy_counter}"
                    self._copy_counter += 1
                self._register_trainable_prefix(
                    new_comp.custom_name or comp.custom_name
                )
            elif comp.role == ParameterRole.INPUT:
                if not share_input:
                    # Generate new input parameter name
                    if comp.custom_name:
                        base_name = comp.custom_name.rstrip("0123456789")
                    else:
                        base_name = "px"
                    new_comp.custom_name = f"{base_name}{self._input_counter}"
                    self._input_counter += 1
                self._register_input_prefix(new_comp.custom_name or comp.custom_name)

        elif isinstance(comp, BeamSplitter):
            # Handle beam splitter parameter transformation
            if comp.theta_role == ParameterRole.TRAINABLE and not share_trainable:
                if comp.theta_name:
                    new_comp.theta_name = f"{comp.theta_name}_copy{self._copy_counter}"
                else:
                    new_comp.theta_name = f"theta_bs_copy_{self._copy_counter}"
                self._copy_counter += 1
            if comp.theta_role == ParameterRole.TRAINABLE:
                self._register_trainable_prefix(new_comp.theta_name or comp.theta_name)
            elif comp.theta_role == ParameterRole.INPUT and not share_input:
                if comp.theta_name:
                    base_name = comp.theta_name.rstrip("0123456789")
                else:
                    base_name = "x_bs"
                new_comp.theta_name = f"{base_name}{self._input_counter}"
                self._input_counter += 1
                self._register_input_prefix(new_comp.theta_name)
            elif comp.theta_role == ParameterRole.INPUT:
                self._register_input_prefix(comp.theta_name)

            # Same for phi
            if comp.phi_role == ParameterRole.TRAINABLE and not share_trainable:
                if comp.phi_name:
                    new_comp.phi_name = f"{comp.phi_name}_copy{self._copy_counter}"
                else:
                    new_comp.phi_name = f"phi_bs_copy_{self._copy_counter}"
                self._copy_counter += 1
            if comp.phi_role == ParameterRole.TRAINABLE:
                self._register_trainable_prefix(new_comp.phi_name or comp.phi_name)
            elif comp.phi_role == ParameterRole.INPUT and not share_input:
                if comp.phi_name:
                    base_name = comp.phi_name.rstrip("0123456789")
                else:
                    base_name = "x_phi"
                new_comp.phi_name = f"{base_name}{self._input_counter}"
                self._input_counter += 1
                self._register_input_prefix(new_comp.phi_name)
            elif comp.phi_role == ParameterRole.INPUT:
                self._register_input_prefix(comp.phi_name)

        # EntanglingBlock doesn't need special handling as its parameters
        # are generated during compilation

        return new_comp

    def end_section(self) -> "CircuitBuilder":
        """Mark the end of the current circuit section.

        Returns:
            CircuitBuilder: ``self`` so builder calls can be chained.
        """
        if self._current_section:
            self._current_section["end_idx"] = len(self.circuit.components)
            self._section_markers.append(self._current_section)
            self._current_section = None
        else:
            warnings.warn("No section to end", stacklevel=2)
        return self

    def build(self) -> Circuit:
        """Build and return the circuit, finalising any open sections.

        Returns:
            Circuit: Circuit instance populated with components and metadata.
        """
        # Close any open section
        if self._current_section is not None:
            warnings.warn(
                f"Section '{self._current_section['name']}' was not closed. Closing it now.",
                stacklevel=2,
            )
            self.end_section()

        # Finalize the circuit to ensure metadata is complete
        return self.finalize_circuit()

    def finalize_circuit(self):
        """Ensure metadata reflects defined sections before returning the circuit.

        Returns:
            Circuit: Circuit with updated section metadata.
        """
        # Ensure 'sections' key is always added to metadata
        self.circuit.metadata["sections"] = self._section_markers or []

        return self.circuit

    def to_pcvl_circuit(self, pcvl_module=None):
        """Convert the constructed circuit into a Perceval circuit.

        Args:
            pcvl_module: Optional Perceval module. If ``None``, attempts to import ``perceval``.

        Returns:
            A ``pcvl.Circuit`` instance mirroring the components tracked by this builder.

        Raises:
            ImportError: If ``perceval`` is not installed and no module is provided.
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
                        component.custom_name or f"theta_{component.target}_{idx}"
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

                def _mzi_factory(
                    i: int, *, trainable: bool = component.trainable, base: str = prefix
                ):
                    """Build a Mach-Zehnder interferometer optionally parameterised per index."""
                    if trainable:
                        phi_inner = pcvl_module.P(f"{base}_li{i}")
                        phi_outer = pcvl_module.P(f"{base}_lo{i}")
                    else:
                        phi_inner = 0.0
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

            else:
                # Components like Measurement are metadata only and do not map to a pcvl operation
                continue

        return pcvl_circuit

    @classmethod
    def from_circuit(cls, circuit: Circuit) -> "CircuitBuilder":
        """Create a builder from an existing circuit.

        Args:
            circuit: Circuit object whose components should seed the builder.

        Returns:
            CircuitBuilder: A new builder instance wrapping the provided circuit.
        """
        builder = cls(circuit.n_modes)
        builder.circuit = circuit
        return builder

    @property
    def trainable_parameter_prefixes(self) -> list[str]:
        """Expose the unique set of trainable prefixes in insertion order.

        Returns:
            List[str]: Trainable parameter stems discovered so far.
        """
        return list(self._trainable_prefixes)

    @property
    def input_parameter_prefixes(self) -> list[str]:
        """Expose the order-preserving set of input prefixes.

        Returns:
            List[str]: Input parameter stems emitted during encoding.
        """
        return list(self._input_prefixes)

    @property
    def angle_encoding_specs(self) -> dict[str, dict[str, Any]]:
        """Return metadata describing configured angle encodings.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping from encoding prefix to combination metadata.
        """
        return {
            prefix: {
                "combinations": list(combos),
                "scales": dict(self._angle_encoding_scales.get(prefix, {})),
            }
            for prefix, combos in self._angle_encoding_specs.items()
        }
