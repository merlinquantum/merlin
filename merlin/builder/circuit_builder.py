"""
Circuit builder for constructing quantum circuits declaratively.
"""

from typing import List, Optional, Union, Tuple, Dict, Any
import warnings
from copy import deepcopy
from ..core.circuit import Circuit
from ..core.components import (
    Component,
    Rotation,
    BeamSplitter,
    EntanglingBlock,
    Measurement,
    ParameterRole
)
from ..core.observables import parse_observable


class ModuleGroup:
    """Helper class for grouping modules."""

    def __init__(self, modes: List[int]):
        self.modes = modes


class CircuitBuilder:
    """
    Builder for quantum circuits using a declarative API.
    """

    def __init__(self, n_modes: int, n_photons: Optional[int] = None):
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.circuit = Circuit(n_modes)

        # Track component counts for naming - these should NEVER reset
        self._layer_counter = 0
        self._trainable_counter = 0
        self._input_counter = 0
        self._copy_counter = 0

        # Section tracking for adjoint support
        self._section_markers = []
        self._current_section = None

        # Track components before any sections for "_all_" reference
        self._pre_section_end_idx = 0

    def add_rotation(
            self,
            target: int,
            angle: float = 0.0,
            trainable: bool = False,
            name: Optional[str] = None
    ) -> "CircuitBuilder":
        """Add a single rotation."""
        role = ParameterRole.TRAINABLE if trainable else ParameterRole.FIXED

        if name is None and trainable:
            name = f"theta_{self._trainable_counter}"
            self._trainable_counter += 1

        rotation = Rotation(
            target=target,
            role=role,
            value=angle,
            custom_name=name
        )

        self.circuit.add(rotation)
        return self

    def add_rotation_layer(
            self,
            modes: Optional[Union[List[int], ModuleGroup]] = None,
            *,
            axis: str = "z",
            trainable: bool = False,
            as_input: bool = False,
            value: Optional[float] = None,
            name: Optional[str] = None,
            role: Optional[Union[str, ParameterRole]] = None,
    ) -> "CircuitBuilder":
        """
        Add rotation layer with clear role specification.
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
                    "trainable": ParameterRole.TRAINABLE
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

        for idx, mode in enumerate(target_modes):
            if mode >= self.n_modes:
                continue

            # Generate appropriate name based on role
            if name is not None:
                # For input parameters with custom name, create simple indexed names
                if final_role == ParameterRole.INPUT:
                    # Use global input counter for unique naming
                    custom_name = f"{name}{self._input_counter + 1}"
                    self._input_counter += 1
                else:
                    # Keep existing naming for non-input params
                    custom_name = f"{name}_{mode}" if len(target_modes) > 1 else name
            elif final_role == ParameterRole.INPUT:
                # Default input naming: px1, px2, etc. - using global counter
                custom_name = f"px{self._input_counter + 1}"
                self._input_counter += 1
            elif final_role == ParameterRole.TRAINABLE:
                custom_name = f"theta_{self._trainable_counter}_{mode}"
                self._trainable_counter += 1
            else:
                custom_name = None

            rotation = Rotation(
                target=mode,
                role=final_role,
                value=final_value,
                axis=axis,
                custom_name=custom_name
            )
            self.circuit.add(rotation)

        self._layer_counter += 1
        return self

    def add_input_encoding(self, modes: Optional[List[int]] = None, name: Optional[str] = None) -> "CircuitBuilder":
        """Convenience method for input encoding."""
        # Use 'px' as default name to match backend expectations
        if name is None:
            name = "px"
        return self.add_rotation_layer(modes=modes, role=ParameterRole.INPUT, name=name)

    def add_trainable_layer(self, modes: Optional[List[int]] = None, name: Optional[str] = None) -> "CircuitBuilder":
        """Convenience method for trainable rotation layer."""
        return self.add_rotation_layer(modes=modes, trainable=True, name=name)

    def add_beam_splitter(
            self,
            targets: Tuple[int, int],
            theta: float = 0.785398,
            phi: float = 0.0,
            trainable_theta: bool = False,
            trainable_phi: bool = False,
            name: Optional[str] = None
    ) -> "CircuitBuilder":
        """Add a beam splitter."""
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
            phi_name=phi_name
        )

        self.circuit.add(bs)
        return self

    def add_entangling_layer(
            self,
            depth: int = 1,
            trainable: bool = False,
            name: Optional[str] = None
    ) -> "CircuitBuilder":
        """Add entangling layer(s)."""
        block = EntanglingBlock(
            depth=depth,
            trainable=trainable,
            name_prefix=name
        )

        self.circuit.add(block)
        return self

    def add_measurement(
            self,
            observable: Union[str, Any],
            name: Optional[str] = None
    ) -> "CircuitBuilder":
        """
        Add a measurement to the circuit.

        Args:
            observable: String representation or observable object
            name: Optional name for the measurement
        """
        # Parse string observables
        if isinstance(observable, str):
            observable = parse_observable(observable)

        measurement = Measurement(observable=observable, name=name)
        self.circuit.add(measurement)

        # Store in metadata
        if 'measurements' not in self.circuit.metadata:
            self.circuit.metadata['measurements'] = []

        self.circuit.metadata['measurements'].append({
            'observable': observable,
            'name': name
        })

        return self

    def begin_section(
            self,
            name: str,
            compute_adjoint: bool = False,
            reference: Optional[str] = None,
            share_trainable: bool = True,
            share_input: bool = False
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
            self for chaining
        """
        if self._current_section is not None:
            warnings.warn(f"Section '{self._current_section['name']}' was not closed. Closing it now.")
            self.end_section()

        # Update pre-section end index if this is the first section
        if not self._section_markers and reference != "_all_":
            self._pre_section_end_idx = len(self.circuit.components)

        self._current_section = {
            'name': name,
            'compute_adjoint': compute_adjoint,
            'reference': reference,
            'share_trainable': share_trainable,
            'share_input': share_input,
            'start_idx': len(self.circuit.components)
        }

        # If referencing, copy components now
        if reference:
            self._copy_from_reference(reference)
            # End section immediately after copying
            self._current_section['end_idx'] = len(self.circuit.components)
            self._section_markers.append(self._current_section)
            self._current_section = None

        return self

    def add_adjoint_section(
            self,
            name: str,
            reference: str,
            share_trainable: bool = True,
            share_input: bool = True
    ) -> "CircuitBuilder":
        """Convenience method for adding adjoint of existing section."""
        return self.begin_section(
            name=name,
            compute_adjoint=True,
            reference=reference,
            share_trainable=share_trainable,
            share_input=share_input
        )

    def _copy_from_reference(self, ref_name: str):
        """Copy components from referenced section with parameter sharing rules."""
        if ref_name == "_all_":
            # Copy everything before sections started
            start_idx = 0
            end_idx = self._pre_section_end_idx
        else:
            # Find the referenced section
            ref_section = None
            for section in self._section_markers:
                if section['name'] == ref_name:
                    ref_section = section
                    break

            if not ref_section:
                raise ValueError(f"Section '{ref_name}' not found")

            start_idx = ref_section['start_idx']
            end_idx = ref_section['end_idx']

        # Copy components with parameter transformation
        for idx in range(start_idx, end_idx):
            comp = self.circuit.components[idx]
            new_comp = self._transform_component(
                comp,
                self._current_section['share_trainable'],
                self._current_section['share_input']
            )
            self.circuit.add(new_comp)

    def _transform_component(self, comp, share_trainable, share_input):
        """Transform component based on sharing rules."""
        new_comp = deepcopy(comp)

        if isinstance(comp, Rotation):
            if comp.role == ParameterRole.TRAINABLE:
                if not share_trainable:
                    # Generate new trainable parameter name
                    if comp.custom_name:
                        new_comp.custom_name = f"{comp.custom_name}_copy{self._copy_counter}"
                    else:
                        new_comp.custom_name = f"theta_copy_{self._copy_counter}"
                    self._copy_counter += 1
            elif comp.role == ParameterRole.INPUT:
                if not share_input:
                    # Generate new input parameter name
                    if comp.custom_name:
                        base_name = comp.custom_name.rstrip('0123456789')
                    else:
                        base_name = "px"
                    new_comp.custom_name = f"{base_name}{self._input_counter}"
                    self._input_counter += 1

        elif isinstance(comp, BeamSplitter):
            # Handle beam splitter parameter transformation
            if comp.theta_role == ParameterRole.TRAINABLE and not share_trainable:
                if comp.theta_name:
                    new_comp.theta_name = f"{comp.theta_name}_copy{self._copy_counter}"
                else:
                    new_comp.theta_name = f"theta_bs_copy_{self._copy_counter}"
                self._copy_counter += 1
            elif comp.theta_role == ParameterRole.INPUT and not share_input:
                if comp.theta_name:
                    base_name = comp.theta_name.rstrip('0123456789')
                else:
                    base_name = "x_bs"
                new_comp.theta_name = f"{base_name}{self._input_counter}"
                self._input_counter += 1

            # Same for phi
            if comp.phi_role == ParameterRole.TRAINABLE and not share_trainable:
                if comp.phi_name:
                    new_comp.phi_name = f"{comp.phi_name}_copy{self._copy_counter}"
                else:
                    new_comp.phi_name = f"phi_bs_copy_{self._copy_counter}"
                self._copy_counter += 1
            elif comp.phi_role == ParameterRole.INPUT and not share_input:
                if comp.phi_name:
                    base_name = comp.phi_name.rstrip('0123456789')
                else:
                    base_name = "x_phi"
                new_comp.phi_name = f"{base_name}{self._input_counter}"
                self._input_counter += 1

        # EntanglingBlock doesn't need special handling as its parameters
        # are generated during compilation

        return new_comp

    def end_section(self) -> "CircuitBuilder":
        """
        Mark the end of a circuit section.

        Returns:
            self for chaining
        """
        if self._current_section:
            self._current_section['end_idx'] = len(self.circuit.components)
            self._section_markers.append(self._current_section)
            self._current_section = None
        else:
            warnings.warn("No section to end")
        return self

    def build(self) -> Circuit:
        """
        Build and return the circuit.

        Returns:
            The constructed circuit with metadata
        """
        # Close any open section
        if self._current_section is not None:
            warnings.warn(f"Section '{self._current_section['name']}' was not closed. Closing it now.")
            self.end_section()

        # Add section markers to metadata if present
        if self._section_markers:
            self.circuit.metadata['sections'] = self._section_markers

        # Set photon number if specified
        if self.n_photons is not None:
            self.circuit.metadata['n_photons'] = self.n_photons

        return self.circuit

    @classmethod
    def from_circuit(cls, circuit: Circuit) -> "CircuitBuilder":
        """Create a builder from an existing circuit."""
        builder = cls(circuit.n_modes)
        builder.circuit = circuit
        return builder