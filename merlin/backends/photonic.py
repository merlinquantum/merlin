"""
Photonic backend using Perceval + pcvl_pytorch.
Clean compilation from descriptive components to Perceval circuits.
Now with amplitude output support and proper dtype handling.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import torch
import numpy as np
import random
import warnings

try:
    import perceval as pcvl

    PERCEVAL_AVAILABLE = True
except Exception:
    PERCEVAL_AVAILABLE = False
    pcvl = None

from ..backends.computation_process import ComputationProcessFactory
from ..core.components import ParameterRole
from ..measurements.processor import PhotonicMeasurementProcessor
from ..core.observables import CompositeObservable
from ..backends.pcvl_pytorch.locirc_to_tensor import CircuitConverter


class PhotonicBackend:
    """
    Photonic backend for quantum circuit execution.
    Now supports both probability and amplitude outputs with proper dtype handling.
    """

    def __init__(
            self,
            n_modes: int,
            n_photons: Optional[int] = None,
            no_bunching: bool = True,
            index_photons: Optional[List[Tuple[int, int]]] = None,
            reservoir_mode: bool = False,
            input_state: Optional[Union[List[int], Dict]] = None,
            trainable_parameters: Optional[List[str]] = None,
            input_parameters: Optional[List[str]] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            **kwargs,
    ):
        self.name = "photonic"
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32

        self.no_bunching = no_bunching
        self.index_photons = index_photons
        self.reservoir_mode = reservoir_mode
        self.input_state = input_state or self._default_input_state()

        # Parameter specs for direct Perceval circuits
        self.trainable_parameters = trainable_parameters or []
        self.input_parameters = input_parameters or []

        # Computation process (our execution engine)
        self.computation_process = None

        # Track parameter information
        self.param_info = {}

        # Measurements
        self.measurements = []
        self.measurement_processor = None

        # Section tracking
        self.circuit = None
        self._section_converters = {}  # Cache for section converters

    def _default_input_state(self) -> List[int]:
        """Create default input state."""
        if self.n_photons is None:
            return [1] * self.n_modes

        state = [0] * self.n_modes
        for i in range(min(self.n_photons, self.n_modes)):
            state[i] = 1
        return state

    def _generate_interferometer(self, n_modes: int, stage_idx: int, reservoir_mode: bool = False):
        """Generate a rectangular interferometer."""
        if not PERCEVAL_AVAILABLE:
            raise ImportError("Perceval is required for PhotonicBackend")

        if reservoir_mode:
            return pcvl.GenericInterferometer(
                n_modes,
                lambda idx: pcvl.BS(theta=np.pi * 2 * random.random())
                            // (0, pcvl.PS(phi=np.pi * 2 * random.random())),
                shape=pcvl.InterferometerShape.RECTANGLE,
                depth=2 * n_modes,
                phase_shifter_fun_gen=lambda idx: pcvl.PS(
                    phi=np.pi * 2 * random.random()
                ),
            )
        else:
            def mzi(P1, P2):
                return (
                    pcvl.Circuit(2)
                    .add((0, 1), pcvl.BS())
                    .add(0, pcvl.PS(P1))
                    .add((0, 1), pcvl.BS())
                    .add(0, pcvl.PS(P2))
                )

            offset = stage_idx * (n_modes * (n_modes - 1) // 2)

            return pcvl.GenericInterferometer(
                n_modes,
                fun_gen=lambda idx: mzi(
                    pcvl.P(f"phi_0{offset + idx}"),
                    pcvl.P(f"phi_1{offset + idx}")
                ),
                shape=pcvl.InterferometerShape.RECTANGLE,
                phase_shifter_fun_gen=lambda idx: pcvl.PS(
                    phi=pcvl.P(f"phi_02{stage_idx}_{idx}")
                ),
            )

    def setup_circuit(self, circuit: Any) -> Dict[str, Any]:
        """
        Setup circuit for execution.
        Returns parameter info for the model to register.
        """
        # Store circuit reference
        self.circuit = circuit

        # Extract measurements from circuit metadata if present
        if hasattr(circuit, 'metadata') and 'measurements' in circuit.metadata:
            self.measurements = circuit.metadata['measurements']

        # Check if it's a Perceval circuit
        if hasattr(circuit, "__module__") and "perceval" in circuit.__module__:
            return self._setup_perceval_direct(circuit)
        else:
            # Platform-agnostic circuit
            return self._setup_platform_agnostic(circuit)

    def _setup_perceval_direct(self, circuit: "pcvl.Circuit") -> Dict[str, Any]:
        """Setup direct Perceval circuit."""
        # Create computation process
        self.computation_process = ComputationProcessFactory.create(
            circuit=circuit,
            input_state=self.input_state,
            trainable_parameters=self.trainable_parameters,
            input_parameters=self.input_parameters,
            reservoir_mode=self.reservoir_mode,
            device=self.device,
            dtype=self.dtype,
            no_bunching=self.no_bunching,
            index_photons=self.index_photons,
        )

        # Setup measurement processor if needed
        if self.measurements:
            self.measurement_processor = PhotonicMeasurementProcessor(
                self.computation_process
            )

        # Get spec mappings for parameter grouping
        spec_mappings = self.computation_process.converter.spec_mappings

        # Return info for model to register parameters
        trainable_info = {}
        for param_name in self.computation_process.trainable_parameters:
            if param_name in spec_mappings:
                trainable_info[param_name] = len(spec_mappings[param_name])

        return {
            'trainable': trainable_info,
            'input': self.computation_process.input_parameters
        }

    def _setup_platform_agnostic(self, circuit: "Circuit") -> Dict[str, Any]:
        """Compile platform-agnostic circuit."""
        compiled = self._compile_to_perceval(circuit)

        # Create computation process
        self.computation_process = ComputationProcessFactory.create(
            circuit=compiled['pcvl_circuit'],
            input_state=self.input_state,
            trainable_parameters=compiled['trainable_params'],
            input_parameters=compiled['input_params'],
            reservoir_mode=self.reservoir_mode,
            device=self.device,
            dtype=self.dtype,
            no_bunching=self.no_bunching,
            index_photons=self.index_photons,
        )

        # Setup measurement processor if needed
        if self.measurements:
            self.measurement_processor = PhotonicMeasurementProcessor(
                self.computation_process
            )

        # Get spec mappings
        spec_mappings = self.computation_process.converter.spec_mappings

        # Return info for parameter registration
        trainable_info = {}
        for param_name in self.computation_process.trainable_parameters:
            if param_name in spec_mappings:
                trainable_info[param_name] = len(spec_mappings[param_name])

        # Store parameter info for debugging
        self.param_info = {
            'trainable': trainable_info,
            'input': self.computation_process.input_parameters,
            'spec_mappings': spec_mappings
        }

        return self.param_info

    def _compile_to_perceval(self, circuit: "Circuit") -> Dict[str, Any]:
        """
        Compile circuit to Perceval with clear parameter categorization.
        """
        if not PERCEVAL_AVAILABLE:
            raise ImportError("Perceval is required for PhotonicBackend")

        from ..core.components import Rotation, BeamSplitter, EntanglingBlock, Measurement

        pcvl_circ = pcvl.Circuit(circuit.n_modes)

        # Clear tracking of parameters by role
        input_params = []
        trainable_params = []

        # Track all parameters for debugging
        all_params = {}  # name -> role

        # Counters for auto-naming when needed
        param_counter = {'input': 0, 'trainable': 0, 'bs': 0}
        interferometer_counter = 0

        for comp in circuit.components:
            if isinstance(comp, Rotation):
                t = comp.target
                if t >= pcvl_circ.m:
                    continue

                if comp.role == ParameterRole.FIXED:
                    # Fixed value - no parameter
                    pcvl_circ.add(t, pcvl.PS(comp.value))

                elif comp.role == ParameterRole.INPUT:
                    # Input parameter
                    param_name = comp.custom_name or f"x_{param_counter['input']}_{t}"
                    param_counter['input'] += 1

                    pcvl_circ.add(t, pcvl.PS(pcvl.P(param_name)))
                    if param_name not in input_params:
                        input_params.append(param_name)
                    all_params[param_name] = "input"

                elif comp.role == ParameterRole.TRAINABLE:
                    # Trainable parameter
                    param_name = comp.custom_name or f"theta_{param_counter['trainable']}_{t}"
                    param_counter['trainable'] += 1

                    pcvl_circ.add(t, pcvl.PS(pcvl.P(param_name)))
                    if param_name not in trainable_params:
                        trainable_params.append(param_name)
                    all_params[param_name] = "trainable"

            elif isinstance(comp, BeamSplitter):
                t0, t1 = sorted(comp.targets)
                if t1 >= pcvl_circ.m or abs(t1 - t0) != 1:
                    continue

                # Handle theta
                if comp.theta_role == ParameterRole.FIXED:
                    theta = comp.theta_value
                elif comp.theta_role == ParameterRole.INPUT:
                    theta_name = comp.theta_name or f"x_bs_{param_counter['bs']}_{t0}_{t1}"
                    param_counter['bs'] += 1
                    theta = pcvl.P(theta_name)
                    if theta_name not in input_params:
                        input_params.append(theta_name)
                    all_params[theta_name] = "input"
                else:  # TRAINABLE
                    theta_name = comp.theta_name or f"theta_bs_{param_counter['bs']}_{t0}_{t1}"
                    param_counter['bs'] += 1
                    theta = pcvl.P(theta_name)
                    if theta_name not in trainable_params:
                        trainable_params.append(theta_name)
                    all_params[theta_name] = "trainable"

                pcvl_circ.add((t0, t1), pcvl.BS(theta))

                # Handle phi if needed
                if comp.phi_role != ParameterRole.FIXED or comp.phi_value != 0:
                    if comp.phi_role == ParameterRole.FIXED:
                        pcvl_circ.add(t0, pcvl.PS(comp.phi_value))
                    else:
                        phi_name = comp.phi_name or f"phi_bs_{t0}_{t1}"
                        pcvl_circ.add(t0, pcvl.PS(pcvl.P(phi_name)))
                        if comp.phi_role == ParameterRole.INPUT:
                            if phi_name not in input_params:
                                input_params.append(phi_name)
                            all_params[phi_name] = "input"
                        else:
                            if phi_name not in trainable_params:
                                trainable_params.append(phi_name)
                            all_params[phi_name] = "trainable"

            elif isinstance(comp, EntanglingBlock):
                # Generate interferometer with appropriate parameters
                for d in range(comp.depth):
                    interferometer = self._generate_interferometer(
                        circuit.n_modes,
                        interferometer_counter,
                        reservoir_mode=(not comp.trainable or self.reservoir_mode)
                    )
                    pcvl_circ.add(0, interferometer)

                    if comp.trainable and not self.reservoir_mode:
                        # Register all generated parameters as trainable
                        offset = interferometer_counter * (circuit.n_modes * (circuit.n_modes - 1) // 2)

                        # MZI internal phases
                        for idx in range(circuit.n_modes * (circuit.n_modes - 1) // 2):
                            param_0 = f"phi_0{offset + idx}"
                            param_1 = f"phi_1{offset + idx}"
                            if param_0 not in trainable_params:
                                trainable_params.append(param_0)
                            if param_1 not in trainable_params:
                                trainable_params.append(param_1)
                            all_params[param_0] = "trainable"
                            all_params[param_1] = "trainable"

                        # External phase shifters
                        for idx in range(circuit.n_modes):
                            param = f"phi_02{interferometer_counter}_{idx}"
                            if param not in trainable_params:
                                trainable_params.append(param)
                            all_params[param] = "trainable"

                    interferometer_counter += 1

            elif isinstance(comp, Measurement):
                # Measurements are metadata only
                pass

        return {
            'pcvl_circuit': pcvl_circ,
            'input_params': input_params,
            'trainable_params': trainable_params,
            'param_roles': all_params
        }

    def _compile_to_perceval(self, circuit: "Circuit") -> Dict[str, Any]:
        """
        Compile circuit to Perceval with clear parameter categorization.
        """
        if not PERCEVAL_AVAILABLE:
            raise ImportError("Perceval is required for PhotonicBackend")

        from ..core.components import Rotation, BeamSplitter, EntanglingBlock, Measurement

        pcvl_circ = pcvl.Circuit(circuit.n_modes)

        # Clear tracking of parameters by role
        input_params = []
        trainable_params = []

        # Track all parameters for debugging
        all_params = {}  # name -> role

        # GLOBAL counters that never reset
        param_counter = {'input': 0, 'trainable': 0, 'bs': 0}
        interferometer_counter = 0

        for comp in circuit.components:
            if isinstance(comp, Rotation):
                t = comp.target
                if t >= pcvl_circ.m:
                    continue

                if comp.role == ParameterRole.FIXED:
                    # Fixed value - no parameter
                    pcvl_circ.add(t, pcvl.PS(comp.value))

                elif comp.role == ParameterRole.INPUT:
                    # Input parameter - use the name from the component
                    param_name = comp.custom_name or f"x_{param_counter['input']}_{t}"
                    param_counter['input'] += 1

                    pcvl_circ.add(t, pcvl.PS(pcvl.P(param_name)))
                    if param_name not in input_params:
                        input_params.append(param_name)
                    all_params[param_name] = "input"

                elif comp.role == ParameterRole.TRAINABLE:
                    # Trainable parameter
                    param_name = comp.custom_name or f"theta_{param_counter['trainable']}_{t}"
                    param_counter['trainable'] += 1

                    pcvl_circ.add(t, pcvl.PS(pcvl.P(param_name)))
                    if param_name not in trainable_params:
                        trainable_params.append(param_name)
                    all_params[param_name] = "trainable"

            elif isinstance(comp, BeamSplitter):
                # ... beam splitter handling remains the same ...
                pass

            elif isinstance(comp, EntanglingBlock):
                # Generate interferometer with appropriate parameters
                for d in range(comp.depth):
                    interferometer = self._generate_interferometer(
                        circuit.n_modes,
                        interferometer_counter,  # Use global counter
                        reservoir_mode=(not comp.trainable or self.reservoir_mode)
                    )
                    pcvl_circ.add(0, interferometer)

                    if comp.trainable and not self.reservoir_mode:
                        # Register all generated parameters as trainable
                        offset = interferometer_counter * (circuit.n_modes * (circuit.n_modes - 1) // 2)

                        # MZI internal phases
                        for idx in range(circuit.n_modes * (circuit.n_modes - 1) // 2):
                            param_0 = f"phi_0{offset + idx}"
                            param_1 = f"phi_1{offset + idx}"
                            if param_0 not in trainable_params:
                                trainable_params.append(param_0)
                            if param_1 not in trainable_params:
                                trainable_params.append(param_1)
                            all_params[param_0] = "trainable"
                            all_params[param_1] = "trainable"

                        # External phase shifters
                        for idx in range(circuit.n_modes):
                            param = f"phi_02{interferometer_counter}_{idx}"
                            if param not in trainable_params:
                                trainable_params.append(param)
                            all_params[param] = "trainable"

                    interferometer_counter += 1  # Always increment globally

            elif isinstance(comp, Measurement):
                # Measurements are metadata only
                pass

        return {
            'pcvl_circuit': pcvl_circ,
            'input_params': input_params,
            'trainable_params': trainable_params,
            'param_roles': all_params
        }

    def execute(
            self,
            params: Dict[str, torch.Tensor],
            input_data: Optional[torch.Tensor] = None,
            shots: int = 0,
            return_amplitudes: bool = False,
            **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Execute circuit with given parameters.
        Handles sections with adjoints transparently.
        """
        if self.computation_process is None:
            raise RuntimeError("No circuit setup. Call setup_circuit() first.")

        # Prepare parameter list
        param_list = self._prepare_parameters(params, input_data)

        # Get unitary - either composed from sections or directly
        if hasattr(self.circuit, 'metadata') and 'sections' in self.circuit.metadata:
            unitary = self._compose_section_unitaries(params, input_data)
        else:
            unitary = self.computation_process.converter.to_tensor(*param_list)

        # Compute amplitudes using SLOS
        if isinstance(self.computation_process.input_state, dict):
            # Superposition state
            amplitudes = self.computation_process.compute_superposition_state_from_unitary(unitary)
        else:
            # Regular state
            keys, amplitudes = self.computation_process.simulation_graph.compute(
                unitary, self.input_state
            )

        # Convert to probabilities
        if torch.is_complex(amplitudes):
            probabilities = amplitudes.real ** 2 + amplitudes.imag ** 2
        else:
            probabilities = amplitudes ** 2

        # Normalize if no_bunching
        if self.no_bunching:
            sum_probs = probabilities.sum(dim=1 if probabilities.dim() > 1 else 0, keepdim=True)
            valid_entries = sum_probs > 0
            if valid_entries.any():
                probabilities = torch.where(
                    valid_entries,
                    probabilities / torch.where(valid_entries, sum_probs, torch.ones_like(sum_probs)),
                    probabilities
                )

        # Sample if requested
        if shots > 0:
            probabilities = self._sample(probabilities, shots)

        # Process measurements if present
        if self.measurements and self.measurement_processor is not None:
            results = []
            for meas_dict in self.measurements:
                observable = meas_dict['observable']

                if isinstance(observable, CompositeObservable):
                    result = self.measurement_processor.process_composite(
                        probabilities, observable
                    )
                else:
                    result = self.measurement_processor.process_observable(
                        probabilities, observable
                    )
                results.append(result)

            # Stack results
            if len(results) == 1:
                return results[0]
            return torch.cat(results, dim=-1)

        # Return based on amplitude request
        if return_amplitudes:
            # Ensure amplitudes are complex for consistency
            if not torch.is_complex(amplitudes):
                complex_dtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128
                amplitudes = amplitudes.to(dtype=complex_dtype)
            return probabilities, amplitudes
        else:
            return probabilities

    def _compose_section_unitaries(
            self,
            params: Dict[str, torch.Tensor],
            input_data: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compose unitaries from circuit sections."""
        sections = self.circuit.metadata['sections']

        # Determine batch size
        batch_size = 1
        if input_data is not None:
            if input_data.dim() == 1:
                input_data = input_data.unsqueeze(0)
            batch_size = input_data.shape[0]

        # Compute unitary for each section
        section_unitaries = []
        for section in sections:
            U_section = self._compute_section_unitary(section, params, input_data)

            if section.get('compute_adjoint', False):
                # Apply adjoint
                U_section = U_section.conj().transpose(-2, -1)

            section_unitaries.append(U_section)

        # Compose all section unitaries: U_total = U_n @ ... @ U_2 @ U_1
        if len(section_unitaries) == 1:
            U_total = section_unitaries[0]
        else:
            U_total = section_unitaries[0]
            for U in section_unitaries[1:]:
                # Handle batched matrix multiplication
                if U_total.dim() == 3 and U.dim() == 3:
                    U_total = torch.bmm(U, U_total)
                elif U_total.dim() == 2 and U.dim() == 2:
                    U_total = U @ U_total
                else:
                    # Ensure both have same dimensions
                    if U_total.dim() == 2:
                        U_total = U_total.unsqueeze(0).expand(batch_size, -1, -1)
                    if U.dim() == 2:
                        U = U.unsqueeze(0).expand(batch_size, -1, -1)
                    U_total = torch.bmm(U, U_total)

        return U_total

    def _compute_section_unitary(
            self,
            section: Dict,
            params: Dict[str, torch.Tensor],
            input_data: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute unitary for a specific circuit section."""
        # Extract section components
        section_components = self.circuit.components[section['start_idx']:section['end_idx']]

        # Build temporary circuit
        from ..core.circuit import Circuit
        temp_circuit = Circuit(self.circuit.n_modes)
        for comp in section_components:
            temp_circuit.add(comp)

        # Compile to Perceval
        compiled = self._compile_to_perceval(temp_circuit)

        # Get or create converter for this section
        section_key = f"{section['name']}_{section['start_idx']}_{section['end_idx']}"
        if section_key not in self._section_converters:
            self._section_converters[section_key] = CircuitConverter(
                compiled['pcvl_circuit'],
                compiled['trainable_params'] + compiled['input_params'],
                dtype=self.dtype,
                device=self.device
            )
        converter = self._section_converters[section_key]

        # Prepare parameters for this section
        section_param_list = []

        # Add trainable parameters
        for param_name in compiled['trainable_params']:
            if param_name in params:
                section_param_list.append(params[param_name])

        # Add input parameters
        if compiled['input_params'] and input_data is not None:
            # For sections, input parameters get the same data
            # This allows compute-uncompute patterns
            for _ in compiled['input_params']:
                section_param_list.append(input_data)

        # Get unitary
        return converter.to_tensor(*section_param_list)



    def _prepare_parameters(
            self,
            params: Dict[str, torch.Tensor],
            input_data: Optional[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Prepare ordered parameter list for computation process.
        """
        spec_mappings = self.computation_process.converter.spec_mappings
        param_list = []

        # Determine batch size and ensure correct dtype
        batch_size = 1
        if input_data is not None:
            if input_data.dim() == 1:
                input_data = input_data.unsqueeze(0)
            batch_size = input_data.shape[0]
            # Ensure correct dtype
            input_data = input_data.to(dtype=self.dtype, device=self.device)

        # Add trainable parameters in order
        for param_name in self.computation_process.trainable_parameters:
            if param_name in params:
                p = params[param_name]
                # Ensure correct dtype
                p = p.to(dtype=self.dtype, device=self.device)
                if batch_size > 1 and p.dim() == 1:
                    p = p.unsqueeze(0).expand(batch_size, -1)
            else:
                # Default to zeros for missing trainable params
                if param_name in spec_mappings:
                    size = len(spec_mappings[param_name])
                else:
                    size = 1
                p = torch.zeros(size, dtype=self.dtype, device=self.device)
                if batch_size > 1:
                    p = p.unsqueeze(0).expand(batch_size, -1)
            param_list.append(p)

        # Add input parameters
        if len(self.computation_process.input_parameters) > 0:
            if input_data is not None:
                n_features = input_data.shape[-1]

                # For each input parameter spec
                for i, input_param_name in enumerate(self.computation_process.input_parameters):
                    if input_param_name in spec_mappings:
                        # Get the actual parameter names this spec maps to
                        param_names = spec_mappings[input_param_name]
                        n_params = len(param_names)

                        # Create tensor for these parameters
                        p = torch.zeros(batch_size, n_params, dtype=self.dtype, device=self.device)

                        # Map input features to parameters
                        if n_features == 1:
                            # Single feature - broadcast to all params
                            p[:, :] = input_data[:, 0:1]
                        elif n_params == 1:
                            # Single parameter - take first feature or cycle
                            feature_idx = i % n_features
                            p = input_data[:, feature_idx:feature_idx + 1]
                        else:
                            # Multiple parameters - map cyclically from features
                            for j in range(n_params):
                                feat_idx = j % n_features
                                p[:, j] = input_data[:, feat_idx]

                        # Squeeze if single batch
                        if batch_size == 1:
                            p = p.squeeze(0)

                        # Ensure gradient flow
                        if input_data.requires_grad:
                            if not p.requires_grad:
                                p = p.requires_grad_(True)
                    else:
                        # Single parameter
                        if i < n_features:
                            p = input_data[:, i:i + 1]
                        else:
                            p = input_data[:, 0:1]

                        if batch_size == 1:
                            p = p.squeeze()

                        # Ensure correct dtype
                        p = p.to(dtype=self.dtype, device=self.device)

                        if input_data.requires_grad:
                            p = p.requires_grad_(True)

                    param_list.append(p)
            else:
                # No input data - use zeros
                for input_param_name in self.computation_process.input_parameters:
                    if input_param_name in spec_mappings:
                        n_params = len(spec_mappings[input_param_name])
                    else:
                        n_params = 1
                    p = torch.zeros(n_params, dtype=self.dtype, device=self.device)
                    if batch_size > 1:
                        p = p.unsqueeze(0).expand(batch_size, -1)
                    param_list.append(p)

        return param_list

    def _sample(self, probs: torch.Tensor, shots: int) -> torch.Tensor:
        """Sample from probability distribution."""
        if probs.dim() == 1:
            idx = torch.multinomial(probs, shots, replacement=True)
            out = torch.zeros_like(probs)
            out.index_add_(0, idx, torch.ones_like(idx, dtype=probs.dtype))
            return out / shots
        else:
            b = probs.shape[0]
            out = torch.zeros_like(probs)
            for i in range(b):
                idx = torch.multinomial(probs[i], shots, replacement=True)
                out[i].index_add_(0, idx, torch.ones_like(idx, dtype=probs.dtype))
            return out / shots

    def get_info(self) -> Dict[str, Any]:
        """Get backend-specific information."""
        info = {
            'n_modes': self.n_modes,
            'n_photons': self.n_photons,
            'platform': 'photonic',
            'no_bunching': self.no_bunching,
            'has_measurements': len(self.measurements) > 0,
            'has_sections': hasattr(self.circuit, 'metadata') and 'sections' in self.circuit.metadata,
        }

        if self.computation_process:
            info['trainable_params'] = self.computation_process.trainable_parameters
            info['input_params'] = self.computation_process.input_parameters
            if hasattr(self.computation_process.converter, 'spec_mappings'):
                info['spec_mappings'] = self.computation_process.converter.spec_mappings

        return info

    @property
    def supports_gradients(self) -> bool:
        return True

    @property
    def supports_amplitudes(self) -> bool:
        """Indicates this backend supports amplitude output."""
        return True