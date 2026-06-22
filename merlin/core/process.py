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
Quantum computation processes and factories.
"""

import math
from dataclasses import dataclass
from typing import Any, Literal, overload

import perceval as pcvl
import torch

from merlin.core.sectored_distribution import SectoredDistribution, SectorResult
from merlin.pcvl_pytorch.noisy_slos import (
    NoisyG2SLOSComputeGraph,
    NoisySLOSComputeGraph,
)

from ..algorithms.layer_utils import (
    NoiseGroups,
    _circuit_has_phase_error,
    _with_component_phase_error,
    has_circuit_noise,
    has_phase_error,
    has_source_noise,
)
from ..pcvl_pytorch import (
    CircuitConverter,
    build_slos_distribution_computegraph,
)
from ..utils.combinadics import Combinadics
from ..utils.deprecations import raise_no_bunching_removed
from ..utils.normalization import normalize_probabilities, probabilities_from_amplitudes
from .base import AbstractComputationProcess
from .computation_space import ComputationSpace
from .state import _generate_default_input_state

_DEFAULT_SUPERPOSITION_CHUNK_SIZE = 32


@dataclass(frozen=True)
class _SuperpositionSupport:
    """Compact internal representation of active superposition components."""

    basis_indices: list[int]
    coefficients: torch.Tensor
    basis_size: int

    @property
    def batch_size(self) -> int:
        return self.coefficients.shape[0]

    @property
    def device(self) -> torch.device:
        return self.coefficients.device

    @property
    def is_sparse(self) -> bool:
        return True

    @property
    def nnz(self) -> int:
        return len(self.basis_indices)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.batch_size, self.basis_size)


class ComputationProcess(AbstractComputationProcess):
    """Handle quantum circuit computation and state evolution.

    Parameters
    ----------
    circuit : pcvl.Circuit
        Circuit used to build the unitary and simulation graphs.
    input_state : list[int] | torch.Tensor
        Input Fock state or superposition tensor.
    trainable_parameters : list[str]
        Prefixes of trainable circuit parameters.
    input_parameters : list[str]
        Prefixes of input-driven circuit parameters.
    n_photons : int | None
        Number of photons represented by the process.
    dtype : torch.dtype
        Real dtype used for internal tensor conversions.
    device : torch.device | None
        Device on which computation graphs are materialized.
    computation_space : ComputationSpace | None
        Computation space used for basis enumeration.
    no_bunching : bool | None
        Removed legacy parameter.
    output_map_func : Any
        Optional output mapping function.
    noise_groups : NoiseGroups | None
        The noise groups applied to the circuit to be ran.
    n_phase_error_samples : int
        Number of Monte Carlo unitary samples used when active
        ``phase_error`` is present. For each sample, the converter draws fresh
        perturbations around the commanded phases after any
        ``phase_imprecision`` quantization, computes probabilities for that
        sampled unitary, and averages probabilities. Amplitudes are not
        averaged.
    """

    def __init__(
        self,
        circuit: pcvl.Circuit,
        input_state: list[int] | torch.Tensor,
        trainable_parameters: list[str],
        input_parameters: list[str],
        n_photons: int = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        computation_space: ComputationSpace | None = None,
        memristive_metadata: list[dict] | None = None,
        no_bunching: bool | None = None,
        output_map_func=None,
        noise_groups: NoiseGroups | None = None,
        n_phase_error_samples: int = 1,
    ):
        """Initialize a computation process.

        Parameters
        ----------
        circuit : pcvl.Circuit
            Circuit used to build the unitary and simulation graphs.
        input_state : list[int] | torch.Tensor
            Input Fock state or superposition tensor.
        trainable_parameters : list[str]
            Prefixes of trainable circuit parameters.
        input_parameters : list[str]
            Prefixes of input-driven circuit parameters.
        n_photons : int | None
            Number of photons represented by the process.
        dtype : torch.dtype
            Real dtype used for internal tensor conversions.
        device : torch.device | None
            Device on which computation graphs are materialized.
        computation_space : ComputationSpace | None
            Computation space used for basis enumeration.
        memristive_metadata: list[dict] | None
            The memristive phase shifter metadata. If None, it will be stored as an empty list.
        no_bunching : bool | None
            Removed legacy parameter.
        output_map_func : Any
            Optional output mapping function.
        noise_groups : NoiseGroups | None
            The noise groups applied to the circuit to be ran.
        n_phase_error_samples : int
            Number of Monte Carlo unitary samples used when active
            ``phase_error`` is present. Each sample computes probabilities for
            one perturbed unitary; the process averages those probabilities.
            This matters for coherent tensor superpositions: amplitudes are
            converted to probabilities per sample before averaging. Default
            value is 1.

        Raises
        ------
        TypeError
            If ``n_phase_error_samples`` is not an integer.
        ValueError
            If ``n_phase_error_samples`` is lower than 1.
        """
        if not isinstance(n_phase_error_samples, int) or isinstance(
            n_phase_error_samples, bool
        ):
            raise TypeError("n_phase_error_samples must be an integer.")
        if n_phase_error_samples < 1:
            raise ValueError("n_phase_error_samples must be at least 1.")

        self.circuit = circuit
        self.input_state = input_state
        self.n_photons = n_photons
        self.trainable_parameters = trainable_parameters
        self.input_parameters = input_parameters
        self.dtype = dtype
        self.device = device
        self.memristive_metadata = (
            [] if memristive_metadata is None else memristive_metadata
        )
        self.noise_groups = (
            _with_component_phase_error(noise_groups)
            if _circuit_has_phase_error(circuit)
            else noise_groups
        )
        self._n_phase_error_samples = n_phase_error_samples
        self._phase_imprecision = 0.0
        self._phase_error = 0.0
        self._setup_phase_noise()

        if no_bunching is not None:
            raise_no_bunching_removed()

        if computation_space is None:
            computation_space = ComputationSpace.UNBUNCHED

        self.computation_space = computation_space
        self.output_map_func = output_map_func
        self._input_basis_states_cache: list[tuple[int, ...]] | None = None

        # Extract circuit parameters for graph building

        self.m = circuit.m  # Number of modes
        if n_photons is None:
            if type(input_state) is list:
                self.n_photons = sum(input_state)  # Total number of photons
            else:
                raise ValueError("The number of photons should be provided")
        else:
            self.n_photons = n_photons
        # Build computation graphs
        self._setup_computation_graphs()
        # Updating the computation space if the simulation is noisy
        self.computation_space = self.simulation_graph.computation_space

        # validate initial input state shape when provided as tensor
        if isinstance(self.input_state, torch.Tensor):
            state_tensor: torch.Tensor = self.input_state
            self._validate_superposition_state_shape(state_tensor)

    def _input_basis_states(self) -> list[tuple[int, ...]]:
        """Enumerate the full Fock basis used for superposition inputs."""
        if self._input_basis_states_cache is None:
            self._input_basis_states_cache = Combinadics(
                "fock", self.n_photons, self.m
            ).enumerate_states()
        return self._input_basis_states_cache

    def _setup_phase_noise(self) -> None:
        """Resolve process-local phase-noise settings from classified noise groups."""
        if not has_circuit_noise(self.noise_groups):
            return

        circuit_noise = self.noise_groups.circuit
        if circuit_noise is None:
            return
        if "phase_imprecision" in circuit_noise:
            self._phase_imprecision = float(circuit_noise["phase_imprecision"])
        if "phase_error" in circuit_noise and circuit_noise["phase_error"] is not None:
            self._phase_error = float(circuit_noise["phase_error"])

    def _has_source_noise(self) -> bool:
        """Return whether this process uses a source-noise simulation graph."""
        return has_source_noise(self.noise_groups)

    def _has_phase_error(self) -> bool:
        """Return whether this process must sample stochastic phase-error unitaries."""
        return has_phase_error(self.noise_groups)

    def _returns_probabilities(self) -> bool:
        """Return whether :meth:`compute` returns probabilities instead of amplitudes."""
        return self._has_source_noise() or self._has_phase_error()

    def _setup_computation_graphs(self):
        """Setup unitary and simulation computation graphs."""
        # Determine parameter specs
        parameter_specs = self.trainable_parameters + self.input_parameters

        # Build unitary graph
        self.converter = CircuitConverter(
            self.circuit,
            parameter_specs,
            memristive_metadata=self.memristive_metadata,
            dtype=self.dtype,
            device=self.device,
            phase_imprecision=self._phase_imprecision,
            phase_error=self._phase_error,
        )

        # Build simulation graph with correct parameters
        self.simulation_graph = build_slos_distribution_computegraph(
            m=self.m,  # Number of modes
            n_photons=self.n_photons,  # Total number of photons
            computation_space=self.computation_space,
            keep_keys=True,  # Usually want to keep keys for output interpretation
            noise_groups=self.noise_groups,
            device=self.device,
            dtype=self.dtype,
        )
        self.noisy_simulation = isinstance(
            self.simulation_graph, NoisySLOSComputeGraph
        ) or isinstance(self.simulation_graph, NoisyG2SLOSComputeGraph)

    def to(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device | None,
    ) -> "ComputationProcess":
        """Move runtime computation state to a new dtype or device.

        Parameters
        ----------
        dtype : torch.dtype
            Target real tensor dtype used by the unitary converter and SLOS
            simulation graph.
        device : torch.device | None
            Target tensor device. If omitted, runtime tensors keep PyTorch's
            implicit CPU placement.

        Returns
        -------
        ComputationProcess
            Updated computation process instance.

        Raises
        ------
        TypeError
            If the converter rejects the target dtype or device.
        ValueError
            If the SLOS simulation graph cannot be rebuilt for the target dtype.
        """
        dtype_changed = dtype != self.dtype
        self.dtype = dtype
        self.device = device

        if dtype_changed:
            self._setup_computation_graphs()
            return self

        converter_device = device if device is not None else torch.device("cpu")
        self.converter = self.converter.to(dtype, converter_device)
        if device is not None:
            self.simulation_graph = self.simulation_graph.to(device)
        return self

    def _default_fixed_input_state(self) -> list[int]:
        """Return the default fixed input state for tensor input placeholders.

        Returns
        -------
        list[int]
            Occupation-number input state generated from the process mode count,
            photon count, and computation space.
        """
        return _generate_default_input_state(
            self.m,
            self.n_photons,
            self.computation_space,
        )

    def _fixed_input_state_for_compute(self) -> list[int] | torch.Tensor:
        """Return the fixed input state used by direct SLOS graph calls.

        Tensor ``input_state`` values represent superposition data only when the
        caller explicitly requests superposition handling. Direct fixed-state
        graph calls therefore use the deterministic default Fock input instead
        of passing the tensor through.

        Returns
        -------
        list[int] | torch.Tensor
            Fixed Fock input state for direct SLOS graph calls.
        """
        if isinstance(self.input_state, torch.Tensor):
            return self._default_fixed_input_state()
        return self.input_state

    def _compute_source_probabilities_for_unitary(
        self,
        unitary: torch.Tensor,
        *,
        amplitude_encoding: bool,
    ) -> torch.Tensor | SectoredDistribution:
        """Compute source-noise probabilities for one precomputed unitary.

        When ``input_state`` is a tensor and ``amplitude_encoding`` is True, the
        tensor is treated as a superposition over input basis states. Source
        noise is applied as an incoherent mixture: every active input basis state
        is propagated independently and weighted by the corresponding
        :math:`|c_i|^2`. For g2 noise, each photon-number sector is accumulated
        separately and aligned by sector keys before addition.

        Parameters
        ----------
        unitary : torch.Tensor
            Precomputed circuit unitary. A 2D tensor represents one parameter
            set and a 3D tensor represents a batch of parameter sets.
        amplitude_encoding : bool
            Whether tensor ``input_state`` values should be interpreted as a
            superposition over input basis states.

        Returns
        -------
        torch.Tensor | SectoredDistribution
            Output probabilities. g2 source noise returns a
            ``SectoredDistribution`` while other source-noise paths return a
            probability tensor.

        Raises
        ------
        RuntimeError
            If tensor-superposition handling finds no active input basis states.
        ValueError
            If g2 sector outputs cannot be aligned by basis keys while
            accumulating active input states.
        """
        if isinstance(self.input_state, torch.Tensor) and amplitude_encoding:
            # Amplitude-encoded input: treat each row as a probability distribution
            # over Fock basis states and produce a weighted mixture of noisy output
            # probabilities. The mixture weight for each basis state is |c_i|^2.
            prepared_state = self._prepare_superposition_support()
            weights = prepared_state.coefficients.abs().pow(2)

            active_indices = prepared_state.basis_indices
            input_basis_states = self._input_basis_states()
            if isinstance(self.simulation_graph, NoisyG2SLOSComputeGraph):
                output_distribution: SectoredDistribution | None = None
                for weight_index, idx in enumerate(active_indices):
                    input_fock_state = input_basis_states[idx]
                    probs = self.simulation_graph.compute_probs(
                        unitary, input_fock_state
                    )

                    for sector in probs.sectors:
                        if sector.tensor.ndim == 1:
                            sector.tensor = sector.tensor.unsqueeze(0)
                        selected_weights = weights[:, weight_index].to(
                            sector.tensor.dtype
                        )
                        sector.tensor = torch.einsum(
                            "i,bo->bio", selected_weights, sector.tensor
                        )
                        if sector.tensor.shape[0] == 1:
                            sector.tensor = sector.tensor.squeeze(0)
                        if sector.tensor.ndim == 3 and sector.tensor.shape[1] == 1:
                            sector.tensor = sector.tensor.squeeze(1)

                    if output_distribution is None:
                        output_distribution = probs
                    else:
                        for sector in output_distribution.sectors:
                            next_sector = probs.get_sector(sector.n_photons)
                            sector.tensor = sector.tensor + self._align_sector_tensor(
                                next_sector,
                                sector,
                            )

                if output_distribution is None:
                    raise RuntimeError("No active input states were found.")
                return output_distribution

            tensor_probs_per_state: list[torch.Tensor] = []
            for idx in active_indices:
                input_fock_state = input_basis_states[idx]
                _, probs = self.simulation_graph.compute_probs(
                    unitary, input_fock_state
                )
                if probs.ndim == 1:
                    probs = probs.unsqueeze(0)
                tensor_probs_per_state.append(probs)

            if not tensor_probs_per_state:
                raise RuntimeError("No active input states were found.")

            probs_stacked = torch.stack(tensor_probs_per_state, dim=0)
            selected_weights = weights.to(probs_stacked.dtype)
            mixed_probs = torch.einsum("is,sbo->bio", selected_weights, probs_stacked)

            if mixed_probs.shape[0] == 1:
                mixed_probs = mixed_probs.squeeze(0)
            if mixed_probs.ndim == 3 and mixed_probs.shape[1] == 1:
                mixed_probs = mixed_probs.squeeze(1)

            return mixed_probs

        result = self.simulation_graph.compute_probs(unitary, self.input_state)
        if isinstance(result, SectoredDistribution):
            return result
        _keys, probs = result
        return probs

    def _compute_superposition_amplitudes_for_unitary(
        self,
        unitary: torch.Tensor,
        *,
        simultaneous_processes: int = 1,
    ) -> torch.Tensor:
        """Compute coherent superposition amplitudes for one precomputed unitary.

        The stored tensor ``input_state`` is normalized and interpreted as
        complex amplitudes over the configured input basis. Each active basis
        state is propagated through the SLOS graph, the per-input amplitudes are
        normalized, and the final output amplitudes are formed by the coherent
        weighted sum over input coefficients.

        Parameters
        ----------
        unitary : torch.Tensor
            Precomputed circuit unitary. A 2D tensor represents one parameter
            set and a 3D tensor represents a batch of parameter sets.
        simultaneous_processes : int
            Maximum number of active input basis states propagated in one
            ``compute_batch`` call. Default is 1.

        Returns
        -------
        torch.Tensor
            Coherent output amplitudes. The leading batch dimensions are
            squeezed when they contain a single element to preserve existing
            caller expectations.

        Raises
        ------
        TypeError
            If ``input_state`` is not a tensor.
        ValueError
            If the tensor ``input_state`` does not match the configured
            computation-space basis size.
        """
        prepared_state = self._prepare_superposition_support()
        _keys_out, final_amplitudes = self._compute_chunked_superposition(
            prepared_state,
            unitary if unitary.dim() == 3 else unitary.unsqueeze(0),
            simultaneous_processes=simultaneous_processes,
        )

        if final_amplitudes.shape[0] == 1:
            final_amplitudes = final_amplitudes.squeeze(0)
        if final_amplitudes.ndim == 3 and final_amplitudes.shape[1] == 1:
            final_amplitudes = final_amplitudes.squeeze(1)

        return final_amplitudes

    def _compute_probabilities_for_unitary(
        self,
        unitary: torch.Tensor,
        *,
        amplitude_encoding: bool,
    ) -> torch.Tensor | SectoredDistribution:
        """Compute output probabilities for one precomputed unitary.

        If ``self.input_state`` is a tensor superposition and no source noise is
        active, this computes coherent output amplitudes for the sampled unitary,
        converts those amplitudes to probabilities independently, and returns
        those probabilities. Phase-error Monte Carlo averaging therefore uses
        ``mean_k |SLOS(U_k) @ psi|^2`` rather than
        ``|mean_k SLOS(U_k) @ psi|^2``.

        Parameters
        ----------
        unitary : torch.Tensor
            Precomputed circuit unitary used for the current probability sample.
        amplitude_encoding : bool
            Whether tensor ``input_state`` values should be interpreted as a
            coherent superposition for this sample.

        Returns
        -------
        torch.Tensor | SectoredDistribution
            Output probabilities for the supplied unitary. g2 source noise may
            return a ``SectoredDistribution``.

        Raises
        ------
        ValueError
            If coherent superposition handling is requested and ``input_state``
            does not match the configured computation-space basis size.
        """
        if self._has_source_noise():
            return self._compute_source_probabilities_for_unitary(
                unitary, amplitude_encoding=amplitude_encoding
            )

        if isinstance(self.input_state, torch.Tensor) and amplitude_encoding:
            amplitudes = self._compute_superposition_amplitudes_for_unitary(unitary)
            probabilities = probabilities_from_amplitudes(amplitudes)
            return normalize_probabilities(probabilities, self.computation_space)

        input_state = self._fixed_input_state_for_compute()
        keys, probs = self.simulation_graph.compute_probs(unitary, input_state)
        self._validate_probability_keys(keys)
        return probs

    def _validate_probability_keys(self, keys: Any) -> None:
        """Validate probability tensor keys against the simulation graph order.

        Parameters
        ----------
        keys : Any
            Keys returned with a probability tensor by the simulation graph.

        Raises
        ------
        ValueError
            If the returned keys do not match
            ``self.simulation_graph.mapped_keys``.
        """
        if keys != self.simulation_graph.mapped_keys:
            raise ValueError(
                "Probability keys returned by the simulation graph do not match "
                "the mapped output-key order."
            )

    @staticmethod
    def _validate_sector_keys(sector: SectorResult) -> tuple[tuple[int, ...], ...]:
        """Return sector keys after validating they align with the tensor basis.

        Parameters
        ----------
        sector : SectorResult
            Sector whose key metadata should be validated.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            Basis keys associated with the final tensor dimension.

        Raises
        ------
        ValueError
            If keys are missing, duplicated, or inconsistent with the sector
            tensor basis dimension.
        """
        if sector.keys is None:
            raise ValueError("Sector basis keys are required to align probabilities.")
        if sector.tensor.ndim == 0:
            raise ValueError("Sector tensor must have a basis dimension.")
        if sector.tensor.shape[-1] != len(sector.keys):
            raise ValueError(
                "Sector tensor basis dimension does not match the number of keys."
            )
        if len(set(sector.keys)) != len(sector.keys):
            raise ValueError("Sector basis keys must be unique.")
        return sector.keys

    @classmethod
    def _align_sector_tensor(
        cls,
        source_sector: SectorResult,
        target_sector: SectorResult,
    ) -> torch.Tensor:
        """Return a source sector tensor reordered to target sector keys.

        Parameters
        ----------
        source_sector : SectorResult
            Sector containing the tensor to reorder.
        target_sector : SectorResult
            Sector whose basis-key order defines the desired output order.

        Returns
        -------
        torch.Tensor
            ``source_sector.tensor`` indexed along its final dimension so it
            matches ``target_sector.keys``.

        Raises
        ------
        ValueError
            If sector metadata differs, batch dimensions differ, keys are
            invalid, or the two sectors do not contain the same basis keys.
        """
        if source_sector.n_photons != target_sector.n_photons:
            raise ValueError("Cannot align sectors with different photon numbers.")
        if source_sector.n_modes != target_sector.n_modes:
            raise ValueError("Cannot align sectors with different mode counts.")
        if source_sector.computation_space != target_sector.computation_space:
            raise ValueError("Cannot align sectors with different computation spaces.")

        source_keys = cls._validate_sector_keys(source_sector)
        target_keys = cls._validate_sector_keys(target_sector)
        if source_sector.tensor.shape[:-1] != target_sector.tensor.shape[:-1]:
            raise ValueError(
                "Sector tensor batch dimensions differ while aligning probabilities."
            )

        if source_keys == target_keys:
            return source_sector.tensor

        source_index_by_key = {key: idx for idx, key in enumerate(source_keys)}
        if set(source_index_by_key) != set(target_keys):
            raise ValueError("Sector basis keys mismatch while aligning probabilities.")

        indices = torch.tensor(
            [source_index_by_key[key] for key in target_keys],
            dtype=torch.long,
            device=source_sector.tensor.device,
        )
        return source_sector.tensor.index_select(source_sector.tensor.ndim - 1, indices)

    def _add_sectored_distributions(
        self,
        left: SectoredDistribution,
        right: SectoredDistribution,
    ) -> SectoredDistribution:
        """Add two sectored distributions by matching photon-number sectors.

        Parameters
        ----------
        left : SectoredDistribution
            Accumulated distribution.
        right : SectoredDistribution
            New distribution to add.

        Returns
        -------
        SectoredDistribution
            Distribution whose sector tensors are the pairwise sums.

        Raises
        ------
        ValueError
            If the two distributions do not contain the same photon-number
            sectors or if matching sectors cannot be aligned by basis keys.
        """
        left_photons = {sector.n_photons for sector in left.sectors}
        right_photons = {sector.n_photons for sector in right.sectors}
        if left_photons != right_photons:
            raise ValueError(
                "Phase-error sector mismatch while averaging SectoredDistribution samples."
            )

        sectors: list[SectorResult] = []
        for left_sector in left.sectors:
            right_sector = right.get_sector(left_sector.n_photons)
            right_tensor = self._align_sector_tensor(right_sector, left_sector)
            sectors.append(
                SectorResult(
                    left_sector.tensor + right_tensor,
                    n_modes=left_sector.n_modes,
                    n_photons=left_sector.n_photons,
                    computation_space=left_sector.computation_space,
                    keys=left_sector.keys,
                )
            )

        return SectoredDistribution(tuple(sectors))

    def _divide_sectored_distribution(
        self,
        distribution: SectoredDistribution,
        divisor: int,
    ) -> SectoredDistribution:
        """Divide every sector tensor by a scalar divisor.

        Parameters
        ----------
        distribution : SectoredDistribution
            Sectored distribution to scale.
        divisor : int
            Positive divisor applied to each sector tensor.

        Returns
        -------
        SectoredDistribution
            Distribution with each sector tensor divided by ``divisor`` while
            preserving sector metadata and basis keys.
        """
        return SectoredDistribution(
            tuple(
                SectorResult(
                    sector.tensor / divisor,
                    n_modes=sector.n_modes,
                    n_photons=sector.n_photons,
                    computation_space=sector.computation_space,
                    keys=sector.keys,
                )
                for sector in distribution.sectors
            )
        )

    def _compute_phase_error_probabilities(
        self,
        parameters: list[torch.Tensor],
        *,
        amplitude_encoding: bool,
        memristive_current_state: list[torch.Tensor] | None = None,
    ) -> torch.Tensor | SectoredDistribution:
        """Average probabilities over stochastic phase-error unitary samples.

        The averaging order is:

        ``mean_k probabilities(U(phi_quantized + epsilon_k), input_state)``

        where ``epsilon_k`` is drawn independently from
        ``Uniform(-phase_error, phase_error)`` for each sampled phase shifter.
        If ``phase_imprecision`` is also active, ``phi_quantized`` is the
        nearest-grid value produced by the converter before perturbation. This
        method never averages amplitudes. For tensor superposition inputs, each
        sampled unitary first produces coherent output amplitudes; those
        amplitudes are converted to probabilities for that same sample, and
        only then are probability tensors averaged.

        Parameters
        ----------
        parameters : list[torch.Tensor]
            Circuit parameters passed to the converter.
        amplitude_encoding : bool
            Whether tensor input states should be interpreted as coherent
            amplitude-encoded superpositions for each sampled unitary.
        memristive_current_state : list[torch.Tensor] | None
            The memristive phase shifters current states. Defaults to None
            and will be treated as an empty list.

        Returns
        -------
        torch.Tensor | SectoredDistribution
            Probability distribution averaged over ``n_phase_error_samples``.
            When g2 source noise is active, each sample returns a
            ``SectoredDistribution`` with one tensor per photon-number sector,
            for example sectors with ``n_photons == 0`` and ``n_photons == 1``.
            The same photon-number sectors must be present in every sample;
            matching sector tensors are accumulated and divided by the sample
            count. A missing or additional sector in any sample is a layout
            mismatch and raises ``ValueError`` instead of silently dropping
            probabilities.

        Raises
        ------
        ValueError
            If phase-error samples do not all return tensors or do not all
            return matching ``SectoredDistribution`` layouts.
        RuntimeError
            If no phase-error samples were computed.
        """
        accumulated: torch.Tensor | SectoredDistribution | None = None

        for _sample_index in range(self._n_phase_error_samples):
            unitary = self.converter.to_tensor(
                *parameters,
                apply_phase_error=True,
                memristive_current_state=(
                    [] if memristive_current_state is None else memristive_current_state
                ),
            )
            self.unitary = unitary
            probabilities = self._compute_probabilities_for_unitary(
                unitary, amplitude_encoding=amplitude_encoding
            )

            if accumulated is None:
                accumulated = probabilities
                continue

            if isinstance(accumulated, SectoredDistribution):
                if not isinstance(probabilities, SectoredDistribution):
                    raise ValueError(
                        "Phase-error sample type mismatch while averaging probabilities."
                    )
                accumulated = self._add_sectored_distributions(
                    accumulated, probabilities
                )
            else:
                if isinstance(probabilities, SectoredDistribution):
                    raise ValueError(
                        "Phase-error sample type mismatch while averaging probabilities."
                    )
                accumulated = accumulated + probabilities

        if accumulated is None:
            raise RuntimeError("No phase-error samples were computed.")

        if isinstance(accumulated, SectoredDistribution):
            return self._divide_sectored_distribution(
                accumulated, self._n_phase_error_samples
            )
        return accumulated / self._n_phase_error_samples

    def compute(
        self,
        parameters: list[torch.Tensor],
        amplitude_encoding: bool = False,
        memristive_current_state: list[torch.Tensor] | None = None,
    ) -> torch.Tensor | SectoredDistribution:
        """Compute output amplitudes or probabilities for the configured input state.

        Parameters
        ----------
        parameters : list[torch.Tensor]
            Circuit parameters passed to the converter.
        amplitude_encoding : bool
            If True and ``input_state`` is a tensor, use tensor-superposition
            handling. Source-noise simulations mix probabilities over Fock basis
            states weighted by :math:`|c_i|^2`; phase-error simulations without
            source noise compute coherent probabilities per sampled unitary.
            Default is False.
        memristive_current_state : list[torch.Tensor] | None
            The memristive phase shifters current states. Defaults to None
            and will be treated as an empty list.

        Returns
        -------
        torch.Tensor | SectoredDistribution
            Output probabilities if source noise or stochastic phase error is
            active (a :class:`~merlin.core.sectored_distribution.SectoredDistribution`
            when g2 noise is present) and amplitudes otherwise.
        """
        if self._has_phase_error():
            return self._compute_phase_error_probabilities(
                parameters,
                amplitude_encoding=amplitude_encoding,
                memristive_current_state=memristive_current_state,
            )

        unitary = self.converter.to_tensor(
            *parameters,
            memristive_current_state=(
                [] if memristive_current_state is None else memristive_current_state
            ),
        )
        self.unitary = unitary

        if self._has_source_noise():
            return self._compute_source_probabilities_for_unitary(
                unitary, amplitude_encoding=amplitude_encoding
            )

        input_state = self._fixed_input_state_for_compute()
        _keys, amplitudes = self.simulation_graph.compute(unitary, input_state)
        return amplitudes

    @overload
    def compute_superposition_state(
        self,
        parameters: list[torch.Tensor],
        *,
        simultaneous_processes: int | None = None,
        return_keys: Literal[True] = True,
        memristive_current_state: list[torch.Tensor] | None = None,
    ) -> tuple[list[tuple[int, ...]], torch.Tensor]: ...

    @overload
    def compute_superposition_state(
        self,
        parameters: list[torch.Tensor],
        *,
        simultaneous_processes: int | None = None,
        return_keys: Literal[False] = False,
        memristive_current_state: list[torch.Tensor] | None = None,
    ) -> torch.Tensor: ...

    def compute_superposition_state(
        self,
        parameters: list[torch.Tensor],
        *,
        simultaneous_processes: int | None = None,
        return_keys: bool = False,
        memristive_current_state: list[torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[list[tuple[int, ...]], torch.Tensor]:
        """Compute amplitudes for a tensor superposition input state.

        Parameters
        ----------
        parameters : list[torch.Tensor]
            Circuit parameters passed to the converter.
        simultaneous_processes : int | None
            Maximum number of active input components propagated in one
            chunk. If omitted, an internal default chunk size is used.
        return_keys : bool
            Whether to return the output basis keys with the amplitudes.
            Default value is False.
        memristive_current_state : list[torch.Tensor] | None
            The memristive phase shifters current states. Defaults to None
            and will be treated as an empty list.

        Returns
        -------
        torch.Tensor | tuple[list[tuple[int, ...]], torch.Tensor]
            Output amplitudes, optionally paired with their basis keys.

        Raises
        ------
        RuntimeError
            If phase error or source noise is active, because those paths return
            probabilities rather than amplitudes.
        """
        if self._has_phase_error():
            raise RuntimeError(
                "Active phase_error returns probabilities; compute_superposition_state cannot return amplitudes."
            )
        if self.noisy_simulation:
            raise RuntimeError(
                "Noisy simulations with source noise can only call the `compute` and `compute_with_keys` methods to compute probabilities"
            )
        prepared_state = self._prepare_superposition_support()
        memristive_current_state=(
                [] if memristive_current_state is None else memristive_current_state
            )
        if prepared_state.batch_size>1 and (not (memristive_current_state == [])):
            unitaries=[self.converter.to_tensor(
                *parameters,
                memristive_current_state=[torch.tensor(memristor[index]) for memristor in memristive_current_state],
            ) for index in range(prepared_state.batch_size)]
            #TODO COMPLETE
            _keys_out, final_amplitudes = self._compute_chunked_superposition(
                prepared_state,
                unitary if unitary.dim() == 3 else unitary.unsqueeze(0),
                simultaneous_processes=simultaneous_processes,
            )
        else:
            unitary = self.converter.to_tensor(
                *parameters,
                memristive_current_state=memristive_current_state,
            )
            _keys_out, final_amplitudes = self._compute_chunked_superposition(
                prepared_state,
                unitary if unitary.dim() == 3 else unitary.unsqueeze(0),
                simultaneous_processes=simultaneous_processes,
            )

        if final_amplitudes.shape[0] == 1:
            final_amplitudes = final_amplitudes.squeeze(0)

        if return_keys:
            return _keys_out, final_amplitudes

        return final_amplitudes

    def compute_ebs_simultaneously(
        self,
        parameters: list[torch.Tensor],
        simultaneous_processes: int = 1,
        memristive_current_state: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Evaluate a single circuit parametrisation against all superposed input
        states by chunking them in groups and delegating the heavy work to the
        TorchScript-enabled batch kernel.

        The method converts the trainable parameters into a unitary matrix,
        normalises the input state (if it is not already normalised), filters
        out components with zero amplitude, and then queries the simulation
        graph for batches of Fock states. Each batch feeds
        :meth:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph.compute_batch`, producing a tensor that contains
        the amplitudes of all reachable output states for the selected input
        components. The partial results are accumulated into a preallocated
        tensor and finally weighted by the complex coefficients of
        ``self.input_state`` to produce the global output amplitudes.

        Parameters
        ----------
        parameters : list[torch.Tensor]
            Differentiable parameters that encode the photonic circuit.
        simultaneous_processes : int
            Maximum number of non-zero input components propagated in a single
            call to ``compute_batch``.
        memristive_current_state : list[torch.Tensor] | None
            The memristive phase shifters current states. Defaults to None
            and will be treated as an empty list.

        Returns
        -------
        torch.Tensor
            Superposed output amplitudes with shape
            ``[batch_size, num_output_states]``.

        Raises
        ------
        TypeError
            If ``self.input_state`` is not a ``torch.Tensor``.

        Notes
        -----
            - ``self.input_state`` is normalized in place to avoid an extra
              allocation. Parameters are forwarded to ``self.converter`` to
              build the unitary matrix used during the simulation.
            - Zero-amplitude components are skipped to minimise the number of
              calls to ``compute_batch``.
            - The method is agnostic to the device: tensors remain on the device
              they already occupy, so callers should ensure ``parameters`` and
              ``self.input_state`` live on the same device.
        """
        if self._has_phase_error():
            raise RuntimeError(
                "Active phase_error returns probabilities; compute_ebs_simultaneously cannot return amplitudes."
            )
        if self.noisy_simulation:
            raise RuntimeError(
                "Noisy simulations with source noise can only call the `compute` and `compute_with_keys` methods to compute probabilities"
            )

        prepared_state = self._prepare_superposition_support()
        unitary = self.converter.to_tensor(
            *parameters,
            memristive_current_state=(
                [] if memristive_current_state is None else memristive_current_state
            ),
        )
        _keys_out, final_amplitudes = self._compute_chunked_superposition(
            prepared_state,
            unitary if unitary.dim() == 3 else unitary.unsqueeze(0),
            simultaneous_processes=simultaneous_processes,
        )

        if final_amplitudes.shape[0] == 1:
            final_amplitudes = final_amplitudes.squeeze(0)
        if final_amplitudes.ndim == 3 and final_amplitudes.shape[1] == 1:
            final_amplitudes = final_amplitudes.squeeze(1)

        return final_amplitudes

    def compute_with_keys(
        self,
        parameters: list[torch.Tensor],
        *,
        use_input_state_superposition: bool = False,
        memristive_current_state: list[torch.Tensor] | None = None,
    ) -> tuple[Any, torch.Tensor | SectoredDistribution]:
        """Compute output values and return them with basis keys.

        Parameters
        ----------
        parameters : list[torch.Tensor]
            Circuit parameters passed to the converter.
        use_input_state_superposition : bool
            If True and ``input_state`` is a tensor, use tensor-superposition
            handling when phase error is active. If omitted, tensor
            ``input_state`` follows the same default fixed-Fock-state behavior
            as :meth:`compute`.
            Default is False.
        memristive_current_state : list[torch.Tensor] | None
            The memristive phase shifters current states. Defaults to None
            and will be treated as an empty list.

        Returns
        -------
        tuple[Any, torch.Tensor | SectoredDistribution]
            Simulation-graph keys and corresponding probabilities if source
            noise or phase error is active, and amplitudes otherwise.
        """
        if self._has_phase_error():
            probabilities = self._compute_phase_error_probabilities(
                parameters,
                amplitude_encoding=use_input_state_superposition,
                memristive_current_state=memristive_current_state,
            )
            if isinstance(probabilities, SectoredDistribution):
                keys = [list(sector.keys) for sector in probabilities.sectors]
                return keys, probabilities
            return self.simulation_graph.mapped_keys, probabilities

        unitary = self.converter.to_tensor(
            *parameters,
            memristive_current_state=(
                [] if memristive_current_state is None else memristive_current_state
            ),
        )

        if self._has_source_noise():
            result = self.simulation_graph.compute_probs(unitary, self.input_state)
            if isinstance(result, SectoredDistribution):
                keys = [list(sector.keys) for sector in result.sectors]
                return keys, result
            keys, probs = result
            return keys, probs

        input_state = self._fixed_input_state_for_compute()
        keys, amplitudes = self.simulation_graph.compute(unitary, input_state)
        return keys, amplitudes

    def _expected_superposition_size(self) -> int:
        """Return the expected superposition basis size.

        Returns
        -------
        int
            Number of input basis states expected for tensor superposition data
            in the configured computation space.

        Raises
        ------
        ValueError
            If the photon count or mode count is incompatible with the selected
            computation space.
        """
        if self.n_photons < 0:
            raise ValueError("Number of photons must be non-negative.")
        if self.computation_space is ComputationSpace.DUAL_RAIL:
            if self.n_photons is None:
                raise ValueError("Dual-rail encoding requires 'n_photons'.")
            if self.m != 2 * self.n_photons:
                raise ValueError(
                    "Dual-rail encoding requires the number of modes to equal 2 * n_photons."
                )
            # Dual-rail limits to 2**n logical states (one photon per rail pair).
            return 2**self.n_photons
        if self.computation_space is ComputationSpace.UNBUNCHED:
            if self.n_photons > self.m:
                raise ValueError(
                    "Invalid configuration: ComputationSpace.UNBUNCHED requires "
                    "n_photons to be less than or equal to the number of modes."
                )
            return math.comb(self.m, self.n_photons)
        return math.comb(self.m + self.n_photons - 1, self.n_photons)

    def _full_fock_superposition_size(self) -> int:
        """Return the full Fock input basis size used by the SLOS engine.

        Returns
        -------
        int
            Number of full Fock input basis states for ``m`` modes and
            ``n_photons`` photons.
        """
        return math.comb(self.m + self.n_photons - 1, self.n_photons)

    def _validate_superposition_state_shape(self, input_state: torch.Tensor) -> None:
        """Validate a tensor superposition input shape.

        Parameters
        ----------
        input_state : torch.Tensor
            Tensor whose final dimension should match the expected
            superposition basis size.

        Raises
        ------
        TypeError
            If ``input_state`` is not a tensor.
        ValueError
            If ``input_state`` is not 1D or 2D, or if its final dimension does
            not match the configured computation-space basis size.
        """
        if not isinstance(input_state, torch.Tensor):
            raise TypeError("Input state should be a tensor")

        if input_state.dim() == 1:
            state_dim = input_state.shape[0]
        elif input_state.dim() == 2:
            state_dim = input_state.shape[1]
        else:
            raise ValueError(
                f"Superposed input state must be 1D or 2D tensor, got shape {tuple(input_state.shape)}"
            )

        logical_expected = self._expected_superposition_size()
        full_fock_expected = self._full_fock_superposition_size()
        if state_dim not in {logical_expected, full_fock_expected}:
            if self.computation_space is ComputationSpace.DUAL_RAIL:
                explanation = (
                    f"expected 2**n_photons = 2**{self.n_photons} = {logical_expected}"
                )
            elif self.computation_space is ComputationSpace.UNBUNCHED:
                explanation = (
                    f"expected C(m, n_photons) = C({self.m}, "
                    f"{self.n_photons}) = {logical_expected}"
                )
            else:
                explanation = (
                    f"expected C(m + n_photons - 1, n_photons) = "
                    f"C({self.m + self.n_photons - 1}, {self.n_photons}) = "
                    f"{logical_expected}"
                )
            if full_fock_expected != logical_expected:
                explanation = (
                    f"{explanation}, or full Fock size "
                    f"C({self.m + self.n_photons - 1}, {self.n_photons}) = "
                    f"{full_fock_expected}"
                )
            raise ValueError(
                "Input state dimension mismatch for computation_space "
                f"'{self.computation_space}': got {state_dim}, {explanation}."
            )

    def _should_defer_state_validation(self, tensor: torch.Tensor) -> bool:
        """Detect tensors that need delayed dual-rail validation.

        Parameters
        ----------
        tensor : torch.Tensor
            Candidate superposition tensor.

        Returns
        -------
        bool
            True when an UNBUNCHED tensor has the dual-rail logical dimension
            and should be validated after the computation space is configured.
        """
        if tensor.dim() == 1:
            state_dim = tensor.shape[0]
        elif tensor.dim() == 2:
            state_dim = tensor.shape[1]
        else:
            return False

        if self.n_photons is None or self.m is None:
            return False

        return (
            self.computation_space is ComputationSpace.UNBUNCHED
            and self.m == 2 * self.n_photons
            and state_dim == 2**self.n_photons
        )

    def _coerce_superposition_tensor_shape(
        self, tensor: torch.Tensor
    ) -> torch.Tensor | None:
        """Lift compatible compact-basis tensors into the full Fock basis.

        Parameters
        ----------
        tensor : torch.Tensor
            Candidate tensor encoded either in the full Fock basis or in a
            compact logical basis compatible with ``computation_space``.

        Returns
        -------
        torch.Tensor | None
            Tensor expanded into the configured Fock basis when the reduced
            basis can be mapped unambiguously. Returns None when no coercion
            applies.
        """
        if self.n_photons is None or self.m is None:
            return None

        if tensor.dim() == 1:
            feature_dim = tensor.shape[0]
        elif tensor.dim() == 2:
            feature_dim = tensor.shape[1]
        else:
            return None

        full_fock_size = self._full_fock_superposition_size()
        if feature_dim == full_fock_size:
            return None

        if self.computation_space is ComputationSpace.DUAL_RAIL:
            compact_schemes = ["dual_rail"]
        elif self.computation_space is ComputationSpace.UNBUNCHED:
            compact_schemes = ["unbunched"]
        else:
            compact_schemes = ["unbunched"]

        fock_combinator = Combinadics("fock", self.n_photons, self.m)
        for scheme in compact_schemes:
            try:
                compact_combinator = Combinadics(scheme, self.n_photons, self.m)
            except ValueError:
                continue
            if feature_dim != compact_combinator.compute_space_size():
                continue
            fock_indices = [
                fock_combinator.fock_to_index(state)
                for state in compact_combinator.iter_states()
            ]
            return self._expand_compact_superposition_tensor(
                tensor, fock_indices, full_fock_size
            )

        return None

    @staticmethod
    def _expand_compact_superposition_tensor(
        tensor: torch.Tensor,
        fock_indices: list[int],
        target_dim: int,
    ) -> torch.Tensor:
        """Expand compact-basis amplitudes into full Fock-basis amplitudes.

        Parameters
        ----------
        tensor : torch.Tensor
            One-dimensional or batched compact amplitude tensor.
        fock_indices : list[int]
            Full Fock-basis indices matching the compact tensor ordering.
        target_dim : int
            Full Fock-basis dimension.

        Returns
        -------
        torch.Tensor
            Tensor with the same leading shape and ``target_dim`` as its final
            dimension.
        """
        index_tensor = torch.tensor(
            fock_indices,
            dtype=torch.long,
            device=tensor.device,
        )

        if tensor.is_sparse:
            coalesced = tensor.coalesce()
            indices = coalesced.indices().clone()
            indices[-1] = index_tensor[indices[-1]]
            shape = (
                (target_dim,) if tensor.dim() == 1 else (tensor.shape[0], target_dim)
            )
            return torch.sparse_coo_tensor(
                indices,
                coalesced.values(),
                shape,
                dtype=coalesced.dtype,
                device=coalesced.device,
            ).coalesce()

        if tensor.dim() == 1:
            expanded = tensor.new_zeros(target_dim)
            expanded[index_tensor] = tensor
            return expanded

        expanded = tensor.new_zeros(tensor.shape[0], target_dim)
        expanded[:, index_tensor] = tensor
        return expanded

    def _prepare_superposition_tensor(self) -> torch.Tensor:
        """Validate, normalize, and store tensor superposition input.

        Returns
        -------
        torch.Tensor
            Normalized 2D complex tensor whose final dimension matches the
            configured input basis.

        Raises
        ------
        TypeError
            If ``input_state`` is not a tensor or has an unsupported dtype.
        ValueError
            If ``input_state`` does not match the configured computation-space
            basis size.
        """
        if not isinstance(self.input_state, torch.Tensor):
            raise TypeError("Input state should be a tensor")

        tensor = self.input_state

        coerced = self._coerce_superposition_tensor_shape(tensor)
        if coerced is not None:
            tensor = coerced

        self._validate_superposition_state_shape(tensor)

        tensor = self._unsqueeze_superposition_tensor(tensor)

        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.complex64)
        elif tensor.dtype == torch.float64:
            tensor = tensor.to(torch.complex128)
        elif tensor.dtype not in (torch.complex64, torch.complex128):
            raise TypeError(
                f"Unsupported dtype for superposition state: {tensor.dtype}"
            )

        tensor = self._normalize_superposition_tensor(tensor)
        self.input_state = tensor
        return tensor

    def _prepare_superposition_support(self) -> _SuperpositionSupport:
        """Return active support for the stored superposition state."""
        tensor = self._prepare_superposition_tensor()
        return self._superposition_support_from_tensor(tensor)

    def _compute_chunked_superposition(
        self,
        prepared_state: _SuperpositionSupport,
        unitary: torch.Tensor,
        *,
        simultaneous_processes: int | None,
    ) -> tuple[list[tuple[int, ...]], torch.Tensor]:
        """Evaluate a superposition by streaming chunked kernel calls into the final tensor."""
        if unitary.dim() != 3:
            raise ValueError(
                "Expected batched unitary tensor for chunked superposition evaluation."
            )

        keys_out = list(self.simulation_graph.mapped_keys)
        if not prepared_state.basis_indices:
            final = torch.zeros(
                (
                    unitary.shape[0],
                    prepared_state.batch_size,
                    len(keys_out),
                ),
                dtype=unitary.dtype,
                device=unitary.device,
            )
            return keys_out, final

        fock_basis_states = self._input_basis_states()
        input_states = [
            (index, fock_basis_states[index]) for index in prepared_state.basis_indices
        ]
        chunk_size = self._resolve_superposition_chunk_size(simultaneous_processes)
        final_amplitudes = torch.zeros(
            (
                unitary.shape[0],
                prepared_state.batch_size,
                len(keys_out),
            ),
            dtype=unitary.dtype,
            device=unitary.device,
        )

        for start in range(0, len(input_states), chunk_size):
            batch = input_states[start : start + chunk_size]
            batch_fock_states = [state for _, state in batch]
            coeffs = prepared_state.coefficients[:, start : start + len(batch)].to(
                device=unitary.device,
                dtype=unitary.dtype,
            )
            _, batch_amplitudes = self.simulation_graph.compute_batch(
                unitary, batch_fock_states
            )
            batch_amplitudes = batch_amplitudes / batch_amplitudes.norm(
                p=2, dim=1, keepdim=True
            ).clamp_min(1e-12)
            final_amplitudes += torch.einsum("se,boe->bso", coeffs, batch_amplitudes)

        return keys_out, final_amplitudes

    @staticmethod
    def _resolve_superposition_chunk_size(simultaneous_processes: int | None) -> int:
        if simultaneous_processes is None:
            return _DEFAULT_SUPERPOSITION_CHUNK_SIZE
        if simultaneous_processes <= 0:
            raise ValueError("simultaneous_processes must be a positive integer.")
        return simultaneous_processes

    @staticmethod
    def _unsqueeze_superposition_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Add a batch dimension while preserving sparse COO storage."""
        if tensor.dim() != 1:
            return tensor
        if not tensor.is_sparse:
            return tensor.unsqueeze(0)

        coalesced = tensor.coalesce()
        nnz = coalesced.values().shape[0]
        batch_indices = torch.zeros((1, nnz), dtype=torch.long, device=coalesced.device)
        indices = torch.cat((batch_indices, coalesced.indices()), dim=0)
        return torch.sparse_coo_tensor(
            indices,
            coalesced.values(),
            (1, tensor.shape[0]),
            dtype=coalesced.dtype,
            device=coalesced.device,
        ).coalesce()

    @staticmethod
    def _normalize_superposition_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize batched superposition tensors without forcing densification."""
        if not tensor.is_sparse:
            norm = tensor.abs().pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
            return tensor / norm

        coalesced = tensor.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        row_indices = indices[0]
        magnitude_sq = values.real.pow(2) + values.imag.pow(2)
        norm_sq = torch.zeros(
            tensor.shape[0], dtype=magnitude_sq.dtype, device=magnitude_sq.device
        )
        norm_sq.scatter_add_(0, row_indices, magnitude_sq)
        norms = norm_sq.sqrt().clamp_min(1e-12)
        scaled_values = values / norms[row_indices]
        return torch.sparse_coo_tensor(
            indices,
            scaled_values,
            coalesced.shape,
            dtype=coalesced.dtype,
            device=coalesced.device,
        ).coalesce()

    @staticmethod
    def _superposition_support_from_tensor(
        tensor: torch.Tensor,
    ) -> _SuperpositionSupport:
        """Extract active basis indices and compact coefficients."""
        if tensor.is_sparse:
            return ComputationProcess._sparse_superposition_support(tensor)
        return ComputationProcess._dense_superposition_support(tensor)

    @staticmethod
    def _dense_superposition_support(tensor: torch.Tensor) -> _SuperpositionSupport:
        """Extract active support from a dense batched superposition tensor."""
        magnitude_sq = tensor.real.pow(2) + tensor.imag.pow(2)
        active_mask = (magnitude_sq >= 1e-13).any(dim=0)
        active_indices_tensor = active_mask.nonzero(as_tuple=False).flatten()
        if active_indices_tensor.numel() == 0:
            coefficients = tensor.new_zeros((tensor.shape[0], 0))
            basis_indices: list[int] = []
        else:
            coefficients = tensor[:, active_indices_tensor]
            basis_indices = [int(idx) for idx in active_indices_tensor.tolist()]
        return _SuperpositionSupport(
            basis_indices=basis_indices,
            coefficients=coefficients,
            basis_size=tensor.shape[-1],
        )

    @staticmethod
    def _sparse_superposition_support(tensor: torch.Tensor) -> _SuperpositionSupport:
        """Extract active support from a sparse COO batched superposition tensor."""
        coalesced = tensor.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        if values.numel() == 0:
            coefficients = values.new_zeros((tensor.shape[0], 0))
            return _SuperpositionSupport(
                basis_indices=[],
                coefficients=coefficients,
                basis_size=tensor.shape[-1],
            )

        magnitude_sq = values.real.pow(2) + values.imag.pow(2)
        active_values_mask = magnitude_sq >= 1e-13
        if not bool(active_values_mask.any().item()):
            coefficients = values.new_zeros((tensor.shape[0], 0))
            return _SuperpositionSupport(
                basis_indices=[],
                coefficients=coefficients,
                basis_size=tensor.shape[-1],
            )

        active_indices = indices[:, active_values_mask]
        active_values = values[active_values_mask]
        rows = active_indices[0]
        cols = active_indices[-1]
        basis_indices_tensor = torch.unique(cols, sorted=True)
        positions = torch.searchsorted(basis_indices_tensor, cols)
        coefficients = values.new_zeros((tensor.shape[0], basis_indices_tensor.numel()))
        coefficients[rows, positions] = active_values
        basis_indices = [int(idx) for idx in basis_indices_tensor.tolist()]
        return _SuperpositionSupport(
            basis_indices=basis_indices,
            coefficients=coefficients,
            basis_size=tensor.shape[-1],
        )


class ComputationProcessFactory:
    """Factory for creating computation processes."""

    @staticmethod
    def create(
        circuit: pcvl.Circuit,
        input_state: list[int] | torch.Tensor,
        trainable_parameters: list[str],
        input_parameters: list[str],
        computation_space: ComputationSpace | None = None,
        noise_groups: NoiseGroups | None = None,
        n_phase_error_samples: int = 1,
        memristive_metadata: list[dict] | None = None,
        **kwargs,
    ) -> ComputationProcess:
        """Create a computation process.

        Parameters
        ----------
        circuit : pcvl.Circuit
            Circuit used to build the process.
        input_state : list[int] | torch.Tensor
            Input Fock state or superposition tensor.
        trainable_parameters : list[str]
            Prefixes of trainable circuit parameters.
        input_parameters : list[str]
            Prefixes of input-driven circuit parameters.
        computation_space : ComputationSpace | None
            Computation space used for basis enumeration.
        noise_groups : NoiseGroups | None
            The noise groups applied to the circuit to be ran.
        n_phase_error_samples : int
            Number of Monte Carlo unitary samples used when active
            ``phase_error`` is present. Default value is 1.
        memristive_metadata: list[dict] | None
            The memristive phase shifter metadata. If None, it will be stored as an empty list.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`ComputationProcess`.

        Returns
        -------
        ComputationProcess
            Created computation process.
        """
        return ComputationProcess(
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=trainable_parameters,
            input_parameters=input_parameters,
            computation_space=computation_space,
            noise_groups=noise_groups,
            n_phase_error_samples=n_phase_error_samples,
            memristive_metadata=(
                [] if memristive_metadata is None else memristive_metadata
            ),
            **kwargs,
        )
