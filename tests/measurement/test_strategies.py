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

from dataclasses import FrozenInstanceError

import perceval as pcvl
import pytest
import torch

from merlin.algorithms.layer import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.core.computation_space import ComputationSpace
from merlin.core.partial_measurement import PartialMeasurement
from merlin.core.state_vector import StateVector
from merlin.measurement.strategies import MeasurementKind, MeasurementStrategy
from merlin.utils.combinadics import Combinadics
from merlin.utils.grouping import LexGrouping, ModGrouping


class TestMeasurementStrategy:
    def test_factory_probs_creates_correct_instance(self):
        strategy = MeasurementStrategy.probs(ComputationSpace.FOCK)
        assert strategy.type == MeasurementKind.PROBABILITIES
        assert strategy.computation_space == ComputationSpace.FOCK

    def test_factory_mode_expectations_creates_correct_instance(self):
        strategy = MeasurementStrategy.mode_expectations(ComputationSpace.DUAL_RAIL)
        assert strategy.type == MeasurementKind.MODE_EXPECTATIONS
        assert strategy.computation_space == ComputationSpace.DUAL_RAIL

    def test_factory_amplitudes_creates_correct_instance(self):
        strategy = MeasurementStrategy.amplitudes()
        assert strategy.type == MeasurementKind.AMPLITUDES
        assert strategy.computation_space == ComputationSpace.UNBUNCHED

    def test_factory_equality(self):
        s1 = MeasurementStrategy.probs(ComputationSpace.FOCK)
        s2 = MeasurementStrategy.probs(ComputationSpace.FOCK)
        assert s1 == s2

    def test_partial_factory_creation_noncontiguous_modes(self):
        """Ensure factory wires fields without reordering sparse modes."""
        strategy = MeasurementStrategy.partial(
            modes=[0, 2, 5],
            computation_space=ComputationSpace.DUAL_RAIL,
            grouping=None,
        )
        assert strategy.type == MeasurementKind.PARTIAL
        assert strategy.measured_modes == (0, 2, 5)
        assert strategy.computation_space == ComputationSpace.DUAL_RAIL
        assert strategy.grouping is None

    def test_partial_empty_modes(self):
        with pytest.raises(ValueError, match="modes cannot be empty"):
            MeasurementStrategy.partial(
                modes=[],
                computation_space=ComputationSpace.FOCK,
            )
        with pytest.raises(ValueError, match="modes cannot be empty"):
            MeasurementStrategy.partial(
                modes=[],
                computation_space=ComputationSpace.UNBUNCHED,
            )
        with pytest.raises(ValueError, match="modes cannot be empty"):
            MeasurementStrategy.partial(
                modes=[],
                computation_space=ComputationSpace.DUAL_RAIL,
            )

    def test_partial_duplicate_modes(self):
        with pytest.raises(ValueError, match="Duplicate mode indices"):
            MeasurementStrategy.partial(
                modes=[0, 1, 1],
                computation_space=ComputationSpace.FOCK,
            )
        with pytest.raises(ValueError, match="Duplicate mode indices"):
            MeasurementStrategy.partial(
                modes=[2, 2, 3],
                computation_space=ComputationSpace.UNBUNCHED,
            )
        with pytest.raises(ValueError, match="Duplicate mode indices"):
            MeasurementStrategy.partial(
                modes=[0, 0],
                computation_space=ComputationSpace.DUAL_RAIL,
            )

    def test_partial_negative_mode_index(self):
        with pytest.raises(ValueError, match="Negative mode index"):
            MeasurementStrategy.partial(
                modes=[0, -1, 2],
                computation_space=ComputationSpace.FOCK,
            )
        with pytest.raises(ValueError, match="Negative mode index"):
            MeasurementStrategy.partial(
                modes=[2, 1, -3],
                computation_space=ComputationSpace.UNBUNCHED,
            )
        with pytest.raises(ValueError, match="Negative mode index"):
            MeasurementStrategy.partial(
                modes=[-1],
                computation_space=ComputationSpace.DUAL_RAIL,
            )

    def test_validate_modes_out_of_bounds(self):
        strategy = MeasurementStrategy.partial(
            modes=[0, 2],
            computation_space=ComputationSpace.FOCK,
        )
        with pytest.raises(ValueError, match="Invalid mode indices"):
            strategy.validate_modes(
                n_modes=2
            )  # mode index 2 is out of bounds for n_modes=2

        strategy = MeasurementStrategy.partial(
            modes=[1, 10],
            computation_space=ComputationSpace.UNBUNCHED,
        )
        with pytest.raises(ValueError, match="Invalid mode indices"):
            strategy.validate_modes(
                n_modes=3
            )  # mode index 10 is out of bounds for n_modes=3

        strategy = MeasurementStrategy.partial(
            modes=[0, 4],
            computation_space=ComputationSpace.DUAL_RAIL,
        )
        with pytest.raises(ValueError, match="Invalid mode indices"):
            strategy.validate_modes(
                n_modes=4
            )  # mode index 4 is out of bounds for n_modes=4

    def test_partial_returns_immutable_object(self):
        """The strategy dataclass is frozen to avoid mutation after construction."""
        strategy = MeasurementStrategy.partial(
            modes=[0, 2],
            computation_space=ComputationSpace.FOCK,
        )

        with pytest.raises(FrozenInstanceError):
            strategy.measured_modes = (1, 3)

    def test_get_unmeasured_modes(self):
        """Unmeasured modes should be the complement of measured modes."""
        strategy = MeasurementStrategy.partial(
            modes=[0, 2, 4],
            computation_space=ComputationSpace.FOCK,
        )

        assert strategy.get_unmeasured_modes(n_modes=6) == (1, 3, 5)

    def test_all_modes_measured_warning(self):
        """Both helpers surface the warning when the selection covers all modes."""
        strategy = MeasurementStrategy.partial(
            modes=[0, 1, 2],
            computation_space=ComputationSpace.FOCK,
        )

        with pytest.warns(
            UserWarning,
            match="All modes are measured",
        ):
            strategy.validate_modes(n_modes=3)

        with pytest.warns(
            UserWarning,
            match="All modes are measured",
        ):
            strategy.get_unmeasured_modes(n_modes=3)

    def test_partial_measurement_result_structure(self):
        """Test that PartialMeasurement result has correct structure."""
        empty_circuit = pcvl.Circuit(8)
        strategy = MeasurementStrategy.partial(
            modes=[0, 1, 2, 3],
        )
        state = [1, 0, 1, 0, 1, 0, 1, 0]

        layer = QuantumLayer(
            circuit=empty_circuit,
            input_size=0,
            input_state=state,
            measurement_strategy=strategy,
        )
        result = layer()

        assert type(result.outcomes) is list
        assert type(result.probabilities) is torch.Tensor
        assert torch.allclose(result.tensor, result.probabilities)
        assert type(result.amplitudes) is list
        assert all(type(amp) is StateVector for amp in result.amplitudes)
        assert all(amplitude.n_modes == 4 for amplitude in result.amplitudes)
        assert result.measured_modes == (0, 1, 2, 3)
        assert result.unmeasured_modes == (4, 5, 6, 7)

    def test_partial_measurement_output(self):
        """Test that using partial measurement returns a PartialMeasurement object."""
        empty_circuit = pcvl.Circuit(4)
        amplitudes = torch.randn(2, 10, dtype=torch.cfloat)
        state = StateVector(tensor=amplitudes, n_modes=4, n_photons=2)
        strategy = MeasurementStrategy.partial(
            modes=[0, 2],
            computation_space=ComputationSpace.FOCK,
        )

        quantum_layer = QuantumLayer(
            circuit=empty_circuit,
            input_size=0,
            input_state=state,
            measurement_strategy=strategy,
        )

        result = quantum_layer()
        assert isinstance(result, PartialMeasurement)
        assert len(result.branches) > 0
        assert all(
            type(branch.amplitudes) is StateVector and branch.amplitudes.n_modes == 2
            for branch in result.branches
        )
        assert result.measured_modes == (0, 2)
        assert result.unmeasured_modes == (1, 3)
        assert type(result.tensor) is torch.Tensor and result.tensor.shape[0] == 2
        assert all(type(amp) is StateVector for amp in result.amplitudes)
        assert all(len(outcome) == 2 for outcome in result.outcomes)
        assert all(sum(outcome) <= 2 for outcome in result.outcomes)

    def test_partial_measurement_grouping(self):
        """Test that grouping is only applied on probabilities when specified with partial measurement."""
        empty_circuit = pcvl.Circuit(4)
        strategy_g1 = MeasurementStrategy.partial(
            modes=[0, 1, 3],
            computation_space=ComputationSpace.FOCK,
            grouping=LexGrouping(10, 2),
        )
        strategy_g2 = MeasurementStrategy.partial(
            modes=[0, 1, 3],
            computation_space=ComputationSpace.FOCK,
            grouping=ModGrouping(10, 5),
        )
        strategy_no_grouping = MeasurementStrategy.partial(
            modes=[0, 1, 3],
            computation_space=ComputationSpace.FOCK,
        )

        amplitudes = torch.randn(3, 10, dtype=torch.cfloat)
        state = StateVector(tensor=amplitudes, n_modes=4, n_photons=2)

        layer_g1 = QuantumLayer(
            circuit=empty_circuit,
            input_size=0,
            input_state=state,
            measurement_strategy=strategy_g1,
        )
        result_g1 = layer_g1()

        layer_g2 = QuantumLayer(
            circuit=empty_circuit,
            input_size=0,
            input_state=state,
            measurement_strategy=strategy_g2,
        )
        result_g2 = layer_g2()

        layer_no_grouping = QuantumLayer(
            circuit=empty_circuit,
            input_size=0,
            input_state=state,
            measurement_strategy=strategy_no_grouping,
        )
        result_no_grouping = layer_no_grouping()

        assert type(result_g1) is PartialMeasurement
        assert type(result_g2) is PartialMeasurement
        assert type(result_no_grouping) is PartialMeasurement
        assert result_g1.measured_modes == (0, 1, 3)
        assert result_g1.unmeasured_modes == (2,)
        assert type(result_g1.tensor) is torch.Tensor and result_g1.tensor.shape == (
            3,
            2,
        )
        assert type(result_g2.tensor) is torch.Tensor and result_g2.tensor.shape == (
            3,
            5,
        )
        assert type(
            result_no_grouping.tensor
        ) is torch.Tensor and result_no_grouping.tensor.shape == (3, 10)
        # Grouping does not affect amplitudes
        assert (
            len(result_g1.amplitudes)
            == len(result_g2.amplitudes)
            == len(result_no_grouping.amplitudes)
        )
        for i in range(len(result_g1.amplitudes)):
            assert torch.allclose(
                result_g1.amplitudes[i].tensor,
                result_g2.amplitudes[i].tensor,
            )
            assert torch.allclose(
                result_g1.amplitudes[i].tensor,
                result_no_grouping.amplitudes[i].tensor,
            )

    def test_partial_measurement_gradients_flow_probabilities_and_amplitudes(self):
        """Ensure gradients flow through probabilities and branch amplitudes."""
        empty_circuit = pcvl.Circuit(4)
        strategy = MeasurementStrategy.partial(
            modes=[0, 2],
            computation_space=ComputationSpace.FOCK,
        )

        amplitudes = torch.randn(2, 10, dtype=torch.cfloat, requires_grad=True)
        state = StateVector(tensor=amplitudes, n_modes=4, n_photons=2)

        layer = QuantumLayer(
            circuit=empty_circuit,
            input_size=0,
            input_state=state,
            measurement_strategy=strategy,
        )
        result = layer()

        assert result.tensor.requires_grad
        targets = torch.tensor([1, 3], device=result.tensor.device)
        probs = result.tensor.clamp_min(1e-12)
        loss_prob = -probs[torch.arange(probs.shape[0]), targets].log().sum()
        loss_prob.backward()

        assert amplitudes.grad is not None
        assert torch.any(amplitudes.grad.abs() > 0)

        amplitudes_amp = torch.randn(2, 10, dtype=torch.cfloat, requires_grad=True)
        state_amp = StateVector(tensor=amplitudes_amp, n_modes=4, n_photons=2)

        layer_amps = QuantumLayer(
            circuit=empty_circuit,
            input_size=0,
            input_state=state_amp,
            measurement_strategy=strategy,
        )
        result_amp = layer_amps()

        assert all(
            branch.amplitudes.tensor.requires_grad for branch in result_amp.branches
        )
        amp_loss = torch.stack([
            (branch.amplitudes.tensor.real**2).sum() for branch in result_amp.branches
        ]).sum()
        amp_loss.backward()

        assert amplitudes_amp.grad is not None
        assert torch.any(amplitudes_amp.grad.abs() > 0)

    def test_chaining_guarantee(self):
        """Ensure that we can chain results after partial measurement into another quantum layer."""
        empty_circuit_1 = pcvl.Circuit(4)
        empty_circuit_2 = pcvl.Circuit(2)

        strategy = MeasurementStrategy.partial(
            modes=[0, 2],
            computation_space=ComputationSpace.FOCK,
        )

        amplitudes = torch.randn(2, 10, dtype=torch.cfloat, requires_grad=True)
        state = StateVector(tensor=amplitudes, n_modes=4, n_photons=2)

        layer_1 = QuantumLayer(
            circuit=empty_circuit_1,
            input_size=0,
            input_state=state,
            measurement_strategy=strategy,
        )
        result_1 = layer_1()

        assert isinstance(result_1, PartialMeasurement)

        # Now use the unmeasured amplitudes as input to another quantum layer
        layer_2 = QuantumLayer(
            circuit=empty_circuit_2,
            input_size=0,
            input_state=result_1.amplitudes[0],  # Take the first branch for simplicity
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.FOCK
            ),
        )
        result_2 = layer_2()

        assert not isinstance(result_2, PartialMeasurement)
        assert type(result_2) is torch.Tensor

        loss = result_2.sum()
        loss.backward()
        assert amplitudes.grad is not None
        assert torch.any(amplitudes.grad.abs() > 0)

    def test_partial_measurement_exactitude(self):
        """
        Test that compares the amplitudes outputed by partial measurement with the
        amplitudes outputed by amplitude measurement on the same circuit.
        """
        # Build Gan et al. style circuit
        builder = CircuitBuilder(n_modes=4)
        builder.add_entangling_layer()
        builder.add_angle_encoding([0, 1], name="data_")
        builder.add_entangling_layer()
        input_state = [1, 0, 1, 0]
        partial_measurement_strat = MeasurementStrategy.partial(modes=[0, 2])
        amplitude_measurement_strat = MeasurementStrategy.amplitudes()

        layer_partial = QuantumLayer(
            builder=builder,
            input_size=2,
            input_state=input_state,
            measurement_strategy=partial_measurement_strat,
        )

        layer_amplitude = QuantumLayer(
            builder=builder,
            input_size=2,
            input_state=input_state,
            measurement_strategy=amplitude_measurement_strat,
            return_object=True,
        )
        layer_amplitude.load_state_dict(
            layer_partial.state_dict()
        )  # Ensure both layers have the same parameters for a fair comparison

        input_data = torch.rand((5, 2), requires_grad=True)

        result_partial = layer_partial(input_data)
        result_amplitude = layer_amplitude(input_data)

        # Compare the amplitudes of the first branch of the partial measurement with the amplitude measurement
        first_branch = result_partial.branches[0]
        first_outcome = first_branch.outcome
        first_branch_amplitudes = first_branch.amplitudes

        assert type(first_branch_amplitudes) is StateVector
        assert type(result_amplitude) is StateVector

        output_keys = layer_amplitude.output_keys
        measured_modes = result_partial.measured_modes
        unmeasured_modes = result_partial.unmeasured_modes

        # Build expected probability and conditional amplitudes without detector effects
        amps_full = result_amplitude.tensor
        remaining_n = first_branch.amplitudes.n_photons
        remaining_basis = Combinadics("fock", remaining_n, len(unmeasured_modes))

        expected_prob = torch.zeros_like(first_branch.probability)
        expected_conditional = torch.zeros(
            (amps_full.shape[0], len(remaining_basis)),
            dtype=amps_full.dtype,
            device=amps_full.device,
        )

        for key_idx, key in enumerate(output_keys):
            remaining_key = tuple(key[m] for m in unmeasured_modes)
            if sum(remaining_key) != remaining_n:
                continue
            if tuple(key[m] for m in measured_modes) != first_outcome:
                continue
            amp_col = amps_full[:, key_idx]
            expected_prob += amp_col.abs().pow(2)
            local_index = remaining_basis.fock_to_index(remaining_key)
            expected_conditional[:, local_index] += amp_col

        matching_branches = [
            branch
            for branch in result_partial.branches
            if branch.outcome == first_outcome
            and branch.amplitudes.n_photons == remaining_n
        ]
        combined_branch_prob = torch.stack(
            [branch.probability for branch in matching_branches], dim=0
        ).sum(dim=0)

        assert torch.allclose(
            expected_prob,
            combined_branch_prob,
            atol=1e-6,
        ), "Branch probability does not match amplitude marginal."

        # Normalize to get the conditional state (partial measurement returns normalized amplitudes)
        norm = expected_prob.sqrt().unsqueeze(-1)
        expected_conditional = torch.where(
            norm == 0,
            expected_conditional,
            expected_conditional / norm,
        )

        # Align global phase before comparing amplitudes
        if len(matching_branches) != 1:
            # If there are multiple branches with the same outcome and remaining photon
            # number, amplitude comparison is not well-defined due to potential superposition
            # of branches. We can only check probabilities in this case, unless we compare
            # the mixed-state density matrix instead of pure state amplitudes, which is out
            # of scope for this test.
            return
        actual = matching_branches[0].amplitudes.tensor
        phase = torch.sum(expected_conditional.conj() * actual, dim=1, keepdim=True)
        phase = torch.where(
            phase.abs() == 0,
            torch.ones_like(phase),
            phase / phase.abs(),
        )
        expected_aligned = expected_conditional * phase

        assert torch.allclose(
            expected_aligned,
            actual,
            atol=1e-6,
        ), "Conditional amplitudes do not match (up to global phase)."
