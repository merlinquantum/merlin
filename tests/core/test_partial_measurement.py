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

import perceval as pcvl
import torch

from merlin.core.partial_measurement import PartialMeasurement, PartialMeasurementBranch
from merlin.core.state_vector import StateVector
from merlin.measurement import DetectorTransform


class TestPartialMeasurementBranch:
    def test_branch_creation(self):
        prob = torch.tensor([0.12, 0.34], dtype=torch.float32, requires_grad=True)
        amps = torch.randn(2, 4, dtype=torch.cfloat)  # (batch, remaining_basis_dim)
        sv = StateVector(tensor=amps, n_modes=3, n_photons=2)
        b = PartialMeasurementBranch(
            outcome=(0, 2),
            probability=prob,
            amplitudes=sv,
        )
        assert b.outcome == (0, 2)
        assert torch.equal(b.probability, prob)
        assert b.amplitudes is sv


class TestPartialMeasurement:
    def test_constructor_orders_branches(self):
        # Outcomes intentionally out of order
        prob_a = torch.tensor([0.14, 0.20], dtype=torch.float32)
        prob_b = torch.tensor([0.12, 0.10], dtype=torch.float32)
        sv_a = StateVector(
            tensor=torch.randn(2, 3, dtype=torch.cfloat), n_modes=2, n_photons=1
        )
        sv_b = StateVector(
            tensor=torch.randn(2, 3, dtype=torch.cfloat), n_modes=2, n_photons=0
        )
        branches = (
            PartialMeasurementBranch(
                outcome=(1, 0), probability=prob_a, amplitudes=sv_a
            ),
            PartialMeasurementBranch(
                outcome=(0, 2), probability=prob_b, amplitudes=sv_b
            ),
        )
        result = PartialMeasurement(
            branches=branches,
            measured_modes=(0, 1),
            unmeasured_modes=(2, 3),
        )
        assert [b.outcome for b in result.branches] == [(0, 2), (1, 0)], (
            "Branches are not ordered lexicographically by outcome"
        )

    def test_constructor_orders_branches_with_duplicate_outcomes(self):
        prob_first = torch.tensor([0.10, 0.11], dtype=torch.float32)
        prob_second = torch.tensor([0.20, 0.21], dtype=torch.float32)
        prob_third = torch.tensor([0.30, 0.31], dtype=torch.float32)

        sv_first = StateVector(
            tensor=torch.randn(2, 3, dtype=torch.cfloat), n_modes=2, n_photons=1
        )
        sv_second = StateVector(
            tensor=torch.randn(2, 3, dtype=torch.cfloat), n_modes=2, n_photons=0
        )
        sv_third = StateVector(
            tensor=torch.randn(2, 3, dtype=torch.cfloat), n_modes=2, n_photons=0
        )

        branches = (
            PartialMeasurementBranch(
                outcome=(1, 0), probability=prob_first, amplitudes=sv_first
            ),
            PartialMeasurementBranch(
                outcome=(0, 2), probability=prob_second, amplitudes=sv_second
            ),
            PartialMeasurementBranch(
                outcome=(0, 2), probability=prob_third, amplitudes=sv_third
            ),
        )
        result = PartialMeasurement(
            branches=branches,
            measured_modes=(0, 1),
            unmeasured_modes=(2, 3),
        )

        assert [b.outcome for b in result.branches] == [(0, 2), (0, 2), (1, 0)]
        assert torch.equal(result.branches[0].probability, prob_second)
        assert torch.equal(result.branches[1].probability, prob_third)
        assert torch.equal(result.branches[2].probability, prob_first)

    def test_tensor_stacks_probabilities_in_branch_order(self):
        prob_a = torch.tensor([0.12, 0.10], dtype=torch.float32)
        prob_b = torch.tensor([0.14, 0.20], dtype=torch.float32)
        sv_a = StateVector(
            tensor=torch.randn(2, 3, dtype=torch.cfloat), n_modes=2, n_photons=0
        )
        sv_b = StateVector(
            tensor=torch.randn(2, 3, dtype=torch.cfloat), n_modes=2, n_photons=1
        )
        branches = (
            PartialMeasurementBranch(
                outcome=(0, 2), probability=prob_a, amplitudes=sv_a
            ),
            PartialMeasurementBranch(
                outcome=(1, 0), probability=prob_b, amplitudes=sv_b
            ),
        )
        result = PartialMeasurement(
            branches=branches,
            measured_modes=(0, 1),
            unmeasured_modes=(2, 3),
        )
        expected = torch.stack([prob_a, prob_b], dim=-1)
        assert torch.allclose(result.tensor, expected)
        assert result.tensor.shape == (2, 2)

        branches_reversed = (
            PartialMeasurementBranch(
                outcome=(1, 0), probability=prob_a, amplitudes=sv_a
            ),
            PartialMeasurementBranch(
                outcome=(0, 2), probability=prob_b, amplitudes=sv_b
            ),
        )
        result_reversed = PartialMeasurement(
            branches=branches_reversed,
            measured_modes=(0, 1),
            unmeasured_modes=(2, 3),
        )
        expected_reversed = torch.stack([prob_b, prob_a], dim=-1)
        assert torch.allclose(result_reversed.tensor, expected_reversed)
        assert result_reversed.tensor.shape == (2, 2)

    def test_tensor_handles_scalar_probabilities(self):
        prob_a = torch.tensor(0.75)
        prob_b = torch.tensor(0.25)
        sv_a = StateVector(
            tensor=torch.randn(1, 2, dtype=torch.cfloat), n_modes=1, n_photons=1
        )
        sv_b = StateVector(
            tensor=torch.randn(1, 2, dtype=torch.cfloat), n_modes=1, n_photons=0
        )
        branches = (
            PartialMeasurementBranch(outcome=(0,), probability=prob_a, amplitudes=sv_a),
            PartialMeasurementBranch(outcome=(1,), probability=prob_b, amplitudes=sv_b),
        )
        result = PartialMeasurement(
            branches=branches,
            measured_modes=(0,),
            unmeasured_modes=(1,),
        )
        assert result.tensor.shape == (1, 2)
        assert torch.allclose(result.tensor, torch.tensor([[0.75, 0.25]]))

    def test_properties_expose_mode_counts(self):
        prob = torch.tensor([1.0], dtype=torch.float32)
        sv = StateVector(
            tensor=torch.randn(1, 2, dtype=torch.cfloat), n_modes=1, n_photons=1
        )
        branches = (
            PartialMeasurementBranch(outcome=(0, 0), probability=prob, amplitudes=sv),
        )
        result = PartialMeasurement(
            branches=branches,
            measured_modes=(0, 2),
            unmeasured_modes=(1,),
        )
        assert result.n_measured_modes == 2
        assert result.n_unmeasured_modes == 1

    def test_from_detector_transform_output_converts_to_triplets(self):
        batch = 2

        # Total system has 2 modes: mode 0 measured, mode 1 unmeasured
        measured_modes = (0,)
        unmeasured_modes = (1,)

        # Fake DetectorTransform(partial_measurement=True) output
        # Format:
        #   list[remaining_n] -> dict[full_key, list[(prob, remaining_amplitudes)]]
        prob_1 = torch.tensor([0.40, 0.10], dtype=torch.float32, requires_grad=True)
        prob_0 = torch.tensor([0.60, 0.90], dtype=torch.float32, requires_grad=True)

        rem_amp_1 = torch.randn(batch, 3, dtype=torch.cfloat)
        rem_amp_0 = torch.randn(batch, 3, dtype=torch.cfloat)

        detector_output = [
            {(1, None): [(prob_1, rem_amp_1)]},
            {(0, None): [(prob_0, rem_amp_0)]},
        ]

        pm = PartialMeasurement.from_detector_transform_output(
            detector_output=detector_output
        )

        # Branches must be sorted lexicographically by measured-only outcome
        assert [b.outcome for b in pm.branches] == [(0,), (1,)]
        assert pm.measured_modes == measured_modes
        assert pm.unmeasured_modes == unmeasured_modes

        # .tensor stacks probabilities in the same order
        expected = torch.stack([prob_0, prob_1], dim=-1)
        assert torch.allclose(pm.tensor, expected)
        assert pm.tensor.shape == (batch, 2)

        # Each branch holds a conditional StateVector on unmeasured modes
        assert isinstance(pm.branches[0], PartialMeasurementBranch)
        assert isinstance(pm.branches[0].amplitudes, StateVector)
        assert isinstance(pm.branches[1].amplitudes, StateVector)

        assert torch.allclose(pm.branches[0].amplitudes.tensor, rem_amp_0)
        assert torch.allclose(pm.branches[1].amplitudes.tensor, rem_amp_1)

        # Gradient must flow through probability tensor
        loss = pm.tensor.sum()
        loss.backward()

        assert prob_0.grad is not None
        assert prob_1.grad is not None

    def test_from_detector_transform_output_uses_detector_transform_output_directly(
        self,
    ):
        simulation_keys = [(0, 1), (1, 0)]
        detectors = [pcvl.Detector.pnr(), None]
        transform = DetectorTransform(
            simulation_keys, detectors, partial_measurement=True
        )

        amplitudes = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j])
        detector_output = transform(amplitudes)

        pm = PartialMeasurement.from_detector_transform_output(detector_output)

        assert pm.measured_modes == (0,)
        assert pm.unmeasured_modes == (1,)
        assert [b.outcome for b in pm.branches] == [(0,), (1,)]

        expected_by_outcome = {}
        for level in detector_output:
            for full_outcome, entries in level.items():
                measured_only = tuple(elem for elem in full_outcome if elem is not None)
                expected_by_outcome[measured_only] = entries

        for branch in pm.branches:
            output = expected_by_outcome[branch.outcome]
            if len(output) == 1:
                expected_prob, expected_amp = output[0]
                assert torch.allclose(branch.probability, expected_prob)
                assert torch.allclose(branch.amplitudes.tensor, expected_amp)
            else:
                assert (branch.probability, branch.amplitudes) in output
