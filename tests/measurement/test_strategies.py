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

import pytest
import torch

from merlin.core.computation_space import ComputationSpace
from merlin.core.partial_measurement import PartialMeasurement
from merlin.core.state_vector import StateVector
from merlin.measurement.strategies import MeasurementStrategyV3, MeasurementType
from merlin.utils.grouping import LexGrouping, ModGrouping


class TestMeasurementStrategyV3:
    def test_partial_factory_creation_noncontiguous_modes(self):
        """Ensure factory wires fields without reordering sparse modes."""
        strategy = MeasurementStrategyV3.partial(
            modes=[0, 2, 5],
            computation_space=ComputationSpace.DUAL_RAIL,
            grouping=None,
        )
        assert strategy.type == MeasurementType.PARTIAL
        assert strategy.measured_modes == (0, 2, 5)
        assert strategy.computation_space == ComputationSpace.DUAL_RAIL
        assert strategy.grouping is None

    def test_partial_empty_modes(self):
        with pytest.raises(ValueError, match="modes cannot be empty"):
            MeasurementStrategyV3.partial(
                modes=[],
                computation_space=ComputationSpace.FOCK,
            )
        with pytest.raises(ValueError, match="modes cannot be empty"):
            MeasurementStrategyV3.partial(
                modes=[],
                computation_space=ComputationSpace.UNBUNCHED,
            )
        with pytest.raises(ValueError, match="modes cannot be empty"):
            MeasurementStrategyV3.partial(
                modes=[],
                computation_space=ComputationSpace.DUAL_RAIL,
            )

    def test_partial_duplicate_modes(self):
        with pytest.raises(ValueError, match="Duplicate mode indices"):
            MeasurementStrategyV3.partial(
                modes=[0, 1, 1],
                computation_space=ComputationSpace.FOCK,
            )
        with pytest.raises(ValueError, match="Duplicate mode indices"):
            MeasurementStrategyV3.partial(
                modes=[2, 2, 3],
                computation_space=ComputationSpace.UNBUNCHED,
            )
        with pytest.raises(ValueError, match="Duplicate mode indices"):
            MeasurementStrategyV3.partial(
                modes=[0, 0],
                computation_space=ComputationSpace.DUAL_RAIL,
            )

    def test_partial_negative_mode_index(self):
        with pytest.raises(ValueError, match="Negative mode index"):
            MeasurementStrategyV3.partial(
                modes=[0, -1, 2],
                computation_space=ComputationSpace.FOCK,
            )
        with pytest.raises(ValueError, match="Negative mode index"):
            MeasurementStrategyV3.partial(
                modes=[2, 1, -3],
                computation_space=ComputationSpace.UNBUNCHED,
            )
        with pytest.raises(ValueError, match="Negative mode index"):
            MeasurementStrategyV3.partial(
                modes=[-1],
                computation_space=ComputationSpace.DUAL_RAIL,
            )

    def test_validate_modes_out_of_bounds(self):
        strategy = MeasurementStrategyV3.partial(
            modes=[0, 2],
            computation_space=ComputationSpace.FOCK,
        )
        with pytest.raises(ValueError, match="Invalid mode indices"):
            strategy.validate_modes(
                n_modes=2
            )  # mode index 2 is out of bounds for n_modes=2

        strategy = MeasurementStrategyV3.partial(
            modes=[1, 10],
            computation_space=ComputationSpace.UNBUNCHED,
        )
        with pytest.raises(ValueError, match="Invalid mode indices"):
            strategy.validate_modes(
                n_modes=3
            )  # mode index 10 is out of bounds for n_modes=3

        strategy = MeasurementStrategyV3.partial(
            modes=[0, 4],
            computation_space=ComputationSpace.DUAL_RAIL,
        )
        with pytest.raises(ValueError, match="Invalid mode indices"):
            strategy.validate_modes(
                n_modes=4
            )  # mode index 4 is out of bounds for n_modes=4

    def test_partial_returns_immutable_object(self):
        """The strategy dataclass is frozen to avoid mutation after construction."""
        strategy = MeasurementStrategyV3.partial(
            modes=[0, 2],
            computation_space=ComputationSpace.FOCK,
        )

        with pytest.raises(FrozenInstanceError):
            strategy.measured_modes = (1, 3)

    def test_get_unmeasured_modes(self):
        """Unmeasured modes should be the complement of measured modes."""
        strategy = MeasurementStrategyV3.partial(
            modes=[0, 2, 4],
            computation_space=ComputationSpace.FOCK,
        )

        assert strategy.get_unmeasured_modes(n_modes=6) == (1, 3, 5)

    def test_all_modes_measured_warning(self):
        """Both helpers surface the warning when the selection covers all modes."""
        strategy = MeasurementStrategyV3.partial(
            modes=[0, 1, 2],
            computation_space=ComputationSpace.FOCK,
        )

        with pytest.raises(
            Warning,
            match="All modes are measured",
        ):
            strategy.validate_modes(n_modes=3)

        with pytest.raises(
            Warning,
            match="All modes are measured",
        ):
            strategy.get_unmeasured_modes(n_modes=3)

    def test_partial_measurement_output(self):
        """Test that _process_partial_measurement returns a PartialMeasurement object."""
        strategy = MeasurementStrategyV3.partial(
            modes=[0, 2],
        )

        amplitudes = torch.randn(2, 10, dtype=torch.cfloat)
        state = StateVector(tensor=amplitudes, n_modes=4, n_photons=2)
        result = strategy.process_measurement(state)
        assert isinstance(result, PartialMeasurement)
        assert len(result.branches) > 0
        assert all(
            type(branch.amplitudes) is StateVector and branch.amplitudes.n_modes == 2
            for branch in result.branches
        )
        assert result.measured_modes == (0, 2)
        assert result.unmeasured_modes == (1, 3)
        assert type(result.tensor) is torch.Tensor and result.tensor.shape[0] == 2

    def test_partial_measurement_grouping(self):
        """Test that grouping is only applied on probabilities when specified with partial measurement."""
        strategy_g1 = MeasurementStrategyV3.partial(
            modes=[0, 1, 3],
            grouping=LexGrouping(10, 2),
        )
        strategy_g2 = MeasurementStrategyV3.partial(
            modes=[0, 1, 3],
            grouping=ModGrouping(10, 5),
        )
        strategy_no_grouping = MeasurementStrategyV3.partial(
            modes=[0, 1, 3],
        )

        amplitudes = torch.randn(3, 10, dtype=torch.cfloat)
        state = StateVector(tensor=amplitudes, n_modes=4, n_photons=2)
        result_g1 = strategy_g1.process_measurement(state)
        result_g2 = strategy_g2.process_measurement(state)
        result_no_grouping = strategy_no_grouping.process_measurement(state)

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
