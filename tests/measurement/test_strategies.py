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

from merlin.core.computation_space import ComputationSpace
from merlin.measurement.strategies import MeasurementKind, MeasurementStrategy


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
            _ = strategy.get_unmeasured_modes(n_modes=3)
