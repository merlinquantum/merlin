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
    @pytest.mark.parametrize(
        ("factory", "expected_kind", "expected_space"),
        [
            (
                lambda: MeasurementStrategy.probs(ComputationSpace.FOCK),
                MeasurementKind.PROBABILITIES,
                ComputationSpace.FOCK,
            ),
            (
                lambda: MeasurementStrategy.mode_expectations(
                    ComputationSpace.DUAL_RAIL
                ),
                MeasurementKind.MODE_EXPECTATIONS,
                ComputationSpace.DUAL_RAIL,
            ),
            (
                MeasurementStrategy.amplitudes,
                MeasurementKind.AMPLITUDES,
                ComputationSpace.UNBUNCHED,
            ),
        ],
    )
    def test_factory_creates_correct_instance(
        self, factory, expected_kind, expected_space
    ):
        strategy = factory()
        assert strategy.type == expected_kind
        assert strategy.computation_space == expected_space

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

    @pytest.mark.parametrize(
        "computation_space",
        [
            ComputationSpace.FOCK,
            ComputationSpace.UNBUNCHED,
            ComputationSpace.DUAL_RAIL,
        ],
    )
    def test_partial_empty_modes(self, computation_space):
        with pytest.raises(ValueError, match="modes cannot be empty"):
            MeasurementStrategy.partial(
                modes=[],
                computation_space=computation_space,
            )

    @pytest.mark.parametrize(
        ("modes", "computation_space"),
        [
            ([0, 1, 1], ComputationSpace.FOCK),
            ([2, 2, 3], ComputationSpace.UNBUNCHED),
            ([0, 0], ComputationSpace.DUAL_RAIL),
        ],
    )
    def test_partial_duplicate_modes(self, modes, computation_space):
        with pytest.raises(ValueError, match="Duplicate mode indices"):
            MeasurementStrategy.partial(
                modes=modes,
                computation_space=computation_space,
            )

    @pytest.mark.parametrize(
        ("modes", "computation_space"),
        [
            ([0, -1, 2], ComputationSpace.FOCK),
            ([2, 1, -3], ComputationSpace.UNBUNCHED),
            ([-1], ComputationSpace.DUAL_RAIL),
        ],
    )
    def test_partial_negative_mode_index(self, modes, computation_space):
        with pytest.raises(ValueError, match="Negative mode index"):
            MeasurementStrategy.partial(
                modes=modes,
                computation_space=computation_space,
            )

    @pytest.mark.parametrize(
        ("modes", "computation_space", "n_modes"),
        [
            ([0, 2], ComputationSpace.FOCK, 2),
            ([1, 10], ComputationSpace.UNBUNCHED, 3),
            ([0, 4], ComputationSpace.DUAL_RAIL, 4),
        ],
    )
    def test_validate_modes_out_of_bounds(self, modes, computation_space, n_modes):
        strategy = MeasurementStrategy.partial(
            modes=modes,
            computation_space=computation_space,
        )
        with pytest.raises(ValueError, match="Invalid mode indices"):
            strategy.validate_modes(n_modes=n_modes)

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
