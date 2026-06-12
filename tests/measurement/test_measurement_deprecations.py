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
import pytest

from merlin import MeasurementStrategy, QuantumLayer
from merlin.core.computation_space import ComputationSpace


class TestMeasurementStrategyDeprecations:
    def test_probabilities_enum_raises_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="v0.4"):
            _ = MeasurementStrategy.PROBABILITIES

    def test_mode_expectations_enum_raises_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="v0.4"):
            _ = MeasurementStrategy.MODE_EXPECTATIONS

    def test_amplitudes_enum_raises_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="v0.4"):
            _ = MeasurementStrategy.AMPLITUDES

    def test_deprecation_warning_includes_migration_hint(self):
        with pytest.warns(DeprecationWarning, match="Use MeasurementStrategy.probs"):
            _ = MeasurementStrategy.PROBABILITIES

    def test_deprecated_enum_still_works_in_quantum_layer(self):
        circuit = pcvl.Circuit(2)
        with pytest.warns(DeprecationWarning):
            layer = QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=[1, 0],
                computation_space=ComputationSpace.FOCK,
                measurement_strategy=MeasurementStrategy.PROBABILITIES,
            )
        assert layer.measurement_strategy is not None

    def test_multiple_enum_accesses_warn_each_time(self):
        with pytest.warns(DeprecationWarning):
            _ = MeasurementStrategy.PROBABILITIES
        with pytest.warns(DeprecationWarning):
            _ = MeasurementStrategy.PROBABILITIES
