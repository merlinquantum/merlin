import perceval as pcvl
import pytest

from merlin.algorithms.layer import QuantumLayer
from merlin.core.computation_space import ComputationSpace


def test_no_bunching_deprecation_in_init():
    """Passing no_bunching explicitly to QuantumLayer.__init__ must warn + fail."""
    circuit = pcvl.Circuit(2)
    # Provide an explicit input_state so the layer can initialize from the custom circuit
    with pytest.warns(DeprecationWarning):
        with pytest.raises(TypeError):
            QuantumLayer(circuit=circuit, input_state=[1, 0], no_bunching=True)


def test_simple_no_bunching_deprecation_and_conversion():
    """QuantumLayer.simple should warn + fail on no_bunching."""
    with pytest.warns(DeprecationWarning):
        with pytest.raises(TypeError):
            QuantumLayer.simple(input_size=2, no_bunching=True)


def test_simple_accepts_computation_space_with_deprecation_warning():
    with pytest.warns(
        DeprecationWarning,
        match=r"Parameter 'computation_space' is deprecated",
    ):
        model = QuantumLayer.simple(
            input_size=2,
            computation_space=ComputationSpace.FOCK,
        )

    assert model.quantum_layer.computation_space == ComputationSpace.FOCK


def test_simple_warns_on_n_params():
    with pytest.warns(DeprecationWarning, match=r"Parameter 'n_params' is deprecated"):
        obj = QuantumLayer.simple(input_size=2, n_params=95)
    assert obj is not None
    assert obj.circuit.m == 2
    assert obj.quantum_layer.input_state == [0, 1]
