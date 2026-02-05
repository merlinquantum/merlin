import perceval as pcvl
import pytest

from merlin.algorithms.layer import QuantumLayer
from merlin.core.computation_space import ComputationSpace


def test_init_defaults_to_unbunched():
    """QuantumLayer.__init__ defaults to UNBUNCHED computation space."""
    circuit = pcvl.Circuit(2)
    # Provide an explicit input_state so the layer can initialize from the custom circuit
    layer = QuantumLayer(circuit=circuit, input_state=[1, 0])
    assert layer.computation_space is ComputationSpace.UNBUNCHED


def test_simple_defaults_to_unbunched():
    """QuantumLayer.simple defaults to UNBUNCHED computation space."""
    model = QuantumLayer.simple(input_size=2)
    assert model.quantum_layer.computation_space is ComputationSpace.UNBUNCHED


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
