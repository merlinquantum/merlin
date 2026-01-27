import perceval as pcvl
import pytest

from merlin.algorithms.layer import QuantumLayer
from merlin.core.computation_space import ComputationSpace


def test_no_bunching_deprecation_in_init():
    """Passing no_bunching explicitly to QuantumLayer.__init__ must issue a DeprecationWarning."""
    circuit = pcvl.Circuit(2)
    # Provide an explicit input_state so the layer can initialize from the custom circuit
    with pytest.warns(DeprecationWarning):
        layer = QuantumLayer(circuit=circuit, input_state=[1, 0], no_bunching=True)
    assert layer.computation_space is ComputationSpace.UNBUNCHED


def test_simple_no_bunching_deprecation_and_conversion():
    """QuantumLayer.simple should warn on no_bunching and convert to computation_space."""
    with pytest.warns(DeprecationWarning):
        # Use n_params matching the entangling budget to avoid unrelated RuntimeWarning.
        model = QuantumLayer.simple(input_size=1, n_params=90, no_bunching=True)

    qlayer = model.quantum_layer
    assert qlayer.computation_space == ComputationSpace.UNBUNCHED


def test_simple_accepts_computation_space_with_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        model = QuantumLayer.simple(
            input_size=1,
            n_params=90,
            computation_space=ComputationSpace.FOCK,
        )

    assert model.quantum_layer.computation_space == ComputationSpace.FOCK
