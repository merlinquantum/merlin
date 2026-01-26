import pytest

from merlin.algorithms.kernels import FeatureMap, FidelityKernel


def test_FeatureMap_simple_warns():
    with pytest.warns(DeprecationWarning, match=r"Parameter 'n_photons' is deprecated"):
        obj = FeatureMap.simple(input_size=2, n_photons=2)
    assert obj is not None
    assert obj.circuit.m == 2
    assert obj.is_trainable
    assert "LI_simple" in obj.trainable_parameters
    assert "RI_simple" in obj.trainable_parameters

    with pytest.warns(DeprecationWarning, match=r"Parameter 'n_photons' is deprecated"):
        obj = FeatureMap.simple(input_size=2, n_photons=2, n_modes=6)
    assert obj is not None
    assert obj.circuit.m == 6
    assert obj.is_trainable
    assert "LI_simple" in obj.trainable_parameters
    assert "RI_simple" in obj.trainable_parameters
    with pytest.warns(DeprecationWarning, match=r"Parameter 'trainable' is deprecated"):
        obj = FeatureMap.simple(input_size=2, trainable=True)
    assert obj is not None
    assert obj.circuit.m == 2
    assert obj.is_trainable
    assert "LI_simple" in obj.trainable_parameters
    assert "RI_simple" in obj.trainable_parameters


def test_FidelityKernel_simple_warns():
    with pytest.warns(DeprecationWarning, match=r"Parameter 'n_photons' is deprecated"):
        obj = FidelityKernel.simple(input_size=2, n_photons=2)
    assert obj is not None
    assert obj.feature_map.circuit.m == 2
    assert obj.feature_map.is_trainable
    assert obj.input_state == [0, 1]

    with pytest.warns(DeprecationWarning, match=r"Parameter 'n_photons' is deprecated"):
        obj = FidelityKernel.simple(input_size=2, n_photons=2, n_modes=6)
    assert obj is not None
    assert obj.feature_map.circuit.m == 6
    assert obj.feature_map.is_trainable
    assert obj.input_state == [0, 1, 0, 1, 0, 1]

    with pytest.warns(DeprecationWarning, match=r"Parameter 'trainable' is deprecated"):
        obj = FidelityKernel.simple(input_size=2, trainable=True)
    assert obj is not None
    assert obj is not None
    assert obj.feature_map.circuit.m == 2
    assert obj.feature_map.is_trainable
    assert obj.input_state == [0, 1]

    with pytest.warns(
        DeprecationWarning, match=r"Parameter 'input_state' is deprecated"
    ):
        obj = FidelityKernel.simple(input_size=2, input_state=[1, 1])
    assert obj is not None
    assert obj is not None
    assert obj.feature_map.circuit.m == 2
    assert obj.feature_map.is_trainable
    assert obj.input_state == [0, 1]
