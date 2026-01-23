import pytest

from merlin.algorithms.kernels import FeatureMap, FidelityKernel


def test_FeatureMap_simple_warns():
    with pytest.warns(DeprecationWarning, match=r"Parameter 'n_photons' is deprecated"):
        obj = FeatureMap.simple(input_size=2, n_photons=2)
    assert obj is not None
    with pytest.warns(DeprecationWarning, match=r"Parameter 'trainable' is deprecated"):
        obj = FeatureMap.simple(input_size=2, trainable=True)
    assert obj is not None


def test_FidelityKernel_simple_warns():
    with pytest.warns(DeprecationWarning, match=r"Parameter 'n_photons' is deprecated"):
        obj = FidelityKernel.simple(input_size=2, n_photons=2)
    assert obj is not None
    with pytest.warns(DeprecationWarning, match=r"Parameter 'trainable' is deprecated"):
        obj = FidelityKernel.simple(input_size=2, trainable=True)
    assert obj is not None
    with pytest.warns(
        DeprecationWarning, match=r"Parameter 'input_state' is deprecated"
    ):
        obj = FidelityKernel.simple(input_size=2, input_state=[0, 1])
    assert obj is not None
