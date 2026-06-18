import numpy as np
import perceval as pcvl
import pytest
import torch

from merlin.algorithms.kernels import FeatureMap, FidelityKernel
from merlin.core.computation_space import ComputationSpace


def test_FeatureMap_simple_warns():
    with pytest.warns(DeprecationWarning, match=r"Parameter 'n_photons' is deprecated"):
        obj = FeatureMap.simple(input_size=2, n_photons=2)
    assert obj is not None
    assert obj.circuit.m == 3
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
    assert obj.circuit.m == 3
    assert obj.is_trainable
    assert "LI_simple" in obj.trainable_parameters
    assert "RI_simple" in obj.trainable_parameters


def test_FidelityKernel_simple_warns():
    with pytest.warns(DeprecationWarning, match=r"Parameter 'n_photons' is deprecated"):
        obj = FidelityKernel.simple(input_size=2, n_photons=2)
    assert obj is not None
    assert obj.feature_map.circuit.m == 3
    assert obj.feature_map.is_trainable
    assert obj.input_state == [1, 0, 1]

    with pytest.warns(DeprecationWarning, match=r"Parameter 'n_photons' is deprecated"):
        obj = FidelityKernel.simple(input_size=2, n_photons=2, n_modes=6)
    assert obj is not None
    assert obj.feature_map.circuit.m == 6
    assert obj.feature_map.is_trainable
    assert obj.input_state == [1, 0, 1, 0, 1, 0]

    with pytest.warns(DeprecationWarning, match=r"Parameter 'trainable' is deprecated"):
        obj = FidelityKernel.simple(input_size=2, trainable=True)
    assert obj is not None
    assert obj is not None
    assert obj.feature_map.circuit.m == 3
    assert obj.feature_map.is_trainable
    assert obj.input_state == [1, 0, 1]

    with pytest.warns(
        DeprecationWarning, match=r"Parameter 'input_state' is deprecated"
    ):
        obj = FidelityKernel.simple(input_size=2, input_state=[1, 1])
    assert obj is not None
    assert obj is not None
    assert obj.feature_map.circuit.m == 3
    assert obj.feature_map.is_trainable
    assert obj.input_state == [1, 0, 1]


class TestLegacyFeatureMapUnitaryPath:
    """Deprecated FeatureMap.compute_unitary path kept for backwards compatibility."""

    def setup_method(self):
        x1, x2 = pcvl.P("x1"), pcvl.P("x2")
        self.circuit = (
            pcvl.Circuit(2) // pcvl.PS(x1) // pcvl.BS() // pcvl.PS(x2) // pcvl.BS()
        )
        self.feature_map = FeatureMap(
            circuit=self.circuit,
            input_size=2,
            input_parameters="x",
        )

    def test_trainable_feature_map_keeps_legacy_training_dict(self):
        theta = pcvl.P("theta")
        circuit = (
            pcvl.Circuit(2)
            // pcvl.PS(pcvl.P("x1"))
            // pcvl.BS(theta)
            // pcvl.PS(pcvl.P("x2"))
            // pcvl.BS(theta)
        )

        feature_map = FeatureMap(
            circuit=circuit,
            input_size=2,
            input_parameters="x",
            trainable_parameters=["theta"],
        )

        assert "theta" in feature_map._training_dict

    def test_compute_unitary_emits_deprecation_warning(self):
        x = torch.tensor([0.5, 1.0])

        with pytest.warns(DeprecationWarning) as warning_list:
            self.feature_map.compute_unitary(x)

        assert len(warning_list) == 1
        message = str(warning_list[0].message)
        assert "compute_unitary is deprecated" in message
        assert "legacy compiler state stored on FeatureMap" in message
        assert "descriptor" in message

    def test_compute_unitary_single_datapoint(self):
        x = torch.tensor([0.5, 1.0])
        with pytest.warns(DeprecationWarning, match="compute_unitary is deprecated"):
            unitary = self.feature_map.compute_unitary(x)

        assert isinstance(unitary, torch.Tensor)
        assert unitary.shape == (2, 2)
        assert torch.allclose(
            unitary @ unitary.conj().T, torch.eye(2, dtype=torch.cfloat), atol=1e-6
        )

    def test_compute_unitary_dataset(self):
        X = torch.tensor([[0.5, 1.0], [1.5, 0.5], [0.0, 2.0]])
        with pytest.warns(DeprecationWarning, match="compute_unitary is deprecated"):
            unitaries = [self.feature_map.compute_unitary(x) for x in X]

        assert len(unitaries) == 3
        for unitary in unitaries:
            assert isinstance(unitary, torch.Tensor)
            assert unitary.shape == (2, 2)
            assert torch.allclose(
                unitary @ unitary.conj().T, torch.eye(2, dtype=torch.cfloat), atol=1e-6
            )

    def test_kernel_no_bunching_matches_perceval_and_legacy_unitaries(self):
        from perceval import (
            BS,
            PS,
            BasicState,
            Circuit,
            GenericInterferometer,
            P,
            Processor,
            Unitary,
        )
        from perceval.algorithm import Sampler

        def circ_func(x):
            """Generate a rectangular generic interferometer component."""
            circuit = Circuit(2) // PS(P(f"phi{2 * x}")) // BS()
            circuit.add(0, PS(P(f"phi{2 * x + 1}")))
            circuit.add(0, BS())
            return circuit

        input_state = [1, 1, 0, 0]
        circuit = GenericInterferometer(len(input_state), circ_func)
        input_size = len(circuit.get_parameters())
        feature_map = FeatureMap(circuit, input_size, input_parameters=["phi"])

        unbunching_kernel = FidelityKernel(
            feature_map,
            input_state,
            computation_space=ComputationSpace.UNBUNCHED,
            force_psd=False,
        )
        quantum_kernel = FidelityKernel(
            feature_map,
            input_state,
            computation_space=ComputationSpace.FOCK,
        )

        rng = np.random.default_rng(42)
        X1 = rng.random(input_size)
        X2 = rng.random(input_size)

        merlin_pnr = float(quantum_kernel(X1, X2))
        merlin_thr = float(unbunching_kernel(X1, X2))

        circuit_forward = GenericInterferometer(len(input_state), circ_func)
        for i, p in enumerate(circuit_forward.get_parameters()):
            p.set_value(X1[i])
        forward_unitary = np.asarray(
            circuit_forward.compute_unitary(), dtype=np.complex128
        )

        with pytest.warns(DeprecationWarning, match="compute_unitary is deprecated"):
            feature_forward = (
                feature_map
                .compute_unitary(torch.as_tensor(X1, dtype=feature_map.dtype))
                .detach()
                .cpu()
                .numpy()
            )
        assert np.allclose(feature_forward, forward_unitary, atol=1e-6)

        circuit_backward = GenericInterferometer(len(input_state), circ_func)
        for i, p in enumerate(circuit_backward.get_parameters()):
            p.set_value(X2[i])
        backward_unitary = np.asarray(
            circuit_backward.compute_unitary(), dtype=np.complex128
        )

        with pytest.warns(DeprecationWarning, match="compute_unitary is deprecated"):
            feature_backward = (
                feature_map
                .compute_unitary(torch.as_tensor(X2, dtype=feature_map.dtype))
                .detach()
                .cpu()
                .numpy()
            )
        assert np.allclose(feature_backward, backward_unitary, atol=1e-6)

        circ_unitary = forward_unitary @ backward_unitary.conj().T
        circ_unitary = Unitary(pcvl.Matrix(circ_unitary))

        processor = Processor("SLOS")
        processor.set_circuit(circ_unitary)
        processor.with_input(BasicState(input_state))
        processor.min_detected_photons_filter(0)

        sampler = Sampler(processor)

        def state_to_tuple(state):
            try:
                return tuple(int(n) for n in state.tolist())
            except AttributeError:
                return tuple(int(n) for n in state)

        raw_results = sampler.probs()["results"]
        results = {
            state_to_tuple(state): float(prob) for state, prob in raw_results.items()
        }

        key = tuple(input_state)
        assert key in results
        perceval_pnr = results[key]

        thresholded_results = {
            state: prob for state, prob in results.items() if max(state) == 1
        }
        total_threshold_prob = sum(thresholded_results.values())

        assert total_threshold_prob > 0
        assert key in thresholded_results

        perceval_thr = thresholded_results[key] / total_threshold_prob

        assert merlin_pnr == pytest.approx(perceval_pnr, rel=1e-6, abs=1e-6)
        assert merlin_thr == pytest.approx(perceval_thr, rel=1e-6, abs=1e-6)
