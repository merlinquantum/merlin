"""Deprecation tests for kernel APIs.

Tests in this module verify that deprecated APIs emit the expected
``DeprecationWarning`` and continue to work correctly until removal.

TODO: In release 0.5.x, remove this module along with the deprecated APIs.
"""

import warnings

import numpy as np
import perceval as pcvl
import pytest
import torch

from merlin.algorithms.kernels import (
    FeatureMap,
    FidelityKernel,
    KernelCircuitBuilder,
)
from merlin.builder import CircuitBuilder
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


# ---------------------------------------------------------------------------
# Deprecated no_bunching parameter
# ---------------------------------------------------------------------------


class TestDeprecatedNoBunchingParam:
    """Tests for the removed ``no_bunching`` parameter across kernel APIs."""

    def setup_method(self):
        x1, x2 = pcvl.P("x1"), pcvl.P("x2")
        circuit = (
            pcvl.Circuit(2) // pcvl.PS(x1) // pcvl.BS() // pcvl.PS(x2) // pcvl.BS()
        )
        self.feature_map = FeatureMap(
            circuit=circuit,
            input_size=2,
            input_parameters="x",
        )

    def test_kernel_rejects_no_bunching(self):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError) as exc_info:
                FidelityKernel(
                    feature_map=self.feature_map,
                    input_state=[2, 0],
                    no_bunching=True,
                )
        assert "no_bunching" in str(exc_info.value)

        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError) as exc_info:
                FidelityKernel.simple(input_size=2, no_bunching=True)
        assert "no_bunching" in str(exc_info.value)

        builder = KernelCircuitBuilder().input_size(2).n_modes(4)
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError) as exc_info:
                builder.build_fidelity_kernel(no_bunching=True)
        assert "no_bunching" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Deprecated legacy FidelityKernel paths (encoder, direct-circuit subset)
# ---------------------------------------------------------------------------


class TestDeprecatedLegacyKernelPaths:
    """Tests for deprecated ``FeatureMap.encoder`` and direct-circuit paths."""

    def test_kernel_warns_and_uses_feature_map_encoder(self):
        circuit = pcvl.Circuit(3)
        for mode in range(3):
            circuit.add(mode, pcvl.PS(pcvl.P(f"x{mode}")))

        def encoder(x):
            return torch.stack([x[0], x[1], x[0] + x[1]])

        feature_map = FeatureMap(
            circuit=circuit,
            input_size=2,
            input_parameters="x",
            encoder=encoder,
        )

        with pytest.warns(
            DeprecationWarning,
            match="FeatureMap.encoder support inside FidelityKernel is deprecated",
        ) as warning_record:
            kernel = FidelityKernel(
                feature_map=feature_map,
                input_state=[1, 0, 0],
                computation_space=ComputationSpace.FOCK,
            )
        warning_message = str(warning_record[0].message)
        assert "CircuitBuilder.add_angle_encoding" in warning_message
        assert "pre-encoding the data" in warning_message
        assert "input_size equal to the encoded circuit-parameter count" in warning_message

        encoded = kernel._quantum_layer._encode_single(torch.tensor([0.2, 0.3]))
        expected = torch.tensor([0.2, 0.3, 0.5], dtype=encoded.dtype)
        assert torch.allclose(encoded, expected)

    def test_kernel_warns_and_uses_direct_circuit_subset_expansion(self):
        circuit = pcvl.Circuit(3)
        for mode in range(3):
            circuit.add(mode, pcvl.PS(pcvl.P(f"x{mode}")))

        feature_map = FeatureMap(
            circuit=circuit,
            input_size=2,
            input_parameters="x",
        )

        with pytest.warns(
            DeprecationWarning,
            match="input_size differs from the circuit input parameter count",
        ) as warning_record:
            kernel = FidelityKernel(
                feature_map=feature_map,
                input_state=[1, 0, 0],
                computation_space=ComputationSpace.FOCK,
            )
        warning_message = str(warning_record[0].message)
        assert "CircuitBuilder.add_angle_encoding" in warning_message
        assert "pre-encoding the data" in warning_message
        assert "input_size equal to the encoded circuit-parameter count" in warning_message

        encoded = kernel._quantum_layer._encode_single(torch.tensor([0.2, 0.3]))
        expected = torch.tensor([0.2, 0.3, 0.5], dtype=encoded.dtype)
        assert torch.allclose(encoded, expected)

    def test_simple_kernel_backend_preserves_angle_encoding_scale(self):
        with pytest.warns(DeprecationWarning, match="n_modes"):
            kernel = FidelityKernel.simple(
                input_size=2,
                n_modes=4,
                angle_encoding_scale=0.5,
            )

        encoded = kernel._quantum_layer._encode_single(torch.tensor([0.2, 0.4]))
        expected = torch.tensor([0.1, 0.2], dtype=encoded.dtype)
        assert torch.allclose(encoded, expected)


# ---------------------------------------------------------------------------
# Deprecated n_modes parameter in FeatureMap.simple()
# ---------------------------------------------------------------------------


class TestDeprecatedFeatureMapSimpleNModes:
    """Tests for the deprecated ``n_modes`` parameter in ``FeatureMap.simple()``."""

    def test_simple_n_modes_override_emits_deprecation_warning(self):
        """Passing n_modes to FeatureMap.simple must emit a DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="n_modes"):
            feature_map = FeatureMap.simple(input_size=2, n_modes=6)
        assert feature_map.circuit.m == 6

    def test_simple_factory_raises_when_input_exceeds_modes(self):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(
                ValueError, match="You cannot encore more features than mode with Builder"
            ):
                FeatureMap.simple(input_size=5, n_modes=4)

    def test_simple_factory_raises_when_input_or_modes_exceeds_20(self):
        with pytest.raises(ValueError):
            FeatureMap.simple(input_size=21)
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError):
                FeatureMap.simple(input_size=21, n_modes=21)


# ---------------------------------------------------------------------------
# Deprecated FidelityKernel.simple() factory method
# ---------------------------------------------------------------------------


class TestDeprecatedFidelityKernelSimple:
    """Tests for the deprecated ``FidelityKernel.simple()`` factory method."""

    def test_simple_factory_method(self):
        """Test the simple FidelityKernel factory method."""
        with pytest.warns(DeprecationWarning, match="n_modes"):
            kernel = FidelityKernel.simple(input_size=2, n_modes=4)

        assert kernel.input_size == 2
        assert kernel.feature_map.circuit.m == 4
        assert len(kernel.input_state) == 4
        assert sum(kernel.input_state) == 2
        assert kernel.input_state == [1, 0, 1, 0]

    def test_simple_factory_warns_once_for_forwarded_n_modes(self):
        """FidelityKernel.simple suppresses duplicate forwarded n_modes warnings."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always", DeprecationWarning)
            kernel = FidelityKernel.simple(input_size=2, n_modes=4)

        messages = [str(warning.message) for warning in warning_list]
        n_modes_messages = [
            message
            for message in messages
            if "Parameter 'n_modes' is deprecated" in message
        ]

        assert len(n_modes_messages) == 1
        assert any(
            "FidelityKernel.simple() is deprecated" in message for message in messages
        )
        assert kernel.feature_map.circuit.m == 4

    def test_simple_factory_default_photons(self):
        """Test simple factory with default n_modes (should equal input_size + 1)."""
        kernel = FidelityKernel.simple(input_size=3)

        assert kernel.input_size == 3
        assert kernel.feature_map.circuit.m == 4  # input_size + 1
        assert sum(kernel.input_state) == 2
        assert kernel.input_state == [1, 0, 1, 0]

    def test_simple_num_photons_modes_and_input_state(self):
        for i in range(1, 15):
            kernel = FidelityKernel.simple(input_size=i)
            assert kernel.feature_map.circuit.m == i + 1
            assert np.sum(kernel.input_state) == int(np.ceil((i + 1) / 2))
            assert len(kernel.input_state) == i + 1

            input_state = [0] * (i + 1)
            for j in range(len(input_state)):
                if j % 2 == 0:
                    input_state[j] = 1
            assert kernel.input_state == input_state
        for i in range(1, 15):
            with pytest.warns(DeprecationWarning, match="n_modes"):
                kernel = FidelityKernel.simple(input_size=1, n_modes=i + 1)
            assert kernel.feature_map.circuit.m == i + 1
            assert np.sum(kernel.input_state) == int(np.ceil((i + 1) / 2))
            assert len(kernel.input_state) == i + 1

            input_state = [0] * (i + 1)
            for j in range(len(input_state)):
                if j % 2 == 0:
                    input_state[j] = 1
            assert kernel.input_state == input_state

    def test_simple_parameters(self):
        for i in range(1, 15):
            kernel = FidelityKernel.simple(input_size=i)
            params = list(kernel.parameters())
            named_params = [i[0] for i in kernel.named_parameters()]
            assert params[0].numel() == i * (i + 1)
            assert params[1].numel() == i * (i + 1)
            assert len(params) == 2
            assert kernel.feature_map.is_trainable
            assert "_quantum_layer.LI_simple" in named_params
            assert "_quantum_layer.RI_simple" in named_params


# ---------------------------------------------------------------------------
# Deprecated KernelCircuitBuilder
# ---------------------------------------------------------------------------


class TestDeprecatedKernelCircuitBuilder:
    """Tests for the deprecated ``KernelCircuitBuilder`` fluent API."""

    def test_builder_basic_usage(self):
        """Test basic KernelCircuitBuilder usage."""
        builder = KernelCircuitBuilder()
        feature_map = builder.input_size(2).n_modes(4).build_feature_map()

        assert feature_map.input_size == 2
        assert feature_map.circuit.m == 4

    def test_builder_with_device_and_dtype(self):
        """Test builder with device and dtype configuration."""
        device = torch.device("cpu")
        builder = KernelCircuitBuilder()
        feature_map = (
            builder
            .input_size(2)
            .n_modes(4)
            .device(device)
            .dtype(torch.float64)
            .build_feature_map()
        )

        assert feature_map.input_size == 2
        assert feature_map.device == device

    def test_builder_trainable_toggle(self):
        """Builder can enable or disable training dynamically."""
        builder = KernelCircuitBuilder()
        feature_map = (
            builder.input_size(2).n_modes(4).trainable(False).build_feature_map()
        )

        assert feature_map.input_size == 2
        assert not feature_map.is_trainable

        feature_map = (
            builder
            .input_size(2)
            .n_modes(4)
            .trainable(True, prefix="phi_")
            .build_feature_map()
        )

        assert feature_map.is_trainable
        assert "phi_" in feature_map.trainable_parameters

    def test_builder_build_fidelity_kernel(self):
        """Test building a FidelityKernel directly."""
        builder = KernelCircuitBuilder()
        kernel = builder.input_size(2).n_modes(4).build_fidelity_kernel()

        assert kernel.input_size == 2
        assert kernel.feature_map.circuit.m == 4
        assert len(kernel.input_state) == 4
        assert sum(kernel.input_state) == 2
        assert kernel.input_state == [1, 0, 1, 0]

    def test_builder_fidelity_kernel_with_custom_input_state(self):
        """Test building FidelityKernel with custom input state."""
        builder = KernelCircuitBuilder()
        custom_state = [2, 0, 0, 0]
        kernel = (
            builder
            .input_size(2)
            .n_modes(4)
            .build_fidelity_kernel(input_state=custom_state)
        )

        assert kernel.input_state == custom_state

    def test_builder_fidelity_kernel_with_shots(self):
        """Test building FidelityKernel with sampling configuration."""
        builder = KernelCircuitBuilder()
        kernel = (
            builder
            .input_size(2)
            .n_modes(4)
            .build_fidelity_kernel(
                shots=1000,
                sampling_method="multinomial",
                computation_space=ComputationSpace.UNBUNCHED,
            )
        )

        assert kernel.shots == 1000
        assert kernel.sampling_method == "multinomial"
        assert kernel.computation_space is ComputationSpace.UNBUNCHED

    def test_builder_default_values(self):
        """Test builder with default values for optional parameters."""
        builder = KernelCircuitBuilder()
        feature_map = builder.input_size(2).build_feature_map()

        assert feature_map.input_size == 2
        # Should use defaults: n_modes = max(input_size + 1, 4) = 4
        assert feature_map.circuit.m == 4

    def test_builder_angle_encoding_configuration(self):
        builder = KernelCircuitBuilder()
        feature_map = (
            builder
            .input_size(3)
            .n_modes(4)
            .angle_encoding(scale=0.5)
            .build_feature_map()
        )

        x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        encoded = feature_map._encode_x(x)

        assert encoded.shape == (3,)

        expected = torch.tensor([0.05, 0.1, 0.15], dtype=torch.float32)
        assert torch.allclose(encoded.detach(), expected, atol=1e-6)

    def test_builder_missing_input_size(self):
        """Test builder error when input_size is not specified."""
        builder = KernelCircuitBuilder()

        with pytest.raises(ValueError, match="Input size must be specified"):
            builder.n_modes(4).build_feature_map()

    def test_deprecation_warning_on_init(self):
        """KernelCircuitBuilder emits DeprecationWarning on instantiation."""
        with pytest.warns(DeprecationWarning, match="KernelCircuitBuilder is deprecated"):
            KernelCircuitBuilder()

    def test_deprecation_warning_build_feature_map(self):
        """build_feature_map emits DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="KernelCircuitBuilder"):
            KernelCircuitBuilder().input_size(2).build_feature_map()

    def test_deprecation_warning_build_fidelity_kernel(self):
        """build_fidelity_kernel emits DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="KernelCircuitBuilder"):
            KernelCircuitBuilder().input_size(2).n_modes(4).build_fidelity_kernel()


class TestDeprecatedConstructorConsistency:
    """Tests comparing deprecated kernel constructors with current constructors."""

    def test_feature_map_unitary_consistency(self):
        """Feature maps built through legacy APIs share the same topology."""
        x1, x2 = pcvl.P("x1"), pcvl.P("x2")
        manual_feature_map = FeatureMap(
            circuit=pcvl.Circuit(3)
            // pcvl.PS(x1)
            // pcvl.BS()
            // pcvl.PS(x2)
            // pcvl.BS(),
            input_size=2,
            input_parameters="x",
        )

        with pytest.warns(DeprecationWarning, match="n_modes"):
            simple_feature_map = FeatureMap.simple(input_size=2, n_modes=3)

        legacy_builder = KernelCircuitBuilder()
        builder_feature_map = legacy_builder.input_size(2).n_modes(3).build_feature_map()

        assert (
            manual_feature_map.input_size
            == simple_feature_map.input_size
            == builder_feature_map.input_size
        )
        assert (
            manual_feature_map.circuit.m
            == simple_feature_map.circuit.m
            == builder_feature_map.circuit.m
        )

    def test_kernel_computation_consistency(self):
        """Deprecated constructors yield kernels compatible with current kernels."""
        builder = CircuitBuilder(n_modes=4)
        builder.add_superpositions(depth=1, name="phi_1_")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_superpositions(depth=1, name="phi_2_")
        current_feature_map = FeatureMap(
            builder=builder,
            input_size=2,
            input_parameters=None,
        )
        current_kernel = FidelityKernel(
            feature_map=current_feature_map,
            input_state=[1, 1, 0, 0],
        )

        with pytest.warns(DeprecationWarning, match="n_modes"):
            simple_kernel = FidelityKernel.simple(
                input_size=2,
                n_modes=4,
            )

        legacy_builder = KernelCircuitBuilder()
        builder_kernel = (
            legacy_builder
            .input_size(2)
            .n_modes(4)
            .trainable(False)
            .build_fidelity_kernel()
        )

        assert (
            current_kernel.input_size
            == simple_kernel.input_size
            == builder_kernel.input_size
            == 2
        )
        assert (
            current_kernel.feature_map.circuit.m
            == simple_kernel.feature_map.circuit.m
            == builder_kernel.feature_map.circuit.m
            == 4
        )
        assert (
            len(current_kernel.input_state)
            == len(simple_kernel.input_state)
            == len(builder_kernel.input_state)
        )
        assert (
            sum(current_kernel.input_state)
            == sum(simple_kernel.input_state)
            == sum(builder_kernel.input_state)
        )

        X = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        for kernel in (current_kernel, simple_kernel, builder_kernel):
            result = kernel(X)
            assert result.shape == (1, 1)
            assert torch.isfinite(result).all()
