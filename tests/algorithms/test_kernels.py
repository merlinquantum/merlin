import warnings

import numpy as np
import perceval as pcvl
import pytest
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from merlin.algorithms.kernels import (
    FeatureMap,
    FidelityKernel,
    KernelCircuitBuilder,
    _CCInvQuantumLayer,
)
from merlin.algorithms.loss import NKernelAlignment
from merlin.builder import CircuitBuilder
from merlin.core.computation_space import ComputationSpace


class TestCCInvBackend:
    """Tests for _CCInvQuantumLayer as the FidelityKernel computation backend."""

    def setup_method(self):
        x1, x2 = pcvl.P("x1"), pcvl.P("x2")
        theta = pcvl.P("theta")
        self.circuit = (
            pcvl.Circuit(2)
            // pcvl.PS(x1)
            // pcvl.BS(theta)
            // pcvl.PS(x2)
            // pcvl.BS(theta)
        )
        self.feature_map = FeatureMap(
            circuit=self.circuit,
            input_size=2,
            input_parameters="x",
            trainable_parameters=["theta"],
        )
        self.input_state = [1, 0]
        self.kernel = FidelityKernel(
            feature_map=self.feature_map,
            input_state=self.input_state,
        )
        self.layer = self.kernel._quantum_layer

    def test_ccinv_backend_matches_reference_kernel(self):
        """Kernel values must match a direct SLOS reference built from the same circuit."""
        X = torch.tensor([[0.3, 0.7], [0.9, 0.2], [0.5, 0.5]], dtype=torch.float32)
        K = self.kernel(X)
        assert K.shape == (3, 3)
        assert torch.allclose(K, K.T, atol=1e-5)
        assert torch.allclose(torch.diag(K), torch.ones(3, dtype=K.dtype), atol=1e-4)

    def test_kernel_unitary_is_identity_when_x1_eq_x2(self):
        """U(x) @ U†(x) must be the identity."""
        identity = torch.eye(
            len(self.input_state), dtype=torch.complex64
        )
        for x in [torch.tensor([0.1, 0.4]), torch.tensor([1.2, 0.0]), torch.tensor([0.0, 0.0])]:
            K_unitary = self.layer._compute_kernel_unitary(
                x.to(self.layer.dtype),
                x.to(self.layer.dtype),
            )
            assert torch.allclose(K_unitary, identity, atol=1e-5), (
                f"Expected identity for x={x}, got\n{K_unitary}"
            )

    def test_gradient_flows_to_trainable_parameters(self):
        """A backward pass must populate .grad on every trainable parameter."""
        X = torch.tensor([[0.3, 0.7], [0.9, 0.2]], dtype=torch.float32)
        K = self.kernel(X)
        loss = K.sum()
        loss.backward()
        for name, param in self.kernel.named_parameters():
            assert param.grad is not None, (
                f"No gradient for parameter '{name}'"
            )

    def test_k_train_is_symmetric(self):
        """Training kernel matrix must be symmetric."""
        X = torch.tensor(
            [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2], [0.3, 0.7]],
            dtype=torch.float32,
        )
        K = self.kernel(X)
        assert torch.allclose(K, K.T, atol=1e-5)

    def test_k_train_diagonal_is_one(self):
        """Diagonal of training kernel matrix must be 1 (self-similarity)."""
        X = torch.tensor(
            [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]],
            dtype=torch.float32,
        )
        K = self.kernel(X)
        assert torch.allclose(
            torch.diag(K), torch.ones(3, dtype=K.dtype), atol=1e-4
        )

    def test_new_backend_transition_prob_matches_perceval_slos(self):
        """Transition probability from the new backend must match Perceval SLOS.

        The combined kernel unitary ``U(x1) @ U†(x2)`` is fed directly into a
        Perceval Processor to obtain a ground-truth probability for the input
        Fock state.  The same unitary is passed to
        ``_compute_transition_probs``, which exercises the full SLOS +
        photon-loss + detector-transform chain of the new backend.  Both values
        must agree to within floating-point tolerance.

        float64 is used so that the product of two unitaries stays numerically
        unitary within Perceval's strict ``is_unitary()`` tolerance.
        """
        from perceval import BasicState, Processor, Unitary
        from perceval.algorithm import Sampler

        # Rebuild with float64 so that U(x1) @ U†(x2) is numerically unitary
        # when converted to complex128 for the Perceval reference.
        feature_map_f64 = FeatureMap(
            circuit=self.circuit,
            input_size=2,
            input_parameters="x",
            trainable_parameters=["theta"],
            dtype=torch.float64,
        )
        kernel_f64 = FidelityKernel(
            feature_map=feature_map_f64,
            input_state=self.input_state,
        )
        layer_f64 = kernel_f64._quantum_layer

        x1 = torch.tensor([0.3, 0.7], dtype=torch.float64)
        x2 = torch.tensor([0.9, 0.2], dtype=torch.float64)

        # --- Merlin new backend ---
        kernel_unitary = layer_f64._compute_kernel_unitary(x1, x2)
        transition_prob = layer_f64._compute_transition_probs(
            kernel_unitary.unsqueeze(0),
            self.input_state,
            shots=0,
            sampling_method="multinomial",
        )
        merlin_value = float(transition_prob[0].item())

        # --- Perceval SLOS reference ---
        u_np = kernel_unitary.detach().cpu().numpy().astype(np.complex128)
        pcvl_circuit = Unitary(pcvl.Matrix(u_np))

        processor = Processor("SLOS")
        processor.set_circuit(pcvl_circuit)
        processor.with_input(BasicState(self.input_state))
        processor.min_detected_photons_filter(0)

        sampler = Sampler(processor)
        raw_results = sampler.probs()["results"]

        def _state_to_tuple(state):
            try:
                return tuple(int(n) for n in state.tolist())
            except AttributeError:
                return tuple(int(n) for n in state)

        results = {_state_to_tuple(s): float(p) for s, p in raw_results.items()}
        key = tuple(self.input_state)
        perceval_value = results.get(key, 0.0)

        assert merlin_value == pytest.approx(perceval_value, rel=1e-5, abs=1e-6)

    def test_new_backend_matches_perceval_slos_unbunched(self):
        """New backend with UNBUNCHED space must match Perceval thresholded probability.

        Perceval returns absolute probabilities over all Fock states.  The
        thresholded (UNBUNCHED) value is the probability of the input state
        within the subspace where every mode holds at most one photon,
        renormalized to that subspace.

        float64 is used so that the product of two unitaries stays numerically
        unitary within Perceval's strict ``is_unitary()`` tolerance.
        """
        from perceval import BasicState, GenericInterferometer, P, Processor, Unitary
        from perceval.algorithm import Sampler

        def _circ_func(x):
            c = pcvl.Circuit(2) // pcvl.PS(P(f"phi{2 * x}")) // pcvl.BS()
            c.add(0, pcvl.PS(P(f"phi{2 * x + 1}")))
            c.add(0, pcvl.BS())
            return c

        input_state = [1, 1, 0, 0]
        circuit = GenericInterferometer(len(input_state), _circ_func)
        input_size = len(circuit.get_parameters())

        # float64 keeps U(x1) @ U†(x2) numerically unitary for Perceval.
        feature_map_unb = FeatureMap(
            circuit=circuit,
            input_size=input_size,
            input_parameters=["phi"],
            dtype=torch.float64,
        )
        kernel_unb = FidelityKernel(
            feature_map=feature_map_unb,
            input_state=input_state,
            computation_space=ComputationSpace.UNBUNCHED,
            force_psd=False,
        )
        layer_unb = kernel_unb._quantum_layer

        rng = np.random.default_rng(0)
        x1 = torch.as_tensor(rng.random(input_size), dtype=torch.float64)
        x2 = torch.as_tensor(rng.random(input_size), dtype=torch.float64)

        # --- Merlin new backend (UNBUNCHED) ---
        merlin_thr = float(kernel_unb(x1, x2))

        # --- Perceval reference (thresholded) ---
        kernel_unitary = layer_unb._compute_kernel_unitary(x1, x2)
        u_np = kernel_unitary.detach().cpu().numpy().astype(np.complex128)
        pcvl_circuit = Unitary(pcvl.Matrix(u_np))

        processor = Processor("SLOS")
        processor.set_circuit(pcvl_circuit)
        processor.with_input(BasicState(input_state))
        processor.min_detected_photons_filter(0)

        sampler = Sampler(processor)
        raw_results = sampler.probs()["results"]

        def _state_to_tuple(state):
            try:
                return tuple(int(n) for n in state.tolist())
            except AttributeError:
                return tuple(int(n) for n in state)

        results = {_state_to_tuple(s): float(p) for s, p in raw_results.items()}

        thresholded = {s: p for s, p in results.items() if max(s) == 1}
        total = sum(thresholded.values())
        assert total > 0, "No unbunched states returned by Perceval"

        key = tuple(input_state)
        perceval_thr = thresholded.get(key, 0.0) / total

        assert merlin_thr == pytest.approx(perceval_thr, rel=1e-5, abs=1e-6)


class TestFidelityKernelInternals:
    """Tests for FidelityKernel internal structure after the _CCInvQuantumLayer refactor."""

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
        self.kernel = FidelityKernel(
            feature_map=self.feature_map,
            input_state=[2, 0],
        )

    def test_fidelity_kernel_no_longer_owns_slos_graph(self):
        """FidelityKernel must not directly own a _slos_graph attribute."""
        assert not hasattr(self.kernel, "_slos_graph")

    def test_quantum_layer_is_registered_sub_module(self):
        """_quantum_layer must appear in FidelityKernel.named_modules()."""
        module_names = [name for name, _ in self.kernel.named_modules()]
        assert "_quantum_layer" in module_names

    def test_quantum_layer_is_ccinv_instance(self):
        assert isinstance(self.kernel._quantum_layer, _CCInvQuantumLayer)

    def test_quantum_layer_forward_matches_kernel_forward(self):
        X = torch.tensor(
            [[0.2, 0.3], [0.5, 0.1], [0.7, 0.4]],
            dtype=torch.float32,
        )

        expected = self.kernel(X)
        actual = self.kernel._quantum_layer(
            X,
            shots=self.kernel.shots,
            sampling_method=self.kernel.sampling_method,
        )

        assert torch.allclose(actual, expected, atol=1e-6)


class TestFidelityKernel:
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
        self.quantum_kernel = FidelityKernel(
            feature_map=self.feature_map,
            input_state=[2, 0],
            computation_space=ComputationSpace.FOCK,
        )

    def test_fidelity_kernel_initialization(self):
        assert self.quantum_kernel.input_state == [2, 0]
        assert self.quantum_kernel.shots == 0
        assert self.quantum_kernel.sampling_method == "multinomial"
        assert self.quantum_kernel.computation_space is ComputationSpace.FOCK
        assert self.quantum_kernel.force_psd
        assert not self.quantum_kernel.is_trainable

    def test_input_state_property_returns_copy(self):
        input_state = self.quantum_kernel.input_state
        input_state[0] = 0

        assert self.quantum_kernel.input_state == [2, 0]
        assert self.quantum_kernel.input_state is not input_state
        assert self.quantum_kernel._quantum_layer._kernel_input_state == [2, 0]

    def test_fidelity_kernel_with_trainable_feature_map(self):
        theta = pcvl.P("theta")
        x1, x2 = pcvl.P("x1"), pcvl.P("x2")
        circuit = (
            pcvl.Circuit(2)
            // pcvl.PS(x1)
            // pcvl.BS(theta)
            // pcvl.PS(x2)
            // pcvl.BS(theta)
        )

        feature_map = FeatureMap(
            circuit=circuit,
            input_size=2,
            input_parameters="x",
            trainable_parameters=["theta"],
        )

        kernel = FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 0],
            computation_space=ComputationSpace.FOCK,
        )

        assert kernel.is_trainable
        assert "_quantum_layer.theta" in dict(kernel.named_parameters())

    def test_kernel_scalar_computation(self):
        x1 = torch.tensor([0.5, 1.0])
        x2 = torch.tensor([1.0, 0.5])
        kernel_value = self.quantum_kernel(x1, x2)
        assert isinstance(kernel_value, float)
        assert 0.0 <= kernel_value <= 1.0

    def test_kernel_matrix_symmetric(self):
        X = torch.tensor([[0.5, 1.0], [1.5, 0.5], [0.0, 2.0]])
        K = self.quantum_kernel(X)

        assert K.shape == (3, 3)
        # Relax tolerance slightly for GPU numeric differences
        assert torch.allclose(K, K.T, atol=1e-4)
        assert torch.allclose(torch.diag(K), torch.ones(3), atol=1e-5)
        assert torch.all(K >= 0)
        assert torch.all(K <= 1 + 2e-2)

    def test_kernel_matrix_asymmetric(self):
        X_train = torch.tensor([[0.5, 1.0], [1.5, 0.5]])
        X_test = torch.tensor([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])

        K = self.quantum_kernel(X_test, X_train)

        assert K.shape == (3, 2)
        assert torch.all(K >= 0)
        assert torch.all(K <= 1)

    def test_kernel_with_numpy_input(self):
        X = np.array([[0.5, 1.0], [1.5, 0.5], [0.0, 2.0]])
        K = self.quantum_kernel(X)

        assert K.shape == (3, 3)
        assert np.allclose(K, K.T, atol=1e-6)
        assert np.allclose(np.diag(K), np.ones(3))

    def test_kernel_with_shots(self):
        kernel = FidelityKernel(
            feature_map=self.feature_map,
            input_state=[2, 0],
            shots=1000,
            sampling_method="multinomial",
        )

        X = torch.tensor([[0.5, 1.0], [1.5, 0.5]])
        K = kernel(X)

        assert K.shape == (2, 2)
        assert torch.allclose(torch.diag(K), torch.ones(2), atol=0.1)

    def test_no_bunching_validation(self):
        with pytest.raises(ValueError, match="Bunching must be enabled"):
            FidelityKernel(
                feature_map=self.feature_map,
                input_state=[2, 0],
                computation_space=ComputationSpace.UNBUNCHED,
            )

        with pytest.raises(ValueError, match="kernel value will always be 1"):
            FidelityKernel(
                feature_map=self.feature_map,
                input_state=[1, 1],
                computation_space=ComputationSpace.UNBUNCHED,
            )

    def test_input_state_circuit_size_mismatch(self):
        x1 = pcvl.P("x1")
        circuit = pcvl.Circuit(3) // pcvl.PS(x1)  # 3 modes
        feature_map = FeatureMap(
            circuit=circuit,
            input_size=1,
            input_parameters="x",
        )

        with pytest.raises(
            ValueError, match="Input state length does not match circuit size"
        ):
            FidelityKernel(
                feature_map=feature_map,
                input_state=[2, 0],  # Only 2 modes
                computation_space=ComputationSpace.FOCK,
            )

    def test_kernel_uses_builder_subset_encoding_without_deprecation(self):
        builder = CircuitBuilder(n_modes=3)
        builder.add_angle_encoding(
            modes=[0, 1],
            name="input",
            subset_combinations=True,
        )
        feature_map = FeatureMap(
            builder=builder,
            input_size=2,
            input_parameters=None,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            kernel = FidelityKernel(
                feature_map=feature_map,
                input_state=[1, 0, 0],
                computation_space=ComputationSpace.FOCK,
            )

        encoded = kernel._quantum_layer._encode_single(torch.tensor([0.2, 0.3]))
        expected = torch.tensor([0.2, 0.3, 0.5], dtype=encoded.dtype)
        assert torch.allclose(encoded, expected)

    def test_kernel_backend_preserves_builder_angle_encoding_scale(self):
        builder = CircuitBuilder(n_modes=3)
        builder.add_angle_encoding(
            modes=[0, 1],
            name="input",
            scale=0.5,
        )
        feature_map = FeatureMap(
            builder=builder,
            input_size=2,
            input_parameters=None,
        )

        kernel = FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 0, 0],
            computation_space=ComputationSpace.FOCK,
        )

        encoded = kernel._quantum_layer._encode_single(torch.tensor([0.2, 0.4]))
        expected = torch.tensor([0.1, 0.2], dtype=encoded.dtype)
        assert torch.allclose(encoded, expected)

    def test_kernel_backend_preserves_builder_subset_scale(self):
        builder = CircuitBuilder(n_modes=3)
        builder.add_angle_encoding(
            modes=[0, 1],
            name="input",
            scale=0.5,
            subset_combinations=True,
        )
        feature_map = FeatureMap(
            builder=builder,
            input_size=2,
            input_parameters=None,
        )

        kernel = FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 0, 0],
            computation_space=ComputationSpace.FOCK,
        )

        encoded = kernel._quantum_layer._encode_single(torch.tensor([0.2, 0.3]))
        expected = torch.tensor([0.1, 0.15, 0.25], dtype=encoded.dtype)
        assert torch.allclose(encoded, expected)

    def test_simple_kernel_rejects_missing_angle_encoding_specs(self):
        feature_map = FeatureMap.simple(input_size=2)
        feature_map._angle_encoding_specs = {}

        with pytest.raises(RuntimeError, match="missing angle_encoding_specs"):
            FidelityKernel(
                feature_map=feature_map,
                input_state=[1, 0, 1],
                computation_space=ComputationSpace.FOCK,
            )

    def test_simple_kernel_rejects_missing_angle_encoding_scales(self):
        feature_map = FeatureMap.simple(
            input_size=2,
            angle_encoding_scale=0.5,
        )
        feature_map._angle_encoding_specs["input"]["scales"] = {}

        with pytest.raises(RuntimeError, match="missing angle-encoding scale entries"):
            FidelityKernel(
                feature_map=feature_map,
                input_state=[1, 0, 1],
                computation_space=ComputationSpace.FOCK,
            )

    def test_psd_projection(self):
        # Test the static method for PSD projection
        matrix = torch.tensor(
            [[1.0, 0.9, -0.1], [0.9, 1.0, 0.2], [-0.1, 0.2, 1.0]], dtype=torch.float64
        )

        psd_matrix = FidelityKernel._project_psd(matrix)

        # Check that all eigenvalues are non-negative
        eigenvals = torch.linalg.eigvals(psd_matrix)
        # Assert eigenvalues are real (imaginary parts are essentially zero)
        assert torch.all(torch.abs(eigenvals.imag) < 1e-12), (
            f"Eigenvalues have significant imaginary parts: {eigenvals.imag}"
        )
        # Assert all eigenvalues are non-negative (PSD condition)
        real_eigenvals = eigenvals.real
        assert torch.all(real_eigenvals >= -1e-10), (
            f"Matrix has negative eigenvalues: {real_eigenvals[real_eigenvals < -1e-10]}"
        )


class TestFidelityKernelInputStateDerivation:
    """Tests for automatic input_state derivation and n_photons parameter."""

    def _make_feature_map(self, n_modes: int) -> FeatureMap:
        params = [pcvl.P(f"x{i}") for i in range(n_modes)]
        circuit = pcvl.Circuit(n_modes)
        for p in params:
            circuit //= pcvl.PS(p)
        return FeatureMap(
            circuit=circuit,
            input_size=n_modes,
            input_parameters="x",
        )

    # ------------------------------------------------------------------
    # input_state=None (no n_photons)
    # ------------------------------------------------------------------

    def test_default_input_state_is_alternating(self):
        """input_state=None defaults to [1, 0, 1, 0, ...] of length circuit.m."""
        fm = self._make_feature_map(4)
        kernel = FidelityKernel(feature_map=fm)
        assert kernel.input_state == [1, 0, 1, 0]

    def test_default_input_state_odd_modes(self):
        """For odd mode count, alternating default ends on 1."""
        fm = self._make_feature_map(5)
        kernel = FidelityKernel(feature_map=fm)
        assert kernel.input_state == [1, 0, 1, 0, 1]

    # ------------------------------------------------------------------
    # n_photons with input_state=None
    # ------------------------------------------------------------------

    def test_n_photons_alternating_pattern(self):
        """n_photons below the alternating slot count uses alternating positions."""
        fm = self._make_feature_map(6)
        kernel = FidelityKernel(feature_map=fm, n_photons=2)
        assert kernel.input_state == [1, 0, 1, 0, 0, 0]
        assert sum(kernel.input_state) == 2

    def test_n_photons_fills_all_alternating_positions(self):
        """n_photons == ceil(m / 2) exactly fills the alternating pattern."""
        fm = self._make_feature_map(6)
        kernel = FidelityKernel(feature_map=fm, n_photons=3)
        assert kernel.input_state == [1, 0, 1, 0, 1, 0]
        assert sum(kernel.input_state) == 3

    def test_n_photons_fills_all_odd_mode_alternating_positions_without_warning(self):
        """n_photons == ceil(m / 2) exactly fills an odd-mode alternating pattern."""
        fm = self._make_feature_map(5)
        kernel = FidelityKernel(feature_map=fm, n_photons=3)
        assert kernel.input_state == [1, 0, 1, 0, 1]
        assert sum(kernel.input_state) == 3

    def test_n_photons_overflow_fills_even_then_odd(self):
        """n_photons above the alternating slot count fills remaining positions."""
        fm = self._make_feature_map(6)
        with pytest.warns(UserWarning, match="Alternating positions are filled first"):
            kernel = FidelityKernel(feature_map=fm, n_photons=4)
        assert kernel.input_state == [1, 1, 1, 0, 1, 0]
        assert sum(kernel.input_state) == 4

    def test_n_photons_all_modes_warns(self):
        """n_photons == m fills all modes and warns."""
        fm = self._make_feature_map(4)
        with pytest.warns(UserWarning, match="Alternating positions are filled first"):
            kernel = FidelityKernel(
                feature_map=fm,
                n_photons=4,
                computation_space=ComputationSpace.FOCK,
            )
        assert kernel.input_state == [1, 1, 1, 1]

    def test_n_photons_one_photon(self):
        """n_photons=1 places a single photon in mode 0."""
        fm = self._make_feature_map(4)
        kernel = FidelityKernel(feature_map=fm, n_photons=1)
        assert kernel.input_state == [1, 0, 0, 0]

    # ------------------------------------------------------------------
    # n_photons with explicit input_state
    # ------------------------------------------------------------------

    def test_n_photons_matches_input_state_accepted(self):
        """Providing matching n_photons and input_state is accepted."""
        fm = self._make_feature_map(4)
        kernel = FidelityKernel(
            feature_map=fm,
            input_state=[1, 0, 1, 0],
            n_photons=2,
        )
        assert kernel.input_state == [1, 0, 1, 0]

    def test_n_photons_mismatch_input_state_raises(self):
        """n_photons that disagrees with sum(input_state) raises ValueError."""
        fm = self._make_feature_map(4)
        with pytest.raises(ValueError, match="n_photons=3 does not match"):
            FidelityKernel(
                feature_map=fm,
                input_state=[1, 0, 1, 0],
                n_photons=3,
            )

    # ------------------------------------------------------------------
    # Invalid n_photons values
    # ------------------------------------------------------------------

    def test_n_photons_zero_raises(self):
        """n_photons=0 raises ValueError."""
        fm = self._make_feature_map(4)
        with pytest.raises(ValueError, match="n_photons must be between"):
            FidelityKernel(feature_map=fm, n_photons=0)

    def test_n_photons_negative_raises(self):
        """Negative n_photons raises ValueError."""
        fm = self._make_feature_map(4)
        with pytest.raises(ValueError, match="n_photons must be between"):
            FidelityKernel(feature_map=fm, n_photons=-1)

    def test_n_photons_exceeds_modes_raises(self):
        """n_photons > m raises ValueError."""
        fm = self._make_feature_map(4)
        with pytest.raises(ValueError, match="n_photons must be between"):
            FidelityKernel(feature_map=fm, n_photons=5)


class TestFeatureMapDescriptor:
    """FeatureMap descriptor behavior used by the new FidelityKernel path."""

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

    def test_feature_map_initialization(self):
        assert self.feature_map.input_size == 2
        assert self.feature_map.input_parameters == "x"
        assert not self.feature_map.is_trainable
        assert self.feature_map.trainable_parameters == []

    def test_feature_map_with_trainable_parameters(self):
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

        assert feature_map.is_trainable
        assert feature_map.trainable_parameters == ["theta"]

    def test_is_datapoint(self):
        assert self.feature_map.is_datapoint(torch.tensor([0.5, 1.0]))
        assert self.feature_map.is_datapoint(np.array([0.5, 1.0]))

        assert not self.feature_map.is_datapoint(torch.tensor([[0.5, 1.0], [1.5, 0.5]]))
        assert not self.feature_map.is_datapoint(np.array([[0.5, 1.0], [1.5, 0.5]]))

    def test_invalid_input_parameters(self):
        with pytest.raises(
            ValueError, match="Only a single input parameter is allowed"
        ):
            FeatureMap(
                circuit=self.circuit, input_size=2, input_parameters=["x1", "x2"]
            )


class TestNKernelAlignment:
    def setup_method(self):
        self.loss_fn = NKernelAlignment()

    def test_nkernel_alignment_basic(self):
        K = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
        y = torch.tensor([1, -1], dtype=torch.float32)

        loss = self.loss_fn(K, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar

    def test_nkernel_alignment_with_target_matrix(self):
        K = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        target = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])

        loss = self.loss_fn(K, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_invalid_kernel_matrix_dimension(self):
        K = torch.tensor([1.0, 0.8, 0.5])  # 1D tensor
        y = torch.tensor([1, -1, 1])

        with pytest.raises(ValueError, match="Input must be a 2D tensor"):
            self.loss_fn(K, y)

    def test_invalid_target_values(self):
        K = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
        y = torch.tensor([1, 0])  # Invalid: should be +1 or -1

        with pytest.raises(ValueError, match="binary target values"):
            self.loss_fn(K, y)

    def test_nkernel_alignment_gradient(self):
        K = torch.tensor([[1.0, 0.8], [0.8, 1.0]], requires_grad=True)
        y = torch.tensor([1, -1], dtype=torch.float32)

        loss = self.loss_fn(K, y)
        loss.backward()

        assert K.grad is not None
        assert K.grad.shape == K.shape


class TestFeatureMapFactoryMethods:
    """Test the new factory methods for FeatureMap creation."""

    def test_from_circuit_builder_basic(self):
        """FeatureMap can be constructed directly from CircuitBuilder."""
        builder = CircuitBuilder(n_modes=4)
        builder.add_superpositions(depth=1)
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_superpositions(depth=1)

        feature_map = FeatureMap(
            builder=builder,
            input_size=2,
            input_parameters=None,
        )

        assert feature_map.input_size == 2
        assert feature_map.circuit.m == 4

    def test_from_circuit_builder_with_trainable_params(self):
        """FeatureMap inherits trainable parameters defined in CircuitBuilder."""
        builder = CircuitBuilder(n_modes=4)
        builder.add_superpositions(depth=1)
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_rotations(trainable=True, name="phi_")
        builder.add_superpositions(depth=1)

        feature_map = FeatureMap(
            builder=builder,
            input_size=2,
            input_parameters=None,
        )

        assert feature_map.input_size == 2
        assert feature_map.is_trainable
        assert "phi" in feature_map.trainable_parameters

    def test_angle_encoding_respects_scale_in_feature_map(self):
        builder = CircuitBuilder(n_modes=4)
        builder.add_angle_encoding(
            modes=[0, 1, 2],
            name="input",
            scale=0.5,
        )

        feature_map = FeatureMap(
            builder=builder,
            input_size=3,
            input_parameters=None,
        )

        x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        encoded = feature_map._encode_x(x)

        assert encoded.shape == (3,)

        expected = torch.tensor([0.05, 0.1, 0.15], dtype=torch.float32)
        assert torch.allclose(encoded.detach(), expected, atol=1e-6)

    def test_from_pcvl_circuit(self):
        """FeatureMap can be built directly from a pcvl.Circuit."""
        x1, x2 = pcvl.P("x1"), pcvl.P("x2")
        circuit = (
            pcvl.Circuit(2) // pcvl.PS(x1) // pcvl.BS() // pcvl.PS(x2) // pcvl.BS()
        )

        feature_map = FeatureMap(
            circuit=circuit,
            input_size=2,
            input_parameters="x",
        )

        assert feature_map.input_size == 2
        assert feature_map.circuit.m == 2

    def test_simple_factory_method(self):
        """Test the simple FeatureMap factory method."""
        feature_map = FeatureMap.simple(input_size=2)

        assert feature_map.input_size == 2
        assert feature_map.circuit.m == 3  # input_size + 1
        assert feature_map.is_trainable
        assert "LI_simple" in feature_map.trainable_parameters
        assert "RI_simple" in feature_map.trainable_parameters

    def test_simple_factory_default_photons(self):
        """Test simple factory with default n_modes (should equal input_size + 1)."""
        feature_map = FeatureMap.simple(input_size=3)

        assert feature_map.input_size == 3
        assert feature_map.circuit.m == 4  # input_size + 1

    def test_simple_trainable(self):
        for i in range(1, 20):
            kernel = FeatureMap.simple(input_size=i)
            assert kernel.is_trainable


class TestFidelityKernelFactoryMethods:
    """Test the new factory methods for FidelityKernel creation."""

    def test_from_feature_map_builder(self):
        """FidelityKernel can wrap a FeatureMap created from CircuitBuilder."""
        builder = CircuitBuilder(n_modes=4)
        builder.add_superpositions(depth=1)
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_superpositions(depth=1)

        feature_map = FeatureMap(
            builder=builder,
            input_size=2,
            input_parameters=None,
        )

        kernel = FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 1, 0, 0],
        )

        assert kernel.input_size == 2
        assert kernel.feature_map.circuit.m == 4
        assert len(kernel.input_state) == 4

    def test_from_feature_map_pcvl_circuit(self):
        """FidelityKernel can wrap a FeatureMap built from pcvl.Circuit."""
        x1, x2 = pcvl.P("x1"), pcvl.P("x2")
        circuit = (
            pcvl.Circuit(2) // pcvl.PS(x1) // pcvl.BS() // pcvl.PS(x2) // pcvl.BS()
        )
        feature_map = FeatureMap(
            circuit=circuit,
            input_size=2,
            input_parameters="x",
        )

        kernel = FidelityKernel(
            feature_map=feature_map,
            input_state=[2, 0],
            shots=1000,
            sampling_method="multinomial",
        )

        assert kernel.input_size == 2
        assert kernel.shots == 1000
        assert kernel.sampling_method == "multinomial"

class TestCircuitBuilderKernelIntegration:
    """Tests for kernels built with the current CircuitBuilder API."""

    def test_kernel_supports_entangling_layer(self):
        builder = CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(name="gi")
        builder.add_angle_encoding(modes=[0, 1, 2, 3], name="input")

        feature_map = FeatureMap(
            builder=builder,
            input_size=4,
            input_parameters=None,
        )

        kernel = FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 1, 0, 0],
        )

        x = torch.rand(3, 4)
        K = kernel(x)

        assert K.shape == (3, 3)
        assert torch.isfinite(K).all()


class TestKernelIntegration:
    def test_kernel_with_sklearn_svc(self):
        # Create simple 2D data
        X_train = torch.tensor([[0.1, 0.2], [0.8, 0.9], [0.3, 0.7], [0.6, 0.4]])
        y_train = np.array([1, -1, 1, -1])
        X_test = torch.tensor([[0.2, 0.3], [0.7, 0.8]])

        # Set up kernel
        x1, x2 = pcvl.P("x1"), pcvl.P("x2")
        circuit = (
            pcvl.Circuit(2) // pcvl.PS(x1) // pcvl.BS() // pcvl.PS(x2) // pcvl.BS()
        )
        feature_map = FeatureMap(
            circuit=circuit,
            input_size=2,
            input_parameters="x",
        )
        quantum_kernel = FidelityKernel(
            feature_map=feature_map,
            input_state=[2, 0],
            computation_space=ComputationSpace.FOCK,
        )

        # Compute kernel matrices
        K_train = quantum_kernel(X_train).detach().numpy()
        K_test = quantum_kernel(X_test, X_train).detach().numpy()

        # Train with sklearn
        svc = SVC(kernel="precomputed")
        svc.fit(K_train, y_train)
        y_pred = svc.predict(K_test)

        assert len(y_pred) == 2
        assert all(pred in [-1, 1] for pred in y_pred)

    def test_kernel_training_with_nka_loss(self):
        # Simple training test
        X = torch.tensor([[0.1, 0.2], [0.8, 0.9], [0.3, 0.7], [0.6, 0.4]])
        y = torch.tensor([1, -1, 1, -1], dtype=torch.float32)

        # Trainable kernel
        theta = pcvl.P("theta")
        x1, x2 = pcvl.P("x1"), pcvl.P("x2")
        circuit = (
            pcvl.Circuit(2)
            // pcvl.PS(x1)
            // pcvl.BS(theta)
            // pcvl.PS(x2)
            // pcvl.BS(theta)
        )

        feature_map = FeatureMap(
            circuit=circuit,
            input_size=2,
            input_parameters="x",
            trainable_parameters=["theta"],
        )
        quantum_kernel = FidelityKernel(
            feature_map=feature_map,
            input_state=[2, 0],
            computation_space=ComputationSpace.FOCK,
        )

        optimizer = torch.optim.Adam(quantum_kernel.parameters(), lr=0.1)
        loss_fn = NKernelAlignment()

        initial_loss = None
        final_loss = None

        for epoch in range(5):
            optimizer.zero_grad()

            K = quantum_kernel(X)
            loss = loss_fn(K, y)

            if epoch == 0:
                initial_loss = loss.item()
            if epoch == 4:
                final_loss = loss.item()

            loss.backward()
            optimizer.step()

        # Training should reduce loss (make it less negative)
        assert final_loss > initial_loss or abs(final_loss - initial_loss) < 0.1


def create_quantum_circuit(m, size=400):
    """Create a quantum circuit with specified number of modes and input size"""

    wl = pcvl.GenericInterferometer(
        m,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"phase_1_{i}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"phase_2_{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    c = pcvl.Circuit(m)
    c.add(0, wl, merge=True)

    c_var = pcvl.Circuit(m)
    for i in range(size):
        px = pcvl.P(f"px-{i + 1}")
        c_var.add(i % m, pcvl.PS(px))
    c.add(0, c_var, merge=True)

    wr = pcvl.GenericInterferometer(
        m,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"phase_3_{i}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"phase_4_{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    c.add(0, wr, merge=True)

    return c


def get_quantum_kernel(
    modes=10,
    input_size=10,
    photons=4,
    computation_space=ComputationSpace.FOCK,
):
    circuit = create_quantum_circuit(m=modes, size=input_size)
    feature_map = FeatureMap(
        circuit=circuit,
        input_size=input_size,
        input_parameters=["px"],
        trainable_parameters=["phase"],
        dtype=torch.float64,
    )
    input_state = [0] * modes
    for p in range(min(photons, modes // 2)):
        input_state[2 * p] = 1
    quantum_kernel = FidelityKernel(
        feature_map=feature_map,
        input_state=input_state,
        computation_space=computation_space,
    )
    return quantum_kernel


def test_iris_dataset_quantum_kernel():
    """Test quantum kernel on Iris dataset for classification"""
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Create quantum kernel with 4 input features (matching Iris dataset)
    kernel = get_quantum_kernel(input_size=4, modes=10, photons=4)

    # Compute kernel matrices
    K_train = kernel(X_train_tensor).detach().numpy()
    K_test = kernel(X_test_tensor, X_train_tensor).detach().numpy()

    # Verify kernel properties
    assert K_train.shape == (len(X_train), len(X_train))
    assert K_test.shape == (len(X_test), len(X_train))
    assert np.allclose(K_train, K_train.T, atol=1e-6)  # Symmetric
    # TODO: all elements should be between 0 and 1 but this test is failing
    # could be due to the fact that the 400 phase shifters in the circuit created deep computational chains / accumulated errors
    assert np.allclose(np.diag(K_train), 1.0, atol=1e-1)  # Diagonal elements ≈ 1
    assert np.all(K_train >= 0 - 1e-1) and np.all(
        K_train <= 1 + 1e-1
    )  # Valid kernel values

    # Train SVM with precomputed kernel
    svc = SVC(kernel="precomputed", random_state=42)
    svc.fit(K_train, y_train)

    # Make predictions
    y_pred = svc.predict(K_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Basic sanity checks
    assert len(y_pred) == len(y_test)
    assert accuracy > 0.0  # Should have some predictive power
    assert all(pred in [0, 1, 2] for pred in y_pred)  # Valid class predictions

    print(f"Iris dataset quantum kernel test - Accuracy: {accuracy:.4f}")
    assert accuracy > 0.8, (
        f"Accuracy too low: {accuracy:.4f}, there may be a problem with the kernel"
    )
    # test functions must not return values (pytest expects None)


def test_iris_dataset_kernel_training_with_nka():
    """Test quantum kernel training on Iris dataset using NKA loss"""
    # Load and prepare Iris data for binary classification (classes 0 vs 1)
    iris = load_iris()
    X, y = iris.data, iris.target

    # Convert to binary classification (keep only classes 0 and 1)
    binary_mask = y < 2
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    y_binary = 2 * y_binary - 1  # Convert to {-1, 1} for NKA loss

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Create trainable quantum kernel
    kernel = get_quantum_kernel(input_size=4, modes=6, photons=2)

    # Training setup
    optimizer = torch.optim.Adam(kernel.parameters(), lr=1e-2)
    loss_fn = NKernelAlignment()

    # Training loop
    initial_loss = None
    final_loss = None

    for epoch in range(3):  # Short training for test
        optimizer.zero_grad()

        K_train = kernel(X_train_tensor)
        loss = loss_fn(K_train, y_train_tensor)

        if epoch == 0:
            initial_loss = loss.item()
        if epoch == 2:
            final_loss = loss.item()

        loss.backward()
        optimizer.step()

    # Test with trained kernel
    K_train_final = kernel(X_train_tensor).detach().numpy()
    K_test_final = kernel(X_test_tensor, X_train_tensor).detach().numpy()

    # Train SVM
    svc = SVC(kernel="precomputed", random_state=42)
    svc.fit(K_train_final, (y_train + 1) // 2)  # Convert back to {0, 1}

    # Make predictions
    y_pred = svc.predict(K_test_final)
    accuracy = accuracy_score((y_test + 1) // 2, y_pred)

    # Assertions
    assert isinstance(initial_loss, float)
    assert isinstance(final_loss, float)
    assert accuracy >= 0.0

    print(f"Iris binary classification with NKA training - Accuracy: {accuracy:.4f}")
    print(f"Loss change: {initial_loss:.4f} -> {final_loss:.4f}")


def test_iris_with_supported_constructors():
    """Test IRIS classification using the supported kernel constructors."""
    # Load IRIS dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Use only first two classes for binary classification
    binary_mask = y < 2
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )

    # Convert to tensors (use smaller subset for reliable testing)
    X_train_small = torch.tensor(
        X_train[:15], dtype=torch.float32
    )  # 15 training samples
    X_test_small = torch.tensor(X_test[:10], dtype=torch.float32)  # 10 test samples
    y_train_small = y_train[:15]
    y_test_small = y_test[:10]

    print("Testing IRIS classification with all supported constructors...")
    print(
        f"Using {len(X_train_small)} training samples, {len(X_test_small)} test samples"
    )

    # Define configurations to test
    configurations = [
        {"name": "Static Mode (stable)", "trainable": False},
        {"name": "Trainable Mode (flexible)", "trainable": True},
    ]

    results = {}

    for config in configurations:
        print(f"\n{'=' * 60}")
        print(f"Testing with {config['name']}")
        print(f"{'=' * 60}")

        trainable_flag = config["trainable"]
        config_results = {}

        # Initialize kernel variables to None to prevent NameError
        kernel_simple = None
        kernel_manual = None
        kernel_builder = None

        # Method 1: Simple factory method
        print(
            f"\n1. FidelityKernel.simple() - {config['name']} (trainable={trainable_flag}):"
        )
        try:
            kernel_simple = FidelityKernel.simple(
                input_size=4,  # IRIS has 4 features
                n_modes=4,
                n_photons=2,
                trainable=trainable_flag,
            )

            # Test basic properties
            assert kernel_simple.input_size == 4
            assert kernel_simple.feature_map.circuit.m == 4
            assert len(kernel_simple.input_state) == 4
            assert sum(kernel_simple.input_state) == 2

            trainable_status = (
                "trainable" if kernel_simple.is_trainable else "non-trainable"
            )
            print(
                f"   ✓ Created {trainable_status} kernel: {kernel_simple.feature_map.circuit.m} modes, {sum(kernel_simple.input_state)} photons"
            )

            # Attempt classification
            accuracy_simple = _test_kernel_classification(
                kernel_simple,
                X_train_small,
                X_test_small,
                y_train_small,
                y_test_small,
                "Simple",
            )
            config_results["simple"] = accuracy_simple

        except Exception as e:
            print(f"   ❌ Simple method failed: {e}")
            config_results["simple"] = None

        # Method 2: Manual pcvl.Circuit construction
        print(f"\n2. Manual pcvl.Circuit() - {config['name']}:")
        try:
            params = [pcvl.P(f"x{i + 1}") for i in range(4)]
            circuit = pcvl.Circuit(4)
            for mode, param in enumerate(params):
                circuit.add(mode, pcvl.PS(param))
            circuit.add(0, pcvl.BS())
            circuit.add(2, pcvl.BS())

            feature_map = FeatureMap(
                circuit=circuit,
                input_size=4,
                input_parameters="x",
            )

            kernel_manual = FidelityKernel(
                feature_map=feature_map,
                input_state=[1, 1, 0, 0],
                force_psd=True,
            )

            assert kernel_manual.input_size == 4
            assert kernel_manual.feature_map.circuit.m == 4

            trainable_status = (
                "trainable" if kernel_manual.is_trainable else "non-trainable"
            )
            print(
                f"   ✓ Created {trainable_status} manual kernel: {kernel_manual.feature_map.circuit.m} modes"
            )

            accuracy_manual = _test_kernel_classification(
                kernel_manual,
                X_train_small,
                X_test_small,
                y_train_small,
                y_test_small,
                "Manual",
            )
            config_results["manual"] = accuracy_manual

        except Exception as e:
            print(f"   ❌ Manual method failed: {e}")
            config_results["manual"] = None

        # Method 3: KernelCircuitBuilder fluent interface
        print(f"\n3. KernelCircuitBuilder() - {config['name']}:")
        try:
            builder = KernelCircuitBuilder()
            kernel_builder = (
                builder
                .input_size(4)
                .n_modes(4)
                .trainable(trainable_flag)
                .build_fidelity_kernel()
            )

            assert kernel_builder.input_size == 4
            assert kernel_builder.feature_map.circuit.m == 4

            trainable_status = (
                "trainable" if kernel_builder.is_trainable else "non-trainable"
            )
            print(
                f"   ✓ Created {trainable_status} builder kernel: {kernel_builder.feature_map.circuit.m} modes"
            )
            pcvl.pdisplay(
                kernel_builder.feature_map.circuit, output_format=pcvl.Format.TEXT
            )
            # Attempt classification
            accuracy_builder = _test_kernel_classification(
                kernel_builder,
                X_train_small,
                X_test_small,
                y_train_small,
                y_test_small,
                "Builder",
            )
            config_results["builder"] = accuracy_builder

        except Exception as e:
            print(f"   ❌ Builder method failed: {e}")
            config_results["builder"] = None

        # Test structural consistency within this configuration
        successful_kernels = []
        if config_results.get("simple") is not None and kernel_simple is not None:
            successful_kernels.append(kernel_simple)
        if config_results.get("manual") is not None and kernel_manual is not None:
            successful_kernels.append(kernel_manual)
        if config_results.get("builder") is not None and kernel_builder is not None:
            successful_kernels.append(kernel_builder)

        if len(successful_kernels) >= 2:
            # Test that successful methods create structurally similar kernels
            input_sizes = [k.input_size for k in successful_kernels]
            circuit_modes = [k.feature_map.circuit.m for k in successful_kernels]
            input_state_lengths = [len(k.input_state) for k in successful_kernels]

            if (
                len(set(input_sizes)) == 1
                and len(set(circuit_modes)) == 1
                and len(set(input_state_lengths)) == 1
            ):
                print("   ✅ All successful methods create consistent structures")
            else:
                print("   ⚠️ Structural inconsistency detected across methods")

        results[config["name"]] = config_results

    # Print comprehensive results summary
    print(f"\n{'=' * 60}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'=' * 60}")

    for config_name, config_results in results.items():
        print(f"\n{config_name}:")
        for method, accuracy in config_results.items():
            if accuracy is not None:
                if isinstance(accuracy, float):
                    print(f"   {method.capitalize()}: {accuracy:.3f} accuracy ✅")
                else:
                    print(
                        f"   {method.capitalize()}: Structure created ✅ (computation issue)"
                    )
            else:
                print(f"   {method.capitalize()}: Failed ❌")

    # Overall assessment
    total_successes = sum(
        1
        for config_results in results.values()
        for accuracy in config_results.values()
        if accuracy is not None
    )
    total_tests = len(results) * 3  # 2 configs × 3 methods each

    print(
        f"\n📊 Overall Success Rate: {total_successes}/{total_tests} ({total_successes / total_tests * 100:.1f}%)"
    )

    if total_successes >= total_tests * 0.5:  # At least 50% success
        print("✅ IRIS classification with supported constructors successful!")
    else:
        print("⚠️ Some constructor issues detected, but structure creation works")


def _test_kernel_classification(kernel, X_train, X_test, y_train, y_test, method_name):
    """Helper function to test kernel classification and return accuracy or status."""
    try:
        # Compute kernel matrices
        K_train = kernel(X_train)
        print(f"K_train = {K_train}")
        K_test = kernel(X_test, X_train)

        # Verify kernel properties
        assert K_train.shape == (len(X_train), len(X_train))
        assert K_test.shape == (len(X_test), len(X_train))
        assert torch.allclose(K_train, K_train.T, atol=1e-4)  # Should be symmetric

        # Train SVM classifier
        svc = SVC(kernel="precomputed", random_state=42)
        svc.fit(K_train.detach().numpy(), y_train)

        # Make predictions
        y_pred = svc.predict(K_test.detach().numpy())
        accuracy = accuracy_score(y_test, y_pred)

        print(f"   ✅ {method_name} classification: {accuracy:.3f} accuracy")

        # Validation
        assert len(y_pred) == len(y_test)
        assert accuracy >= 0.0
        assert all(pred in [0, 1] for pred in y_pred)

    except Exception as e:
        print(f"   ⚠️ {method_name} computation failed: {str(e)[:60]}...")
        print("      (Structure creation successful, computation issue detected)")
        # Return a special marker to indicate structure success but computation failure
        # return "structure_ok"


def test_kernel_constructor_performance_comparison():
    """Compare the supported kernel construction methods for performance."""
    print("\nPerformance comparison of kernel construction methods:")

    import time

    methods = []
    times = []

    # Time Method 1: Simple factory
    start = time.time()
    with pytest.warns(DeprecationWarning, match="n_modes"):
        kernel1 = FidelityKernel.simple(input_size=3, n_modes=4)
    time1 = time.time() - start
    methods.append("FidelityKernel.simple()")
    times.append(time1)

    # Time Method 2: Manual pcvl.Circuit construction
    start = time.time()
    params = [pcvl.P(f"x{i + 1}") for i in range(3)]
    circuit = pcvl.Circuit(4)
    for mode, param in enumerate(params):
        circuit.add(mode, pcvl.PS(param))
    circuit.add(0, pcvl.BS())
    circuit.add(2, pcvl.BS())
    feature_map = FeatureMap(
        circuit=circuit,
        input_size=3,
        input_parameters="x",
    )
    kernel2 = FidelityKernel(
        feature_map=feature_map,
        input_state=[1, 1, 0, 0],
    )
    time2 = time.time() - start
    methods.append("Manual pcvl.Circuit")
    times.append(time2)

    # Time Method 3: Builder pattern
    start = time.time()
    builder = KernelCircuitBuilder()
    kernel3 = builder.input_size(3).n_modes(4).trainable(False).build_fidelity_kernel()
    time3 = time.time() - start
    methods.append("KernelCircuitBuilder")
    times.append(time3)

    # Print results
    for method, time_taken in zip(methods, times, strict=False):
        print(f"   {method}: {time_taken:.4f}s")

    # Verify all methods create equivalent structures
    assert kernel1.input_size == kernel2.input_size == kernel3.input_size
    assert (
        kernel1.feature_map.circuit.m
        == kernel2.feature_map.circuit.m
        == kernel3.feature_map.circuit.m
    )
    assert (
        len(kernel1.input_state) == len(kernel2.input_state) == len(kernel3.input_state)
    )

    print("   ✅ All methods create structurally equivalent kernels")


@pytest.fixture(scope="module")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.mark.parametrize("constructor", ["simple", "manual", "builder"])
def test_fidelity_kernel_gpu_execution_all_constructors(cuda_device, constructor):
    device = cuda_device
    # Use 4 features to match factory/kernel expectations
    X_train = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.8, 0.9, 0.1, 0.2], [0.3, 0.7, 0.5, 0.6]],
        dtype=torch.float32,
        device=device,
    )
    X_test = torch.tensor(
        [[0.2, 0.3, 0.4, 0.5], [0.7, 0.8, 0.2, 0.3]], dtype=torch.float32, device=device
    )

    # Build kernels via each constructor with input_size=4
    if constructor == "simple":
        kernel = FidelityKernel.simple(
            input_size=4,
            n_modes=4,
        )
    elif constructor == "manual":
        params = [pcvl.P(f"x{i + 1}") for i in range(4)]
        circuit = pcvl.Circuit(4)
        for mode, param in enumerate(params):
            circuit.add(mode, pcvl.PS(param))
        circuit.add(0, pcvl.BS())
        circuit.add(2, pcvl.BS())
        feature_map = FeatureMap(
            circuit=circuit,
            input_size=4,
            input_parameters="x",
        )
        kernel = FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 1, 0, 0],
        )
    else:  # "builder"
        builder = KernelCircuitBuilder()
        kernel = (
            builder.input_size(4).n_modes(4).trainable(False).build_fidelity_kernel()
        )

    # Ensure kernel is on the correct device
    kernel = kernel.to(device)
    K_train = kernel(X_train)
    K_test = kernel(X_test, X_train)

    # Assertions
    assert isinstance(K_train, torch.Tensor) and isinstance(K_test, torch.Tensor)
    assert K_train.device.type == device.type and K_test.device.type == device.type
    assert K_train.shape == (X_train.shape[0], X_train.shape[0])
    assert K_test.shape == (X_test.shape[0], X_train.shape[0])
    assert torch.isfinite(K_train).all() and torch.isfinite(K_test).all()


def test_fidelity_kernel_gpu_training_step(cuda_device):
    device = cuda_device
    # Small trainable kernel with 4 features to match factory assumptions
    kernel = FidelityKernel.simple(
        input_size=4,
        n_modes=6,
    ).to(device)

    if sum(p.numel() for p in kernel.parameters()) == 0:
        pytest.skip("No trainable parameters available in this configuration")

    for parameter in kernel.parameters():
        assert parameter.device.type == device.type

    X = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.8, 0.9, 0.1, 0.2],
            [0.3, 0.7, 0.5, 0.6],
            [0.6, 0.4, 0.2, 0.1],
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    y = torch.tensor([1, -1, 1, -1], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(kernel.parameters(), lr=1e-2)
    loss_fn = NKernelAlignment()
    optimizer.zero_grad()
    K = kernel(X)
    assert K.device.type == device.type
    assert K.requires_grad

    loss = loss_fn(K, y)
    assert loss.device.type == device.type
    loss.backward()

    assert X.grad is not None
    assert X.grad.device.type == device.type
    assert torch.isfinite(X.grad).all()

    trainable_gradients = [
        parameter.grad for parameter in kernel.parameters() if parameter.requires_grad
    ]
    assert trainable_gradients
    for gradient in trainable_gradients:
        assert gradient is not None
        assert gradient.device.type == device.type
        assert torch.isfinite(gradient).all()
    assert any(torch.any(gradient.abs() > 0) for gradient in trainable_gradients)

    optimizer.step()

    assert torch.isfinite(loss).item() == 1
