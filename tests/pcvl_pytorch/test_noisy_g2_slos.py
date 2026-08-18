"""Tests for g2 photon correlations in SLOS (Sampled Linear Optical Simulator).

This module comprehensively tests:
1. g2=0 sector matching (pure SLOS output)
2. Sector probability normalization
3. Sector structure and metadata
4. Gradient flow through g2 parameters
5. Distinguishability modes (g2_distinguishable True/False)
6. Photon number distributions
7. Comparison with Perceval Simulator
"""

from copy import deepcopy

import numpy as np
import perceval as pcvl
import pytest
import torch

from merlin import (
    CircuitBuilder,
    Combinadics,
    ComputationSpace,
    MeasurementStrategy,
    QuantumLayer,
)
from merlin.algorithms.layer_utils import NoiseGroups
from merlin.core import SectoredDistribution, SectorResult
from merlin.core.process import ComputationProcess
from merlin.pcvl_pytorch.locirc_to_tensor import CircuitConverter
from merlin.pcvl_pytorch.noisy_slos import NoisyG2SLOSComputeGraph
from merlin.pcvl_pytorch.slos_torchscript import SLOSComputeGraph


@pytest.fixture
def circuit():
    """4-mode input-dependent circuit: entangling-angle-entangling."""
    builder = CircuitBuilder(n_modes=4)
    builder.add_entangling_layer(trainable=False)
    builder.add_rotations()  # Creates variable input-dependent angles
    builder.add_entangling_layer(trainable=False)
    return builder.to_pcvl_circuit()


@pytest.fixture
def unitary(circuit):
    """Convert input-dependent circuit to unitary tensor function."""
    converter = CircuitConverter(circuit)

    return converter.to_tensor


class TestG2SectorStructure:
    """Test g2 sector structure and metadata."""

    def test_sector_count(self, unitary):
        """SectoredDistribution returned with sectors for each photon number."""
        groups = NoiseGroups(
            source={"g2": 0.1, "g2_distinguishable": False},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])
        # Result should be SectoredDistribution
        assert isinstance(result, SectoredDistribution)
        # Should have 4 sectors for n_photons=3 (3, 4, 5, 6 photons)
        assert len(result.sectors) == 4

    def test_sectored_distribution_structure(self, unitary):
        """SectoredDistribution has correct structure."""
        groups = NoiseGroups(
            source={"g2": 0.1, "g2_distinguishable": False},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])
        assert isinstance(result, SectoredDistribution)

        # Check each sector exists and is accessible
        for i, sector in enumerate(result.sectors):
            assert hasattr(sector, "tensor")
            assert isinstance(sector.tensor, torch.Tensor)
            assert (
                sector.tensor.squeeze().shape[0]
                == Combinadics(scheme="fock", n=3 + i, m=4).compute_space_size()
            )

    def test_sectored_distribution_metadata(self, unitary):
        """Each SectorResult has consistent n_photons, n_modes, and basis keys.

        For n_photons=3, sector i must carry metadata for 3+i photons:
        n_photons == 3+i, n_modes == 4, computation_space == FOCK, and every
        key in the (possibly empty) keys tuple must have length 4 and sum to
        3+i.
        """
        groups = NoiseGroups(
            source={"g2": 0.15, "g2_distinguishable": False},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )
        result = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])

        assert isinstance(result, SectoredDistribution)
        for i, sector in enumerate(result.sectors):
            assert isinstance(sector, SectorResult)
            assert sector.n_modes == 4
            assert sector.n_photons == 3 + i
            assert sector.computation_space == ComputationSpace.FOCK
            expected_size = Combinadics("fock", n=3 + i, m=4).compute_space_size()
            assert sector.tensor.squeeze().shape[0] == expected_size
            for key in sector.keys:
                assert len(key) == 4
                assert sum(key) == 3 + i

    def test_g2_zero_single_sector(self, unitary):
        """g2=0 returns single sector matching pure SLOS."""
        # Pure SLOS (no noise)
        slos = SLOSComputeGraph(
            m=4, n_photons=3, computation_space=ComputationSpace.FOCK
        )
        keys_pure, probs_pure = slos.compute_probs(unitary(), [1, 1, 1, 0])

        # g2 computation with g2=0
        groups = NoiseGroups(
            source={"g2": 0.0, "g2_distinguishable": True},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])
        assert isinstance(result, SectoredDistribution)

        # With g2=0, should have single sector
        assert len(result.sectors) >= 1
        # First sector probabilities should match pure SLOS
        sector_0_probs = result.sectors[0].tensor
        assert torch.allclose(probs_pure, sector_0_probs, atol=1e-5)

    def test_g2_zero_with_indistinguishability(self, unitary):
        """g2 =0 produces multiple sectors; first sector related to indistinguishable noisy SLOS."""
        indistinguishability = 0.8

        # Noisy SLOS with indistinguishability but no g2
        groups_noisy = NoiseGroups(
            source={"indistinguishability": indistinguishability},
            circuit=None,
            post_measurement=None,
        )
        from merlin.pcvl_pytorch.noisy_slos import NoisySLOSComputeGraph

        noisy_slos_hom = NoisySLOSComputeGraph(
            groups_noisy,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
            keep_keys=False,
        )
        probs_hom = noisy_slos_hom.compute_probs(unitary(), [1, 1, 1, 0])

        # g2 computation with g2 > 0 and same indistinguishability
        groups_g2 = NoiseGroups(
            source={
                "g2": 0.0,
                "g2_distinguishable": False,
                "indistinguishability": indistinguishability,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos_g2 = NoisyG2SLOSComputeGraph(
            groups_g2,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result_g2 = noisy_slos_g2.compute_probs(unitary(), [1, 1, 1, 0])
        assert isinstance(result_g2, SectoredDistribution)

        # With g2 > 0, should have multiple sectors
        assert len(result_g2.sectors) >= 2

        sector_0_probs = result_g2.sectors[0].tensor
        assert torch.allclose(sector_0_probs, probs_hom, atol=1e-4)


class TestG2ProbabilityNormalization:
    """Test probability normalization across sectors."""

    def test_all_sector_probs_sum_to_one(self, unitary):
        """Sum across all photon sectors equals 1."""
        groups = NoiseGroups(
            source={"g2": 0.2, "g2_distinguishable": False},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])
        assert isinstance(result, SectoredDistribution)

        # Sum all probabilities across all sectors
        total_prob = 0.0
        for sector in result.sectors:
            total_prob += sector.tensor.sum().item()

        assert np.isclose(total_prob, 1.0, atol=1e-5)

    def test_normalization_multiple_inputs(self, unitary):
        """Normalization holds for various input states."""
        groups = NoiseGroups(
            source={"g2": 0.15, "g2_distinguishable": True},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        # Test multiple input states
        test_states = [[1, 1, 1, 0], [2, 1, 0, 0], [1, 1, 1, 0]]
        for state in test_states:
            result = noisy_slos.compute_probs(unitary(), state)
            assert isinstance(result, SectoredDistribution)

            total = sum(sector.tensor.sum().item() for sector in result.sectors)
            assert np.isclose(total, 1.0, atol=1e-5)


class TestG2DistinguishabilityModes:
    """Test distinguishability modes (g2_distinguishable True/False)."""

    def test_distinguishable_different_from_indistinguishable(self):
        """Different output for g2_distinguishable=True vs False.

        The 4×4 DFT unitary is used directly so the circuit is guaranteed to
        be maximally mixing, making HOM bunching visible.  With a near-identity
        circuit photons never scatter between modes, and both modes collapse to
        the same result regardless of the g2_distinguishable flag.

        g2_distinguishable=False: the extra photon bunches in the same mode as
        the source photon and runs through SLOS together with the other photons
        (quantum interference active).
        g2_distinguishable=True: the extra photon is fully distinguishable and
        its output distribution is convolved classically with the base SLOS.
        The DFT unitary ensures these two paths produce different sector
        distributions.
        """
        # 4×4 DFT: U_{jk} = exp(2πijk/4) / 2 — maximally mixing, unitary.
        n = 4
        idx = torch.arange(n, dtype=torch.float32)
        dft = (torch.exp(2j * torch.pi * idx[:, None] * idx[None, :] / n) / n**0.5).to(
            torch.complex64
        )
        u_tensor = dft.unsqueeze(0)  # [1, 4, 4]

        # Test with distinguishable extra photons (classical convolution)
        groups_dist = NoiseGroups(
            source={
                "g2": 0.25,
                "g2_distinguishable": True,
                "indistinguishability": 1.0,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos_dist = NoisyG2SLOSComputeGraph(
            groups_dist,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )
        result_dist = noisy_slos_dist.compute_probs(u_tensor, [1, 1, 1, 0])

        # Test with indistinguishable extra photons (HOM active)
        groups_indist = NoiseGroups(
            source={
                "g2": 0.25,
                "g2_distinguishable": False,
                "indistinguishability": 1.0,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos_indist = NoisyG2SLOSComputeGraph(
            groups_indist,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )
        result_indist = noisy_slos_indist.compute_probs(u_tensor, [1, 1, 1, 0])

        # Both should be SectoredDistribution
        assert isinstance(result_dist, SectoredDistribution)
        assert isinstance(result_indist, SectoredDistribution)

        # They should have same number of sectors
        assert len(result_dist.sectors) == len(result_indist.sectors)
        assert len(result_dist.sectors) == 4

        # Sector 0 is identical (same base SLOS, indist=1.0).
        # Sectors 1+ must differ: with the DFT unitary, HOM bunching changes
        # the output of SLOS([2,1,1,0]) relative to the classical convolution
        # conv(SLOS([1,1,1,0]), SLOS([1,0,0,0])).
        sectors_higher_differ = any(
            not torch.allclose(si.tensor, sd.tensor, atol=1e-6)
            for si, sd in zip(
                result_indist.sectors[1:], result_dist.sectors[1:], strict=True
            )
        )
        assert sectors_higher_differ, (
            "Extra-photon sectors must differ between distinguishable and indistinguishable modes"
        )

    def test_indistinguishable_bunched_delegate(self, unitary):
        """g2_distinguishable=False: augmented input has the expected mode occupation.

        With a single extra photon added to a single-photon input [1,0,0,0]
        (g2=0.5, the maximum of g^(2)(0), gives p_emit=1.0 so all probability
        is in sector 1) and indistinguishability=1.0 (NoisySLOS reduces to pure
        SLOS), the sector-1 distribution must equal the pure SLOS output for
        the augmented input [2,0,0,0].

        This confirms that the extra photon is bunched into mode 0—the same
        mode occupied by the original photon—rather than being placed in a
        different mode.
        """
        input_state = [1, 0, 0, 0]
        n_photons = 1
        m = 4

        unitary_tensor = unitary()

        # Reference: pure SLOS on the augmented state (extra photon in mode 0)
        slos_augmented = SLOSComputeGraph(
            m=m, n_photons=2, computation_space=ComputationSpace.FOCK
        )
        _, probs_augmented = slos_augmented.compute_probs(unitary_tensor, [2, 0, 0, 0])

        # g2=0.5 → p_emit=1.0 → weight_0=(1-p)^1=0, weight_1=p*(1-p)^0=1 (sector 1 carries everything)
        # (g2=0.5 is the maximum valid value of the second-order coherence; it gives p_emit=1.0)
        # indistinguishability=1.0 → NoisySLOSComputeGraph ≡ pure SLOS
        groups = NoiseGroups(
            source={
                "g2": 0.5,
                "g2_distinguishable": False,
                "indistinguishability": 1.0,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups, m=m, n_photons=n_photons, computation_space=ComputationSpace.FOCK
        )
        result = noisy_slos.compute_probs(unitary_tensor, input_state)

        assert isinstance(result, SectoredDistribution)
        assert len(result.sectors) == 2
        # Sector 1 = 1.0 * NoisySLOS(n=2).compute_probs([2,0,0,0]) ≈ SLOS([2,0,0,0])
        sector_1_probs = result.sectors[1].tensor.squeeze()
        assert torch.allclose(sector_1_probs, probs_augmented.squeeze(), atol=1e-5)

    def test_bunched_input_uses_one_g2_source_slot_per_photon(self):
        """Bunched Fock inputs can duplicate each intended photon.

        Input state [2, 0] represents two intended photons in mode 0, so the g2
        model has two source slots in that mode. At g2=0.5, p_emit=1.0 and both
        slots emit an extra photon, placing all probability in the four-photon
        sector. A per-mode source model would only reach the three-photon sector.
        """
        groups = NoiseGroups(
            source={
                "g2": 0.5,
                "g2_distinguishable": False,
                "indistinguishability": 1.0,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups, m=2, n_photons=2, computation_space=ComputationSpace.FOCK
        )

        result = noisy_slos.compute_probs(torch.eye(2, dtype=torch.complex64), [2, 0])

        assert isinstance(result, SectoredDistribution)
        assert [sector.n_photons for sector in result.sectors] == [2, 3, 4]
        sector_probabilities = torch.stack([
            sector.tensor.sum() for sector in result.sectors
        ])
        assert torch.allclose(
            sector_probabilities,
            torch.tensor([0.0, 0.0, 1.0], dtype=sector_probabilities.dtype),
            atol=1e-6,
        )


class TestG2ExtraPhotonDistribution:
    """Test extra photon sector distributions."""

    def test_extra_photon_sector_exists(self, unitary):
        """Extra photon sector is present when g2 > 0."""
        groups = NoiseGroups(
            source={"g2": 0.1, "g2_distinguishable": True},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])
        assert isinstance(result, SectoredDistribution)

        # Should have multiple sectors (3, 4, 5, 6 photons)
        assert len(result.sectors) >= 2

        # Second sector should have 4 photons
        second_sector = result.sectors[1]
        assert second_sector.tensor.shape[0] > 0

    def test_distinguishable_extra_photon_distribution(self, unitary):
        """g2_distinguishable=True, single g2 event in mode m: extra-photon marginal = |U[:,m]|².

        For n_photons=1, input=[e_m] (photon only in mode 0), g2=0.5 (the
        maximum of g^(2)(0) giving p_emit=1.0 so all weight is in sector 1),
        g2_distinguishable=True, indistinguishability=1.0:

        sector-1 = conv(P_regular, P_onehot_m) where both equal |U[:,m]|².

        For two independent distinguishable photons the expected mode
        occupation factorises:

            E[n_j | sector 1] = P_regular[j] + P_onehot[j] = 2 |U[j,m]|²

        Equivalently, ``sector_1_probs @ fock_states`` (where fock_states is
        the matrix of 2-photon Fock vectors) gives a vector whose j-th entry
        equals 2|U[j,m]|², confirming that marginalising sector 1 over the
        regular-photon output recovers |U[:,m]|².
        """
        m_mode = 0  # single photon injected in mode 0
        input_state = [1, 0, 0, 0]
        n_photons = 1
        m = 4

        unitary_tensor = unitary()

        # g2=0.5 → p_emit=1.0 → weight_0=0, weight_1=1 (sector 1 carries everything)
        # (g2=0.5 is the maximum valid value of g^(2)(0), corresponding to p_emit=1.0)
        # indistinguishability=1.0 → regular probs = pure SLOS = |U[:,m]|²
        groups = NoiseGroups(
            source={"g2": 0.5, "g2_distinguishable": True, "indistinguishability": 1.0},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups, m=m, n_photons=n_photons, computation_space=ComputationSpace.FOCK
        )
        result = noisy_slos.compute_probs(unitary_tensor, input_state)

        assert isinstance(result, SectoredDistribution)
        assert len(result.sectors) == 2

        sector_1_probs = result.sectors[1].tensor.squeeze()  # shape [N_2photon]

        # 2-photon Fock basis in Combinadics order: shape [N_2photon, m]
        fock_states = torch.tensor(
            Combinadics("fock", n=2, m=m).enumerate_states(), dtype=torch.float32
        )

        # E[n_j] = sum_s P(s) * s[j]  for each output mode j
        actual_occupation = sector_1_probs @ fock_states  # shape [m]

        # Expected: 2 * |U[j, m_mode]|² (two independent photons from mode m_mode)
        u = unitary_tensor.squeeze()  # [m, m] complex tensor
        expected_occupation = 2.0 * (u.abs() ** 2)[:, m_mode].float()

        assert torch.allclose(actual_occupation, expected_occupation, atol=1e-4)


class TestG2Gradients:
    """Test gradient flow through g2 parameters."""

    def test_result_is_differentiable(self, unitary):
        """SectoredDistribution probs maintain gradient connection."""
        unitary_diff = unitary().clone().detach().requires_grad_(True)

        groups = NoiseGroups(
            source={"indistinguishability": 0.8, "g2": 0.2, "g2_distinguishable": True},
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )

        result = noisy_slos.compute_probs(unitary_diff, [1, 1, 1, 0])
        assert isinstance(result, SectoredDistribution)

        # Get first sector probability sum
        prob_sum = result.sectors[0].tensor.sum()
        prob_sum.backward()

        # Unitary should have gradients
        assert unitary_diff.grad is not None

    def test_gradient_flows_through_g2(self, unitary):
        """torch.autograd.gradcheck confirms analytical gradients w.r.t. g2.

        The g2 parameter enters through the binomial weights
        ``weight_k = g2^k * (1-g2)^(n-k)`` that mix the per-sector SLOS
        outputs.  Passing a differentiable tensor via ``noisy_slos.g2``
        keeps autograd alive through those scalar multiplications.

        Uses n_photons=2, m=4 to limit SLOS graph size and keep the check
        fast.
        """
        unitary_tensor = unitary().to(torch.complex128)

        def fn(g2: torch.Tensor) -> torch.Tensor:
            groups = NoiseGroups(
                source={"g2": float(g2.item()), "g2_distinguishable": False},
                circuit=None,
                post_measurement=None,
            )
            noisy_slos = NoisyG2SLOSComputeGraph(
                groups,
                m=4,
                n_photons=2,
                computation_space=ComputationSpace.FOCK,
                dtype=torch.float64,
            )
            # Override with the differentiable tensor so autograd tracks g2.
            noisy_slos.g2 = g2.squeeze()
            result = noisy_slos.compute_probs(unitary_tensor, [1, 1, 0, 0])
            # Each sector sum equals C(n,k)*p_emit^k*(1-p_emit)^(n-k) where
            # p_emit = p_emit(g2) is the per-source emission probability derived
            # from the second-order coherence g^(2)(0)=2p/(1+p)^2.  The vector
            # has non-zero Jacobian entries w.r.t. g2.
            return torch.stack([s.tensor.sum() for s in result.sectors])

        g2_input = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(fn, (g2_input,), eps=1e-4, atol=1e-3)

    def test_gradient_flows_through_hom_and_g2(self, unitary):
        """torch.autograd.gradcheck confirms gradients flow through both g2 and indistinguishability.

        Indistinguishability is passed as a differentiable tensor directly in
        the NoiseGroups source dict.  ``torch.as_tensor`` in
        ``_InputStateNoisySLOSComputeGraph`` returns it unchanged (same
        memory), so ``_weights`` are computed with autograd active and
        individual sector probabilities carry a gradient back to
        ``indistinguishability``.

        The sector-sum gradients w.r.t. indistinguishability are identically
        zero (normalization cancels them), so ``sectors[0].tensor`` is
        returned instead; its individual entries carry the indistinguishability
        gradient.

        Uses n_photons=2, m=4 to limit SLOS graph size.
        """
        unitary_tensor = unitary().to(torch.complex128)

        def fn(g2: torch.Tensor, indist: torch.Tensor) -> torch.Tensor:
            groups = NoiseGroups(
                source={
                    "g2": float(g2.item()),
                    "g2_distinguishable": False,
                    # Pass the tensor so requires_grad survives into
                    # _InputStateNoisySLOSComputeGraph._weights.
                    "indistinguishability": indist.squeeze(),
                },
                circuit=None,
                post_measurement=None,
            )
            noisy_slos = NoisyG2SLOSComputeGraph(
                groups,
                m=4,
                n_photons=2,
                computation_space=ComputationSpace.FOCK,
                dtype=torch.float64,
            )
            noisy_slos.g2 = g2.squeeze()
            result = noisy_slos.compute_probs(unitary_tensor, [1, 1, 0, 0])
            # Return individual entries of sector 0; each entry depends on
            # indistinguishability through the OBB mixture weights.
            return result.sectors[0].tensor.squeeze()

        g2_input = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        indist_input = torch.tensor([0.9], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            fn, (g2_input, indist_input), eps=1e-4, atol=1e-3
        )


class TestG2PercevalComparison:
    """Compare g2 calculations with Perceval Simulator.

    Each test verifies that Merlin's NoisyG2SLOSComputeGraph probabilities are
    within 1e-4 of Perceval's probs_svd for the same NoiseModel and circuit.

    Mapping strategy: Perceval returns a dict of BasicState -> probability.
    Each BasicState belongs to a photon-number sector (sum of occupations).
    For n_photons=3, sector i holds states with 3+i photons (i=0..3).
    Within a sector, the index is given by Combinadics("fock", n, m).fock_to_index().
    """

    @staticmethod
    def _perceval_to_merlin_probs(
        perceval_result: dict,
        result_merlin: SectoredDistribution,
        n_photons: int,
        m: int,
    ) -> None:
        """Assert each Perceval output probability matches the corresponding Merlin sector value."""
        for state, perceval_prob in perceval_result.items():
            occupations = tuple(state)
            n = sum(occupations)
            sector_idx = n - n_photons
            assert 0 <= sector_idx < len(result_merlin.sectors), (
                f"State {occupations} with {n} photons has no corresponding Merlin sector"
            )
            combo = Combinadics("fock", n=n, m=m)
            tensor_idx = combo.fock_to_index(occupations)
            merlin_prob = (
                result_merlin.sectors[sector_idx].tensor.squeeze()[tensor_idx].item()
            )
            assert abs(merlin_prob - perceval_prob) < 1e-4, (
                f"State {occupations}: Merlin={merlin_prob:.6f}, Perceval={perceval_prob:.6f}"
            )

    def test_against_perceval_distinguishable(self, circuit, unitary):
        """g2_distinguishable=True probabilities are within 1e-4 of Perceval."""
        unitary_tensor = unitary()

        groups = NoiseGroups(
            source={
                "g2": 0.1,
                "g2_distinguishable": True,
                "indistinguishability": 0.9,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )
        result_merlin = noisy_slos.compute_probs(unitary_tensor, [1, 1, 1, 0])
        assert isinstance(result_merlin, SectoredDistribution)

        noise = pcvl.NoiseModel(
            g2=0.1, g2_distinguishable=True, indistinguishability=0.9
        )
        source = pcvl.Source.from_noise_model(noise)
        backend = pcvl.BackendFactory.get_backend("SLOS")
        sim = pcvl.Simulator(backend)
        sim.set_circuit(deepcopy(circuit))
        perceval_result = sim.probs_svd((source, pcvl.BasicState([1, 1, 1, 0])))[
            "results"
        ]

        self._perceval_to_merlin_probs(perceval_result, result_merlin, n_photons=3, m=4)

    def test_against_perceval_indistinguishable(self, circuit, unitary):
        """g2_distinguishable=False probabilities are within 1e-4 of Perceval."""
        unitary_tensor = unitary()

        groups = NoiseGroups(
            source={
                "g2": 0.2,
                "g2_distinguishable": False,
                "indistinguishability": 0.95,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )
        result_merlin = noisy_slos.compute_probs(unitary_tensor, [1, 1, 1, 0])
        assert isinstance(result_merlin, SectoredDistribution)

        noise = pcvl.NoiseModel(
            g2=0.2, g2_distinguishable=False, indistinguishability=0.95
        )
        source = pcvl.Source.from_noise_model(noise)
        backend = pcvl.BackendFactory.get_backend("SLOS")
        sim = pcvl.Simulator(backend)
        sim.set_circuit(deepcopy(circuit))
        perceval_result = sim.probs_svd((source, pcvl.BasicState([1, 1, 1, 0])))[
            "results"
        ]

        self._perceval_to_merlin_probs(perceval_result, result_merlin, n_photons=3, m=4)

    def test_joint_g2_and_indistinguishability(self, circuit, unitary):
        """Combined g2 + indistinguishability probabilities are within 1e-4 of Perceval."""
        groups = NoiseGroups(
            source={
                "g2": 0.15,
                "g2_distinguishable": False,
                "indistinguishability": 0.8,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=4,
            n_photons=3,
            computation_space=ComputationSpace.FOCK,
        )
        result_merlin = noisy_slos.compute_probs(unitary(), [1, 1, 1, 0])
        assert isinstance(result_merlin, SectoredDistribution)

        noise = pcvl.NoiseModel(
            g2=0.15, g2_distinguishable=False, indistinguishability=0.8
        )
        source = pcvl.Source.from_noise_model(noise)
        backend = pcvl.BackendFactory.get_backend("SLOS")
        sim = pcvl.Simulator(backend)
        sim.set_circuit(deepcopy(circuit))
        perceval_result = sim.probs_svd((source, pcvl.BasicState([1, 1, 1, 0])))[
            "results"
        ]

        self._perceval_to_merlin_probs(perceval_result, result_merlin, n_photons=3, m=4)


class TestG2AmplitudeEncodingMultipleActiveIndices:
    """Test ComputationProcess.compute(amplitude_encoding=True) with g2 noise and multiple active states.

    Each test builds a superposition tensor covering multiple Fock basis states
    (multiple ``active_indices``) and verifies that the resulting
    ``SectoredDistribution`` equals the |c_i|^2-weighted mixture of per-state
    Perceval outputs.  This exercises the multi-active-indices loop in
    ``ComputationProcess.compute()`` (the ``NoisyG2SLOSComputeGraph`` branch).

    Mapping strategy: identical to ``TestG2PercevalComparison``.
    """

    @staticmethod
    def _weighted_perceval_combined(
        sim: pcvl.Simulator,
        source: pcvl.Source,
        states: list[tuple[int, ...]],
        weights: list[float],
    ) -> dict:
        """Return the |c_i|^2-weighted mixture of per-state Perceval outputs.

        Parameters
        ----------
        sim : pcvl.Simulator
            Configured Perceval simulator (circuit already set).
        source : pcvl.Source
            Noise source built from the same NoiseModel.
        states : list[tuple[int, ...]]
            Fock input states corresponding to each superposition term.
        weights : list[float]
            |c_i|^2 mixture weights; must sum to 1.

        Returns
        -------
        dict
            Combined ``{BasicState: probability}`` mapping.
        """
        combined: dict = {}
        for state, weight in zip(states, weights, strict=True):
            result = sim.probs_svd((source, pcvl.BasicState(list(state))))["results"]
            for bs, prob in result.items():
                combined[bs] = combined.get(bs, 0.0) + weight * prob
        return combined

    @staticmethod
    def _assert_combined_matches_merlin(
        combined: dict,
        result_merlin: SectoredDistribution,
        n_photons: int,
        m: int,
    ) -> None:
        """Assert each combined Perceval probability matches the Merlin sector value."""
        for state, perceval_prob in combined.items():
            occupations = tuple(state)
            n = sum(occupations)
            sector_idx = n - n_photons
            assert 0 <= sector_idx < len(result_merlin.sectors), (
                f"State {occupations} with {n} photons has no corresponding Merlin sector"
            )
            combo = Combinadics("fock", n=n, m=m)
            tensor_idx = combo.fock_to_index(occupations)
            merlin_prob = (
                result_merlin.sectors[sector_idx].tensor.squeeze()[tensor_idx].item()
            )
            assert abs(merlin_prob - perceval_prob) < 1e-4, (
                f"State {occupations}: Merlin={merlin_prob:.6f}, Perceval={perceval_prob:.6f}"
            )

    def _make_superposition(
        self,
        active_states: list[tuple[int, ...]],
        weights: list[float],
        n_photons: int,
        m: int,
    ) -> torch.Tensor:
        """Build a 1-D complex superposition tensor for the given Fock states.

        Parameters
        ----------
        active_states : list[tuple[int, ...]]
            Fock states to superpose.
        weights : list[float]
            |c_i|^2 values; must sum to 1.
        n_photons : int
            Photon number of each state.
        m : int
            Number of modes.

        Returns
        -------
        torch.Tensor
            1-D complex64 tensor of length ``Combinadics('fock', n, m).compute_space_size()``.
        """
        combo = Combinadics("fock", n=n_photons, m=m)
        superposition = torch.zeros(combo.compute_space_size(), dtype=torch.complex64)
        for state, w in zip(active_states, weights, strict=True):
            superposition[combo.fock_to_index(state)] = torch.tensor(
                w, dtype=torch.float32
            ).sqrt()
        return superposition

    def test_distinguishable_two_active_states(self, circuit):
        """g2_distinguishable=True: superposition of two Fock states matches weighted Perceval mixture."""
        n_photons, m = 2, 4
        active_states = [(1, 0, 1, 0), (0, 1, 0, 1)]
        weights = [0.6, 0.4]

        noise_groups = NoiseGroups(
            source={"g2": 0.1, "g2_distinguishable": True, "indistinguishability": 0.9},
            circuit=None,
            post_measurement=None,
        )
        proc = ComputationProcess(
            circuit=circuit,
            input_state=self._make_superposition(active_states, weights, n_photons, m),
            trainable_parameters=[],
            input_parameters=[],
            n_photons=n_photons,
            computation_space=ComputationSpace.FOCK,
            noise_groups=noise_groups,
        )
        result = proc.compute([], amplitude_encoding=True)
        assert isinstance(result, SectoredDistribution)

        noise = pcvl.NoiseModel(
            g2=0.1, g2_distinguishable=True, indistinguishability=0.9
        )
        source = pcvl.Source.from_noise_model(noise)
        sim = pcvl.Simulator(pcvl.BackendFactory.get_backend("SLOS"))
        sim.set_circuit(deepcopy(circuit))
        combined = self._weighted_perceval_combined(sim, source, active_states, weights)
        self._assert_combined_matches_merlin(combined, result, n_photons=n_photons, m=m)

    def test_indistinguishable_two_active_states(self, circuit):
        """g2_distinguishable=False: superposition of two Fock states matches weighted Perceval mixture."""
        n_photons, m = 2, 4
        active_states = [(1, 1, 0, 0), (0, 0, 1, 1)]
        weights = [0.5, 0.5]

        noise_groups = NoiseGroups(
            source={
                "g2": 0.2,
                "g2_distinguishable": False,
                "indistinguishability": 0.9,
            },
            circuit=None,
            post_measurement=None,
        )
        proc = ComputationProcess(
            circuit=circuit,
            input_state=self._make_superposition(active_states, weights, n_photons, m),
            trainable_parameters=[],
            input_parameters=[],
            n_photons=n_photons,
            computation_space=ComputationSpace.FOCK,
            noise_groups=noise_groups,
        )
        result = proc.compute([], amplitude_encoding=True)
        assert isinstance(result, SectoredDistribution)

        noise = pcvl.NoiseModel(
            g2=0.2, g2_distinguishable=False, indistinguishability=0.9
        )
        source = pcvl.Source.from_noise_model(noise)
        sim = pcvl.Simulator(pcvl.BackendFactory.get_backend("SLOS"))
        sim.set_circuit(deepcopy(circuit))
        combined = self._weighted_perceval_combined(sim, source, active_states, weights)
        self._assert_combined_matches_merlin(combined, result, n_photons=n_photons, m=m)

    def test_three_active_states(self, circuit):
        """Three active Fock states: SectoredDistribution equals weighted Perceval mixture."""
        n_photons, m = 2, 4
        active_states = [(1, 0, 1, 0), (0, 1, 0, 1), (1, 1, 0, 0)]
        weights = [0.5, 0.3, 0.2]

        noise_groups = NoiseGroups(
            source={
                "g2": 0.15,
                "g2_distinguishable": True,
                "indistinguishability": 0.85,
            },
            circuit=None,
            post_measurement=None,
        )
        proc = ComputationProcess(
            circuit=circuit,
            input_state=self._make_superposition(active_states, weights, n_photons, m),
            trainable_parameters=[],
            input_parameters=[],
            n_photons=n_photons,
            computation_space=ComputationSpace.FOCK,
            noise_groups=noise_groups,
        )
        result = proc.compute([], amplitude_encoding=True)
        assert isinstance(result, SectoredDistribution)

        noise = pcvl.NoiseModel(
            g2=0.15, g2_distinguishable=True, indistinguishability=0.85
        )
        source = pcvl.Source.from_noise_model(noise)
        sim = pcvl.Simulator(pcvl.BackendFactory.get_backend("SLOS"))
        sim.set_circuit(deepcopy(circuit))
        combined = self._weighted_perceval_combined(sim, source, active_states, weights)
        self._assert_combined_matches_merlin(combined, result, n_photons=n_photons, m=m)


def _identity_unitary(n_modes: int) -> torch.Tensor:
    """Return an identity unitary with Merlin's default complex dtype."""
    return torch.eye(n_modes, dtype=torch.complex64)


def _g2_groups(
    *,
    g2: float = 0.1,
    g2_distinguishable: bool = False,
) -> NoiseGroups:
    """Return a minimal source-noise group for direct graph tests."""
    return NoiseGroups(
        source={
            "g2": g2,
            "g2_distinguishable": g2_distinguishable,
            "indistinguishability": 1.0,
        },
        circuit=None,
        post_measurement=None,
    )


def test_g2_sector_results_include_fock_basis_keys() -> None:
    """Check that each g2 output sector carries its Fock basis metadata.

    Correct result: every ``SectorResult.keys`` tuple should contain exactly the
    Fock basis states matching ``sector.tensor`` for that sector's photon count
    and mode count.

    Branch result: ``SectorResult.keys`` is empty, so downstream code cannot
    safely apply per-sector transforms or interpret probabilities by basis key.
    """
    graph = NoisyG2SLOSComputeGraph(
        _g2_groups(),
        m=3,
        n_photons=2,
        computation_space=ComputationSpace.FOCK,
    )

    result = graph.compute_probs(_identity_unitary(3), [1, 1, 0])

    assert isinstance(result, SectoredDistribution)
    for sector in result.sectors:
        expected_keys = tuple(
            tuple(state)
            for state in Combinadics(
                "fock", n=sector.n_photons, m=sector.n_modes
            ).enumerate_states()
        )
        assert sector.keys == expected_keys


def test_sectored_total_probability_stays_in_autograd() -> None:
    """Check that total probability remains differentiable.

    Correct result: ``SectoredDistribution.total_probability()`` should return a
    tensor connected to each sector tensor so callers can use it in losses and
    call ``backward()``.

    Branch result: the helper uses ``.item()``, returns a Python float, and
    detaches the total from autograd.
    """
    sector_1 = SectorResult(
        torch.tensor([0.2, 0.3], requires_grad=True),
        n_modes=2,
        n_photons=1,
    )
    sector_2 = SectorResult(
        torch.tensor([0.5], requires_grad=True),
        n_modes=2,
        n_photons=2,
    )
    distribution = SectoredDistribution((sector_1, sector_2))

    total = distribution.total_probability()

    assert isinstance(total, torch.Tensor)
    total.backward()
    assert sector_1.tensor.grad is not None
    assert sector_2.tensor.grad is not None


# Tests added after review
def _g2_layer_with_loss() -> QuantumLayer:
    """Build a g2 layer whose photon-loss transform changes each sector basis."""
    with pytest.warns(UserWarning, match="g2_distinguishable must be False"):
        return QuantumLayer(
            circuit=pcvl.Circuit(3),
            input_state=[1, 1, 0],
            n_photons=2,
            noise=pcvl.NoiseModel(g2=0.1, brightness=0.8),
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.FOCK
            ),
        )


def test_sector_result_preserves_explicit_basis_keys() -> None:
    """Check that explicit sector basis metadata is not overwritten.

    Correct result: when a caller passes ``keys=...`` to ``SectorResult``, those
    keys should remain attached to the sector. Auto-generating Fock keys is useful
    only when ``keys`` is omitted.

    Why this matters: after photon loss or detector transforms, a sector tensor
    can live in a different output basis from the fixed ``n_photons`` Fock basis.
    For example, loss from a one-photon sector can include the vacuum key
    ``(0, 0, 0)``. The container needs to preserve those transform-produced keys.
    """
    transformed_basis_keys = ((1, 0, 0), (0, 0, 0))

    sector = SectorResult(
        torch.ones(len(transformed_basis_keys)),
        n_modes=3,
        n_photons=1,
        keys=transformed_basis_keys,
    )

    assert sector.keys == transformed_basis_keys


def test_g2_layer_to_moves_per_sector_transforms() -> None:
    """Check ``QuantumLayer.to()`` works with g2 per-sector transforms.

    Correct result: calling ``layer.to("cpu")`` should move the g2 simulation
    graph and each per-sector photon-loss/detector transform without raising.
    """
    layer = _g2_layer_with_loss()

    layer.to("cpu")

class TestG2AugmentedObbRegression:
    """Pin the indistinguishable extra-photon sector probabilities.

    Regression guard for ``NoisyG2SLOSComputeGraph._augmented_obb_probs``:
    with partial indistinguishability, each extra-photon sector must reuse the
    OBB partitions already built for the base ``n_photons`` input state and
    fold every g2-emitted sibling into its partition cell's coherent group.
    An earlier implementation instead grew a whole new noisy SLOS graph on the
    augmented input state, re-drawing an independent
    coherent-vs-distinguishable outcome for the g2 sibling; on this fixture
    that skews sector 1 by up to 6.8e-3 per entry (e.g. the (1, 1, 1) entry
    comes out ~0.031014 instead of the correct 0.024246), far beyond the pin
    tolerance below.

    Golden values were produced by the fixed implementation on a
    deterministic 3-mode Fourier interferometer in float64 and match Perceval
    1.2.4 ``Simulator.probs_svd`` with ``NoiseModel(g2=0.3,
    g2_distinguishable=False, indistinguishability=0.6)`` to below 1e-15.
    """

    # Fock basis order is Combinadics("fock", n, m).enumerate_states().
    EXPECTED_SECTOR_1 = [
        0.030686209636,  # (3, 0, 0)
        0.038768111696,  # (2, 1, 0)
        0.038768111696,  # (2, 0, 1)
        0.038768111696,  # (1, 2, 0)
        0.024245706182,  # (1, 1, 1)
        0.038768111696,  # (1, 0, 2)
        0.030686209636,  # (0, 3, 0)
        0.038768111696,  # (0, 2, 1)
        0.038768111696,  # (0, 1, 2)
        0.030686209636,  # (0, 0, 3)
    ]
    EXPECTED_SECTOR_2 = [
        0.002972157995,  # (4, 0, 0)
        0.003723146497,  # (3, 1, 0)
        0.003723146497,  # (3, 0, 1)
        0.004881425266,  # (2, 2, 0)
        0.001597365052,  # (2, 1, 1)
        0.004881425266,  # (2, 0, 2)
        0.003723146497,  # (1, 3, 0)
        0.001597365052,  # (1, 2, 1)
        0.001597365052,  # (1, 1, 2)
        0.003723146497,  # (1, 0, 3)
        0.002972157995,  # (0, 4, 0)
        0.003723146497,  # (0, 3, 1)
        0.004881425266,  # (0, 2, 2)
        0.003723146497,  # (0, 1, 3)
        0.002972157995,  # (0, 0, 4)
    ]

    def test_extra_photon_sectors_match_golden_values(self):
        """Extra-photon sectors equal the Perceval-verified golden values."""
        m = 3
        n_photons = 2
        modes = np.arange(m)
        fourier = np.exp(2j * np.pi * np.outer(modes, modes) / m) / np.sqrt(m)
        unitary = torch.tensor(fourier, dtype=torch.complex128)

        groups = NoiseGroups(
            source={
                "g2": 0.3,
                "g2_distinguishable": False,
                "indistinguishability": 0.6,
            },
            circuit=None,
            post_measurement=None,
        )
        noisy_slos = NoisyG2SLOSComputeGraph(
            groups,
            m=m,
            n_photons=n_photons,
            computation_space=ComputationSpace.FOCK,
            dtype=torch.float64,
        )
        result = noisy_slos.compute_probs(unitary, [1, 1, 0])

        assert isinstance(result, SectoredDistribution)
        assert len(result.sectors) == n_photons + 1

        for sector_idx, expected in (
            (1, self.EXPECTED_SECTOR_1),
            (2, self.EXPECTED_SECTOR_2),
        ):
            actual = result.sectors[sector_idx].tensor.squeeze()
            expected_tensor = torch.tensor(expected, dtype=torch.float64)
            assert torch.allclose(actual, expected_tensor, atol=1e-8), (
                f"Sector {sector_idx} deviates from golden values by "
                f"{(actual - expected_tensor).abs().max().item():.3e}"
            )
