# tests/core/cloud/test_state_mapping.py
"""
Tests for bunched (FOCK) and unbunched state mapping, SLOS construction ordering,
and cloud execution correctness across both computation spaces.

Unit tests (no cloud required):
    pytest tests/core/cloud/test_state_mapping.py -k "Unit" -v

Cloud tests (Quandela):
    pytest --run-cloud-tests tests/core/cloud/test_state_mapping.py -k "Cloud" -v

Scaleway tests:
    pytest --run-scaleway-tests tests/core/cloud/test_state_mapping.py -k "Scaleway" -v
"""
from __future__ import annotations

import time
from math import comb

import pytest
import torch
import torch.nn as nn
from ._helpers import make_layer

from merlin.core.computation_space import ComputationSpace
from merlin.core.merlin_processor import MerlinProcessor
from merlin.utils.combinadics import Combinadics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_future(fut, timeout_s: float = 120.0):
    end = time.time() + timeout_s
    while not fut.done():
        if time.time() >= end:
            raise TimeoutError("Timeout waiting for Merlin future")
        time.sleep(0.01)
    return fut.value()


def _slos_reference_ordering(
    n_modes: int, n_photons: int, unbunched: bool,
) -> list[tuple[int, ...]]:
    """Reference SLOS graph construction order (mirrors _build_graph_structure)."""
    last_combinations: dict[tuple[int, ...], tuple[int, int]] = {
        tuple([0] * n_modes): (1, 0)
    }
    for _ in range(n_photons):
        combinations: dict[tuple[int, ...], tuple[int, int]] = {}
        for state, (norm_factor, _src_idx) in last_combinations.items():
            nstate = list(state)
            for i in range(n_modes):
                if unbunched and nstate[i]:
                    continue
                nstate[i] += 1
                nstate_t = tuple(nstate)
                if nstate_t not in combinations:
                    combinations[nstate_t] = (norm_factor * nstate[i], len(combinations))
                nstate[i] -= 1
        last_combinations = combinations
    return list(last_combinations.keys())


# ---------------------------------------------------------------------------
# Unit tests — no cloud access needed
# ---------------------------------------------------------------------------


class TestUnitStateOrdering:
    """Unit tests for Combinadics-based state ordering used by MerlinProcessor."""

    @pytest.mark.parametrize(
        "n_modes,n_photons",
        [(2, 1), (2, 2), (3, 2), (3, 3), (4, 2), (4, 3), (5, 2), (6, 2)],
    )
    def test_fock_ordering_matches_slos_reference(self, n_modes, n_photons):
        """FOCK ordering must match the SLOS graph construction order exactly."""
        result = Combinadics("fock", n_photons, n_modes).enumerate_states()
        reference = _slos_reference_ordering(n_modes, n_photons, unbunched=False)
        assert result == reference

    @pytest.mark.parametrize(
        "n_modes,n_photons",
        [(3, 2), (4, 2), (5, 2), (5, 3), (6, 2), (6, 3)],
    )
    def test_unbunched_ordering_matches_slos_reference(self, n_modes, n_photons):
        """UNBUNCHED ordering must match the SLOS graph construction order exactly."""
        result = Combinadics("unbunched", n_photons, n_modes).enumerate_states()
        reference = _slos_reference_ordering(n_modes, n_photons, unbunched=True)
        assert result == reference

    @pytest.mark.parametrize(
        "n_modes,n_photons",
        [(2, 2), (3, 2), (3, 3), (4, 2), (5, 2), (6, 2)],
    )
    def test_fock_state_count(self, n_modes, n_photons):
        """FOCK state count should equal C(m + n - 1, n)."""
        states = Combinadics("fock", n_photons, n_modes).enumerate_states()
        assert len(states) == comb(n_modes + n_photons - 1, n_photons)

    @pytest.mark.parametrize(
        "n_modes,n_photons",
        [(3, 2), (4, 2), (5, 2), (5, 3), (6, 2)],
    )
    def test_unbunched_state_count(self, n_modes, n_photons):
        """UNBUNCHED state count should equal C(m, n)."""
        states = Combinadics("unbunched", n_photons, n_modes).enumerate_states()
        assert len(states) == comb(n_modes, n_photons)

    @pytest.mark.parametrize("n_modes", [4, 6, 8])
    def test_dual_rail_state_count(self, n_modes):
        """DUAL_RAIL state count should equal 2^(m/2)."""
        n_photons = n_modes // 2
        states = Combinadics("dual_rail", n_photons, n_modes).enumerate_states()
        assert len(states) == 2 ** (n_modes // 2)

    def test_fock_states_are_unique(self):
        """Every state in the ordering must be unique."""
        states = Combinadics("fock", 3, 5).enumerate_states()
        assert len(states) == len(set(states))

    def test_unbunched_states_are_unique(self):
        """Every state in the ordering must be unique."""
        states = Combinadics("unbunched", 2, 6).enumerate_states()
        assert len(states) == len(set(states))

    def test_dual_rail_states_are_unique(self):
        """Every state in the ordering must be unique."""
        states = Combinadics("dual_rail", 3, 6).enumerate_states()
        assert len(states) == len(set(states))

    def test_fock_all_photons_first_mode_is_first(self):
        """In descending lex order, all photons in mode 0 is always first."""
        for m in range(2, 6):
            for n in range(1, 4):
                states = Combinadics("fock", n, m).enumerate_states()
                expected_first = tuple([n] + [0] * (m - 1))
                assert states[0] == expected_first

    def test_fock_all_photons_last_mode_is_last(self):
        """In descending lex order, all photons in last mode is always last."""
        for m in range(2, 6):
            for n in range(1, 4):
                states = Combinadics("fock", n, m).enumerate_states()
                expected_last = tuple([0] * (m - 1) + [n])
                assert states[-1] == expected_last

    def test_unbunched_no_bunching_constraint(self):
        """Unbunched states must have at most 1 photon per mode."""
        states = Combinadics("unbunched", 3, 6).enumerate_states()
        for state in states:
            assert all(s <= 1 for s in state), f"State {state} violates no-bunching"

    def test_dual_rail_pair_constraint(self):
        """DUAL_RAIL: each pair of modes must have exactly 1 photon total."""
        states = Combinadics("dual_rail", 3, 6).enumerate_states()
        for state in states:
            for k in range(0, 6, 2):
                assert state[k] + state[k + 1] == 1, (
                    f"State {state} violates dual-rail at modes {k},{k+1}"
                )

    def test_all_states_have_correct_photon_count(self):
        """Every state must have exactly n_photons total."""
        for scheme, n, m in [("fock", 2, 5), ("unbunched", 2, 5), ("dual_rail", 2, 4)]:
            states = Combinadics(scheme, n, m).enumerate_states()
            for state in states:
                assert sum(state) == n, f"State {state} has wrong photon count"


class TestUnitLocalOutputShape:
    """Unit tests verifying local forward produces correct shapes and normalization."""

    @pytest.mark.parametrize(
        "m,n,input_size,computation_space,expected_dist_size",
        [
            (4, 2, 2, ComputationSpace.UNBUNCHED, 6),
            (4, 2, 2, ComputationSpace.FOCK, 10),
            (5, 2, 2, ComputationSpace.UNBUNCHED, 10),
            (5, 2, 2, ComputationSpace.FOCK, 15),
            (5, 3, 3, ComputationSpace.UNBUNCHED, 10),
            (5, 3, 3, ComputationSpace.FOCK, 35),
            (6, 2, 2, ComputationSpace.UNBUNCHED, 15),
            (6, 2, 2, ComputationSpace.FOCK, 21),
        ],
    )
    def test_local_dist_size(self, m, n, input_size, computation_space, expected_dist_size):
        """Local forward output dimension matches combinatorial expectation."""
        layer = make_layer(m, n, input_size, computation_space=computation_space)
        y = layer(torch.rand(2, input_size))
        assert y.shape == (2, expected_dist_size)

    @pytest.mark.parametrize(
        "computation_space", [ComputationSpace.UNBUNCHED, ComputationSpace.FOCK],
    )
    def test_local_output_normalized(self, computation_space):
        """Local forward should produce valid normalized probabilities."""
        layer = make_layer(4, 2, 2, computation_space=computation_space)
        y = layer(torch.rand(4, 2))
        assert torch.all(y >= 0), "Probabilities must be non-negative"
        assert torch.allclose(y.sum(dim=1), torch.ones(4), atol=1e-5), (
            "Probabilities must sum to 1"
        )


# ---------------------------------------------------------------------------
# Cloud tests — Quandela remote_processor
# ---------------------------------------------------------------------------


class TestCloudBunchedUnbunched:
    """Cloud execution tests for both FOCK and UNBUNCHED spaces."""

    @pytest.mark.parametrize(
        "m,n,input_size,computation_space",
        [
            (4, 2, 2, ComputationSpace.FOCK),
            (5, 2, 2, ComputationSpace.FOCK),
            (6, 2, 2, ComputationSpace.FOCK),
            (5, 3, 3, ComputationSpace.FOCK),
        ],
    )
    def test_bunched_forward_shape_and_norm(
        self, remote_processor, m, n, input_size, computation_space
    ):
        """Remote execution with FOCK space should return correct shape and normalized output."""
        layer = make_layer(m, n, input_size, computation_space=computation_space)
        expected_dist = comb(m + n - 1, n)

        proc = MerlinProcessor(remote_processor, timeout=300.0)
        X = torch.rand(4, input_size)
        y = proc.forward(layer, X, nsample=5000)

        assert y.shape == (4, expected_dist), (
            f"Expected shape (4, {expected_dist}), got {y.shape}"
        )
        assert torch.all(y >= 0), "Probabilities must be non-negative"
        assert torch.allclose(y.sum(dim=1), torch.ones(4), atol=0.1), (
            f"Probabilities should sum to ~1, got {y.sum(dim=1)}"
        )

    @pytest.mark.parametrize(
        "m,n,input_size,computation_space",
        [
            (4, 2, 2, ComputationSpace.UNBUNCHED),
            (5, 2, 2, ComputationSpace.UNBUNCHED),
            (6, 2, 2, ComputationSpace.UNBUNCHED),
        ],
    )
    def test_unbunched_forward_shape_and_norm(
        self, remote_processor, m, n, input_size, computation_space
    ):
        """Remote execution with UNBUNCHED space should return correct shape and normalized output."""
        layer = make_layer(m, n, input_size, computation_space=computation_space)
        expected_dist = comb(m, n)

        proc = MerlinProcessor(remote_processor, timeout=300.0)
        X = torch.rand(4, input_size)
        y = proc.forward(layer, X, nsample=5000)

        assert y.shape == (4, expected_dist)
        assert torch.all(y >= 0)
        assert torch.allclose(y.sum(dim=1), torch.ones(4), atol=0.1)

    def test_bunched_local_vs_remote(self, remote_processor):
        """Remote FOCK results should approximate local probabilities."""
        layer = make_layer(4, 2, 2, computation_space=ComputationSpace.FOCK, trainable=True)
        X = torch.rand(4, 2)

        y_local = layer(X)

        proc = MerlinProcessor(remote_processor, timeout=300.0)
        y_remote = proc.forward(layer, X, nsample=20_000)

        assert y_local.shape == y_remote.shape
        assert torch.allclose(y_local.sum(dim=1), torch.ones(4), atol=1e-5)
        assert torch.allclose(y_remote.sum(dim=1), torch.ones(4), atol=0.1)
        diff = (y_local - y_remote).abs().mean().item()
        assert diff < 0.1, f"Mean abs diff between local and remote too large: {diff:.4f}"

    def test_unbunched_local_vs_remote(self, remote_processor):
        """Remote UNBUNCHED results should approximate local probabilities."""
        layer = make_layer(5, 2, 2, computation_space=ComputationSpace.UNBUNCHED, trainable=True)
        X = torch.rand(4, 2)

        y_local = layer(X)

        proc = MerlinProcessor(remote_processor, timeout=300.0)
        y_remote = proc.forward(layer, X, nsample=20_000)

        assert y_local.shape == y_remote.shape
        diff = (y_local - y_remote).abs().mean().item()
        assert diff < 0.1, f"Mean abs diff between local and remote too large: {diff:.4f}"

    def test_bunched_nonzero_output(self, remote_processor):
        """FOCK cloud execution must produce non-trivial (non-zero) output tensors."""
        layer = make_layer(4, 2, 2, computation_space=ComputationSpace.FOCK)
        proc = MerlinProcessor(remote_processor, timeout=300.0)
        X = torch.rand(3, 2)
        y = proc.forward(layer, X, nsample=5000)

        assert y.abs().sum() > 0, "Output is all zeros — state mapping likely broken"

    def test_bunched_async(self, remote_processor):
        """Async forward with FOCK space should work correctly."""
        layer = make_layer(4, 2, 2, computation_space=ComputationSpace.FOCK)
        proc = MerlinProcessor(remote_processor, timeout=300.0)
        X = torch.rand(4, 2)

        fut = proc.forward_async(layer, X, nsample=5000)
        y = _wait_future(fut)

        expected_dist = comb(4 + 2 - 1, 2)  # 10
        assert y.shape == (4, expected_dist)
        assert len(fut.job_ids) >= 1

    def test_bunched_in_sequential_model(self, remote_processor):
        """FOCK layer in a Sequential model pipeline should work end-to-end."""
        q = make_layer(4, 2, 2, computation_space=ComputationSpace.FOCK)
        dist_size = comb(4 + 2 - 1, 2)  # 10

        model = nn.Sequential(
            nn.Linear(3, 2, bias=False),
            q,
            nn.Linear(dist_size, 4, bias=False),
            nn.Softmax(dim=-1),
        ).eval()

        proc = MerlinProcessor(remote_processor, timeout=300.0)
        X = torch.rand(6, 3)
        y = proc.forward(model, X, nsample=5000)

        assert y.shape == (6, 4)
        assert torch.allclose(y.sum(dim=1), torch.ones(6), atol=1e-5)


# ---------------------------------------------------------------------------
# Scaleway tests — same coverage via ISession path
# ---------------------------------------------------------------------------


class TestScalewayBunchedUnbunched:
    """Bunched/unbunched tests via Scaleway session."""

    def test_bunched_forward(self, scaleway_session):
        """FOCK execution through ISession should return correct shape."""
        proc = MerlinProcessor(
            session=scaleway_session,
            timeout=300.0,
            max_shots_per_call=1000,
        )

        layer = make_layer(4, 2, 2, computation_space=ComputationSpace.FOCK)
        expected_dist = comb(4 + 2 - 1, 2)  # 10

        X = torch.rand(4, 2)
        y = proc.forward(layer, X, nsample=1000)

        assert y.shape == (4, expected_dist)
        assert torch.all(y >= 0)
        assert torch.allclose(y.sum(dim=1), torch.ones(4), atol=0.1)

    def test_unbunched_forward(self, scaleway_session):
        """UNBUNCHED execution through ISession should return correct shape."""
        proc = MerlinProcessor(
            session=scaleway_session,
            timeout=300.0,
            max_shots_per_call=1000,
        )

        layer = make_layer(6, 2, 2, computation_space=ComputationSpace.UNBUNCHED)
        expected_dist = comb(6, 2)  # 15

        X = torch.rand(4, 2)
        y = proc.forward(layer, X, nsample=1000)

        assert y.shape == (4, expected_dist)
        assert torch.all(y >= 0)
        assert torch.allclose(y.sum(dim=1), torch.ones(4), atol=0.1)

    def test_bunched_nonzero_output(self, scaleway_session):
        """FOCK output via ISession must be non-trivial."""
        proc = MerlinProcessor(
            session=scaleway_session,
            timeout=300.0,
            max_shots_per_call=1000,
        )

        layer = make_layer(4, 2, 2, computation_space=ComputationSpace.FOCK)
        X = torch.rand(3, 2)
        y = proc.forward(layer, X, nsample=1000)

        assert y.abs().sum() > 0, "Output is all zeros — state mapping likely broken"

    def test_bunched_local_vs_remote(self, scaleway_session):
        """Remote FOCK results via ISession should approximate local probabilities."""
        proc = MerlinProcessor(
            session=scaleway_session,
            timeout=300.0,
            max_shots_per_call=5000,
        )

        layer = make_layer(4, 2, 2, computation_space=ComputationSpace.FOCK, trainable=True)
        X = torch.rand(4, 2)

        y_local = layer(X)
        y_remote = proc.forward(layer, X, nsample=5000)

        assert y_local.shape == y_remote.shape
        diff = (y_local - y_remote).abs().mean().item()
        assert diff < 0.15, f"Mean abs diff between local and remote too large: {diff:.4f}"