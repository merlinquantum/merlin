# tests/core/cloud/test_cloud_estimation_export.py
from __future__ import annotations

from math import comb

import pytest
import torch
from _helpers import make_layer

from merlin.core.computation_space import ComputationSpace
from merlin.core.merlin_processor import MerlinProcessor


class TestShotEstimation:
    def test_single_and_batch_estimates(self, remote_processor):
        q = make_layer(6, 2, 2, computation_space=ComputationSpace.UNBUNCHED)
        proc = MerlinProcessor(remote_processor)

        est_single = proc.estimate_required_shots_per_input(
            q, torch.rand(2), desired_samples_per_input=2_000
        )
        assert (
            isinstance(est_single, list)
            and len(est_single) == 1
            and isinstance(est_single[0], int)
            and est_single[0] >= 0
        )

        X = torch.rand(5, 2)
        est_batch = proc.estimate_required_shots_per_input(
            q, X, desired_samples_per_input=2_000
        )
        assert (
            isinstance(est_batch, list)
            and len(est_batch) == 5
            and all(isinstance(v, int) and v >= 0 for v in est_batch)
        )

    def test_monotonic_with_desired_samples(self, remote_processor):
        q = make_layer(5, 2, 2, computation_space=ComputationSpace.UNBUNCHED)
        proc = MerlinProcessor(remote_processor)
        X = torch.rand(4, 2)
        lo = proc.estimate_required_shots_per_input(
            q, X, desired_samples_per_input=1_000
        )
        hi = proc.estimate_required_shots_per_input(
            q, X, desired_samples_per_input=5_000
        )
        assert len(lo) == len(hi) == 4
        for a, b in zip(lo, hi, strict=False):
            assert b >= a  # larger target -> requires >= shots

    def test_no_jobs_created(self, remote_processor):
        q = make_layer(6, 2, 2, computation_space=ComputationSpace.UNBUNCHED)
        proc = MerlinProcessor(remote_processor)
        before = len(proc.get_job_history())
        _ = proc.estimate_required_shots_per_input(
            q, torch.rand(3, 2), desired_samples_per_input=3_000
        )
        after = len(proc.get_job_history())
        assert after == before


class TestOutputAndExport:
    @pytest.mark.parametrize(
        "m,n,input_size,computation_space,expected",
        [
            (4, 2, 2, ComputationSpace.UNBUNCHED, 6),
            (4, 2, 2, ComputationSpace.FOCK, 10),
            (5, 3, 3, ComputationSpace.UNBUNCHED, 10),
            (5, 3, 3, ComputationSpace.FOCK, 35),
            (6, 2, 2, ComputationSpace.UNBUNCHED, 15),
        ],
    )
    def test_local_distribution_size(
        self, m, n, input_size, computation_space, expected
    ):
        q = make_layer(m, n, input_size, computation_space=computation_space)
        y = q(torch.rand(3, input_size))
        assert y.shape == (3, expected)
        assert torch.allclose(y.sum(dim=1), torch.ones(3), atol=1e-5)

    def test_cloud_distribution_size_matches(self, remote_processor):
        q = make_layer(6, 2, 2, computation_space=ComputationSpace.UNBUNCHED)
        y = MerlinProcessor(remote_processor).forward(q, torch.rand(4, 2), nsample=2000)
        assert y.shape == (4, comb(6, 2))

    def test_export_config_includes_trained_thetas(self):
        q = make_layer(
            5, 3, 3, computation_space=ComputationSpace.UNBUNCHED, trainable=True
        )
        before = {n: p.clone() for n, p in q.named_parameters()}
        q.train()
        opt = torch.optim.Adam(q.parameters(), lr=0.05)
        X = torch.randn(6, 3)
        for _ in range(4):
            opt.zero_grad()
            q(X).sum().backward()
            opt.step()
        q.eval()
        changed = any(
            not torch.allclose(p, before[n], atol=1e-6) for n, p in q.named_parameters()
        )
        assert changed

        cfg = q.export_config()
        names = [p.name for p in cfg["circuit"].get_parameters()]
        assert any("theta" in n for n in names)
