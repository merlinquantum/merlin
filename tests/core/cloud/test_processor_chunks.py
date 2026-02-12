# tests/core/cloud/test_cloud_futures_chunking.py
from __future__ import annotations

import concurrent.futures as _cf

import pytest
import torch
import torch.nn as nn
from ._helpers import make_layer, spin_until

from merlin.core.computation_space import ComputationSpace
from merlin.core.merlin_processor import MerlinProcessor


class TestFuturesAndChunking:
    def test_forward_async_future_and_helpers(self, remote_processor):
        layer = make_layer(6, 2, 2, computation_space=ComputationSpace.UNBUNCHED)
        proc = MerlinProcessor(remote_processor)
        fut = proc.forward_async(layer, torch.rand(3, 2), nsample=1500)

        assert isinstance(fut, torch.futures.Future)
        assert hasattr(fut, "cancel_remote")
        assert hasattr(fut, "status")
        assert hasattr(fut, "job_ids")

        spin_until(lambda f=fut: len(f.job_ids) > 0 or f.done(), timeout_s=240.0)
        out = fut.wait()
        assert out.shape == (3, 15)

    def test_timeout_sets_timeouterror(self, remote_processor):
        layer = make_layer(6, 2, 2, computation_space=ComputationSpace.UNBUNCHED)
        proc = MerlinProcessor(remote_processor)
        fut = proc.forward_async(layer, torch.rand(8, 2), nsample=50_000, timeout=60)

        done_in_time = spin_until(lambda: fut.done(), timeout_s=60.0)
        if not done_in_time:
            with pytest.raises(TimeoutError):
                fut.wait()
        else:
            try:
                _ = fut.value()
            except Exception:
                with pytest.raises(TimeoutError):
                    fut.wait()

    def test_cancel_remote_cancelled_error(self, remote_processor):
        layer = make_layer(6, 2, 2, computation_space=ComputationSpace.UNBUNCHED)
        proc = MerlinProcessor(remote_processor)  # default timeout; per-call infinite
        fut = proc.forward_async(layer, torch.rand(8, 2), nsample=40_000, timeout=None)

        spin_until(lambda f=fut: len(f.job_ids) > 0 or f.done(), timeout_s=120.0)
        if fut.done():
            pytest.skip("Backend finished too quickly to test cancellation")
        fut.cancel_remote()
        with pytest.raises(_cf.CancelledError):
            fut.wait()

    def test_multiple_concurrent_futures(self, remote_processor):
        layer = make_layer(6, 2, 2, computation_space=ComputationSpace.UNBUNCHED)
        proc = MerlinProcessor(remote_processor)
        futs = [
            proc.forward_async(layer, torch.rand(2, 2), nsample=1500) for _ in range(4)
        ]
        for f in futs:
            spin_until(lambda f=f: len(f.job_ids) > 0 or f.done(), timeout_s=120.0)
        outs = [f.wait() for f in futs]
        for y in outs:
            assert y.shape == (2, 15)

    def test_context_manager_auto_cancel_on_exit(self, remote_processor):
        layer = make_layer(6, 2, 2, computation_space=ComputationSpace.UNBUNCHED)
        fut = None
        with MerlinProcessor(remote_processor) as proc:
            fut = proc.forward_async(
                layer, torch.rand(8, 2), nsample=40_000, timeout=None
            )
            spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=120.0)
        assert fut is not None
        with pytest.raises(_cf.CancelledError):
            fut.wait()

    def test_default_timeout_via_constructor(self, remote_processor):
        layer = make_layer(6, 2, 2, computation_space=ComputationSpace.UNBUNCHED)
        proc = MerlinProcessor(remote_processor, timeout=0.03)
        fut = proc.forward_async(layer, torch.rand(8, 2), nsample=50_000)
        done_in_time = spin_until(lambda: fut.done(), timeout_s=120.0)
        if not done_in_time:
            with pytest.raises(TimeoutError):
                fut.wait()
        else:
            try:
                _ = fut.value()
            except Exception:
                with pytest.raises(TimeoutError):
                    fut.wait()

    def test_chunking_end_to_end(self, remote_processor):
        q = make_layer(6, 2, 2, computation_space=ComputationSpace.UNBUNCHED)
        B, max_bs = 5, 2  # -> 3 chunks: [0:2],[2:4],[4:5]
        X = torch.rand(B, 2)
        proc = MerlinProcessor(
            remote_processor,
            microbatch_size=max_bs,
            chunk_concurrency=2,
            max_shots_per_call=50_000,
            timeout=300.0,  # 5 min for chunked jobs
        )
        fut = proc.forward_async(q, X, nsample=2000, timeout=300.0)
        spin_until(lambda f=fut: len(f.job_ids) >= 3 or f.done(), timeout_s=300.0)
        y = fut.wait()
        assert y.shape == (B, 15)
        assert len(fut.job_ids) >= 3
        st = fut.status()
        assert st["chunks_total"] >= 3 and "chunks_done" in st and "active_chunks" in st

    def test_two_quantum_leaves_both_chunked(self, remote_processor):
        # q1: 4m,2p -> 6 ; q2: 5m,2p -> 10
        q1 = make_layer(4, 2, 1, computation_space=ComputationSpace.UNBUNCHED)
        q2 = make_layer(5, 2, 2, computation_space=ComputationSpace.UNBUNCHED)
        model = nn.Sequential(
            nn.Linear(3, 1, bias=False),
            q1,
            nn.Linear(6, 2, bias=False),
            q2,
            nn.Linear(10, 3, bias=False),
            nn.Softmax(dim=-1),
        ).eval()
        X = torch.rand(7, 3)
        proc = MerlinProcessor(
            remote_processor,
            microbatch_size=3,
            chunk_concurrency=2,
            max_shots_per_call=60_000,
        )
        fut = proc.forward_async(model, X, nsample=3000)
        spin_until(lambda f=fut: len(f.job_ids) >= 4 or f.done(), timeout_s=120.0)
        y = fut.wait()
        assert y.shape == (7, 3)
        assert len(fut.job_ids) >= 4
