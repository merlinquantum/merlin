# tests/core/cloud/test_processor_pooling.py
from __future__ import annotations

import time
from itertools import combinations

import torch
from ._helpers import make_layer, spin_until

from merlin.core.computation_space import ComputationSpace
from merlin.core.merlin_processor import MerlinProcessor


def _wait_future(fut, timeout_s: float = 120.0):
    end = time.time() + timeout_s
    while not fut.done():
        if time.time() >= end:
            raise TimeoutError("Timeout waiting for Merlin future")
        time.sleep(0.01)
    return fut.value()


class TestPerCallRemoteProcessorPooling:
    def test_per_call_pool_respects_chunk_concurrency(self, remote_processor):
        """
        With microbatching and chunk_concurrency > 1, ensure that within a single forward_async call
        the number of active chunks never exceeds chunk_concurrency.

        This indirectly verifies that we aren't reusing a single RP/handler across threads
        and that pool-based parallelism is correctly bounded per forward call.
        """
        # Make a layer that offloads
        q = make_layer(6, 2, input_size=2, computation_space=ComputationSpace.UNBUNCHED)
        # Shape: B=7 -> with microbatch_size=2 we get chunks: [0:2],[2:4],[4:6],[6:7] = 4 chunks
        B = 7
        microbatch_size = 2
        chunk_concurrency = 2
        X = torch.rand(B, 2)

        proc = MerlinProcessor(
            remote_processor,
            microbatch_size=microbatch_size,
            chunk_concurrency=chunk_concurrency,
            # raise shots a bit to keep jobs alive long enough to observe concurrency
            max_shots_per_call=60_000,
        )

        fut = proc.forward_async(q, X, nsample=3000, timeout=240.0)

        # Wait until we see at least one job, or it completes very fast.
        assert spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=30.0)

        # Track the maximum number of concurrent active chunks reported.
        max_active = 0
        # Poll status while running
        end = time.time() + 60.0
        while not fut.done() and time.time() < end:
            st = fut.status()
            max_active = max(max_active, int(st.get("active_chunks", 0)))
            # Small sleep to avoid hammering
            time.sleep(0.02)

        # Finish and get result (raises if error/timeout)
        y = _wait_future(fut, timeout_s=240.0)

        # Basic shape check: distribution size is C(6,2)=15
        assert y.shape == (B, 15)

        # The key assertion: active chunks never exceeded the per-call limit
        assert max_active <= chunk_concurrency, (
            f"Observed {max_active} active chunks > {chunk_concurrency}"
        )

        # Also: ensure we actually chunked (chunks_total >= 2)
        st_final = fut.status()
        assert st_final.get("chunks_total", 0) >= 2
        assert st_final.get("chunks_done", 0) >= 2

    def test_concurrent_calls_isolated_job_ids(self, remote_processor):
        """
        Launch several concurrent forward_async calls and verify:
          - All complete successfully.
          - Their job_ids sets are disjoint (strong sign of per-call isolation).
        """
        q = make_layer(6, 2, input_size=2, computation_space=ComputationSpace.UNBUNCHED)
        # Small batches so each call still chunks at least a little when we set microbatch_size low.
        Xs = [torch.rand(5, 2) for _ in range(3)]

        proc = MerlinProcessor(
            remote_processor,
            microbatch_size=2,  # encourage multiple chunks per call
            chunk_concurrency=2,  # pool size per call
            max_shots_per_call=60_000,
        )

        futs = [proc.forward_async(q, X, nsample=3000, timeout=240.0) for X in Xs]

        # Wait until each future has at least one job id (or it's done surprisingly fast)
        for f in futs:
            assert spin_until(
                lambda f=f: len(f.job_ids) > 0 or f.done(), timeout_s=30.0
            )

        # Collect job id sets for each future
        job_sets = []
        for f in futs:
            # Give a moment to accumulate all chunk jobs
            spin_until(lambda f=f: f.done() or len(f.job_ids) >= 2, timeout_s=20.0)
            job_sets.append(set(f.job_ids))

        # All complete
        outs = [_wait_future(f, timeout_s=240.0) for f in futs]
        for y in outs:
            assert y.shape[1] == 15  # comb(6,2)

        # Assert pairwise disjoint job id sets (cloud assigns distinct job IDs per submission)
        # This is a practical, observable proxy for "no cross-talk" between calls.
        for (a_idx, a), (b_idx, b) in combinations(enumerate(job_sets), 2):
            assert a.isdisjoint(b), (
                f"job_ids of future {a_idx} and {b_idx} overlap; expected isolation"
            )
