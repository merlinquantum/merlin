# tests/core/cloud/test_userguide_examples.py
from __future__ import annotations

import time
from concurrent.futures import CancelledError
from math import comb

import perceval as pcvl  # noqa: F401  (import keeps RemoteConfig init consistent with conftest)
import pytest
import torch
import torch.nn as nn

from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.core.computation_space import ComputationSpace
from merlin.core.merlin_processor import MerlinProcessor
from merlin.measurement.strategies import MeasurementStrategy


def _make_layer(
    n_modes: int,
    n_photons: int,
    input_size: int,
    *,
    computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
    trainable: bool = True,
) -> QuantumLayer:
    """Helper that mirrors the builder pattern used in the docs."""
    b = CircuitBuilder(n_modes=n_modes)
    if trainable:
        b.add_rotations(trainable=True, name="theta")
    b.add_angle_encoding(modes=list(range(input_size)), name="px")
    if n_modes >= 3:
        b.add_entangling_layer()
    return QuantumLayer(
        input_size=input_size,
        builder=b,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=computation_space,
        ),
    ).eval()


def _wait_future(fut, timeout_s: float = 30.0):
    end = time.time() + timeout_s
    while not fut.done():
        if time.time() >= end:
            raise TimeoutError("Timeout waiting for Merlin future")
        time.sleep(0.01)
    return fut.value()


class TestUserGuideExamples:
    def test_quick_start_end_to_end(self, remote_processor):
        """Quick Start: end-to-end sync execution and basic shape checks."""
        # RemoteProcessor already configured by fixture
        proc = MerlinProcessor(
            remote_processor,
            microbatch_size=32,
            timeout=3600.0,
            max_shots_per_call=None,
            chunk_concurrency=1,
        )

        # Build quantum layer & model exactly like the doc
        b = CircuitBuilder(n_modes=6)
        b.add_rotations(trainable=True, name="theta")
        b.add_angle_encoding(modes=[0, 1], name="px")
        b.add_entangling_layer()

        q = QuantumLayer(
            input_size=2,
            builder=b,
            n_photons=2,
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.UNBUNCHED,
            ),
        ).eval()

        model = nn.Sequential(
            nn.Linear(3, 2, bias=False),
            q,
            nn.Linear(15, 4, bias=False),
            nn.Softmax(dim=-1),
        ).eval()

        X = torch.rand(8, 3)
        y = proc.forward(model, X, nsample=5000)
        assert y.shape == (8, 4)
        # Softmax layer guarantees normalization
        assert torch.allclose(y.sum(dim=1), torch.ones(8), atol=1e-5)

    def test_local_vs_remote_ab_force_simulation(self, remote_processor):
        """Remote vs forced-local A/B; distributions should be reasonably close."""
        q = _make_layer(
            6,
            2,
            input_size=2,
            computation_space=ComputationSpace.UNBUNCHED,
            trainable=True,
        )
        X = torch.rand(4, 2)
        proc = MerlinProcessor(remote_processor)

        y_remote = proc.forward(q, X, nsample=5000)

        q.force_local = True
        y_local = proc.forward(q, X, nsample=5000)

        assert y_remote.shape == y_local.shape == (4, comb(6, 2))
        # Allow mild sampling noise tolerance
        diff = (y_local - y_remote).abs().mean().item()
        assert diff < 0.15

    def test_monitor_status_and_cancellation(self, remote_processor):
        """Status polling and cooperative cancellation should raise CancelledError."""
        q = _make_layer(
            6,
            2,
            input_size=2,
            computation_space=ComputationSpace.UNBUNCHED,
            trainable=True,
        )
        proc = MerlinProcessor(remote_processor)
        fut = proc.forward_async(q, torch.rand(16, 2), nsample=40_000, timeout=None)

        # Give backend time to allocate & start
        _ = fut.status()
        # Wait until at least one job is visible or it finishes very fast
        end = time.time() + 20.0
        while not fut.done() and len(fut.job_ids) == 0 and time.time() < end:
            time.sleep(0.05)

        # If it already finished, there's nothing to cancel
        if fut.done():
            pytest.skip("Backend finished too quickly to test cancellation.")

        fut.cancel_remote()
        with pytest.raises(CancelledError):
            fut.wait()

    def test_high_throughput_batching_with_chunking(self, remote_processor):
        """Large batch is chunked; counters and output shape match expectations."""
        q = _make_layer(
            6,
            2,
            input_size=2,
            computation_space=ComputationSpace.UNBUNCHED,
            trainable=True,
        )

        proc = MerlinProcessor(
            remote_processor,
            microbatch_size=8,  # as in docs
            chunk_concurrency=2,
            timeout=300.0,  # 5 min for chunked jobs
        )

        X = torch.rand(64, 2)
        fut = proc.forward_async(q, X, nsample=3000, timeout=300.0)
        y = _wait_future(fut, timeout_s=300.0)

        # Expect stitching of 64 x C(6,2)=15
        assert y.shape == (64, comb(6, 2))

        st = fut.status()
        # At least 8 chunks for 64 with max_batch_size=8
        assert st.get("chunks_total", 0) >= 8
        assert st.get("chunks_done", 0) >= 8
        # We don't assert on active_chunks because it can be 0 by the time we check

    def test_device_and_dtype_roundtrip(self, remote_processor):
        """Ensure outputs preserve input device/dtype semantics after remote call."""
        q = _make_layer(
            6,
            2,
            input_size=2,
            computation_space=ComputationSpace.UNBUNCHED,
            trainable=True,
        )
        proc = MerlinProcessor(remote_processor, timeout=300.0)

        x_f32 = torch.rand(3, 2, dtype=torch.float32)
        y_f32 = proc.forward(q, x_f32, nsample=3000)
        assert y_f32.dtype == torch.float32

        # If CUDA is available, verify round-trip back to CUDA device
        if torch.cuda.is_available():
            q_cuda = _make_layer(
                6,
                2,
                input_size=2,
                computation_space=ComputationSpace.UNBUNCHED,
                trainable=True,
            ).to("cuda")
            x_cuda = torch.rand(2, 2, device="cuda", dtype=torch.float32)
            y_cuda = proc.forward(q_cuda, x_cuda, nsample=1500)
            assert y_cuda.device.type == "cuda"
            assert y_cuda.shape[1] == comb(6, 2)
