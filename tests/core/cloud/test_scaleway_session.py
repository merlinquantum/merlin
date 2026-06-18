# tests/core/cloud/test_scaleway_session.py
"""
Tests for MerlinProcessor with Scaleway session backend.

These tests verify that the MerlinProcessor works correctly when initialized
with a Scaleway Session (ISession) instead of a RemoteProcessor.

Requires environment variables:
    SCW_PROJECT_ID: Scaleway project ID
    SCW_SECRET_KEY: Scaleway API token

Run with: pytest --run-scaleway-tests tests/core/cloud/test_scaleway_session.py
"""

from __future__ import annotations

import time
from concurrent.futures import CancelledError
from math import comb

import pytest
import torch
import torch.nn as nn

from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.core.computation_space import ComputationSpace
from merlin.core.merlin_processor import MerlinProcessor
from merlin.measurement.strategies import MeasurementStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_layer(
    n_modes: int,
    n_photons: int,
    input_size: int,
    *,
    computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
    trainable: bool = True,
) -> QuantumLayer:
    """Helper to create a QuantumLayer for testing."""
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


def _wait_future(fut, timeout_s: float = 120.0):
    """Wait for a future to complete with timeout."""
    end = time.time() + timeout_s
    while not fut.done():
        if time.time() >= end:
            raise TimeoutError("Timeout waiting for Merlin future")
        time.sleep(0.01)
    return fut.value()


def _spin_until(pred, timeout_s: float = 10.0, sleep_s: float = 0.02) -> bool:
    """Spin until predicate is true or timeout."""
    start = time.time()
    while not pred():
        if time.time() - start > timeout_s:
            return False
        time.sleep(sleep_s)
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScalewaySessionBasic:
    """Basic functionality tests with Scaleway session."""

    def test_processor_accepts_session(self, scaleway_session):
        """MerlinProcessor should accept a Scaleway Session without error."""
        proc = MerlinProcessor(
            session=scaleway_session,
            microbatch_size=32,
            timeout=300.0,
        )
        assert proc.backend_name is not None
        # Verify backwards-compatible attributes are set
        assert proc.session is not None or proc.remote_processor is not None

    # Uncomment when there is a probs backend
    def test_confirm_backend_has_probs(self, scaleway_session_probs):
        """Confirm backend has probs in available_commands."""
        proc = MerlinProcessor(session=scaleway_session_probs)

        # Check available commands
        print(f"\nBackend available_commands: {proc.available_commands}")
        print(f"Backend capabilities: {proc.backend_capabilities.available_commands}")

        # Confirm probs is available
        assert (
            "probs" in proc.available_commands
        ), f"'probs' not in available commands: {proc.available_commands}"
        assert (
            "probs" in proc.backend_capabilities.available_commands
        ), f"'probs' not in backend capabilities: {proc.backend_capabilities.available_commands}"

    def test_simple_forward_probs_zero(self, scaleway_session_probs):
        """Basic synchronous forward pass with nsample=0 should use probs."""
        proc = MerlinProcessor(
            session=scaleway_session_probs,
            microbatch_size=32,
            timeout=300.0,
            max_shots_per_call=100,
        )

        # Confirm probs is available
        print(f"\nAvailable commands: {proc.available_commands}")
        assert "probs" in proc.available_commands, "probs not available"

        q = _make_layer(
            6, 2, input_size=2, computation_space=ComputationSpace.UNBUNCHED
        )
        X = torch.rand(4, 2)

        print("Running forward with nsample=0 - should use PROBS command")
        y = proc.forward(q, X, nsample=0)

        expected_output_size = comb(6, 2)  # 15
        assert y.shape == (4, expected_output_size)
        # Output should be normalized probabilities
        assert torch.all(y >= 0)
        assert torch.allclose(y.sum(dim=1), torch.ones(4), atol=0.001)

    def test_simple_forward_probs_None(self, scaleway_session_probs):
        """Basic synchronous forward pass with nsample=None should use probs."""
        proc = MerlinProcessor(
            session=scaleway_session_probs,
            microbatch_size=32,
            timeout=300.0,
            max_shots_per_call=100,
        )

        # Confirm probs is available
        print(f"\nAvailable commands: {proc.available_commands}")
        assert "probs" in proc.available_commands, "probs not available"

        q = _make_layer(
            6, 2, input_size=2, computation_space=ComputationSpace.UNBUNCHED
        )
        X = torch.rand(4, 2)

        print("Running forward with nsample=None - should use PROBS command")
        y = proc.forward(q, X, nsample=None)

        expected_output_size = comb(6, 2)  # 15
        assert y.shape == (4, expected_output_size)
        # Output should be normalized probabilities
        assert torch.all(y >= 0)
        assert torch.allclose(y.sum(dim=1), torch.ones(4), atol=0.001)

    def test_simple_forward_sample(self, scaleway_session):
        """Basic synchronous forward pass with nsample=1000 should use sample_count."""
        proc = MerlinProcessor(
            session=scaleway_session,
            microbatch_size=32,
            timeout=300.0,
            max_shots_per_call=100,
        )

        # Check available commands
        print(f"\nAvailable commands: {proc.available_commands}")

        q = _make_layer(
            6, 2, input_size=2, computation_space=ComputationSpace.UNBUNCHED
        )
        X = torch.rand(4, 2)

        print("Running forward with nsample=1000 - should use SAMPLE_COUNT command")
        with pytest.warns(UserWarning, match=r"Number of samples requested"):
            y = proc.forward(q, X, nsample=1000)

        expected_output_size = comb(6, 2)  # 15
        assert y.shape == (4, expected_output_size)
        # Output should be normalized probabilities
        assert torch.all(y >= 0)
        assert torch.allclose(y.sum(dim=1), torch.ones(4), atol=0.001)

    def test_forward_async(self, scaleway_session):
        """Asynchronous forward pass should work."""
        proc = MerlinProcessor(
            session=scaleway_session,
            microbatch_size=32,
            timeout=300.0,
            max_shots_per_call=100,
        )

        q = _make_layer(
            6, 2, input_size=2, computation_space=ComputationSpace.UNBUNCHED
        )
        X = torch.rand(4, 2)
        fut = proc.forward_async(q, X, nsample=100)

        # Future should have expected attributes
        assert hasattr(fut, "cancel_remote")
        assert hasattr(fut, "status")
        assert hasattr(fut, "job_ids")

        y = _wait_future(fut)
        assert y.shape == (4, comb(6, 2))


class TestScalewaySessionPipeline:
    """Test MerlinProcessor with Sequential models using Scaleway session."""

    def test_sequential_model(self, scaleway_session):
        """Full pipeline with classical + quantum layers should work."""
        proc = MerlinProcessor(
            session=scaleway_session,
            microbatch_size=32,
            timeout=300.0,
            max_shots_per_call=100,
        )

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
            nn.Linear(15, 4, bias=False),  # 15 = C(6,2)
            nn.Softmax(dim=-1),
        ).eval()

        X = torch.rand(8, 3)
        y = proc.forward(model, X, nsample=100)

        assert y.shape == (8, 4)
        # Softmax guarantees normalization
        assert torch.allclose(y.sum(dim=1), torch.ones(8), atol=1e-5)

    def test_mixed_local_and_remote(self, scaleway_session):
        """Model with one remote and one local quantum layer."""
        proc = MerlinProcessor(
            session=scaleway_session,
            microbatch_size=32,
            timeout=300.0,
            max_shots_per_call=100,
        )

        q1 = _make_layer(
            5, 2, input_size=2, computation_space=ComputationSpace.UNBUNCHED
        )
        q2 = _make_layer(
            6, 2, input_size=3, computation_space=ComputationSpace.UNBUNCHED
        )
        q2.force_local = True  # Force second layer to run locally

        dist1 = comb(5, 2)  # 10
        dist2 = comb(6, 2)  # 15

        adapter = nn.Linear(dist1, 3, bias=False)
        model = nn.Sequential(q1, adapter, q2).eval()

        X = torch.rand(4, 2)
        fut = proc.forward_async(model, X, nsample=100)
        y = _wait_future(fut)

        assert y.shape == (4, dist2)
        # Only q1 should have been offloaded
        assert len(fut.job_ids) >= 1


class TestScalewaySessionChunking:
    """Test batching and chunking behavior with Scaleway session."""

    def test_chunked_batch(self, scaleway_session):
        """ISession supports chunking — batch is split by microbatch_size."""
        proc = MerlinProcessor(
            session=scaleway_session,
            microbatch_size=8,
            timeout=300.0,
            max_shots_per_call=100,
        )

        q = _make_layer(
            6, 2, input_size=2, computation_space=ComputationSpace.UNBUNCHED
        )
        X = torch.rand(32, 2)

        fut = proc.forward_async(q, X, nsample=100)
        y = _wait_future(fut)

        assert y.shape == (32, comb(6, 2))

        st = fut.status()
        assert st.get("chunks_total", 0) == 4  # 32 / 8
        assert st.get("chunks_done", 0) == 4

    def test_concurrent_chunks(self, scaleway_session):
        """ISession with chunk_concurrency > 1 runs chunks in parallel."""
        proc = MerlinProcessor(
            session=scaleway_session,
            microbatch_size=4,
            timeout=300.0,
            max_shots_per_call=100,
            chunk_concurrency=4,
        )

        q = _make_layer(
            6, 2, input_size=2, computation_space=ComputationSpace.UNBUNCHED
        )
        X = torch.rand(16, 2)  # 4 chunks of 4

        fut = proc.forward_async(q, X, nsample=100)
        y = _wait_future(fut)

        assert y.shape == (16, comb(6, 2))
        assert len(fut.job_ids) >= 1

        st = fut.status()
        assert st.get("chunks_total", 0) == 4  # 16 / 4
        assert st.get("chunks_done", 0) == 4


class TestScalewaySessionStatus:
    """Test status monitoring and cancellation with Scaleway session."""

    def test_status_polling(self, scaleway_session):
        """Status should be queryable during execution."""
        proc = MerlinProcessor(
            session=scaleway_session,
            microbatch_size=32,
            timeout=300.0,
            max_shots_per_call=1000,
        )

        q = _make_layer(
            6, 2, input_size=2, computation_space=ComputationSpace.UNBUNCHED
        )
        X = torch.rand(8, 2)

        fut = proc.forward_async(q, X, nsample=1000)

        # Poll status a few times
        statuses = []
        for _ in range(5):
            st = fut.status()
            statuses.append(st)
            if fut.done():
                break
            time.sleep(0.1)

        # Status should have expected keys
        assert all("state" in st for st in statuses)
        assert all("chunks_total" in st for st in statuses)

        # Wait for completion
        y = _wait_future(fut)
        assert y.shape == (8, comb(6, 2))

    def test_cancellation(self, scaleway_session):
        """Cancellation should raise CancelledError."""
        proc = MerlinProcessor(
            session=scaleway_session,
            microbatch_size=32,
            timeout=36000.0,  # Large timeout - we'll cancel manually
            max_shots_per_call=50000,
        )

        q = _make_layer(
            6, 2, input_size=2, computation_space=ComputationSpace.UNBUNCHED
        )
        X = torch.rand(16, 2)

        fut = proc.forward_async(q, X, nsample=50000)

        # Wait for job to start
        end = time.time() + 20.0
        while not fut.done() and len(fut.job_ids) == 0 and time.time() < end:
            time.sleep(0.05)

        if fut.done():
            pytest.skip("Job finished too quickly to test cancellation")

        fut.cancel_remote()

        with pytest.raises(CancelledError):
            fut.wait()


class TestScalewaySessionContextManager:
    """Test MerlinProcessor context manager with Scaleway session."""

    def test_context_manager_cleanup(self, scaleway_session):
        """Context manager should clean up properly."""
        with MerlinProcessor(
            session=scaleway_session,
            microbatch_size=32,
            timeout=300.0,
            max_shots_per_call=100,
        ) as proc:
            q = _make_layer(
                6, 2, input_size=2, computation_space=ComputationSpace.UNBUNCHED
            )
            X = torch.rand(4, 2)
            y = proc.forward(q, X, nsample=100)
            assert y.shape == (4, comb(6, 2))

        # After exiting, processor should be closed
        with pytest.raises(RuntimeError, match="closed"):
            proc.forward(q, X, nsample=100)


class TestScalewaySessionBackwardsCompat:
    """Test backwards compatibility attributes with Scaleway session."""

    def test_public_session_attribute(self, scaleway_session):
        """The public .session attribute should be accessible."""
        proc = MerlinProcessor(session=scaleway_session)

        # When initialized with ISession, .session should be set
        # and .remote_processor should be None
        assert proc.session is not None
        assert proc.remote_processor is None

    def test_backend_name_set(self, scaleway_session):
        """backend_name should be set from session."""
        proc = MerlinProcessor(session=scaleway_session)
        assert proc.backend_name is not None
        assert isinstance(proc.backend_name, str)
