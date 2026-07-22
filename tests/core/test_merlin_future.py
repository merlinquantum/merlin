"""No-cloud unit tests for the typed MerlinFuture async handle (PML-304).

Covers the explicit async contract that replaced the runtime monkey-patched
future attributes: return type, status payload, job id exposure, cooperative
cancellation, and the synchronous forward() built on top of the handle.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from unittest.mock import MagicMock

import perceval as pcvl
import pytest
import torch
import torch.nn as nn

from merlin.algorithms.module import MerlinModule
from merlin.core.merlin_processor import CallState, MerlinFuture, MerlinProcessor


def make_future(**overrides) -> tuple[MerlinFuture, CallState, MagicMock]:
    """Build a MerlinFuture around a fresh CallState and recorded cancel_all."""
    state = overrides.get("state", CallState.new())
    cancel_all = overrides.get("cancel_all", MagicMock(name="cancel_all"))
    return MerlinFuture(state, cancel_all), state, cancel_all


def make_local_processor() -> MerlinProcessor:
    """Build a MerlinProcessor around a local mocked AProcessor."""
    from perceval.runtime import AProcessor

    processor = MagicMock(spec=AProcessor)
    processor.is_remote = False
    processor.name = "local:slos"
    processor.available_commands = ["probs"]
    return MerlinProcessor(processor=processor)


class PassthroughLeaf(MerlinModule):
    """Quantum leaf that never offloads, so pipelines run locally."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def should_offload(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0

    def export_config(self):
        return {
            "circuit": pcvl.Circuit(m=2),
            "input_state": [1, 0],
            "input_param_order": ["theta_0", "theta_1"],
        }


class TestReturnType:
    def test_forward_async_returns_merlin_future(self):
        proc = make_local_processor()
        module = PassthroughLeaf()
        module.eval()

        fut = proc.forward_async(module, torch.ones(1, 2))

        assert isinstance(fut, MerlinFuture)
        assert isinstance(fut, torch.futures.Future)
        fut.wait()

    def test_contract_methods_are_declared_not_patched(self):
        """The async contract lives on the class, not on instances."""
        assert callable(MerlinFuture.status)
        assert callable(MerlinFuture.cancel_remote)
        assert isinstance(MerlinFuture.job_ids, property)

        fut, _, _ = make_future()
        assert "status" not in fut.__dict__
        assert "cancel_remote" not in fut.__dict__
        assert "job_ids" not in fut.__dict__


class TestStatus:
    def test_status_payload_shape_when_idle(self):
        fut, _, _ = make_future()

        assert fut.status() == {
            "state": "IDLE",
            "progress": 0.0,
            "message": None,
            "chunks_total": 0,
            "chunks_done": 0,
            "active_chunks": 0,
        }

    def test_status_reflects_call_state_progress(self):
        fut, state, _ = make_future()
        state.add_planned_chunks(2)
        state.mark_chunk_started()
        state.set_current_status(state="RUNNING", progress=0.5, message="halfway")

        status = fut.status()

        assert status["state"] == "RUNNING"
        assert status["progress"] == 0.5
        assert status["message"] == "halfway"
        assert status["chunks_total"] == 2
        assert status["active_chunks"] == 1

    def test_status_reports_complete_when_done_without_backend_status(self):
        fut, _, _ = make_future()
        fut.set_result(torch.ones(1))

        assert fut.status()["state"] == "COMPLETE"


class TestJobIds:
    def test_job_ids_is_live_view_of_call_state(self):
        fut, state, _ = make_future()

        assert fut.job_ids == []
        state.record_job_id("job-1")
        assert fut.job_ids == ["job-1"]
        assert fut.job_ids is state.job_ids


class TestCancelRemote:
    def test_cancel_remote_requests_cancel_and_cancels_jobs(self):
        fut, state, cancel_all = make_future()

        fut.cancel_remote()

        assert state.cancel_requested is True
        cancel_all.assert_called_once_with()

    def test_cancel_remote_resolves_future_with_cancelled_error(self):
        fut, _, _ = make_future()

        fut.cancel_remote()

        with pytest.raises(CancelledError, match="Remote call was cancelled"):
            fut.wait()

    def test_cancel_remote_after_completion_keeps_result(self):
        """Cancelling a finished call still cancels jobs but keeps the result."""
        fut, state, cancel_all = make_future()
        result = torch.ones(1)
        fut.set_result(result)

        fut.cancel_remote()

        assert state.cancel_requested is True
        cancel_all.assert_called_once_with()
        assert torch.equal(fut.wait(), result)

    def test_forward_async_cancel_propagates_cancelled_error(self):
        proc = make_local_processor()

        class BlockingLeaf(PassthroughLeaf):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                import time

                time.sleep(0.5)
                return x

        module = BlockingLeaf()
        module.eval()

        fut = proc.forward_async(module, torch.ones(1, 2))
        fut.cancel_remote()

        with pytest.raises(CancelledError, match="Remote call was cancelled"):
            fut.wait()


class TestSynchronousForward:
    def test_forward_waits_on_the_typed_handle(self):
        proc = make_local_processor()
        module = PassthroughLeaf()
        module.eval()
        x = torch.ones(2, 2)

        output = proc.forward(module, x)

        torch.testing.assert_close(output, x * 2.0)
