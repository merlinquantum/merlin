"""No-cloud unit tests for the typed CallState per-call state object (PML-303)."""

from __future__ import annotations

import threading

from merlin.core.merlin_processor import CallState, JobStatus


class TestCallStateCreation:
    def test_new_assigns_short_unique_call_id(self):
        """CallState.new() assigns an 8-char hex call id, unique per call."""
        first = CallState.new()
        second = CallState.new()

        assert isinstance(first.call_id, str)
        assert len(first.call_id) == 8
        int(first.call_id, 16)  # hex-parsable
        assert first.call_id != second.call_id

    def test_initial_state_is_idle_and_empty(self):
        """A fresh state carries no cancellation, status, jobs, or chunks."""
        state = CallState.new()

        assert state.cancel_requested is False
        assert state.current_status is None
        assert state.job_ids == []
        assert state.chunks_total == 0
        assert state.chunks_done == 0
        assert state.active_chunks == 0


class TestCancelPropagation:
    def test_request_cancel_sets_flag(self):
        """request_cancel() flips the cooperative cancellation flag."""
        state = CallState.new()

        state.request_cancel()

        assert state.cancel_requested is True

    def test_request_cancel_is_idempotent(self):
        """Repeated cancellation requests keep the flag set."""
        state = CallState.new()

        state.request_cancel()
        state.request_cancel()

        assert state.cancel_requested is True

    def test_cancel_is_visible_across_threads(self):
        """A cancel requested on one thread is observed by another."""
        state = CallState.new()
        observed = threading.Event()

        def worker():
            while not state.cancel_requested:
                pass
            observed.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        state.request_cancel()
        assert observed.wait(timeout=5.0)
        thread.join(timeout=5.0)


class TestJobIdRecording:
    def test_record_job_id_appends(self):
        """Recorded job ids accumulate in observation order."""
        state = CallState.new()

        state.record_job_id("job-1")
        state.record_job_id("job-2")

        assert state.job_ids == ["job-1", "job-2"]

    def test_record_job_id_deduplicates(self):
        """Re-observing the same job id (repeated polls) records it once."""
        state = CallState.new()

        state.record_job_id("job-123")
        state.record_job_id("job-123")
        state.record_job_id("job-456")
        state.record_job_id("job-123")

        assert state.job_ids == ["job-123", "job-456"]

    def test_job_ids_list_identity_is_stable(self):
        """The job_ids list object is stable so future.job_ids stays live."""
        state = CallState.new()
        shared_reference = state.job_ids

        state.record_job_id("job-1")

        assert shared_reference is state.job_ids
        assert shared_reference == ["job-1"]


class TestChunkCounters:
    def test_ticket_example_transitions(self):
        """The PML-303 minimal example holds: one started+finished chunk."""
        state = CallState.new()
        state.record_job_id("job-123")
        state.set_current_status(state="RUNNING", progress=0.5, message=None)
        state.mark_chunk_started()
        state.mark_chunk_finished()

        snapshot = state.status_snapshot()

        assert snapshot["chunks_done"] == 1
        assert snapshot["active_chunks"] == 0

    def test_add_planned_chunks_accumulates(self):
        """Planned chunk counts accumulate across quantum leaves."""
        state = CallState.new()

        state.add_planned_chunks(4)
        state.add_planned_chunks(2)

        assert state.chunks_total == 6

    def test_start_and_finish_track_in_flight_chunks(self):
        """Start/finish transitions drive active and done counters."""
        state = CallState.new()
        state.add_planned_chunks(2)

        state.mark_chunk_started()
        state.mark_chunk_started()
        assert state.active_chunks == 2
        assert state.chunks_done == 0

        state.mark_chunk_finished()
        assert state.active_chunks == 1
        assert state.chunks_done == 1

        state.mark_chunk_finished()
        assert state.active_chunks == 0
        assert state.chunks_done == 2

    def test_finish_never_drives_active_chunks_negative(self):
        """An unmatched finish clamps active_chunks at zero."""
        state = CallState.new()

        state.mark_chunk_finished()

        assert state.active_chunks == 0
        assert state.chunks_done == 1


class TestStatusSnapshot:
    def test_snapshot_defaults_to_idle(self):
        """Before any backend status is recorded, the state is IDLE."""
        snapshot = CallState.new().status_snapshot()

        assert snapshot == {
            "state": "IDLE",
            "progress": 0.0,
            "message": None,
            "chunks_total": 0,
            "chunks_done": 0,
            "active_chunks": 0,
        }

    def test_snapshot_reports_complete_when_future_done_without_status(self):
        """A resolved future with no recorded backend status is COMPLETE."""
        snapshot = CallState.new().status_snapshot(future_done=True)

        assert snapshot["state"] == "COMPLETE"

    def test_snapshot_passes_through_recorded_status(self):
        """Recorded backend status fields flow into the snapshot unchanged."""
        state = CallState.new()
        state.set_current_status(state="RUNNING", progress=0.5, message="halfway")

        snapshot = state.status_snapshot()

        assert snapshot["state"] == "RUNNING"
        assert snapshot["progress"] == 0.5
        assert snapshot["message"] == "halfway"

    def test_snapshot_prefers_recorded_status_over_future_done(self):
        """A recorded backend status wins over the COMPLETE fallback."""
        state = CallState.new()
        state.set_current_status(state="RUNNING", progress=1.0, message=None)

        snapshot = state.status_snapshot(future_done=True)

        assert snapshot["state"] == "RUNNING"

    def test_snapshot_includes_chunk_counters(self):
        """Snapshots surface the chunk counters after transitions."""
        state = CallState.new()
        state.add_planned_chunks(3)
        state.mark_chunk_started()
        state.mark_chunk_started()
        state.mark_chunk_finished()

        snapshot = state.status_snapshot()

        assert snapshot["chunks_total"] == 3
        assert snapshot["chunks_done"] == 1
        assert snapshot["active_chunks"] == 1

    def test_set_current_status_stores_immutable_job_status(self):
        """Recorded status is a frozen JobStatus value object."""
        state = CallState.new()
        state.set_current_status(state="RUNNING", progress=0.25, message=None)

        status = state.current_status

        assert isinstance(status, JobStatus)
        assert status.state == "RUNNING"
        assert status.progress == 0.25
        assert status.message is None
