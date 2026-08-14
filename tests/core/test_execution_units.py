"""No-cloud unit tests for the extracted execution units (PML-305).

BatchChunker and RemoteJobRunner are exercised directly with fake jobs,
samplers, and processors — no MerlinProcessor instance and no cloud access.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import merlin.core.execution as execution_module
import merlin.core.perceval_adapter as perceval_adapter_module
from merlin.core.execution import BatchChunker, RemoteJobRunner
from merlin.core.merlin_processor import CallState
from merlin.core.perceval_adapter import RemoteJobFailedError

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeCommand:
    """Sampler command that records how it was submitted."""

    def __init__(self, job=None) -> None:
        self.executed = False
        self.name = None
        self.execute_kwargs = None
        self._job = job if job is not None else self

    def execute_async(self, **kwargs):
        self.executed = True
        self.execute_kwargs = kwargs
        return self._job


class FakeSampler:
    """Sampler fake exposing the commands and iteration API used by run_chunk."""

    def __init__(self) -> None:
        self.probs = FakeCommand()
        self.sample_count = FakeCommand()
        self.samples = FakeCommand()
        self.cleared = False
        self.iterations: list[dict] = []

    def clear_iterations(self) -> None:
        self.cleared = True

    def add_iteration(self, circuit_params) -> None:
        self.iterations.append(circuit_params)


class FakeJob:
    """Remote job fake with scripted polling behavior."""

    def __init__(
        self,
        *,
        job_id: str | None = "job-1",
        is_complete: bool = True,
        is_failed: bool = False,
        status=None,
        result_events=None,
    ) -> None:
        self.id = job_id
        self.is_complete = is_complete
        self.is_failed = is_failed
        self.status = status
        self.cancelled = False
        self.get_results_calls = 0
        self._result_events = (
            [{"results_list": []}] if result_events is None else list(result_events)
        )

    def cancel(self) -> None:
        self.cancelled = True

    def get_results(self):
        self.get_results_calls += 1
        event = (
            self._result_events.pop(0)
            if len(self._result_events) > 1
            else self._result_events[0]
        )
        if isinstance(event, BaseException):
            raise event
        return event


def make_runner(**overrides) -> RemoteJobRunner:
    """Build a RemoteJobRunner with recording no-op dependencies."""
    deps = {
        "create_processor": MagicMock(name="create_processor"),
        "get_available_commands": lambda: ("probs", "sample_count", "samples"),
        "effective_sample_count": lambda nsample: (
            10_000 if nsample is None else int(nsample)
        ),
        "get_max_shots_per_call": lambda: 100_000,
        "default_shots_per_call": 10_000,
        "map_results": MagicMock(return_value=torch.tensor([[1.0]])),
        "register_job": MagicMock(name="register_job"),
        "unregister_job": MagicMock(name="unregister_job"),
        "get_microbatch_limit": lambda: 32,
    }
    deps.update(overrides)
    return RemoteJobRunner(**deps)


def make_chunk_config() -> SimpleNamespace:
    """Return the validated-config shape consumed by run_chunk."""
    return SimpleNamespace(
        circuit=MagicMock(name="circuit"),
        input_state=[1, 0],
        input_param_order=["theta_0", "theta_1"],
    )


# ---------------------------------------------------------------------------
# BatchChunker
# ---------------------------------------------------------------------------


class TestSplitBatch:
    def test_exact_multiple_produces_equal_chunks(self):
        """An exact multiple splits into equal-size chunks."""
        assert BatchChunker.split_batch(8, 4) == [(0, 4), (4, 8)]

    def test_remainder_produces_short_final_chunk(self):
        """A remainder yields a short final chunk."""
        assert BatchChunker.split_batch(10, 4) == [(0, 4), (4, 8), (8, 10)]

    def test_batch_smaller_than_microbatch_is_one_chunk(self):
        """Small batches stay as a single chunk."""
        assert BatchChunker.split_batch(3, 32) == [(0, 3)]

    def test_empty_batch_produces_no_chunks(self):
        """An empty batch produces no chunks."""
        assert BatchChunker.split_batch(0, 32) == []


class TestBatchChunkerRunChunks:
    def test_outputs_are_stitched_in_chunk_order(self):
        """Chunk outputs are concatenated in submission order."""

        def run_chunk(layer, config, chunk, nsample, state, deadline, job_base_label):
            # Later chunks finish first to prove ordering is positional.
            time.sleep(0.03 if chunk[0, 0] == 0 else 0.0)
            return chunk.clone()

        chunker = BatchChunker(
            run_chunk=run_chunk, chunk_concurrency=4, cancel_all=MagicMock()
        )
        input_tensor = torch.arange(8, dtype=torch.float32).reshape(4, 2)

        output = chunker.run_chunks(
            object(),
            make_chunk_config(),
            input_tensor,
            BatchChunker.split_batch(4, 2),
            None,
            CallState.new(),
            None,
        )

        torch.testing.assert_close(output, input_tensor)

    def test_concurrency_is_bounded(self):
        """No more than chunk_concurrency chunk jobs run at once."""
        lock = threading.Lock()
        active = 0
        max_active = 0

        def run_chunk(layer, config, chunk, nsample, state, deadline, job_base_label):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return torch.ones(chunk.shape[0], 1)

        chunker = BatchChunker(
            run_chunk=run_chunk, chunk_concurrency=2, cancel_all=MagicMock()
        )

        chunker.run_chunks(
            object(),
            make_chunk_config(),
            torch.zeros(8, 2),
            BatchChunker.split_batch(8, 1),
            None,
            CallState.new(),
            None,
        )

        assert max_active <= 2

    def test_chunk_labels_carry_call_id_and_position(self):
        """Job base labels identify the layer, call, and chunk position."""
        labels: list[str] = []

        def run_chunk(layer, config, chunk, nsample, state, deadline, job_base_label):
            labels.append(job_base_label)
            return torch.ones(chunk.shape[0], 1)

        chunker = BatchChunker(
            run_chunk=run_chunk, chunk_concurrency=1, cancel_all=MagicMock()
        )
        state = CallState.new()
        layer = SimpleNamespace(name="qlayer")

        chunker.run_chunks(
            layer,
            make_chunk_config(),
            torch.zeros(4, 2),
            BatchChunker.split_batch(4, 2),
            None,
            state,
            None,
        )

        assert labels == [
            f"mer:qlayer:{state.call_id}:1/2",
            f"mer:qlayer:{state.call_id}:2/2",
        ]

    def test_state_counters_reflect_chunk_lifecycle(self):
        """Planned/done counters are updated and drain to zero active."""
        state = CallState.new()

        chunker = BatchChunker(
            run_chunk=lambda *a, **k: torch.ones(1, 1),
            chunk_concurrency=2,
            cancel_all=MagicMock(),
        )

        chunker.run_chunks(
            object(),
            make_chunk_config(),
            torch.zeros(3, 2),
            BatchChunker.split_batch(3, 1),
            None,
            state,
            None,
        )

        assert state.chunks_total == 3
        assert state.chunks_done == 3
        assert state.active_chunks == 0

    def test_first_chunk_error_is_raised(self):
        """A failing chunk propagates its error after the pool drains."""

        def run_chunk(layer, config, chunk, nsample, state, deadline, job_base_label):
            if chunk[0, 0] == 2:
                raise RuntimeError("chunk exploded")
            return torch.ones(chunk.shape[0], 1)

        chunker = BatchChunker(
            run_chunk=run_chunk, chunk_concurrency=1, cancel_all=MagicMock()
        )

        with pytest.raises(RuntimeError, match="chunk exploded"):
            chunker.run_chunks(
                object(),
                make_chunk_config(),
                torch.tensor([[0.0], [2.0]]),
                BatchChunker.split_batch(2, 1),
                None,
                CallState.new(),
                None,
            )

    def test_deadline_cancels_all_and_raises_timeout(self):
        """An elapsed deadline cancels in-flight jobs and raises TimeoutError."""
        cancel_all = MagicMock()
        release = threading.Event()

        def run_chunk(layer, config, chunk, nsample, state, deadline, job_base_label):
            release.wait(timeout=5.0)
            return torch.ones(chunk.shape[0], 1)

        chunker = BatchChunker(
            run_chunk=run_chunk, chunk_concurrency=1, cancel_all=cancel_all
        )

        try:
            with pytest.raises(TimeoutError, match="remote cancel issued"):
                chunker.run_chunks(
                    object(),
                    make_chunk_config(),
                    torch.zeros(1, 2),
                    BatchChunker.split_batch(1, 1),
                    None,
                    CallState.new(),
                    time.time() + 0.05,
                )
        finally:
            release.set()

        cancel_all.assert_called_once_with()

    def test_concurrency_floor_is_one(self):
        """A non-positive concurrency setting still runs chunks serially."""
        chunker = BatchChunker(
            run_chunk=lambda *a, **k: torch.ones(1, 1),
            chunk_concurrency=0,
            cancel_all=MagicMock(),
        )

        output = chunker.run_chunks(
            object(),
            make_chunk_config(),
            torch.zeros(2, 2),
            BatchChunker.split_batch(2, 1),
            None,
            CallState.new(),
            None,
        )

        assert output.shape == (2, 1)


# ---------------------------------------------------------------------------
# RemoteJobRunner.submit_job
# ---------------------------------------------------------------------------


class TestSubmitJob:
    def test_probs_selected_when_available_and_nsample_none(self):
        """probs is selected when available and no shots requested."""
        runner = make_runner()
        sampler = FakeSampler()

        job, is_probability = runner.submit_job(sampler, None, "label")

        assert is_probability is True
        assert job is sampler.probs
        assert sampler.probs.executed is True
        assert sampler.probs.name == "label:probs"

    def test_sampling_selected_when_nsample_positive(self):
        """A positive nsample forces the sampling command path."""
        runner = make_runner()
        sampler = FakeSampler()

        job, is_probability = runner.submit_job(sampler, 500, "label")

        assert is_probability is False
        assert job is sampler.sample_count
        assert sampler.sample_count.execute_kwargs == {"max_samples": 500}

    def test_samples_used_when_sample_count_unavailable(self):
        """samples is the fallback when sample_count is unavailable."""
        runner = make_runner(get_available_commands=lambda: ("samples",))
        sampler = FakeSampler()

        job, is_probability = runner.submit_job(sampler, 100, None)

        assert job is sampler.samples
        assert is_probability is False

    def test_sample_count_is_default_fallback_without_commands(self):
        """sample_count is attempted when no commands are advertised."""
        runner = make_runner(get_available_commands=lambda: ())
        sampler = FakeSampler()

        job, _ = runner.submit_job(sampler, None, None)

        assert job is sampler.sample_count

    def test_shot_count_flows_through_effective_sample_count(self):
        """Submitted shots come from the injected effective_sample_count."""
        runner = make_runner(effective_sample_count=lambda nsample: 42)
        sampler = FakeSampler()

        runner.submit_job(sampler, 999, None)

        assert sampler.sample_count.execute_kwargs == {"max_samples": 42}

    def test_serializable_iterator_normalized_before_submit(self):
        """Perceval 1.2 iterator payloads are flattened to plain lists."""
        runner = make_runner()
        sampler = FakeSampler()
        iterations = [{"theta_0": 0.1}, {"theta_0": 0.2}]
        iterator = SimpleNamespace(iterations=iterations)
        sampler._iterator = iterator
        sampler.probs._request_data = {"payload": {"iterator": iterator}}

        runner.submit_job(sampler, None, None)

        assert sampler.probs._request_data["payload"]["iterator"] == iterations


class TestCappedName:
    def test_short_names_are_sanitized_verbatim(self):
        """Short job names are sanitized but otherwise kept verbatim."""
        runner = make_runner()

        assert runner._capped_name("mer:layer 1", "probs") == "mer:layer_1:probs"

    def test_long_names_are_capped_with_hash_suffix(self):
        """Long job names are capped with a stable hash suffix."""
        runner = make_runner(job_name_max=20)

        name = runner._capped_name("mer:" + "x" * 60, "sample_count")

        assert len(name) == 20
        assert "~" in name


# ---------------------------------------------------------------------------
# RemoteJobRunner.run_chunk
# ---------------------------------------------------------------------------


def run_chunk_with(runner, *, nsample=None, state=None, deadline=None, rows=1):
    """Invoke run_chunk with a standard fake config and input."""
    return runner.run_chunk(
        object(),
        make_chunk_config(),
        torch.zeros(rows, 2),
        nsample,
        state if state is not None else CallState.new(),
        deadline,
        job_base_label="label",
    )


class TestRunChunk:
    def test_success_registers_job_and_maps_results(self):
        """A successful chunk registers, polls, unregisters, and maps results."""
        raw_results = {"results_list": [{"results": {"|1,0>": 1.0}}]}
        job = FakeJob(result_events=[raw_results])
        sampler = FakeSampler()
        sampler.probs = FakeCommand(job=job)
        mapped = torch.tensor([[0.25, 0.75]])
        layer = object()
        runner = make_runner(map_results=MagicMock(return_value=mapped))
        rp = runner._create_processor.return_value

        with patch.object(perceval_adapter_module, "Sampler", return_value=sampler):
            output = runner.run_chunk(
                layer,
                make_chunk_config(),
                torch.zeros(1, 2),
                None,
                CallState.new(),
                None,
                job_base_label="label",
            )

        assert output is mapped
        runner._register_job.assert_called_once_with(job)
        runner._unregister_job.assert_called_once_with(job)
        runner._map_results.assert_called_once_with(raw_results, 1, layer, None, True)
        rp.set_circuit.assert_called_once()
        rp.with_input.assert_called_once()
        rp.min_detected_photons_filter.assert_called_once_with(1)

    def test_fresh_processor_and_retry_on_failure(self, monkeypatch):
        """Each retry builds a fresh processor; success on a later attempt wins."""
        monkeypatch.setattr(execution_module.time, "sleep", lambda _s: None)
        job = FakeJob()
        good_sampler = FakeSampler()
        good_sampler.probs = FakeCommand(job=job)

        attempts = []

        def create_processor():
            attempts.append(MagicMock(name=f"rp-{len(attempts)}"))
            return attempts[-1]

        submit_results = [RuntimeError("submit failed"), None]

        class FlakySampler(FakeSampler):
            def __init__(self, processor, max_shots_per_call):
                super().__init__()
                failure = submit_results.pop(0)
                if failure is None:
                    self.probs = FakeCommand(job=job)
                else:
                    self.probs = MagicMock()
                    self.probs.execute_async.side_effect = failure

            def clear_iterations(self):
                pass

            def add_iteration(self, circuit_params):
                pass

        runner = make_runner(create_processor=create_processor)

        with patch.object(perceval_adapter_module, "Sampler", FlakySampler):
            output = run_chunk_with(runner)

        assert output.shape == (1, 1)
        assert len(attempts) == 2  # fresh processor per attempt

    def test_gives_up_after_max_retries_and_chains_last_error(self, monkeypatch):
        """Persistent submission failures exhaust retries and chain the cause."""
        monkeypatch.setattr(execution_module.time, "sleep", lambda _s: None)
        sampler = FakeSampler()
        sampler.probs = MagicMock()
        sampler.probs.execute_async.side_effect = RuntimeError("backend down")
        runner = make_runner(max_retries=3)

        with (
            patch.object(perceval_adapter_module, "Sampler", return_value=sampler),
            pytest.raises(RuntimeError, match="failed after 3 attempts") as excinfo,
        ):
            run_chunk_with(runner)

        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert str(excinfo.value.__cause__) == "backend down"
        assert runner._create_processor.call_count == 3  # fresh RP per attempt

    def test_cancellation_short_circuits_before_submission(self):
        """A prior cancel prevents any processor construction."""
        runner = make_runner()
        state = CallState.new()
        state.request_cancel()

        with pytest.raises(CancelledError, match="Remote call was cancelled"):
            run_chunk_with(runner, state=state)

        runner._create_processor.assert_not_called()

    def test_deadline_short_circuits_before_submission(self):
        """An elapsed deadline prevents any processor construction."""
        runner = make_runner()

        with pytest.raises(TimeoutError, match="remote cancel issued"):
            run_chunk_with(runner, deadline=time.time() - 1.0)

        runner._create_processor.assert_not_called()

    def test_oversized_chunk_raises_value_error(self):
        """Chunks beyond the microbatch limit fail loudly."""
        runner = make_runner(get_microbatch_limit=lambda: 2)

        with pytest.raises(ValueError, match="exceeds microbatch"):
            run_chunk_with(runner, rows=3)

    def test_unbounded_microbatch_limit_accepts_large_chunks(self):
        """Session-style backends (limit None) skip the chunk size guard."""
        raw_results = {"results_list": []}
        job = FakeJob(result_events=[raw_results])
        sampler = FakeSampler()
        sampler.probs = FakeCommand(job=job)
        runner = make_runner(get_microbatch_limit=lambda: None)

        with patch.object(perceval_adapter_module, "Sampler", return_value=sampler):
            run_chunk_with(runner, rows=100)

        runner._map_results.assert_called_once()


# ---------------------------------------------------------------------------
# RemoteJobRunner.poll_job
# ---------------------------------------------------------------------------


class TestPollJob:
    def test_success_records_job_id_and_unregisters(self):
        """Successful polling records the job id and unregisters the job."""
        raw_results = {"results_list": [{"results": {"|1,0>": 1.0}}]}
        job = FakeJob(job_id="job-42", result_events=[raw_results])
        state = CallState.new()
        mapped = torch.tensor([[1.0]])
        runner = make_runner(map_results=MagicMock(return_value=mapped))

        result = runner.poll_job(job, state, None, 1, object(), None)

        assert result is mapped
        assert state.job_ids == ["job-42"]
        runner._unregister_job.assert_called_once_with(job)

    def test_cancel_request_cancels_job_and_raises(self):
        """Caller cancellation asks the backend job to cancel before raising."""
        job = FakeJob(is_complete=False)
        state = CallState.new()
        state.request_cancel()
        runner = make_runner()

        with pytest.raises(CancelledError, match="Remote call was cancelled"):
            runner.poll_job(job, state, None, 1, object(), None)

        assert job.cancelled is True

    def test_failed_job_raises_with_message_and_unregisters(self):
        """Failed jobs raise the Merlin error with the backend message."""
        job = FakeJob(
            is_complete=False,
            is_failed=True,
            status=SimpleNamespace(
                state="FAILED", progress=None, stop_message="hardware rejected job"
            ),
        )
        runner = make_runner()

        with pytest.raises(RemoteJobFailedError, match="hardware rejected job"):
            runner.poll_job(job, CallState.new(), None, 1, object(), None)

        runner._unregister_job.assert_called_once_with(job)

    def test_status_snapshot_updated_from_job_status(self):
        """Polling records the backend status into the call state."""
        raw_results = {"results_list": []}
        job = FakeJob(
            result_events=[raw_results],
            status=SimpleNamespace(state="RUNNING", progress=0.5, stop_message=None),
        )
        state = CallState.new()
        runner = make_runner()

        runner.poll_job(job, state, None, 1, object(), None)

        assert state.current_status is not None
        assert state.current_status.state == "RUNNING"
        assert state.current_status.progress == 0.5
