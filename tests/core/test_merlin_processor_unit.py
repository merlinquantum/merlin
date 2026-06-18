"""No-cloud characterization tests for MerlinProcessor remote-job helpers."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from perceval.runtime import RemoteProcessor
from perceval.runtime.session import ISession

import merlin.core.merlin_processor as merlin_processor_module
from merlin.core.merlin_processor import (
    BackendCapabilities,
    MerlinProcessor,
    ValidatedLayerConfig,
    SupportsExportConfig,
)
from collections.abc import Sequence
from numbers import Integral
import perceval as pcvl
import numpy as np
import re
from merlin.core.circuit import Circuit
from merlin.core.state_vector import StateVector


class FakeCommand:
    """Sampler command that records how it was submitted."""

    def __init__(self) -> None:
        self.executed = False
        self.name = None
        self.execute_kwargs = None

    def execute_async(self, **kwargs):
        """Record async execution arguments and return this fake job."""
        self.executed = True
        self.execute_kwargs = kwargs
        return self


class FakeSampler:
    """Minimal sampler exposing the command attributes used by _submit_job."""

    def __init__(self) -> None:
        self.probs = FakeCommand()
        self.sample_count = FakeCommand()
        self.samples = FakeCommand()


class FakePerceval12Command(FakeCommand):
    """Sampler command with a Perceval 1.2-style private request payload."""

    def __init__(self, iterator) -> None:
        super().__init__()
        self._request_data = {"payload": {"iterator": iterator}}

    def execute_async(self, **kwargs):
        """Serialize the payload before recording the async execution call."""
        json.dumps(self._request_data["payload"])
        return super().execute_async(**kwargs)


class FakePerceval12Iterator:
    """Small stand-in for Perceval 1.2 ParameterIterator."""

    def __init__(self) -> None:
        self.iterations = [{"circuit_params": {"px1": 0.25}}]

    def __bool__(self) -> bool:
        return True


class FakePerceval12Sampler:
    """Sampler fake whose commands expose a Perceval 1.2 iterator payload."""

    def __init__(self) -> None:
        self._iterator = FakePerceval12Iterator()
        self.probs = FakePerceval12Command(self._iterator)
        self.sample_count = FakePerceval12Command(self._iterator)
        self.samples = FakePerceval12Command(self._iterator)


@dataclass
class FakeStatus:
    """Small job status object with the fields read by _poll_job."""

    state: str = "SUCCESS"
    progress: float = 1.0
    stop_message: str | None = None


class FakeJob:
    """Remote job fake with deterministic status and result events."""

    def __init__(
        self,
        *,
        job_id: str = "job-1",
        is_complete: bool = True,
        is_failed: bool = False,
        status: FakeStatus | None = None,
        result_events: list | None = None,
    ) -> None:
        self.id = job_id
        self.status = status or FakeStatus()
        self.is_complete = is_complete
        self.is_failed = is_failed
        self.cancelled = False
        self.get_results_calls = 0
        self._result_events = (
            [{"results_list": []}] if result_events is None else list(result_events)
        )

    def cancel(self) -> None:
        """Record that remote cancellation was requested."""
        self.cancelled = True

    def get_results(self):
        """Return or raise the next configured result event."""
        self.get_results_calls += 1
        event = (
            self._result_events.pop(0)
            if len(self._result_events) > 1
            else self._result_events[0]
        )
        if isinstance(event, BaseException):
            raise event
        return event


@dataclass
class FakeComputationSpace:
    """Computation-space shim exposing the enum value used by MerlinProcessor."""

    value: str


class FakeLayer:
    """Layer shim providing just enough state-mapping data for result parsing."""

    def __init__(
        self,
        *,
        final_keys: list[tuple[int, ...]] | None = None,
        computation_scheme: str = "fock",
    ) -> None:
        graph = SimpleNamespace(
            final_keys=[(1, 0), (0, 1)] if final_keys is None else final_keys
        )
        self.computation_process = SimpleNamespace(simulation_graph=graph)
        self.computation_space = FakeComputationSpace(computation_scheme)


def make_processor(available_commands: list[str]) -> MerlinProcessor:
    """Build an uninitialized processor configured for unit helper tests."""
    proc = MerlinProcessor.__new__(MerlinProcessor)
    proc.backend_capabilities = BackendCapabilities(
        name="sim:slos",
        available_commands=available_commands,
    )
    proc._lock = threading.Lock()
    proc._active_jobs = set()
    proc._layer_cache = {}
    proc.microbatch_size = 32
    return proc


def make_poll_processor(output: torch.Tensor | None = None) -> MerlinProcessor:
    """Build a processor whose result parser records the raw payload."""
    proc = make_processor(["probs"])
    proc.processed_calls = []

    def process_results(raw_results, batch_size, layer, nsample, is_probability=False):
        proc.processed_calls.append(
            (raw_results, batch_size, layer, nsample, is_probability)
        )
        return torch.tensor([[1.0]]) if output is None else output

    proc._process_batch_results = process_results
    return proc


def make_state() -> dict:
    """Return the mutable polling state shape expected by _poll_job."""
    return {"cancel_requested": False, "job_ids": []}


# ────── Tests for BackendCapabilities ──────


def test_backend_capabilities_creation():
    """BackendCapabilities stores name and available_commands."""
    caps = BackendCapabilities(name="sim:slos", available_commands=("probs", "samples"))
    assert caps.name == "sim:slos"
    assert caps.available_commands == ("probs", "samples")


def test_backend_capabilities_is_frozen():
    """BackendCapabilities is immutable (frozen dataclass)."""
    caps = BackendCapabilities(name="sim:slos", available_commands=["probs"])
    with pytest.raises(AttributeError):
        caps.name = "new-name"


def test_backend_capabilities_equality():
    """BackendCapabilities instances with same data are equal."""
    caps1 = BackendCapabilities(name="sim:slos", available_commands=["probs"])
    caps2 = BackendCapabilities(name="sim:slos", available_commands=["probs"])
    assert caps1 == caps2


def test_backend_capabilities_inequality():
    """BackendCapabilities instances with different data are not equal."""
    caps1 = BackendCapabilities(name="sim:slos", available_commands=["probs"])
    caps2 = BackendCapabilities(name="sim:other", available_commands=["probs"])
    assert caps1 != caps2


def test_backend_capabilities_repr():
    """BackendCapabilities has a useful string representation."""
    caps = BackendCapabilities(
        name="sim:slos", available_commands=["probs", "sample_count"]
    )
    repr_str = repr(caps)
    assert "BackendCapabilities" in repr_str
    assert "sim:slos" in repr_str
    assert "probs" in repr_str


# ────── Tests for MerlinProcessor with BackendCapabilities ──────


def test_remote_processor_path_stores_backend_capabilities():
    """RemoteProcessor path stores capabilities in backend_capabilities."""
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.name = "sim:slos"
    remote_processor.available_commands = ["probs", "sample_count"]
    remote_processor.proxies = None

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(remote_processor=remote_processor)

    assert proc.backend_capabilities.name == "sim:slos"
    assert proc.backend_capabilities.available_commands == ("probs", "sample_count")


def test_backend_name_property_backward_compatibility():
    """backend_name property provides backward compatibility."""
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.name = "sim:slos"
    remote_processor.available_commands = ["probs"]
    remote_processor.proxies = None

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(remote_processor=remote_processor)

    # Old-style access should still work
    assert proc.backend_name == "sim:slos"
    # New-style access should work too
    assert proc.backend_capabilities.name == "sim:slos"
    # Both should refer to the same value
    assert proc.backend_name == proc.backend_capabilities.name


def test_available_commands_property_backward_compatibility():
    """available_commands property provides backward compatibility."""
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.name = "sim:slos"
    remote_processor.available_commands = ["probs", "samples"]
    remote_processor.proxies = None

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(remote_processor=remote_processor)

    # Old-style access should still work
    assert proc.available_commands == ("probs", "samples")
    # New-style access should work too
    assert proc.backend_capabilities.available_commands == ("probs", "samples")
    # Both should refer to the same value
    assert proc.available_commands == proc.backend_capabilities.available_commands


def test_session_path_stores_backend_capabilities():
    """ISession path also stores capabilities in backend_capabilities."""
    session = MagicMock(spec=ISession)
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.name = "perceval-qpu:scaleway"
    remote_processor.available_commands = ["probs", "sample_count"]
    remote_processor.proxies = None
    session.build_remote_processor.return_value = remote_processor

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(session=session)

    assert proc.backend_capabilities.name == "perceval-qpu:scaleway"
    assert proc.backend_capabilities.available_commands == ("probs", "sample_count")


def test_session_path_does_not_require_remote_processor_token():
    """ISession authentication should stay owned by the session object."""
    session = MagicMock(spec=ISession)
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.name = "perceval-qpu:scaleway"
    remote_processor.available_commands = ["probs", "sample_count"]
    remote_processor.proxies = None
    session.build_remote_processor.return_value = remote_processor

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value=None) as extract:
        proc = MerlinProcessor(session=session)

    extract.assert_not_called()
    assert proc.session is session
    assert proc.remote_processor is None
    assert proc.backend_capabilities.available_commands == ("probs", "sample_count")


def test_remote_processor_path_copies_available_commands():
    """RemoteProcessor construction freezes the current command-detection path."""
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.name = "sim:slos"
    remote_processor.available_commands = ["probs", "sample_count"]
    remote_processor.proxies = None

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(remote_processor=remote_processor)

    assert proc.remote_processor is remote_processor
    assert proc.session is None
    assert proc.available_commands == ("probs", "sample_count")


def test_session_path_with_empty_commands_and_sampling_only():
    """ISession with no command list, so submission samples."""
    session = MagicMock(spec=ISession)
    session.platform_name = "scaleway-like"
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.available_commands = []
    remote_processor.proxies = None
    session.build_remote_processor.return_value = remote_processor
    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        with pytest.warns(
            UserWarning, match=r"Remote processor has no available commands"
        ):
            proc = MerlinProcessor(session=session)
    sampler = FakeSampler()

    assert proc.available_commands == ()

    _, is_probability = proc._submit_job(
        sampler,
        nsample=None,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )
    assert proc.session is session
    # Session path does not store remote_processor; only uses it per chunk
    assert proc.available_commands == ()
    assert is_probability is False
    assert sampler.sample_count.executed is True
    assert sampler.sample_count.execute_kwargs == {
        "max_samples": MerlinProcessor.DEFAULT_SHOTS_PER_CALL
    }
    assert sampler.sample_count.name == "job:sample_count"
    assert sampler.probs.executed is False


def test_session_path_prefers_probs_when_available():
    """ISession path with probs available uses probs for exact probability requests."""
    session = MagicMock(spec=ISession)
    session.platform_name = "scaleway-like"
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.available_commands = ["probs", "sample_count"]
    remote_processor.proxies = None
    session.build_remote_processor.return_value = remote_processor
    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(session=session)
    sampler = FakeSampler()

    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=None,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.probs
    assert is_probability is True
    assert sampler.probs.executed is True
    assert sampler.probs.execute_kwargs == {}
    assert sampler.probs.name == "job:probs"
    assert sampler.sample_count.executed is False


def test_session_path_uses_sample_count_when_probs_unavailable():
    """ISession path without probs available falls back to sample_count."""
    session = MagicMock(spec=ISession)
    session.platform_name = "scaleway-like"
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.available_commands = ["sample_count"]
    remote_processor.proxies = None
    session.build_remote_processor.return_value = remote_processor
    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(session=session)
    sampler = FakeSampler()

    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=None,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.sample_count
    assert is_probability is False
    assert sampler.sample_count.executed is True
    assert sampler.sample_count.execute_kwargs == {
        "max_samples": MerlinProcessor.DEFAULT_SHOTS_PER_CALL
    }
    assert sampler.sample_count.name == "job:sample_count"
    assert sampler.probs.executed is False


def test_session_path_with_probs_and_samples_no_sample_count():
    """ISession with probs+samples but no sample_count uses probs for probability."""
    session = MagicMock(spec=ISession)
    session.platform_name = "scaleway-like"
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.available_commands = ["probs", "samples"]
    remote_processor.proxies = None
    session.build_remote_processor.return_value = remote_processor
    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(session=session)
    sampler = FakeSampler()

    # Probability request should use probs
    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=None,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )
    assert returned_job is sampler.probs
    assert is_probability is True

    # Reset sampler for sampling request
    sampler = FakeSampler()
    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=10,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )
    # Should use samples (not sample_count since unavailable)
    assert returned_job is sampler.samples
    assert is_probability is False


def test_session_path_with_sample_count_and_samples_no_probs():
    """ISession with sample_count+samples but no probs uses sample_count for probability."""
    session = MagicMock(spec=ISession)
    session.platform_name = "scaleway-like"
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.available_commands = ["sample_count", "samples"]
    remote_processor.proxies = None
    session.build_remote_processor.return_value = remote_processor
    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(session=session)
    sampler = FakeSampler()

    # Probability request (nsample=None) should use sample_count as fallback
    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=None,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )
    assert returned_job is sampler.sample_count
    assert is_probability is False  # No probs available


def test_session_path_with_only_probs():
    """ISession with only probs available defaults to sample_count for sampling."""
    session = MagicMock(spec=ISession)
    session.platform_name = "scaleway-like"
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.available_commands = ["probs"]
    remote_processor.proxies = None
    session.build_remote_processor.return_value = remote_processor
    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(session=session)
    sampler = FakeSampler()

    # Probability request should use probs
    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=None,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )
    assert returned_job is sampler.probs
    assert is_probability is True

    # Sampling request with only probs defaults to sample_count (unavailable)
    sampler = FakeSampler()
    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=10,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )
    # Defaults to sample_count since no sampling commands available
    assert returned_job is sampler.sample_count
    assert is_probability is False


def test_session_path_with_only_samples():
    """ISession with only samples available falls back for probability requests."""
    session = MagicMock(spec=ISession)
    session.platform_name = "scaleway-like"
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.available_commands = ["samples"]
    remote_processor.proxies = None
    session.build_remote_processor.return_value = remote_processor
    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(session=session)
    sampler = FakeSampler()

    # Sampling request should use samples
    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=10,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )
    assert returned_job is sampler.samples
    assert is_probability is False


def test_session_path_with_all_three_commands():
    """ISession with all commands prioritizes probs for probability, sample_count for sampling."""
    session = MagicMock(spec=ISession)
    session.platform_name = "scaleway-like"
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.available_commands = ["probs", "sample_count", "samples"]
    remote_processor.proxies = None
    session.build_remote_processor.return_value = remote_processor
    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(session=session)
    sampler = FakeSampler()

    # Probability request should prefer probs
    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=None,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )
    assert returned_job is sampler.probs
    assert is_probability is True
    assert sampler.sample_count.executed is False
    assert sampler.samples.executed is False

    # Reset sampler and test sampling request
    sampler = FakeSampler()
    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=25,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )
    # Should prefer sample_count over samples
    assert returned_job is sampler.sample_count
    assert is_probability is False
    assert sampler.samples.executed is False


def test_session_path_zero_samples_treated_as_probability_request():
    """ISession with nsample=0 is treated the same as nsample=None for probability."""
    session = MagicMock(spec=ISession)
    session.platform_name = "scaleway-like"
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.available_commands = ["probs", "sample_count"]
    remote_processor.proxies = None
    session.build_remote_processor.return_value = remote_processor
    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(session=session)
    sampler = FakeSampler()

    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=0,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.probs
    assert is_probability is True


def test_session_path_backend_name_is_extracted():
    """ISession processors extract and store the backend name from RemoteProcessor."""
    session = MagicMock(spec=ISession)
    session.platform_name = "scaleway-like"
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.name = "perceval-qpu:scaleway"
    remote_processor.available_commands = ["probs"]
    remote_processor.proxies = None
    session.build_remote_processor.return_value = remote_processor
    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(session=session)

    assert proc.backend_name == "perceval-qpu:scaleway"


def test_submit_job_prefers_probs_when_available_without_samples():
    """Probability-capable backends use probs for exact probability requests."""
    proc = make_processor(["probs", "sample_count"])
    sampler = FakeSampler()

    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=None,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.probs
    assert is_probability is True
    assert sampler.probs.executed is True
    assert sampler.probs.execute_kwargs == {}
    assert sampler.probs.name == "job:probs"
    assert sampler.sample_count.executed is False
    assert sampler.samples.executed is False


def test_submit_job_treats_zero_samples_as_exact_probabilities():
    """Zero requested samples currently follows the same path as nsample=None."""
    proc = make_processor(["probs", "sample_count"])
    sampler = FakeSampler()

    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=0,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.probs
    assert is_probability is True
    assert sampler.probs.executed is True
    assert sampler.probs.execute_kwargs == {}
    assert sampler.probs.name == "job:probs"
    assert sampler.sample_count.executed is False


def test_submit_job_uses_sample_count_when_sampling_requested():
    """A positive sample request uses sample_count when the backend exposes it."""
    proc = make_processor(["probs", "sample_count"])
    sampler = FakeSampler()

    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=37,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.sample_count
    assert is_probability is False
    assert sampler.sample_count.executed is True
    assert sampler.sample_count.execute_kwargs == {"max_samples": 37}
    assert sampler.sample_count.name == "job:sample_count"
    assert sampler.probs.executed is False


def test_submit_job_serializes_perceval_12_parameter_iterator_payload():
    """Perceval 1.2 sampler iterations must be JSON-serializable for Scaleway."""
    proc = make_processor(["sample_count"])
    sampler = FakePerceval12Sampler()

    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=37,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.sample_count
    assert is_probability is False
    assert sampler.sample_count.executed is True
    assert sampler.sample_count.execute_kwargs == {"max_samples": 37}
    assert sampler.sample_count._request_data["payload"]["iterator"] == [
        {"circuit_params": {"px1": 0.25}}
    ]


def test_submit_job_falls_back_to_samples_when_sample_count_is_unavailable():
    """Backends without sample_count currently use samples for sampled jobs."""
    proc = make_processor(["samples"])
    sampler = FakeSampler()

    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=11,
        job_base_label="job",
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.samples
    assert is_probability is False
    assert sampler.samples.executed is True
    assert sampler.samples.execute_kwargs == {"max_samples": 11}
    assert sampler.samples.name == "job:samples"
    assert sampler.sample_count.executed is False


def test_submit_job_defaults_to_sample_count_when_commands_are_empty():
    """An empty command list currently means sampling through sample_count."""
    proc = make_processor([])
    sampler = FakeSampler()

    returned_job, is_probability = proc._submit_job(
        sampler,
        nsample=None,
        job_base_label=None,
        _capped_name=lambda base, command: f"{base}:{command}",
    )

    assert returned_job is sampler.sample_count
    assert is_probability is False
    assert sampler.sample_count.executed is True
    assert sampler.sample_count.execute_kwargs == {
        "max_samples": MerlinProcessor.DEFAULT_SHOTS_PER_CALL
    }
    assert sampler.probs.executed is False
    assert sampler.samples.executed is False


def test_poll_job_success_processes_dict_payload_and_records_job_id():
    """Successful polling records the job id and delegates dict parsing."""
    output = torch.tensor([[0.25, 0.75]])
    proc = make_poll_processor(output=output)
    raw_results = {"results_list": [{"results": {"|1,0>": 1.0}}]}
    job = FakeJob(job_id="job-success", result_events=[raw_results])
    proc._active_jobs.add(job)
    state = make_state()
    layer = object()

    result = proc._poll_job(
        job,
        state,
        deadline=None,
        batch_size=3,
        layer=layer,
        nsample=None,
    )

    assert torch.equal(result, output)
    assert state["job_ids"] == ["job-success"]
    assert proc.processed_calls == [(raw_results, 3, layer, None, False)]
    assert job not in proc._active_jobs


def test_poll_job_failed_status_raises_with_stop_message_and_job_id():
    """Failed jobs surface the backend stop message and current job id."""
    proc = make_poll_processor()
    job = FakeJob(
        job_id="job-failed",
        is_complete=False,
        is_failed=True,
        status=FakeStatus(state="FAILED", stop_message="hardware rejected job"),
    )
    proc._active_jobs.add(job)

    with pytest.raises(RuntimeError, match=r"hardware rejected job.*job-failed"):
        proc._poll_job(job, make_state(), None, 1, object(), None)

    assert job not in proc._active_jobs


def test_poll_job_cancel_request_cancels_remote_job():
    """Caller cancellation asks the backend job to cancel before raising."""
    proc = make_poll_processor()
    job = FakeJob(is_complete=False)
    state = make_state()
    state["cancel_requested"] = True

    with pytest.raises(CancelledError, match=r"Remote call was cancelled"):
        proc._poll_job(job, state, None, 1, object(), None)

    assert job.cancelled is True


def test_poll_job_timeout_cancels_remote_job():
    """Timeouts request remote cancellation and raise TimeoutError."""
    proc = make_poll_processor()
    job = FakeJob(is_complete=False)

    with pytest.raises(TimeoutError, match=r"remote cancel issued"):
        proc._poll_job(job, make_state(), time.time() - 1.0, 1, object(), None)

    assert job.cancelled is True


def test_poll_job_cancel_requested_stop_message_raises_cancelled_error():
    """A failed job whose stop message says cancel maps to CancelledError."""
    proc = make_poll_processor()
    job = FakeJob(
        is_complete=False,
        is_failed=True,
        status=FakeStatus(state="FAILED", stop_message="Cancel requested by user"),
    )
    proc._active_jobs.add(job)

    with pytest.raises(CancelledError, match=r"Remote call was cancelled"):
        proc._poll_job(job, make_state(), None, 1, object(), None)

    assert job not in proc._active_jobs


def test_poll_job_cancel_requested_get_results_exception_raises_cancelled_error():
    """A completion-time cancel exception maps to CancelledError."""
    proc = make_poll_processor()
    job = FakeJob(result_events=[RuntimeError("Cancel requested on backend")])
    proc._active_jobs.add(job)

    with pytest.raises(CancelledError, match=r"Remote call was cancelled"):
        proc._poll_job(job, make_state(), None, 1, object(), None)

    assert job not in proc._active_jobs


def test_poll_job_retries_when_results_are_not_available(monkeypatch):
    """Completed jobs retry when Perceval says results are not available yet."""
    monkeypatch.setattr(merlin_processor_module.time, "sleep", lambda _seconds: None)
    output = torch.tensor([[1.0]])
    proc = make_poll_processor(output=output)
    raw_results = {"results_list": [{"results": {"|1,0>": 1.0}}]}
    job = FakeJob(
        result_events=[
            RuntimeError("Results are not available"),
            raw_results,
        ]
    )

    result = proc._poll_job(job, make_state(), None, 1, object(), None)

    assert torch.equal(result, output)
    assert job.get_results_calls == 2


def test_poll_job_retries_complete_non_dict_payloads_then_fails(monkeypatch):
    """Complete non-dict payloads are retried for the current bounded window."""
    monkeypatch.setattr(merlin_processor_module.time, "sleep", lambda _seconds: None)
    proc = make_poll_processor()
    job = FakeJob(job_id="job-nondict", result_events=[["not", "a", "dict"]])
    proc._active_jobs.add(job)

    with pytest.raises(RuntimeError, match=r"not a dict after 60 re-polls"):
        proc._poll_job(job, make_state(), None, 1, object(), None)

    assert job.get_results_calls == 60
    assert job not in proc._active_jobs


def test_process_batch_results_rejects_missing_payload():
    """A missing backend payload currently raises a runtime failure."""
    proc = make_processor(["probs"])

    with pytest.raises(RuntimeError, match=r"returned no results"):
        proc._process_batch_results(None, 1, FakeLayer())


def test_process_batch_results_rejects_non_dict_payload():
    """A non-dict backend payload currently raises a runtime failure."""
    proc = make_processor(["probs"])

    with pytest.raises(RuntimeError, match=r"Unexpected remote results type"):
        proc._process_batch_results(["not", "a", "dict"], 1, FakeLayer())


def test_process_batch_results_normalizes_counts():
    """Integer count payloads are normalized into probabilities per row."""
    proc = make_processor(["probs"])
    layer = FakeLayer()
    raw_results = {
        "results_list": [{"results": {"|1,0>": 3, "|0,1>": 1}}],
    }

    result = proc._process_batch_results(raw_results, 1, layer)

    assert torch.allclose(result, torch.tensor([[0.75, 0.25]]))


def test_process_batch_results_passes_probability_payloads_through():
    """Float payloads no larger than one are treated as probabilities."""
    proc = make_processor(["probs"])
    layer = FakeLayer()
    raw_results = {
        "results_list": [{"results": {"|1,0>": 0.2, "|0,1>": 0.8}}],
    }

    result = proc._process_batch_results(raw_results, 1, layer)

    assert torch.allclose(result, torch.tensor([[0.2, 0.8]]))


def test_process_batch_results_filters_invalid_states_for_non_fock_space():
    """Non-FOCK mappings currently filter states outside final_keys."""
    proc = make_processor(["probs"])
    layer = FakeLayer(
        final_keys=[(1, 0), (0, 1)],
        computation_scheme="unbunched",
    )
    raw_results = {
        "results_list": [{"results": {"|1,0>": 5, "|2,0>": 5}}],
    }

    result = proc._process_batch_results(raw_results, 1, layer)

    assert torch.allclose(result, torch.tensor([[1.0, 0.0]]))


def test_process_batch_results_zero_fills_missing_rows():
    """Missing result rows are padded with zero probability rows."""
    proc = make_processor(["probs"])
    layer = FakeLayer()
    raw_results = {
        "results_list": [
            {"results": {"|0,1>": 1.0}},
            {"metadata": "no results key"},
        ],
    }

    result = proc._process_batch_results(raw_results, 3, layer)

    assert torch.allclose(
        result,
        torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_process_batch_results_probability_heuristic_renormalizes_float_rows():
    """The current heuristic treats first float <= 1 as probability payloads."""
    proc = make_processor(["probs"])
    layer = FakeLayer()
    raw_results = {
        "results_list": [{"results": {"|1,0>": 1.0, "|0,1>": 1.0}}],
    }

    result = proc._process_batch_results(raw_results, 1, layer)

    assert torch.allclose(result, torch.tensor([[0.5, 0.5]]))


# ────── Tests for _create_fresh_rp() ──────


def test_create_fresh_rp_remote_processor_path_clones_with_token():
    """RemoteProcessor path delegates to _clone_remote_processor."""
    original_rp = MagicMock(spec=RemoteProcessor)
    original_rp.name = "sim:slos"
    original_rp.available_commands = ["probs"]
    original_rp.proxies = None

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="test_token"):
        proc = MerlinProcessor(remote_processor=original_rp)

    # Verify session is None (RemoteProcessor path)
    assert proc.session is None
    # Verify the original RP is stored
    assert proc.remote_processor is original_rp

    # Mock _clone_remote_processor to verify it's called by _create_fresh_rp
    cloned_rp = MagicMock(spec=RemoteProcessor)
    with patch.object(
        proc, "_clone_remote_processor", return_value=cloned_rp
    ) as mock_clone:
        fresh_rp = proc._create_fresh_rp()
        mock_clone.assert_called_once_with(original_rp)

    assert fresh_rp is cloned_rp


def test_create_fresh_rp_session_path_calls_build_remote_processor():
    """ISession path calls session.build_remote_processor() for each chunk."""
    session = MagicMock(spec=ISession)
    rp1 = MagicMock(spec=RemoteProcessor)
    rp1.available_commands = ["probs"]
    rp1.proxies = None
    session.build_remote_processor.return_value = rp1

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(session=session)

    # Create first fresh processor
    fresh_rp1 = proc._create_fresh_rp()
    assert fresh_rp1 is rp1
    assert session.build_remote_processor.call_count == 2  # init + this call
    assert proc.session is session


def test_create_fresh_rp_session_path_creates_independent_processors():
    """ISession path creates independent processors on repeated calls."""
    session = MagicMock(spec=ISession)
    rp1 = MagicMock(spec=RemoteProcessor)
    rp1.available_commands = ["probs"]
    rp1.proxies = None
    rp2 = MagicMock(spec=RemoteProcessor)
    rp2.available_commands = ["probs"]
    rp2.proxies = None
    rp3 = MagicMock(spec=RemoteProcessor)
    rp3.available_commands = ["probs"]
    rp3.proxies = None
    rp4 = MagicMock(spec=RemoteProcessor)
    rp4.available_commands = ["probs"]
    rp4.proxies = None

    # Set up the session to return different processor instances each time
    # side_effect needs: 1 for init + 3 for _create_fresh_rp calls = 4 total
    session.build_remote_processor.side_effect = [rp1, rp2, rp3, rp4]

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(session=session)

    # Create multiple fresh processors
    fresh_rp2 = proc._create_fresh_rp()
    fresh_rp3 = proc._create_fresh_rp()
    fresh_rp4 = proc._create_fresh_rp()

    # Each should be a different instance
    assert fresh_rp2 is not fresh_rp3
    assert fresh_rp3 is not fresh_rp4
    assert fresh_rp2 is not fresh_rp4
    assert session.build_remote_processor.call_count == 4  # init + 3 calls


def test_create_fresh_rp_remote_processor_path_maintains_available_commands():
    """RemoteProcessor path preserves available_commands through cloning."""
    original_rp = MagicMock(spec=RemoteProcessor)
    original_rp.name = "sim:slos"
    original_rp.available_commands = ["probs", "sample_count", "samples"]
    original_rp.proxies = None

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(remote_processor=original_rp)

    # Verify available_commands was captured at init
    assert proc.available_commands == ("probs", "sample_count", "samples")

    # Create fresh RPs and verify available_commands is unchanged
    cloned_rp1 = MagicMock(spec=RemoteProcessor)
    cloned_rp1.available_commands = ["probs", "sample_count"]  # Different commands
    cloned_rp2 = MagicMock(spec=RemoteProcessor)
    cloned_rp2.available_commands = []  # Empty commands

    with patch.object(
        proc, "_clone_remote_processor", side_effect=[cloned_rp1, cloned_rp2]
    ):
        proc._create_fresh_rp()
        proc._create_fresh_rp()

    # available_commands should still reflect the original
    assert proc.available_commands == ("probs", "sample_count", "samples")


def test_create_fresh_rp_session_path_maintains_available_commands():
    """ISession path preserves available_commands from first initialization."""
    session = MagicMock(spec=ISession)
    rp_init = MagicMock(spec=RemoteProcessor)
    rp_init.available_commands = ["probs", "sample_count"]
    rp_init.proxies = None

    # Later RPs may have different commands, but shouldn't affect proc's cached value
    rp_chunk1 = MagicMock(spec=RemoteProcessor)
    rp_chunk1.available_commands = ["probs"]
    rp_chunk2 = MagicMock(spec=RemoteProcessor)
    rp_chunk2.available_commands = []

    session.build_remote_processor.side_effect = [rp_init, rp_chunk1, rp_chunk2]

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(session=session)

    # Verify available_commands was captured from init processor
    assert proc.available_commands == ("probs", "sample_count")

    # Create fresh processors
    proc._create_fresh_rp()
    proc._create_fresh_rp()

    # available_commands should still reflect the initial state
    assert proc.available_commands == ("probs", "sample_count")


def test_create_fresh_rp_remote_processor_path_with_cloning_disabled():
    """RemoteProcessor path delegates to _clone_remote_processor correctly."""
    original_rp = MagicMock(spec=RemoteProcessor)
    original_rp.name = "sim:ascella"
    original_rp.available_commands = ["probs"]
    original_rp.proxies = None

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(remote_processor=original_rp)

    # Verify remote_processor is stored
    assert proc.remote_processor is original_rp
    assert proc.session is None

    # Create fresh RP and verify clone method was called
    with patch.object(proc, "_clone_remote_processor") as mock_clone:
        mock_clone.return_value = MagicMock(spec=RemoteProcessor)
        fresh_rp = proc._create_fresh_rp()

    mock_clone.assert_called_once_with(original_rp)
    assert fresh_rp is mock_clone.return_value


def test_different_valid_configs():
    # BasicState input
    config = {
        "circuit": pcvl.Circuit(m=2, name="Circuit"),
        "input_state": pcvl.BasicState("|1,0>"),
        "input_param_order": ["px", "el", "s"],
    }
    v_config = ValidatedLayerConfig(config)
    assert isinstance(v_config.circuit, pcvl.Circuit)
    assert v_config.circuit == config["circuit"]
    assert isinstance(v_config.input_state, pcvl.BasicState)
    assert v_config.input_state == config["input_state"]
    assert isinstance(v_config.input_param_order, Sequence)
    for i in v_config.input_param_order:
        assert isinstance(i, str)
    assert v_config.input_param_order == config["input_param_order"]

    # StateVector input
    config = {
        "circuit": pcvl.Circuit(m=2, name="Circuit"),
        "input_state": pcvl.StateVector("|1,0>"),
        "input_param_order": ["px", "el", "s"],
    }
    v_config = ValidatedLayerConfig(config)
    assert isinstance(v_config.circuit, pcvl.Circuit)
    assert v_config.circuit == config["circuit"]
    assert isinstance(v_config.input_state, pcvl.StateVector)
    assert v_config.input_state == config["input_state"]
    assert isinstance(v_config.input_param_order, Sequence)
    for i in v_config.input_param_order:
        assert isinstance(i, str)
    assert v_config.input_param_order == config["input_param_order"]

    # FockState input
    config = {
        "circuit": pcvl.Circuit(m=2, name="Circuit"),
        "input_state": pcvl.FockState("|1,0>"),
        "input_param_order": ["px", "el", "s"],
    }
    v_config = ValidatedLayerConfig(config)
    assert isinstance(v_config.circuit, pcvl.Circuit)
    assert v_config.circuit == config["circuit"]
    assert isinstance(v_config.input_state, pcvl.FockState)
    assert v_config.input_state == config["input_state"]
    assert isinstance(v_config.input_param_order, Sequence)
    for i in v_config.input_param_order:
        assert isinstance(i, str)
    assert v_config.input_param_order == config["input_param_order"]

    # NoisyFockState input
    config = {
        "circuit": pcvl.Circuit(m=2, name="Circuit"),
        "input_state": pcvl.NoisyFockState(pcvl.FockState([1, 0])),
        "input_param_order": ["px", "el", "s"],
    }
    v_config = ValidatedLayerConfig(config)
    assert isinstance(v_config.circuit, pcvl.Circuit)
    assert v_config.circuit == config["circuit"]
    assert isinstance(v_config.input_state, pcvl.NoisyFockState)
    assert v_config.input_state == config["input_state"]
    assert isinstance(v_config.input_param_order, Sequence)
    for i in v_config.input_param_order:
        assert isinstance(i, str)
    assert v_config.input_param_order == config["input_param_order"]

    # LogicalState input
    config = {
        "circuit": pcvl.Circuit(m=2, name="Circuit"),
        "input_state": pcvl.LogicalState([1, 0]),
        "input_param_order": ["px", "el", "s"],
    }
    v_config = ValidatedLayerConfig(config)
    assert isinstance(v_config.circuit, pcvl.Circuit)
    assert v_config.circuit == config["circuit"]
    assert isinstance(v_config.input_state, pcvl.LogicalState)
    assert v_config.input_state == config["input_state"]
    assert isinstance(v_config.input_param_order, Sequence)
    for i in v_config.input_param_order:
        assert isinstance(i, str)
    assert v_config.input_param_order == config["input_param_order"]

    # Sequence input list
    config = {
        "circuit": pcvl.Circuit(m=2, name="Circuit"),
        "input_state": [1, 0],
        "input_param_order": ["px", "el", "s"],
    }
    v_config = ValidatedLayerConfig(config)
    assert isinstance(v_config.circuit, pcvl.Circuit)
    assert v_config.circuit == config["circuit"]
    assert isinstance(v_config.input_state, Sequence)
    for i in v_config.input_state:
        assert isinstance(i, Integral)
    assert v_config.input_state == config["input_state"]
    assert isinstance(v_config.input_param_order, Sequence)
    for i in v_config.input_param_order:
        assert isinstance(i, str)
    assert v_config.input_param_order == config["input_param_order"]

    # Sequence input torch ints as tuple
    config = {
        "circuit": pcvl.Circuit(m=2, name="Circuit"),
        "input_state": tuple([torch.tensor(1).item(), torch.tensor(0).item()]),
        "input_param_order": ["px", "el", "s"],
    }
    v_config = ValidatedLayerConfig(config)
    assert isinstance(v_config.circuit, pcvl.Circuit)
    assert v_config.circuit == config["circuit"]
    assert isinstance(v_config.input_state, Sequence)
    for i in v_config.input_state:
        assert isinstance(i, Integral)
    assert v_config.input_state == config["input_state"]
    assert isinstance(v_config.input_param_order, Sequence)
    for i in v_config.input_param_order:
        assert isinstance(i, str)
    assert v_config.input_param_order == config["input_param_order"]

    # Sequence input array
    config = {
        "circuit": pcvl.Circuit(m=2, name="Circuit"),
        "input_state": np.array([1, 0]),
        "input_param_order": ["px", "el", "s"],
    }
    v_config = ValidatedLayerConfig(config)
    assert isinstance(v_config.circuit, pcvl.Circuit)
    assert v_config.circuit == config["circuit"]
    assert isinstance(v_config.input_state, Sequence)
    for i in v_config.input_state:
        assert isinstance(i, Integral)
    assert v_config.input_state == tuple(config["input_state"])
    assert isinstance(v_config.input_param_order, Sequence)
    for i in v_config.input_param_order:
        assert isinstance(i, str)
    assert v_config.input_param_order == config["input_param_order"]


def test_missing_required_fiels_in_configs():
    # Missing Circuit
    config = {
        "circuit": pcvl.Circuit(m=2, name="Circuit"),
        "input_state": pcvl.BasicState("|1,0>"),
        "input_param_order": ["px", "el", "s"],
    }

    # Missing Circuit
    config = {
        "input_state": pcvl.BasicState("|1,0>"),
        "input_param_order": ["px", "el", "s"],
    }
    with pytest.raises(
        KeyError,
        match=r"There must be a key 'circuit' in the configs dictionary that is associated with a perceval.ACircuit.",
    ):
        v_config = ValidatedLayerConfig(config)

    # Missing State
    config = {
        "circuit": pcvl.Circuit(m=2, name="Circuit"),
        "input_param_order": ["px", "el", "s"],
    }
    with pytest.raises(
        KeyError,
        match=r".*There must be a key 'input_state' in the configs dictionary.*",
    ):
        v_config = ValidatedLayerConfig(config)

    # Missing input param order
    config = {
        "circuit": pcvl.Circuit(m=2, name="Circuit"),
        "input_state": pcvl.BasicState("|1,0>"),
    }
    with pytest.raises(
        KeyError,
        match=r".*There must be a key 'input_param_order' in the configs dictionary that is associated with a Sequence\[str\] or None\..*",
    ):
        v_config = ValidatedLayerConfig(config)


def test_wrong_types_config():
    # Bad Circuit
    config = {
        "circuit": None,
        "input_state": pcvl.BasicState("|1,0>"),
        "input_param_order": ["px", "el", "s"],
    }
    with pytest.raises(
        ValueError,
        match=r"The 'circuit' key of the config dictionary must be a perceval.ACircuit",
    ):
        v_config = ValidatedLayerConfig(config)

    config = {
        "circuit": Circuit(n_modes=2, components=[pcvl.components.BS()]),
        "input_state": pcvl.BasicState("|1,0>"),
        "input_param_order": ["px", "el", "s"],
    }
    with pytest.raises(
        ValueError,
        match=r"The 'circuit' key of the config dictionary must be a perceval.ACircuit",
    ):
        v_config = ValidatedLayerConfig(config)

    # input_state
    config = {
        "circuit": pcvl.Circuit(m=2),
        "input_state": [1.1, 2.0],
        "input_param_order": ["px", "el", "s"],
    }
    with pytest.raises(
        ValueError,
        match=r"'input_state' must contain only integers when it is a sequence.",
    ):
        v_config = ValidatedLayerConfig(config)

    config = {
        "circuit": pcvl.Circuit(m=2),
        "input_state": StateVector(torch.tensor([1, 0]), n_modes=2, n_photons=1),
        "input_param_order": ["px", "el", "s"],
    }
    with pytest.raises(
        ValueError,
        match=r"'input_state' must be None, a sequence of integers, or an Perceval state object.",
    ):
        v_config = ValidatedLayerConfig(config)

    # input_param_order
    config = {
        "circuit": pcvl.Circuit(m=2),
        "input_state": [2, 0],
        "input_param_order": 3,
    }
    with pytest.raises(
        ValueError,
        match=r"'input_param_order' must be a sequence of strings or None, got int.",
    ):
        v_config = ValidatedLayerConfig(config)

    config = {
        "circuit": pcvl.Circuit(m=2),
        "input_state": [1, 2],
        "input_param_order": [11, 2, 1],
    }
    with pytest.raises(
        ValueError,
        match=r"'input_param_order' must contain only strings.",
    ):
        v_config = ValidatedLayerConfig(config)


def test_has_export_config():
    class GoodLayer:
        def __init__(self):
            pass

        def export_config():
            return {
                "circuit": pcvl.Circuit(m=2, name="Circuit"),
                "input_state": [1, 0],
                "input_param_order": ["px", "el", "s"],
            }

    class BadLayer:
        def __init__(self):
            pass

    assert isinstance(GoodLayer(), SupportsExportConfig)
    assert not isinstance(BadLayer(), SupportsExportConfig)


def test_offload_quantum_layer_with_chunking_validates_and_caches_export_config():
    proc = make_processor(["probs", "sample_count"])

    class LayerWithExportConfig:
        uid = 42

        def __init__(self) -> None:
            self.export_config_calls = 0

        def export_config(self):
            self.export_config_calls += 1
            return {
                "circuit": pcvl.Circuit(m=2, name="Circuit"),
                "input_state": [1, 0],
                "input_param_order": ["px", "el", "s"],
            }

    layer = LayerWithExportConfig()

    def fake_run_chunks_pooled(
        layer_arg, config, input_tensor, chunks, nsample, state, deadline
    ):
        assert layer_arg is layer
        assert isinstance(config, ValidatedLayerConfig)
        assert isinstance(config.circuit, pcvl.Circuit)
        assert config.input_param_order == ["px", "el", "s"]
        return torch.tensor([[1.0]])

    proc._run_chunks_pooled = fake_run_chunks_pooled

    result = proc._offload_quantum_layer_with_chunking(
        layer,
        torch.zeros(1, 2),
        None,
        {},
        None,
    )

    assert torch.equal(result, torch.tensor([[1.0]]))
    assert layer.export_config_calls == 1
    assert layer.uid in proc._layer_cache
    assert isinstance(proc._layer_cache[layer.uid]["config"], ValidatedLayerConfig)

    # Calling again should reuse the cached config and not call export_config again.
    proc._offload_quantum_layer_with_chunking(
        layer,
        torch.zeros(1, 2),
        None,
        {},
        None,
    )
    assert layer.export_config_calls == 1
