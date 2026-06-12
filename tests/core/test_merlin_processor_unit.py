"""No-cloud characterization tests for MerlinProcessor remote-job helpers."""

from __future__ import annotations

import json
import threading
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass
from numbers import Integral
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import perceval as pcvl
import pytest
import torch
from perceval.runtime import AProcessor, Processor, RemoteProcessor
from perceval.runtime.session import ISession

import merlin.core.merlin_processor as merlin_processor_module
from merlin.algorithms.module import MerlinModule
from merlin.core.circuit import Circuit
from merlin.core.merlin_processor import (
    BackendCapabilities,
    MerlinProcessor,
    SupportsExportConfig,
    ValidatedLayerConfig,
)
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


class FakeSyncCommand:
    """Sampler command that records synchronous execution."""

    def __init__(self, result: dict, on_execute=None) -> None:
        self.result = result
        self.on_execute = on_execute
        self.executed = False
        self.execute_kwargs = None

    def execute_sync(self, **kwargs):
        """Record sync execution arguments and return the configured result."""
        self.executed = True
        self.execute_kwargs = kwargs
        if self.on_execute is not None:
            self.on_execute()
        return self.result


class FakeSyncSampler:
    """Minimal sampler exposing sync command attributes used by _run_chunk_local."""

    def __init__(self, result: dict, on_execute=None) -> None:
        self.probs = FakeSyncCommand(result, on_execute=on_execute)
        self.sample_count = FakeSyncCommand(result, on_execute=on_execute)
        self.samples = FakeSyncCommand(result, on_execute=on_execute)
        self.iterations = []
        self.cleared = False

    def clear_iterations(self) -> None:
        """Record that sampler iterations were reset."""
        self.cleared = True
        self.iterations.clear()

    def add_iteration(self, **kwargs) -> None:
        """Record a sampler iteration."""
        self.iterations.append(kwargs)


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
    proc.max_shots_per_call = MerlinProcessor.DEFAULT_MAX_SHOTS
    proc.backend_kind = "remote_processor"
    return proc


def make_poll_processor(output: torch.Tensor | None = None) -> MerlinProcessor:
    """Build a processor whose result parser records the raw payload."""
    proc = make_processor(["probs"])
    proc.processed_calls = []

    def process_results(raw_results, batch_size, layer, nsample, is_probability=False):
        proc.processed_calls.append((
            raw_results,
            batch_size,
            layer,
            nsample,
            is_probability,
        ))
        return torch.tensor([[1.0]]) if output is None else output

    proc._process_batch_results = process_results
    return proc


def make_state() -> dict:
    """Return the mutable polling state shape expected by _poll_job."""
    return {"cancel_requested": False, "job_ids": []}


def make_local_chunk_config() -> SimpleNamespace:
    """Return the config shape consumed by _run_chunk_local."""
    return SimpleNamespace(
        circuit=MagicMock(name="circuit"),
        input_state=[1, 0],
        input_param_order=["theta_0", "theta_1"],
    )


def make_local_chunk_processor(available_commands: list[str]) -> MerlinProcessor:
    """Build a processor configured for local chunk execution tests."""
    proc = make_processor(available_commands)
    proc.backend_kind = "local_processor"
    proc.processor = MagicMock(name="original_processor")
    proc.local_execution_processor = MagicMock(name="local_execution_processor")
    proc._create_fresh_local_processor = MagicMock(
        return_value=proc.local_execution_processor
    )
    proc._process_batch_results = MagicMock(return_value=torch.tensor([[1.0]]))
    return proc


def make_local_aprocessor(
    available_commands: list[str] | None = None,
) -> AProcessor:
    """Build a local AProcessor mock for constructor-routing tests."""
    processor = MagicMock(spec=AProcessor)
    processor.is_remote = False
    processor.name = "local:slos"
    processor.available_commands = (
        ["probs"] if available_commands is None else list(available_commands)
    )
    return processor


def make_remote_processor_mock(
    available_commands: list[str] | None = None,
) -> RemoteProcessor:
    """Build a RemoteProcessor mock for constructor-routing tests."""
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.name = "sim:slos"
    remote_processor.available_commands = (
        ["probs"] if available_commands is None else list(available_commands)
    )
    remote_processor.proxies = None
    return remote_processor


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


@pytest.mark.parametrize(
    ("nsample", "max_shots_per_call", "expected"),
    [
        (
            None,
            MerlinProcessor.DEFAULT_MAX_SHOTS,
            MerlinProcessor.DEFAULT_SHOTS_PER_CALL,
        ),
        (None, 123, 123),
        (123, 456, 123),
        (456, 123, 123),
    ],
)
def test_effective_sample_count_caps_requested_samples(
    nsample: int | None, max_shots_per_call: int, expected: int
):
    """_effective_sample_count caps defaults and explicit requests."""
    proc = make_processor(["sample_count"])
    proc.max_shots_per_call = max_shots_per_call

    assert proc._effective_sample_count(nsample) == expected


# ────── Tests for MerlinProcessor with BackendCapabilities ──────


def test_merlinprocessor_accepts_local_aprocessor_backend():
    """Local AProcessor path stores the processor and backend kind."""
    local_processor = make_local_aprocessor(["probs", "sample_count"])

    proc = MerlinProcessor(processor=local_processor)

    assert proc.processor is local_processor
    assert proc.remote_processor is None
    assert proc.session is None
    assert proc.backend_kind == "local_processor"
    assert proc.backend_capabilities.name == "local:slos"
    assert proc.backend_capabilities.available_commands == ("probs", "sample_count")


def test_merlinprocessor_rejects_non_aprocessor_processor_backend():
    """processor= rejects objects outside the Perceval AProcessor hierarchy."""
    with pytest.raises(TypeError, match="Expected AProcessor"):
        MerlinProcessor(processor=object())


def test_merlinprocessor_rejects_missing_backend():
    """Exactly one explicit backend is required."""
    with pytest.raises(TypeError, match="Exactly one"):
        MerlinProcessor()


def test_merlinprocessor_accepts_remote_processor_through_processor_argument():
    """RemoteProcessor passed through processor= is accepted."""
    remote_processor = make_remote_processor_mock(["probs", "sample_count"])

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            proc = MerlinProcessor(processor=remote_processor)

    assert proc.processor is None
    assert proc.remote_processor is remote_processor
    assert proc.session is None
    assert proc.backend_capabilities.available_commands == ("probs", "sample_count")


def test_merlinprocessor_routes_remote_processor_argument_to_remote_backend():
    """RemoteProcessor passed through processor= uses the remote backend kind."""
    remote_processor = make_remote_processor_mock(["probs"])

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(processor=remote_processor)

    assert proc.backend_kind == "remote_processor"


def test_merlinprocessor_accepts_remote_processor_through_remote_processor_argument():
    """Existing remote_processor= construction remains supported."""
    remote_processor = make_remote_processor_mock(["probs"])

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        with pytest.warns(DeprecationWarning, match="'remote_processor' argument"):
            proc = MerlinProcessor(remote_processor=remote_processor)

    assert proc.processor is None
    assert proc.remote_processor is remote_processor
    assert proc.session is None
    assert proc.backend_kind == "remote_processor"


def test_merlinprocessor_preserves_positional_remote_processor_argument():
    """The first positional argument still binds to remote_processor."""
    remote_processor = make_remote_processor_mock(["probs"])

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        with pytest.warns(DeprecationWarning, match="'remote_processor' argument"):
            proc = MerlinProcessor(remote_processor, None, 16)

    assert proc.processor is None
    assert proc.remote_processor is remote_processor
    assert proc.session is None
    assert proc.microbatch_size == 16
    assert proc.backend_kind == "remote_processor"


def test_merlinprocessor_accepts_session_backend():
    """Existing session= construction remains supported."""
    session = MagicMock(spec=ISession)
    remote_processor = make_remote_processor_mock(["probs", "sample_count"])
    session.build_remote_processor.return_value = remote_processor

    with patch.object(MerlinProcessor, "_extract_rp_token") as extract:
        proc = MerlinProcessor(session=session)

    extract.assert_not_called()
    assert proc.processor is None
    assert proc.remote_processor is None
    assert proc.session is session
    assert proc.backend_kind == "session"
    assert proc.backend_capabilities.available_commands == ("probs", "sample_count")


def test_merlinprocessor_preserves_positional_session_argument():
    """The second positional argument still binds to session."""
    session = MagicMock(spec=ISession)
    remote_processor = make_remote_processor_mock(["probs", "sample_count"])
    session.build_remote_processor.return_value = remote_processor

    proc = MerlinProcessor(None, session)

    assert proc.processor is None
    assert proc.remote_processor is None
    assert proc.session is session
    assert proc.backend_kind == "session"
    assert proc.backend_capabilities.available_commands == ("probs", "sample_count")


def test_merlinprocessor_rejects_processor_and_remote_processor_together():
    """processor= and remote_processor= are mutually exclusive."""
    local_processor = make_local_aprocessor()
    remote_processor = make_remote_processor_mock()

    with pytest.raises(TypeError, match="mutually exclusive"):
        MerlinProcessor(
            processor=local_processor,
            remote_processor=remote_processor,
        )


def test_merlinprocessor_rejects_processor_and_session_together():
    """processor= and session= are mutually exclusive."""
    local_processor = make_local_aprocessor()
    session = MagicMock(spec=ISession)

    with pytest.raises(TypeError, match="mutually exclusive"):
        MerlinProcessor(processor=local_processor, session=session)


def test_merlinprocessor_rejects_remote_processor_and_session_together():
    """remote_processor= and session= are mutually exclusive."""
    remote_processor = make_remote_processor_mock()
    session = MagicMock(spec=ISession)

    with pytest.raises(TypeError, match="mutually exclusive"):
        MerlinProcessor(remote_processor=remote_processor, session=session)


def test_local_aprocessor_backend_does_not_extract_remote_token():
    """Local AProcessor path leaves token extraction to remote backends only."""
    local_processor = make_local_aprocessor(["probs"])

    with patch.object(MerlinProcessor, "_extract_rp_token") as extract:
        proc = MerlinProcessor(processor=local_processor)

    extract.assert_not_called()
    assert proc._token is None
    assert proc.backend_kind == "local_processor"


def test_local_aprocessor_backend_uses_available_commands_from_processor():
    """Local AProcessor capabilities come from processor.available_commands."""
    local_processor = make_local_aprocessor(["samples"])

    proc = MerlinProcessor(processor=local_processor)

    assert proc.available_commands == ("samples",)
    assert proc.backend_capabilities.available_commands == ("samples",)


def test_remote_processor_argument_copies_available_commands():
    """processor=RemoteProcessor copies commands like remote_processor=."""
    remote_processor = make_remote_processor_mock(["sample_count", "samples"])

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(processor=remote_processor)

    assert proc.available_commands == ("sample_count", "samples")


def test_remote_processor_argument_extracts_token_like_remote_processor_argument():
    """processor=RemoteProcessor uses the existing remote token extraction path."""
    remote_processor = make_remote_processor_mock(["probs"])
    resolved_value = "resolved-token-value"

    with patch.object(
        MerlinProcessor, "_extract_rp_token", return_value=resolved_value
    ) as extract:
        proc = MerlinProcessor(processor=remote_processor)

    extract.assert_called_once_with(remote_processor)
    assert proc._token == resolved_value


def test_merlinprocessor_rejects_unknown_remote_aprocessor_subclass():
    """Remote AProcessor subclasses must be supported explicitly."""
    remote_aprocessor = make_local_aprocessor(["probs"])
    remote_aprocessor.is_remote = True

    with pytest.raises(TypeError, match="Unsupported remote AProcessor subclass"):
        MerlinProcessor(processor=remote_aprocessor)


def test_local_aprocessor_backend_does_not_require_copy_method():
    """Local AProcessor construction does not require a copy() method."""
    processor = MagicMock(spec=AProcessor)
    processor.is_remote = False
    processor.name = "local:slos"
    processor.available_commands = ["probs"]

    proc = MerlinProcessor(processor=processor)

    assert proc.processor is processor
    assert proc.backend_kind == "local_processor"


def test_remote_processor_path_stores_backend_capabilities():
    """RemoteProcessor path stores capabilities in backend_capabilities."""
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.name = "sim:slos"
    remote_processor.available_commands = ["probs", "sample_count"]
    remote_processor.proxies = None

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(processor=remote_processor)

    assert proc.backend_capabilities.name == "sim:slos"
    assert proc.backend_capabilities.available_commands == ("probs", "sample_count")


def test_backend_name_property_backward_compatibility():
    """backend_name property provides backward compatibility."""
    remote_processor = MagicMock(spec=RemoteProcessor)
    remote_processor.name = "sim:slos"
    remote_processor.available_commands = ["probs"]
    remote_processor.proxies = None

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(processor=remote_processor)

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
        proc = MerlinProcessor(processor=remote_processor)

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

    with patch.object(
        MerlinProcessor, "_extract_rp_token", return_value=None
    ) as extract:
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
        proc = MerlinProcessor(processor=remote_processor)

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


def test_run_chunk_dispatches_local_processor_backend():
    """Local processor backends are delegated to _run_chunk_local."""
    proc = make_processor(["probs"])
    proc.backend_kind = "local_processor"
    layer = FakeLayer()
    config = MagicMock()
    input_chunk = torch.tensor([[0.25, 0.5]])
    state = make_state()
    deadline = time.time() + 10.0
    expected = torch.tensor([[1.0]])
    proc._run_chunk_local = MagicMock(return_value=expected)

    output = proc._run_chunk(
        layer,
        config,
        input_chunk,
        nsample=None,
        state=state,
        deadline=deadline,
        job_base_label="ignored-local-label",
    )

    assert output is expected
    proc._run_chunk_local.assert_called_once_with(
        layer, config, input_chunk, None, state, deadline
    )


def test_local_aprocessor_backend_executes_quantum_leaf_without_chunking():
    """forward() sends the full local AProcessor batch to _run_chunk_local."""

    class LocalQuantumLeaf(MerlinModule):
        def __init__(self) -> None:
            super().__init__()
            self.uid = "local-leaf"

        def export_config(self):
            return {
                "circuit": pcvl.Circuit(m=2),
                "input_state": [1, 0],
                "input_param_order": ["theta_0", "theta_1"],
            }

    proc = MerlinProcessor(
        processor=make_local_aprocessor(["probs"]),
        microbatch_size=2,
        chunk_concurrency=1,
    )
    layer = LocalQuantumLeaf()
    layer.eval()
    input_tensor = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32
    )
    observed_chunks: list[torch.Tensor] = []

    def fake_run_chunk_local(
        layer_arg,
        config,
        input_chunk,
        nsample,
        state,
        deadline,
    ):
        assert layer_arg is layer
        assert isinstance(config, ValidatedLayerConfig)
        assert nsample is None
        assert state["cancel_requested"] is False
        assert deadline is not None
        observed_chunks.append(input_chunk.clone())
        return torch.ones(input_chunk.shape[0], 2)

    proc._run_chunk_local = MagicMock(side_effect=fake_run_chunk_local)

    output = proc.forward(layer, input_tensor, timeout=10.0)

    torch.testing.assert_close(output, torch.ones(3, 2))
    assert proc._run_chunk_local.call_count == 1
    assert [chunk.shape[0] for chunk in observed_chunks] == [3]


def test_estimate_required_shots_rejects_local_processor_backend():
    """Shot estimation is remote-only and fails clearly for local processors."""

    class LocalQuantumLeaf(MerlinModule):
        def export_config(self):
            return {
                "circuit": pcvl.Circuit(m=2),
                "input_state": [1, 0],
                "input_param_order": ["theta_0", "theta_1"],
            }

    proc = MerlinProcessor(processor=make_local_aprocessor(["probs"]))
    proc._create_fresh_rp = MagicMock(side_effect=AssertionError("unexpected RP"))

    with pytest.raises(RuntimeError, match="remote processor or session backends"):
        proc.estimate_required_shots_per_input(
            LocalQuantumLeaf(),
            torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            desired_samples_per_input=100,
        )

    proc._create_fresh_rp.assert_not_called()


def test_estimate_required_shots_uses_remote_processor_passed_through_processor():
    """RemoteProcessor passed through processor= can use the remote estimator."""

    class RemoteEstimationLayer(MerlinModule):
        def export_config(self):
            return {
                "circuit": pcvl.Circuit(m=2),
                "input_state": [1, 0],
                "input_param_order": ["theta_0", "theta_1"],
            }

    remote_processor = make_remote_processor_mock(["probs"])
    child_remote_processor = MagicMock(spec=RemoteProcessor)
    child_remote_processor.estimate_required_shots.side_effect = [123, 456]

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(processor=remote_processor)

    proc._create_fresh_rp = MagicMock(return_value=child_remote_processor)
    input_tensor = torch.tensor(
        [[0.25, 0.5], [0.75, 1.0]], dtype=torch.float64
    )

    estimates = proc.estimate_required_shots_per_input(
        RemoteEstimationLayer(),
        input_tensor,
        desired_samples_per_input=100,
    )

    assert estimates == [123, 456]
    proc._create_fresh_rp.assert_called_once_with()
    child_remote_processor.set_circuit.assert_called_once()
    child_remote_processor.with_input.assert_called_once_with(pcvl.BasicState([1, 0]))
    child_remote_processor.min_detected_photons_filter.assert_called_once_with(1)
    assert child_remote_processor.estimate_required_shots.call_args_list == [
        call(
            100,
            param_values={"theta_0": 0.25 * np.pi, "theta_1": 0.5 * np.pi},
        ),
        call(
            100,
            param_values={"theta_0": 0.75 * np.pi, "theta_1": 1.0 * np.pi},
        ),
    ]


def test_run_chunk_local_uses_fresh_processor_per_execution():
    """Local execution uses a rebuilt processor per run."""
    proc = make_local_chunk_processor(["probs"])
    layer = FakeLayer()
    config = make_local_chunk_config()
    input_chunk = torch.tensor([[0.25, 0.5], [0.75, 1.0]])
    state = make_state()
    raw_results = {"results_list": [{"results": {"|1,0>": 1.0}}]}
    sampler = FakeSyncSampler(raw_results)
    execution_processor = proc.local_execution_processor
    copied_circuit = config.circuit.copy.return_value

    with (
        patch.object(
            merlin_processor_module, "Sampler", return_value=sampler
        ) as sampler_cls,
    ):
        output = proc._run_chunk_local(
            layer, config, input_chunk, None, state, deadline=None
        )

    assert output is proc._process_batch_results.return_value
    proc._create_fresh_local_processor.assert_called_once_with()
    proc.processor.set_circuit.assert_not_called()
    execution_processor.set_circuit.assert_called_once_with(copied_circuit)
    execution_processor.with_input.assert_called_once()
    execution_processor.min_detected_photons_filter.assert_called_once_with(1)
    sampler_cls.assert_called_once_with(
        execution_processor, max_shots_per_call=MerlinProcessor.DEFAULT_MAX_SHOTS
    )
    assert sampler.cleared is True
    assert sampler.iterations == [
        {"circuit_params": {"theta_0": 0.25, "theta_1": 0.5}},
        {"circuit_params": {"theta_0": 0.75, "theta_1": 1.0}},
    ]
    assert sampler.probs.executed is True
    assert sampler.probs.execute_kwargs == {}
    assert sampler.sample_count.executed is False
    proc._process_batch_results.assert_called_once_with(
        raw_results, 2, layer, None, True
    )


def test_run_chunk_local_uses_sample_count_when_probs_unavailable():
    """Local chunk execution uses sample_count for sampling-capable processors."""
    proc = make_local_chunk_processor(["sample_count"])
    layer = FakeLayer()
    config = make_local_chunk_config()
    raw_results = {"results_list": [{"results": {"|1,0>": 3}}]}
    sampler = FakeSyncSampler(raw_results)

    with patch.object(merlin_processor_module, "Sampler", return_value=sampler):
        proc._run_chunk_local(
            layer,
            config,
            torch.tensor([[0.25, 0.5]]),
            nsample=None,
            state=make_state(),
            deadline=None,
        )

    assert sampler.sample_count.executed is True
    assert sampler.sample_count.execute_kwargs == {
        "max_samples": MerlinProcessor.DEFAULT_SHOTS_PER_CALL
    }
    assert sampler.probs.executed is False
    proc._process_batch_results.assert_called_once_with(
        raw_results, 1, layer, None, False
    )


def test_run_chunk_local_caps_default_sample_count_to_max_shots_per_call():
    """Local sampling caps default shots when max_shots_per_call is lower."""
    proc = make_local_chunk_processor(["sample_count"])
    proc.max_shots_per_call = 123
    layer = FakeLayer()
    config = make_local_chunk_config()
    raw_results = {"results_list": [{"results": {"|1,0>": 3}}]}
    sampler = FakeSyncSampler(raw_results)

    with patch.object(
        merlin_processor_module, "Sampler", return_value=sampler
    ) as sampler_cls:
        proc._run_chunk_local(
            layer,
            config,
            torch.tensor([[0.25, 0.5]]),
            nsample=None,
            state=make_state(),
            deadline=None,
        )

    sampler_cls.assert_called_once()
    assert sampler_cls.call_args.kwargs == {"max_shots_per_call": 123}
    assert sampler.sample_count.executed is True
    assert sampler.sample_count.execute_kwargs == {"max_samples": 123}
    proc._process_batch_results.assert_called_once_with(
        raw_results, 1, layer, None, False
    )


def test_run_chunk_local_uses_samples_when_sample_count_unavailable():
    """Local chunk execution falls back to samples when sample_count is absent."""
    proc = make_local_chunk_processor(["samples"])
    layer = FakeLayer()
    config = make_local_chunk_config()
    raw_results = {"results_list": [{"results": {"|1,0>": 3}}]}
    sampler = FakeSyncSampler(raw_results)

    with patch.object(merlin_processor_module, "Sampler", return_value=sampler):
        proc._run_chunk_local(
            layer,
            config,
            torch.tensor([[0.25, 0.5]]),
            nsample=17,
            state=make_state(),
            deadline=None,
        )

    assert sampler.samples.executed is True
    assert sampler.samples.execute_kwargs == {"max_samples": 17}
    assert sampler.sample_count.executed is False
    proc._process_batch_results.assert_called_once_with(
        raw_results, 1, layer, 17, False
    )


def test_run_chunk_local_defaults_to_sample_count_when_commands_are_empty():
    """Local chunk execution fails explicitly through sample_count when commands are empty."""
    proc = make_local_chunk_processor([])
    layer = FakeLayer()
    config = make_local_chunk_config()
    raw_results = {"results_list": [{"results": {"|1,0>": 3}}]}
    sampler = FakeSyncSampler(raw_results)

    with patch.object(merlin_processor_module, "Sampler", return_value=sampler):
        proc._run_chunk_local(
            layer,
            config,
            torch.tensor([[0.25, 0.5]]),
            nsample=5,
            state=make_state(),
            deadline=None,
        )

    assert sampler.sample_count.executed is True
    assert sampler.sample_count.execute_kwargs == {"max_samples": 5}
    proc._process_batch_results.assert_called_once_with(raw_results, 1, layer, 5, False)


def test_run_chunk_local_raises_cancelled_before_execution():
    """Local chunk execution observes cancellation before starting work."""
    proc = make_local_chunk_processor(["probs"])
    state = make_state()
    state["cancel_requested"] = True

    with pytest.raises(CancelledError, match="Local call was cancelled"):
        proc._run_chunk_local(
            FakeLayer(),
            make_local_chunk_config(),
            torch.tensor([[0.25, 0.5]]),
            nsample=None,
            state=state,
            deadline=None,
        )

    proc._create_fresh_local_processor.assert_not_called()
    proc._process_batch_results.assert_not_called()


def test_run_chunk_local_raises_timeout_before_execution():
    """Local chunk execution observes deadlines before starting work."""
    proc = make_local_chunk_processor(["probs"])

    with pytest.raises(TimeoutError, match="Local call timed out"):
        proc._run_chunk_local(
            FakeLayer(),
            make_local_chunk_config(),
            torch.tensor([[0.25, 0.5]]),
            nsample=None,
            state=make_state(),
            deadline=time.time() - 1.0,
        )

    proc._create_fresh_local_processor.assert_not_called()
    proc._process_batch_results.assert_not_called()


def test_run_chunk_local_raises_cancelled_after_execution():
    """Local chunk execution observes cancellation before returning results."""
    proc = make_local_chunk_processor(["probs"])
    state = make_state()
    raw_results = {"results_list": [{"results": {"|1,0>": 1.0}}]}
    sampler = FakeSyncSampler(
        raw_results, on_execute=lambda: state.__setitem__("cancel_requested", True)
    )

    with (
        patch.object(merlin_processor_module, "Sampler", return_value=sampler),
        pytest.raises(CancelledError, match="Local call was cancelled"),
    ):
        proc._run_chunk_local(
            FakeLayer(),
            make_local_chunk_config(),
            torch.tensor([[0.25, 0.5]]),
            nsample=None,
            state=state,
            deadline=None,
        )

    proc._process_batch_results.assert_not_called()


def test_run_chunk_local_raises_timeout_after_execution(monkeypatch):
    """Local chunk execution observes deadlines before returning results."""
    proc = make_local_chunk_processor(["probs"])
    raw_results = {"results_list": [{"results": {"|1,0>": 1.0}}]}
    sampler = FakeSyncSampler(raw_results)
    time_values = iter([0.0, 2.0])
    monkeypatch.setattr(merlin_processor_module.time, "time", lambda: next(time_values))

    with (
        patch.object(merlin_processor_module, "Sampler", return_value=sampler),
        pytest.raises(TimeoutError, match="Local call timed out"),
    ):
        proc._run_chunk_local(
            FakeLayer(),
            make_local_chunk_config(),
            torch.tensor([[0.25, 0.5]]),
            nsample=None,
            state=make_state(),
            deadline=1.0,
        )

    proc._process_batch_results.assert_not_called()


def test_create_fresh_local_processor_does_not_share_experiment_state():
    """Fresh local processors do not share mutable Perceval experiment state."""
    original_processor = Processor("SLOS")
    original_processor.set_circuit(pcvl.Circuit(2))
    original_processor.with_input(pcvl.BasicState([1, 0]))
    proc = MerlinProcessor(processor=original_processor)

    execution_processor = proc._create_fresh_local_processor()
    execution_processor.set_circuit(pcvl.Circuit(4))
    execution_processor.with_input(pcvl.BasicState([0, 1, 0, 0]))

    assert execution_processor is not original_processor
    assert execution_processor.experiment is not original_processor.experiment
    assert str(original_processor.input_state) == "|1,0>"
    assert str(execution_processor.input_state) == "|0,1,0,0>"
    assert original_processor.circuit_size == 2
    assert execution_processor.circuit_size == 4


def test_run_chunk_local_preserves_processor_experiment_metadata():
    """Local execution keeps caller-provided experiment detector metadata."""
    original_processor = Processor("SLOS")
    original_processor.set_circuit(pcvl.Circuit(4))
    original_processor.add_port(0, pcvl.Port(pcvl.Encoding.DUAL_RAIL, "q0"))
    original_processor.add_herald(2, 1, "h0")
    original_processor.experiment._add_detector(3, pcvl.Detector.threshold())
    original_processor.set_postselection("[3]==1")
    proc = MerlinProcessor(processor=original_processor)
    proc._process_batch_results = MagicMock(return_value=torch.tensor([[1.0]]))
    layer = FakeLayer(final_keys=[(1, 0, 0, 0)])
    config = SimpleNamespace(
        circuit=pcvl.Circuit(4),
        input_state=[1, 0, 0],
        input_param_order=[],
    )
    raw_results = {"results_list": [{"results": {"|1,0,0,0>": 1.0}}]}
    captured_processor = {}

    class CapturingSampler(FakeSyncSampler):
        """Sampler fake that exposes the execution processor under test."""

        def __init__(self, processor, max_shots_per_call) -> None:
            captured_processor["processor"] = processor
            captured_processor["max_shots_per_call"] = max_shots_per_call
            super().__init__(raw_results)

    with patch.object(merlin_processor_module, "Sampler", CapturingSampler):
        output = proc._run_chunk_local(
            layer,
            config,
            torch.empty((1, 0)),
            nsample=None,
            state=make_state(),
            deadline=None,
        )

    execution_processor = captured_processor["processor"]
    execution_experiment = execution_processor.experiment
    assert execution_processor is not original_processor
    assert execution_experiment is not original_processor.experiment
    assert execution_experiment.out_port_names == ["q0", "q0", "h0", ""]
    assert execution_experiment.in_port_names == ["q0", "q0", "h0", ""]
    assert execution_experiment.heralds == {2: 1}
    assert execution_experiment.in_heralds == {2: 1}
    assert execution_experiment.detectors[3] is not None
    assert str(execution_experiment.post_select_fn) == "[3] == 1"
    assert str(execution_processor.input_state) == "|1,0,1,0>"
    assert execution_processor._min_detected_photons_filter == 1
    assert str(original_processor.post_select_fn) == "[3] == 1"
    torch.testing.assert_close(output, torch.tensor([[1.0]]))


def test_run_chunk_local_accepts_layer_with_different_mode_count_than_original_processor():
    """Local execution ignores the caller processor's previous circuit mode count."""
    original_processor = Processor("SLOS")
    original_processor.set_circuit(pcvl.Circuit(2))
    original_processor.with_input(pcvl.BasicState([1, 0]))
    proc = MerlinProcessor(processor=original_processor)
    layer = FakeLayer(final_keys=[(1, 0, 0, 0), (0, 1, 0, 0)])
    config = SimpleNamespace(
        circuit=pcvl.Circuit(4),
        input_state=[1, 0, 0, 0],
        input_param_order=[],
    )

    output = proc._run_chunk_local(
        layer,
        config,
        torch.empty((1, 0)),
        nsample=None,
        state=make_state(),
        deadline=None,
    )

    assert str(original_processor.input_state) == "|1,0>"
    assert original_processor.circuit_size == 2
    torch.testing.assert_close(output, torch.tensor([[1.0, 0.0]]))


def test_run_chunk_local_rejects_postselection_with_different_mode_count():
    """Local execution rejects postselection tied to another mode layout."""
    original_processor = Processor("SLOS")
    original_processor.set_circuit(pcvl.Circuit(3))
    original_processor.set_postselection("[2]==1")
    proc = MerlinProcessor(processor=original_processor)
    layer = FakeLayer(final_keys=[(1, 0), (0, 1)])
    config = SimpleNamespace(
        circuit=pcvl.Circuit(2),
        input_state=[1, 0],
        input_param_order=[],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Local processor experiment metadata is tied to circuit size 3, "
            "but the execution circuit has size 2."
        ),
    ):
        proc._run_chunk_local(
            layer,
            config,
            torch.empty((1, 0)),
            nsample=None,
            state=make_state(),
            deadline=None,
        )

    assert str(original_processor.post_select_fn) == "[2] == 1"


def test_run_chunk_local_does_not_mutate_original_real_processor_input():
    """Local execution does not rewrite the caller's real Perceval Processor."""
    original_processor = Processor("SLOS")
    original_processor.set_circuit(pcvl.Circuit(2))
    original_processor.with_input(pcvl.BasicState([0, 1]))
    proc = MerlinProcessor(processor=original_processor)
    layer = FakeLayer(final_keys=[(1, 0), (0, 1)])
    config = SimpleNamespace(
        circuit=pcvl.Circuit(2),
        input_state=[1, 0],
        input_param_order=[],
    )

    output = proc._run_chunk_local(
        layer,
        config,
        torch.empty((1, 0)),
        nsample=None,
        state=make_state(),
        deadline=None,
    )

    assert str(original_processor.input_state) == "|0,1>"
    torch.testing.assert_close(output, torch.tensor([[1.0, 0.0]]))


def test_run_chunk_local_executes_real_perceval_processor():
    """Local chunk execution works with a real Perceval Processor."""
    pcvl_proc = Processor("SLOS")
    proc = MerlinProcessor(processor=pcvl_proc)
    layer = FakeLayer(final_keys=[(1, 0), (0, 1)])
    config = SimpleNamespace(
        circuit=pcvl.Circuit(2),
        input_state=[1, 0],
        input_param_order=[],
    )

    output = proc._run_chunk_local(
        layer,
        config,
        torch.empty((2, 0)),
        nsample=None,
        state=make_state(),
        deadline=None,
    )

    torch.testing.assert_close(output, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))


def test_run_chunk_local_executes_real_clifford_processor_samples():
    """Local chunk execution handles real Perceval samples results."""
    pcvl_proc = Processor("CliffordClifford2017")
    proc = MerlinProcessor(processor=pcvl_proc)
    layer = FakeLayer(final_keys=[(1, 0), (0, 1)])
    config = SimpleNamespace(
        circuit=pcvl.Circuit(2),
        input_state=[1, 0],
        input_param_order=[],
    )

    assert proc.available_commands == ("samples",)

    output = proc._run_chunk_local(
        layer,
        config,
        torch.empty((2, 0)),
        nsample=8,
        state=make_state(),
        deadline=None,
    )

    torch.testing.assert_close(output, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))


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


def test_submit_job_caps_default_sample_count_to_max_shots_per_call():
    """Remote submission caps default shots when max_shots_per_call is lower."""
    proc = make_processor(["sample_count"])
    proc.max_shots_per_call = 123
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
    assert sampler.sample_count.execute_kwargs == {"max_samples": 123}


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


def test_process_batch_results_normalizes_sample_sequences():
    """Sequence sample payloads are counted and normalized per row."""
    proc = make_processor(["samples"])
    layer = FakeLayer()
    raw_results = {
        "results_list": [
            {
                "results": [
                    pcvl.BasicState([1, 0]),
                    pcvl.BasicState([1, 0]),
                    pcvl.BasicState([0, 1]),
                ]
            }
        ],
    }

    result = proc._process_batch_results(raw_results, 1, layer)

    assert torch.allclose(result, torch.tensor([[2 / 3, 1 / 3]]))


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
        torch.tensor([
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]),
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
        proc = MerlinProcessor(processor=original_rp)

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


def test_create_fresh_rp_rejects_local_processor_backend():
    """Fresh RemoteProcessor creation is invalid for local processor backends."""
    proc = MerlinProcessor(processor=make_local_aprocessor(["probs"]))

    with pytest.raises(RuntimeError, match="Fresh RemoteProcessor creation"):
        proc._create_fresh_rp()


def test_create_fresh_rp_remote_processor_path_maintains_available_commands():
    """RemoteProcessor path preserves available_commands through cloning."""
    original_rp = MagicMock(spec=RemoteProcessor)
    original_rp.name = "sim:slos"
    original_rp.available_commands = ["probs", "sample_count", "samples"]
    original_rp.proxies = None

    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="token"):
        proc = MerlinProcessor(processor=original_rp)

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
        proc = MerlinProcessor(processor=original_rp)

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
