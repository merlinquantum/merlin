"""No-cloud unit tests for the Perceval adapter layer (PML-306).

Every test uses plain fakes or monkeypatching — no cloud access and no real
RemoteProcessor construction.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import perceval as pcvl
import pytest

import merlin.core.perceval_adapter as adapter_module
from merlin.core.perceval_adapter import (
    LOCAL_EXPERIMENT_SNAPSHOT_ATTR,
    JobStatusSnapshot,
    LocalExperimentSnapshot,
    PercevalAdapter,
    RemoteJobFailedError,
    TokenExtractionError,
)


def make_rp(handler=None) -> MagicMock:
    """Build a RemoteProcessor fake exposing an RPC handler."""
    rp = MagicMock(name="remote_processor")
    if handler is None:
        rp.get_rpc_handler.side_effect = RuntimeError("no handler")
    else:
        rp.get_rpc_handler.return_value = handler
    return rp


def patch_remote_config(token: str | None):
    """Patch perceval's RemoteConfig to return the given global token."""
    config = MagicMock()
    config.get_token.return_value = token
    return patch("perceval.runtime.RemoteConfig", return_value=config)


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


class TestExtractToken:
    @pytest.mark.parametrize("attr", ["token", "_token", "auth_token"])
    def test_handler_attribute_variants(self, attr):
        handler = SimpleNamespace(**{attr: "tok-123"})

        assert PercevalAdapter.extract_token(make_rp(handler)) == "tok-123"

    def test_handler_attribute_priority_order(self):
        handler = SimpleNamespace(token="primary", _token="secondary")

        assert PercevalAdapter.extract_token(make_rp(handler)) == "primary"

    def test_empty_handler_attributes_are_skipped(self):
        handler = SimpleNamespace(token="", _token=None, auth_token="fallback")

        assert PercevalAdapter.extract_token(make_rp(handler)) == "fallback"

    def test_bearer_header_fallback(self):
        handler = SimpleNamespace(headers={"Authorization": "Bearer tok-from-header"})

        assert PercevalAdapter.extract_token(make_rp(handler)) == "tok-from-header"

    def test_malformed_bearer_header_is_ignored(self):
        handler = SimpleNamespace(headers={"Authorization": "Basic abc"})

        with patch_remote_config(None):
            assert PercevalAdapter.extract_token(make_rp(handler)) is None

    def test_remote_config_fallback(self):
        handler = SimpleNamespace()

        with patch_remote_config("  global-tok  "):
            assert PercevalAdapter.extract_token(make_rp(handler)) == "global-tok"

    def test_broken_handler_falls_back_to_remote_config(self):
        with patch_remote_config("global-tok"):
            assert PercevalAdapter.extract_token(make_rp(None)) == "global-tok"

    def test_all_strategies_failing_returns_none(self):
        with patch_remote_config(None):
            assert PercevalAdapter.extract_token(make_rp(SimpleNamespace())) is None

    def test_remote_config_error_returns_none(self):
        config = MagicMock()
        config.get_token.side_effect = RuntimeError("config broken")

        with patch("perceval.runtime.RemoteConfig", return_value=config):
            assert PercevalAdapter.extract_token(make_rp(SimpleNamespace())) is None


# ---------------------------------------------------------------------------
# URL / clone / session / capabilities
# ---------------------------------------------------------------------------


class TestProcessorAccess:
    def test_get_url_reads_handler_url(self):
        handler = SimpleNamespace(url="https://api.quandela.cloud")

        assert PercevalAdapter.get_url(make_rp(handler)) == "https://api.quandela.cloud"

    def test_get_url_returns_none_without_url_attribute(self):
        assert PercevalAdapter.get_url(make_rp(SimpleNamespace())) is None

    def test_clone_forwards_name_token_url_and_proxies(self):
        rp = make_rp(SimpleNamespace(url="https://cloud"))
        rp.name = "sim:slos"
        rp.proxies = {"https": "proxy"}
        clone = MagicMock(name="clone")

        with patch.object(
            adapter_module, "RemoteProcessor", return_value=clone
        ) as rp_cls:
            result = PercevalAdapter.clone_remote_processor(rp, "tok")

        assert result is clone
        rp_cls.assert_called_once_with(
            name="sim:slos",
            token="tok",
            url="https://cloud",
            proxies={"https": "proxy"},
        )

    def test_build_from_session_delegates(self):
        session = MagicMock()

        result = PercevalAdapter.build_from_session(session)

        assert result is session.build_remote_processor.return_value

    def test_get_backend_capabilities_snapshots_commands(self):
        processor = SimpleNamespace(
            name="sim:slos", available_commands=["probs", "sample_count"]
        )

        name, commands = PercevalAdapter.get_backend_capabilities(processor)

        assert name == "sim:slos"
        assert commands == ("probs", "sample_count")

    def test_configure_processor_sets_circuit_input_and_filter(self):
        processor = MagicMock()
        circuit = MagicMock(name="circuit")

        PercevalAdapter.configure_processor(processor, circuit, [1, 0, 1])

        processor.set_circuit.assert_called_once_with(circuit)
        processor.with_input.assert_called_once_with(pcvl.BasicState([1, 0, 1]))
        processor.min_detected_photons_filter.assert_called_once_with(2)

    def test_configure_processor_skips_input_when_falsy(self):
        processor = MagicMock()

        PercevalAdapter.configure_processor(processor, MagicMock(), None)

        processor.set_circuit.assert_called_once()
        processor.with_input.assert_not_called()
        processor.min_detected_photons_filter.assert_not_called()

    def test_copy_circuit_returns_independent_copy(self):
        circuit = MagicMock()

        assert PercevalAdapter.copy_circuit(circuit) is circuit.copy.return_value

    def test_estimate_required_shots_delegates(self):
        rp = MagicMock()
        rp.estimate_required_shots.return_value = 4321

        result = PercevalAdapter.estimate_required_shots(rp, 100, {"theta": 0.5})

        assert result == 4321
        rp.estimate_required_shots.assert_called_once_with(
            100, param_values={"theta": 0.5}
        )


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


class FakeCommand:
    def __init__(self) -> None:
        self.name = None
        self.async_kwargs = None
        self.sync_kwargs = None

    def execute_async(self, **kwargs):
        self.async_kwargs = kwargs
        return "async-job"

    def execute_sync(self, **kwargs):
        self.sync_kwargs = kwargs
        return {"results_list": []}


class FakeSampler:
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


class TestSamplerAccess:
    def test_create_sampler_loads_iterations(self):
        processor = MagicMock()
        iterations = [{"theta": 0.1}, {"theta": 0.2}]
        fake = FakeSampler()

        with patch.object(adapter_module, "Sampler", return_value=fake) as sampler_cls:
            sampler = PercevalAdapter.create_sampler(processor, 5000, iterations)

        assert sampler is fake
        sampler_cls.assert_called_once_with(processor, max_shots_per_call=5000)
        assert fake.cleared is True
        assert fake.iterations == iterations

    @pytest.mark.parametrize("command", ["probs", "sample_count", "samples"])
    def test_submit_async_dispatches_each_command(self, command):
        sampler = FakeSampler()

        job = PercevalAdapter.submit_async(sampler, command, name="job-name")

        assert job == "async-job"
        assert getattr(sampler, command).name == "job-name"
        assert getattr(sampler, command).async_kwargs == {}

    def test_submit_async_forwards_max_samples(self):
        sampler = FakeSampler()

        PercevalAdapter.submit_async(sampler, "sample_count", max_samples=777)

        assert sampler.sample_count.async_kwargs == {"max_samples": 777}

    def test_submit_async_without_name_leaves_command_name(self):
        sampler = FakeSampler()

        PercevalAdapter.submit_async(sampler, "probs")

        assert sampler.probs.name is None

    @pytest.mark.parametrize("command", ["probs", "sample_count", "samples"])
    def test_execute_sync_dispatches_each_command(self, command):
        sampler = FakeSampler()

        result = PercevalAdapter.execute_sync(sampler, command)

        assert result == {"results_list": []}
        assert getattr(sampler, command).sync_kwargs == {}

    def test_execute_sync_forwards_max_samples(self):
        sampler = FakeSampler()

        PercevalAdapter.execute_sync(sampler, "samples", max_samples=99)

        assert sampler.samples.sync_kwargs == {"max_samples": 99}

    def test_serializable_iterator_normalized(self):
        sampler = FakeSampler()
        iterations = [{"theta": 0.1}]
        iterator = SimpleNamespace(iterations=iterations)
        sampler._iterator = iterator
        sampler.probs._request_data = {"payload": {"iterator": iterator}}

        PercevalAdapter.submit_async(sampler, "probs")

        assert sampler.probs._request_data["payload"]["iterator"] == iterations

    def test_serializable_iterator_untouched_for_plain_lists(self):
        sampler = FakeSampler()
        sampler._iterator = None
        sampler.probs._request_data = {"payload": {"iterator": ["kept"]}}

        PercevalAdapter.submit_async(sampler, "probs")

        assert sampler.probs._request_data["payload"]["iterator"] == ["kept"]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


class TestJobAccess:
    def test_job_snapshot_maps_all_fields(self):
        job = SimpleNamespace(
            id="job-1",
            status=SimpleNamespace(state="RUNNING", progress=0.4, stop_message="msg"),
            is_complete=False,
            is_failed=False,
        )

        snapshot = PercevalAdapter.job_snapshot(job)

        assert snapshot == JobStatusSnapshot(
            job_id="job-1",
            state="RUNNING",
            progress=0.4,
            stop_message="msg",
            is_complete=False,
            is_failed=False,
        )

    def test_job_snapshot_falls_back_to_job_id_attribute(self):
        job = SimpleNamespace(job_id="alt-id")

        assert PercevalAdapter.job_snapshot(job).job_id == "alt-id"

    def test_job_snapshot_defaults_for_bare_objects(self):
        snapshot = PercevalAdapter.job_snapshot(object())

        assert snapshot == JobStatusSnapshot(
            job_id=None,
            state=None,
            progress=None,
            stop_message=None,
            is_complete=False,
            is_failed=False,
        )

    def test_get_results_propagates_perceval_errors(self):
        job = MagicMock()
        job.get_results.side_effect = RuntimeError("Results are not available")

        with pytest.raises(RuntimeError, match="Results are not available"):
            PercevalAdapter.get_results(job)

    def test_cancel_job_swallows_cancel_errors(self):
        job = MagicMock()
        job.cancel.side_effect = RuntimeError("already finished")

        PercevalAdapter.cancel_job(job)  # must not raise

        job.cancel.assert_called_once_with()

    def test_cancel_job_ignores_objects_without_cancel(self):
        PercevalAdapter.cancel_job(object())  # must not raise


# ---------------------------------------------------------------------------
# Merlin-specific exceptions
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_token_extraction_error_is_value_error(self):
        assert issubclass(TokenExtractionError, ValueError)

    def test_remote_job_failed_error_is_runtime_error(self):
        assert issubclass(RemoteJobFailedError, RuntimeError)


# ---------------------------------------------------------------------------
# Local processors
# ---------------------------------------------------------------------------


def make_local_processor() -> pcvl.Processor:
    """Build a real local Perceval processor with metadata to preserve."""
    processor = pcvl.Processor("SLOS")
    processor.set_circuit(pcvl.Circuit(2))
    processor.set_postselection(pcvl.PostSelect("[0] == 1"))
    return processor


class TestLocalProcessorRebuild:
    def test_rebuild_returns_isolated_processor_with_snapshot(self):
        original = make_local_processor()

        fresh = PercevalAdapter.rebuild_local_processor(original)

        assert fresh is not original
        assert fresh.experiment is not original.experiment
        snapshot = getattr(fresh, LOCAL_EXPERIMENT_SNAPSHOT_ATTR)
        assert isinstance(snapshot, LocalExperimentSnapshot)

    def test_rebuild_rejects_processor_without_experiment(self):
        bare = SimpleNamespace(experiment=None, backend=MagicMock())

        with pytest.raises(TypeError, match="copyable"):
            PercevalAdapter.rebuild_local_processor(bare)

    def test_snapshot_and_restore_round_trip_preserves_postselection(self):
        original = make_local_processor()
        snapshot = PercevalAdapter.snapshot_experiment(original.experiment)

        fresh = PercevalAdapter.rebuild_local_processor(original)
        fresh.set_circuit(pcvl.Circuit(2))
        PercevalAdapter.restore_experiment(fresh.experiment, snapshot)

        assert fresh.experiment.post_select_fn == original.experiment.post_select_fn

    def test_restore_rejects_mismatched_circuit_size(self):
        original = make_local_processor()
        snapshot = PercevalAdapter.snapshot_experiment(original.experiment)

        fresh = PercevalAdapter.rebuild_local_processor(original)
        fresh.set_circuit(pcvl.Circuit(3))

        with pytest.raises(ValueError, match="circuit size"):
            PercevalAdapter.restore_experiment(fresh.experiment, snapshot)

    def test_snapshot_is_independent_of_source_experiment(self):
        original = make_local_processor()

        snapshot = PercevalAdapter.snapshot_experiment(original.experiment)
        original.set_postselection(pcvl.PostSelect("[1] == 1"))

        assert snapshot.postselection != original.experiment.post_select_fn
