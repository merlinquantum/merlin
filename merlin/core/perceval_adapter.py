"""Adapter layer isolating Merlin from Perceval internal APIs (PML-306).

MerlinProcessor and the execution units historically reached directly into
Perceval internals: RPC handler token/URL attributes, sampler command objects
(``probs`` / ``sample_count`` / ``samples``), remote job status fields, local
experiment private state, and ``RemoteConfig``. Any Perceval version bump that
renames or restructures those internals could silently break Merlin at runtime.

:class:`PercevalAdapter` owns every such access. The rest of Merlin talks to
this facade, so a Perceval API change is localized to this module.

The adapter is stateless (static methods) and duck-typed: it reads the same
attributes Perceval exposes today, which also makes it independently testable
with plain fakes.
"""

from __future__ import annotations

import copy
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import perceval as pcvl
from perceval.algorithm import Sampler
from perceval.runtime import AProcessor, Processor, RemoteJob, RemoteProcessor
from perceval.runtime.session import ISession

logger = logging.getLogger(__name__)

#: Attribute name used to carry a local experiment snapshot on a rebuilt
#: processor between :meth:`PercevalAdapter.rebuild_local_processor` and
#: :meth:`PercevalAdapter.restore_experiment`.
LOCAL_EXPERIMENT_SNAPSHOT_ATTR = "_merlin_local_experiment_metadata"


class TokenExtractionError(ValueError):
    """Raised when no auth token can be resolved for a RemoteProcessor.

    Subclasses ``ValueError`` so existing callers catching the historical
    exception type keep working.
    """


class RemoteJobFailedError(RuntimeError):
    """Raised when a remote Perceval job reports failure.

    Subclasses ``RuntimeError`` so existing callers catching the historical
    exception type keep working.
    """


@dataclass(frozen=True)
class JobStatusSnapshot:
    """Merlin-normalized view of a Perceval remote job's status.

    All ``getattr`` guards against Perceval job internals live in
    :meth:`PercevalAdapter.job_snapshot`; consumers only see these fields.
    """

    job_id: str | None
    state: Any
    progress: Any
    stop_message: Any
    is_complete: bool
    is_failed: bool


@dataclass(frozen=True)
class LocalExperimentSnapshot:
    """Experiment-level state that must survive local circuit replacement.

    Captures the Perceval experiment private state (ports, detectors, mode
    types, heralds, postselection) that ``clear_input_and_circuit()`` wipes.
    """

    circuit_size: int
    in_ports: tuple[tuple[Any, tuple[int, ...]], ...]
    out_ports: tuple[tuple[Any, tuple[int, ...]], ...]
    detectors: tuple[Any | None, ...]
    detectors_injected: tuple[int, ...]
    in_mode_type: tuple[Any, ...]
    out_mode_type: tuple[Any, ...]
    anon_herald_num: int
    postselection: Any

    @property
    def has_mode_metadata(self) -> bool:
        """Return whether metadata is tied to a concrete circuit mode layout."""

        return (
            bool(self.in_ports)
            or bool(self.out_ports)
            or any(detector is not None for detector in self.detectors)
            or bool(self.detectors_injected)
            or self.postselection != pcvl.PostSelect()
        )


class PercevalAdapter:
    """Stateless facade owning all direct Perceval-internal access."""

    # ------------------------------------------------------------------
    # Token / handler / URL
    # ------------------------------------------------------------------

    @staticmethod
    def extract_token(rp: RemoteProcessor) -> str | None:
        """Extract the auth token from a RemoteProcessor.

        Perceval stores the token on the RPC handler as ``handler.token``
        and also embeds it in ``handler.headers['Authorization']``.  We
        probe both locations so that inline-token and global-config
        ``RemoteProcessor`` instances are both handled.

        As a last resort, falls back to ``RemoteConfig().get_token()``.
        Returns ``None`` only if every strategy fails.
        """
        try:
            handler = rp.get_rpc_handler()
        except Exception:
            handler = None

        if handler is not None:
            # Primary: handler.token (set by RPCHandler.__init__)
            for attr in ("token", "_token", "auth_token"):
                val = getattr(handler, attr, None)
                if isinstance(val, str) and val:
                    return val

            # Fallback: parse 'Bearer <token>' from Authorization header
            headers = getattr(handler, "headers", None)
            if isinstance(headers, dict):
                auth = headers.get("Authorization", "")
                if auth.startswith("Bearer ") and len(auth) > 7:
                    return auth[7:]

        # Last resort: check the global config
        try:
            from perceval.runtime import RemoteConfig

            global_token = (RemoteConfig().get_token() or "").strip()
            if global_token:
                return global_token
        except Exception:
            logger.debug("RemoteConfig token lookup failed", exc_info=True)

        return None

    @staticmethod
    def get_url(rp: RemoteProcessor) -> str | None:
        """Return the RPC handler URL of a RemoteProcessor, if exposed."""
        handler = rp.get_rpc_handler()
        return handler.url if hasattr(handler, "url") else None

    # ------------------------------------------------------------------
    # Processor creation and configuration
    # ------------------------------------------------------------------

    @staticmethod
    def clone_remote_processor(
        rp: RemoteProcessor, token: str | None
    ) -> RemoteProcessor:
        """Create a sibling RemoteProcessor with its own RPC handler.

        Forwards the provided token so that inline-token RemoteProcessors
        are cloned correctly.
        """
        return RemoteProcessor(
            name=rp.name,
            token=token,
            url=PercevalAdapter.get_url(rp),
            proxies=rp.proxies,
        )

    @staticmethod
    def build_from_session(session: ISession) -> RemoteProcessor:
        """Build a fresh RemoteProcessor from a Perceval session."""
        return session.build_remote_processor()

    @staticmethod
    def get_backend_capabilities(processor: AProcessor) -> tuple[str, tuple[str, ...]]:
        """Return the backend platform name and available command snapshot."""
        return processor.name, tuple(processor.available_commands)

    @staticmethod
    def configure_processor(
        processor: AProcessor,
        circuit: pcvl.ACircuit,
        input_state: Any,
    ) -> None:
        """Set the circuit and, when provided, the input state and photon filter.

        Parameters
        ----------
        processor : AProcessor
            Processor (local or remote) to configure.
        circuit : pcvl.ACircuit
            Circuit to install.
        input_state : Any
            Sequence of photon counts per mode, or falsy to skip input setup.
            When set, ``min_detected_photons_filter`` is set to the total
            photon count.
        """
        processor.set_circuit(circuit)
        PercevalAdapter.set_input(processor, input_state)

    @staticmethod
    def set_input(processor: AProcessor, input_state: Any) -> None:
        """Set the input state and matching photon filter, if provided.

        Split out from :meth:`configure_processor` because the local
        execution path must restore experiment metadata between installing
        the circuit and setting the input.
        """
        if input_state:
            state = pcvl.BasicState(input_state)
            processor.with_input(state)
            n_photons = sum(input_state)
            processor.min_detected_photons_filter(n_photons)

    @staticmethod
    def copy_circuit(circuit: pcvl.ACircuit) -> pcvl.ACircuit:
        """Return an independent copy of a circuit for one execution."""
        return circuit.copy()

    @staticmethod
    def estimate_required_shots(
        rp: RemoteProcessor, desired_samples: int, param_values: dict[str, float]
    ) -> int | None:
        """Ask the remote platform estimator for the required shot count."""
        return rp.estimate_required_shots(desired_samples, param_values=param_values)

    # ------------------------------------------------------------------
    # Samplers
    # ------------------------------------------------------------------

    @staticmethod
    def create_sampler(
        processor: AProcessor,
        max_shots_per_call: int,
        iterations: list[dict[str, float]],
    ) -> Sampler:
        """Create a Sampler on ``processor`` loaded with the given iterations."""
        sampler = Sampler(processor, max_shots_per_call=max_shots_per_call)
        sampler.clear_iterations()
        for params in iterations:
            sampler.add_iteration(circuit_params=params)
        return sampler

    @staticmethod
    def submit_async(
        sampler: Sampler,
        command: str,
        name: str | None = None,
        max_samples: int | None = None,
    ) -> RemoteJob:
        """Submit a sampler command asynchronously and return the job handle.

        Parameters
        ----------
        sampler : Sampler
            Sampler prepared with circuit and iterations.
        command : str
            Sampler command to dispatch: ``"probs"``, ``"sample_count"``,
            or ``"samples"``.
        name : str | None
            Remote job name to assign before submission, if any.
        max_samples : int | None
            Shots to request. ``None`` submits without a shot argument
            (exact probabilities).
        """
        job = getattr(sampler, command)
        if name:
            job.name = name
        PercevalAdapter.ensure_serializable_sampler_iterator(job, sampler)
        if max_samples is None:
            return job.execute_async()
        return job.execute_async(max_samples=max_samples)

    @staticmethod
    def execute_sync(
        sampler: Sampler,
        command: str,
        max_samples: int | None = None,
    ) -> Any:
        """Execute a sampler command synchronously and return the raw results."""
        job = getattr(sampler, command)
        if max_samples is None:
            return job.execute_sync()
        return job.execute_sync(max_samples=max_samples)

    @staticmethod
    def ensure_serializable_sampler_iterator(job: RemoteJob, sampler: Sampler) -> None:
        """Replace Perceval 1.2 iterator objects with JSON-serializable data.

        Perceval 1.1 stores sampler iterations as a plain list. Perceval 1.2
        stores them in a ``ParameterIterator`` object, but the Scaleway session
        handler still serializes ``payload["payload"]`` with ``json.dumps``.
        Until Perceval exposes a public serializer for that object, Merlin
        normalizes the remote-job payload back to the list shape accepted by
        the cloud side.
        """
        iterator = getattr(sampler, "_iterator", None)
        iterations = getattr(iterator, "iterations", None)
        if not iterations:
            return

        request_data = getattr(job, "_request_data", None)
        if not isinstance(request_data, dict):
            return

        payload = request_data.get("payload")
        if isinstance(payload, dict) and payload.get("iterator") is iterator:
            payload["iterator"] = list(iterations)

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    @staticmethod
    def job_snapshot(job: RemoteJob) -> JobStatusSnapshot:
        """Read a job's status fields into a Merlin-normalized snapshot."""
        status = getattr(job, "status", None)
        return JobStatusSnapshot(
            job_id=getattr(job, "id", None) or getattr(job, "job_id", None),
            state=getattr(status, "state", None) if status else None,
            progress=getattr(status, "progress", None) if status else None,
            stop_message=getattr(status, "stop_message", None) if status else None,
            is_complete=bool(getattr(job, "is_complete", False)),
            is_failed=bool(getattr(job, "is_failed", False)),
        )

    @staticmethod
    def get_results(job: RemoteJob) -> Any:
        """Retrieve a job's raw results, propagating Perceval errors."""
        return job.get_results()

    @staticmethod
    def cancel_job(job: RemoteJob) -> None:
        """Request best-effort cancellation of a job, swallowing errors."""
        cancel = getattr(job, "cancel", None)
        if callable(cancel):
            with suppress(Exception):
                cancel()

    # ------------------------------------------------------------------
    # Local processors
    # ------------------------------------------------------------------

    @staticmethod
    def rebuild_local_processor(processor: AProcessor) -> AProcessor:
        """Create an isolated local Perceval processor for one execution.

        The returned processor carries a :class:`LocalExperimentSnapshot`
        under :data:`LOCAL_EXPERIMENT_SNAPSHOT_ATTR` so the caller can restore
        experiment metadata after installing the execution circuit.

        Raises
        ------
        TypeError
            If the configured local processor cannot be reconstructed safely.
        """
        experiment = getattr(processor, "experiment", None)
        backend_object = getattr(processor, "backend", None)
        experiment_copy = getattr(experiment, "copy", None)
        if (
            experiment is None
            or backend_object is None
            or not callable(experiment_copy)
        ):
            raise TypeError(
                "Local execution requires a Perceval processor with copyable "
                "experiment state and a reconstructable local backend."
            )

        backend_name = getattr(backend_object, "name", None)
        backend: str | object
        if isinstance(backend_name, str):
            backend = backend_name
        else:
            try:
                backend = type(backend_object)()
            except Exception as exc:
                raise TypeError(
                    "Local processor backend cannot be reconstructed safely."
                ) from exc

        experiment_snapshot = PercevalAdapter.snapshot_experiment(experiment)
        copied_experiment = experiment_copy()
        copied_experiment.clear_input_and_circuit()

        fresh_processor = Processor(backend, copied_experiment)
        setattr(fresh_processor, LOCAL_EXPERIMENT_SNAPSHOT_ATTR, experiment_snapshot)
        return fresh_processor

    @staticmethod
    def snapshot_experiment(experiment: Any) -> LocalExperimentSnapshot:
        """Copy non-circuit local experiment metadata before Perceval clears it.

        Parameters
        ----------
        experiment : Any
            Perceval experiment owned by the caller's local processor.

        Returns
        -------
        LocalExperimentSnapshot
            Deep-copied metadata that is independent from the caller's
            processor.
        """
        in_ports = tuple(
            (port, tuple(modes))
            for port, modes in copy.deepcopy(experiment._in_ports).items()
        )
        out_ports = tuple(
            (port, tuple(modes))
            for port, modes in copy.deepcopy(experiment._out_ports).items()
        )
        return LocalExperimentSnapshot(
            circuit_size=int(experiment.circuit_size),
            in_ports=in_ports,
            out_ports=out_ports,
            detectors=tuple(copy.deepcopy(experiment.detectors)),
            detectors_injected=tuple(copy.deepcopy(experiment.detectors_injected)),
            in_mode_type=tuple(copy.deepcopy(experiment._in_mode_type)),
            out_mode_type=tuple(copy.deepcopy(experiment._out_mode_type)),
            anon_herald_num=int(experiment._anon_herald_num),
            postselection=copy.copy(experiment.post_select_fn),
        )

    @staticmethod
    def restore_experiment(experiment: Any, snapshot: LocalExperimentSnapshot) -> None:
        """Restore local experiment metadata after the execution circuit is set.

        Parameters
        ----------
        experiment : Any
            Perceval experiment owned by the fresh local execution processor.
        snapshot : LocalExperimentSnapshot
            Metadata copied from the caller's local processor.

        Raises
        ------
        ValueError
            If mode-indexed metadata cannot be applied to the execution
            circuit because the circuit sizes differ.
        """
        if snapshot.has_mode_metadata:
            circuit_size = int(experiment.circuit_size)
            if circuit_size != snapshot.circuit_size:
                raise ValueError(
                    "Local processor experiment metadata is tied to circuit size "
                    f"{snapshot.circuit_size}, but the execution circuit has size "
                    f"{circuit_size}."
                )
            experiment._in_ports = {
                port: list(modes) for port, modes in snapshot.in_ports
            }
            experiment._out_ports = {
                port: list(modes) for port, modes in snapshot.out_ports
            }
            experiment._detectors = list(snapshot.detectors)
            experiment.detectors_injected = list(snapshot.detectors_injected)
            experiment._in_mode_type = list(snapshot.in_mode_type)
            experiment._out_mode_type = list(snapshot.out_mode_type)
            experiment._anon_herald_num = snapshot.anon_herald_num

        experiment._postselect = copy.copy(snapshot.postselection)
        experiment._circuit_changed()
