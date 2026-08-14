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


class TokenExtractionError(ValueError):
    """Signals that no auth token could be resolved for a RemoteProcessor.

    Raised by *callers* of :meth:`PercevalAdapter.extract_token` when it returns
    ``None`` (see ``MerlinProcessor.__init__``). ``extract_token`` itself returns
    ``None`` rather than raising, so its multi-strategy fallback (handler token,
    Bearer header, global ``RemoteConfig``) can run to completion before the
    caller decides the token is genuinely unresolvable.

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

        Parameters
        ----------
        rp : perceval.runtime.RemoteProcessor
            Remote processor to probe for authentication material.

        Returns
        -------
        str | None
            The resolved token, or ``None`` if every strategy fails.

        Notes
        -----
        ``get_rpc_handler()`` is wrapped defensively here — unlike in
        :meth:`get_url` — precisely because this method has a downstream
        fallback: if the handler is unavailable it can still resolve a token from
        the global ``RemoteConfig``. Swallowing the handler error is therefore
        part of the control flow, not error hiding; a genuinely unresolvable
        token surfaces as ``None`` (which the caller turns into a
        :class:`TokenExtractionError`).
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
        """Return the RPC handler URL of a RemoteProcessor, if exposed.

        Parameters
        ----------
        rp : perceval.runtime.RemoteProcessor
            Remote processor whose RPC handler is inspected.

        Returns
        -------
        str | None
            The handler URL, or ``None`` when the handler has no ``url``
            attribute.

        Notes
        -----
        ``get_rpc_handler()`` is intentionally left unguarded here — unlike in
        :meth:`extract_token` — because this method has no fallback. A broken
        handler should fail fast at the real fault rather than yield ``url=None``
        and a silently misconfigured clone downstream in
        :meth:`clone_remote_processor`.
        """
        # Intentionally unguarded (see Notes): no fallback, so fail fast.
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

        Parameters
        ----------
        rp : perceval.runtime.RemoteProcessor
            Processor whose platform name, URL, and proxies are copied.
        token : str | None
            Authentication token forwarded to the clone.

        Returns
        -------
        perceval.runtime.RemoteProcessor
            Independent processor targeting the same platform.
        """
        return RemoteProcessor(
            name=rp.name,
            token=token,
            url=PercevalAdapter.get_url(rp),
            proxies=rp.proxies,
        )

    @staticmethod
    def build_from_session(session: ISession) -> RemoteProcessor:
        """Build a fresh RemoteProcessor from a Perceval session.

        Parameters
        ----------
        session : perceval.runtime.session.ISession
            Provider session (e.g. Scaleway) able to build processors.

        Returns
        -------
        perceval.runtime.RemoteProcessor
            Independent processor with its own handler state.
        """
        return session.build_remote_processor()

    @staticmethod
    def get_backend_capabilities(processor: AProcessor) -> tuple[str, tuple[str, ...]]:
        """Return the backend platform name and available command snapshot.

        Parameters
        ----------
        processor : perceval.runtime.AProcessor
            Local or remote processor to inspect.

        Returns
        -------
        tuple[str, tuple[str, ...]]
            Platform name and immutable snapshot of supported commands.
        """
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
        PercevalAdapter.set_circuit(processor, circuit)
        PercevalAdapter.set_input(processor, input_state)

    @staticmethod
    def set_circuit(processor: AProcessor, circuit: pcvl.ACircuit) -> None:
        """Install a circuit on a processor without touching its input state.

        Split out from :meth:`configure_processor` so the local execution path
        can install the circuit, restore experiment metadata, and only then set
        the input — instead of passing a ``None`` input-state sentinel.

        Parameters
        ----------
        processor : perceval.runtime.AProcessor
            Processor (local or remote) to configure.
        circuit : pcvl.ACircuit
            Circuit to install.
        """
        processor.set_circuit(circuit)

    @staticmethod
    def set_input(processor: AProcessor, input_state: Any) -> None:
        """Set the input state and matching photon filter, if provided.

        Split out from :meth:`configure_processor` because the local
        execution path must restore experiment metadata between installing
        the circuit and setting the input.

        Parameters
        ----------
        processor : perceval.runtime.AProcessor
            Processor to receive the input state.
        input_state : Any
            Sequence of photon counts per mode, or falsy to skip input setup.
        """
        if input_state:
            state = pcvl.BasicState(input_state)
            processor.with_input(state)
            n_photons = sum(input_state)
            processor.min_detected_photons_filter(n_photons)

    @staticmethod
    def copy_circuit(circuit: pcvl.ACircuit) -> pcvl.ACircuit:
        """Return an independent copy of a circuit for one execution.

        Parameters
        ----------
        circuit : pcvl.ACircuit
            Circuit exported by the quantum layer.

        Returns
        -------
        pcvl.ACircuit
            Independent circuit object used by a single backend execution.
        """
        return circuit.copy()

    @staticmethod
    def estimate_required_shots(
        rp: RemoteProcessor, desired_samples: int, param_values: dict[str, float]
    ) -> int | None:
        """Ask the remote platform estimator for the required shot count.

        Parameters
        ----------
        rp : perceval.runtime.RemoteProcessor
            Configured remote processor exposing the platform estimator.
        desired_samples : int
            Target number of usable samples.
        param_values : dict[str, float]
            Circuit parameter values for the input row being estimated.

        Returns
        -------
        int | None
            Estimated shots, or ``None`` when the platform gives no answer.
        """
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
        """Create a Sampler on ``processor`` loaded with the given iterations.

        Parameters
        ----------
        processor : perceval.runtime.AProcessor
            Configured processor (circuit and input already set).
        max_shots_per_call : int
            Shot cap forwarded to the Perceval sampler.
        iterations : list[dict[str, float]]
            One circuit-parameter mapping per batch row.

        Returns
        -------
        perceval.algorithm.Sampler
            Sampler ready for command dispatch.
        """
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

        Returns
        -------
        perceval.runtime.RemoteJob
            Handle of the submitted asynchronous job.
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
        """Execute a sampler command synchronously and return the raw results.

        Parameters
        ----------
        sampler : perceval.algorithm.Sampler
            Sampler prepared with circuit and iterations.
        command : str
            Sampler command to dispatch: ``"probs"``, ``"sample_count"``,
            or ``"samples"``.
        max_samples : int | None
            Shots to request. ``None`` executes without a shot argument
            (exact probabilities). Default value is None.

        Returns
        -------
        Any
            Raw Perceval results object for the executed command.
        """
        job = getattr(sampler, command)
        if max_samples is None:
            return job.execute_sync()
        return job.execute_sync(max_samples=max_samples)

    @staticmethod
    def ensure_serializable_sampler_iterator(job: RemoteJob, sampler: Sampler) -> None:
        """Replace Perceval 1.2 iterator objects with JSON-serializable data.

        Parameters
        ----------
        job : perceval.runtime.RemoteJob
            Prepared job whose private request payload may hold an iterator.
        sampler : perceval.algorithm.Sampler
            Sampler used to prepare the job.

        Notes
        -----
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
        """Read a job's status fields into a Merlin-normalized snapshot.

        Parameters
        ----------
        job : perceval.runtime.RemoteJob
            Job to inspect. Missing attributes map to ``None``/``False``.

        Returns
        -------
        JobStatusSnapshot
            Immutable view of the job's id, state, and completion flags.
        """
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
        """Retrieve a job's raw results, propagating Perceval errors.

        Parameters
        ----------
        job : perceval.runtime.RemoteJob
            Completed job to read.

        Returns
        -------
        Any
            Raw Perceval results object.

        Raises
        ------
        RuntimeError
            Propagated unchanged from Perceval (e.g. results not yet
            available, cancel requested); the polling loop interprets it.
        """
        return job.get_results()

    @staticmethod
    def cancel_job(job: RemoteJob) -> None:
        """Request best-effort cancellation of a job, swallowing errors.

        Parameters
        ----------
        job : perceval.runtime.RemoteJob
            Job to cancel. Objects without a callable ``cancel`` are ignored;
            cancellation errors are suppressed by design (best-effort path).
        """
        cancel = getattr(job, "cancel", None)
        if callable(cancel):
            with suppress(Exception):
                cancel()

    # ------------------------------------------------------------------
    # Local processors
    # ------------------------------------------------------------------

    @staticmethod
    def rebuild_local_processor(
        processor: AProcessor,
    ) -> tuple[AProcessor, LocalExperimentSnapshot]:
        """Create an isolated local Perceval processor for one execution.

        Returns the fresh processor together with the
        :class:`LocalExperimentSnapshot` the caller must apply (via
        :meth:`restore_experiment`) after installing the execution circuit. The
        snapshot is an explicit return value rather than hidden state on the
        processor, so a caller cannot silently forget to restore it.

        Parameters
        ----------
        processor : perceval.runtime.AProcessor
            Local processor whose experiment and backend are copied.

        Returns
        -------
        tuple[perceval.runtime.AProcessor, LocalExperimentSnapshot]
            A fresh local processor (copied non-circuit experiment state and a
            fresh backend instance) and the experiment snapshot to restore once
            the execution circuit is installed.

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
        return fresh_processor, experiment_snapshot

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
