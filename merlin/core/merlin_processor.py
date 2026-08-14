import logging
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from perceval.runtime import AProcessor, RemoteJob, RemoteProcessor
from perceval.runtime.session import ISession
from torch.futures import Future

from ..algorithms.module import MerlinModule
from ..utils.combinadics import Combinadics
from .execution import BatchChunker, RemoteJobRunner
from .perceval_adapter import (
    LocalExperimentSnapshot,
    PercevalAdapter,
    TokenExtractionError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendCapabilities:
    """Encapsulate Perceval backend capabilities.

    Attributes
    ----------
    name : str
        Backend platform name (e.g., "sim:slos", "perceval-qpu:scaleway").
    available_commands : tuple[str]
        Immutable snapshot of supported commands (e.g., ["probs", "sample_count"]).
    """

    name: str
    available_commands: tuple[str]


@dataclass(frozen=True)
class JobStatus:
    """Immutable snapshot of the most recently observed backend job status.

    Attributes
    ----------
    state : Any
        Backend-reported job state (e.g. ``"RUNNING"``), or ``None`` when the
        backend did not expose one.
    progress : Any
        Backend-reported progress value, or ``None``.
    message : Any
        Backend-reported stop/status message, or ``None``.
    """

    state: Any = None
    progress: Any = None
    message: Any = None


class CallState:
    """Typed per-call execution state for one :meth:`MerlinProcessor.forward_async` call.

    Replaces the anonymous mutable ``state`` dict previously threaded through
    ``forward_async()``, chunk orchestration, chunk execution, and job polling.
    All per-call runtime state is owned by this object and mutated only through
    named helpers, making the contract explicit and searchable.

    **Thread ownership**

    - ``call_id``: immutable after creation; readable from any thread.
    - ``cancel_requested``: set (never cleared) by the caller thread through
      :meth:`request_cancel`; read cooperatively by the pipeline, chunk, and
      polling threads.
    - ``current_status``: written by polling threads through
      :meth:`set_current_status`; read by :meth:`status_snapshot`.
    - ``job_ids``: appended (with deduplication) by polling threads through
      :meth:`record_job_id`. The list object is intentionally shared with
      ``future.job_ids`` so recorded ids appear on the future as they arrive.
    - Chunk counters (``chunks_total``, ``chunks_done``, ``active_chunks``):
      mutated by the chunk orchestration thread through
      :meth:`add_planned_chunks`, :meth:`mark_chunk_started`, and
      :meth:`mark_chunk_finished` under the internal lock.
    """

    def __init__(self, call_id: str) -> None:
        """Initialize an empty call state.

        Parameters
        ----------
        call_id : str
            Short identifier embedded in remote job names for traceability.
            Use :meth:`new` to generate one automatically.
        """
        self.call_id = call_id
        self.job_ids: list[str] = []
        self._lock = threading.Lock()
        self._cancel_requested = False
        self._current_status: JobStatus | None = None
        self._chunks_total = 0
        self._chunks_done = 0
        self._active_chunks = 0

    @classmethod
    def new(cls) -> "CallState":
        """Create a fresh call state with a short unique call identifier.

        Returns
        -------
        CallState
            Empty state carrying an 8-character hex ``call_id``.
        """
        return cls(call_id=uuid.uuid4().hex[:8])

    # ---- cancellation ----

    @property
    def cancel_requested(self) -> bool:
        """Whether cooperative cancellation has been requested for this call."""
        return self._cancel_requested

    def request_cancel(self) -> None:
        """Request cooperative cancellation of this call (irreversible)."""
        self._cancel_requested = True

    # ---- job ids ----

    def record_job_id(self, job_id: str) -> None:
        """Record a remote job id, deduplicating repeated observations.

        Parameters
        ----------
        job_id : str
            Identifier reported by the backend for a submitted job. Re-observing
            an already-recorded id (each polling cycle re-reads it) is a no-op.
        """
        with self._lock:
            if job_id not in self.job_ids:
                self.job_ids.append(job_id)

    # ---- backend job status ----

    @property
    def current_status(self) -> JobStatus | None:
        """Most recent backend job status, or ``None`` before first poll."""
        return self._current_status

    def set_current_status(
        self, *, state: Any = None, progress: Any = None, message: Any = None
    ) -> None:
        """Record the latest backend job status observed while polling.

        Parameters
        ----------
        state : Any
            Backend-reported job state (e.g. ``"RUNNING"``), or ``None``.
            Default value is None.
        progress : Any
            Backend-reported progress value, or ``None``. Default value is None.
        message : Any
            Backend-reported stop/status message, or ``None``. Default value
            is None.
        """
        self._current_status = JobStatus(
            state=state, progress=progress, message=message
        )

    # ---- chunk counters ----

    @property
    def chunks_total(self) -> int:
        """Total number of chunks planned so far for this call."""
        return self._chunks_total

    @property
    def chunks_done(self) -> int:
        """Number of chunks that finished (successfully or not)."""
        return self._chunks_done

    @property
    def active_chunks(self) -> int:
        """Number of chunk jobs currently in flight."""
        return self._active_chunks

    def add_planned_chunks(self, count: int) -> None:
        """Register additional chunks planned for submission.

        Parameters
        ----------
        count : int
            Number of chunks about to be submitted for one quantum leaf.
        """
        with self._lock:
            self._chunks_total += count

    def mark_chunk_started(self) -> None:
        """Mark one chunk job as submitted and in flight."""
        with self._lock:
            self._active_chunks += 1

    def mark_chunk_finished(self) -> None:
        """Mark one in-flight chunk job as finished."""
        with self._lock:
            self._active_chunks = max(0, self._active_chunks - 1)
            self._chunks_done += 1

    # ---- snapshots ----

    def status_snapshot(self, future_done: bool = False) -> dict:
        """Return the public status dict exposed through ``future.status()``.

        Parameters
        ----------
        future_done : bool
            Whether the owning future has already resolved. A resolved future
            with no recorded backend status reports state ``"COMPLETE"``.

        Returns
        -------
        dict
            ``{"state", "progress", "message", "chunks_total", "chunks_done",
            "active_chunks"}`` with the same semantics as before the CallState
            refactor.
        """
        js = self._current_status
        return {
            "state": (
                "COMPLETE"
                if future_done and js is None
                else (js.state if js else "IDLE")
            ),
            "progress": js.progress if js else 0.0,
            "message": js.message if js else None,
            "chunks_total": self._chunks_total,
            "chunks_done": self._chunks_done,
            "active_chunks": self._active_chunks,
        }


class MerlinFuture(Future):
    """Typed async handle returned by :meth:`MerlinProcessor.forward_async`.

    Extends ``torch.futures.Future[torch.Tensor]`` with the Merlin-specific
    async contract that was previously monkey-patched onto plain Future
    instances at runtime: remote-job visibility (:attr:`job_ids`), progress
    reporting (:meth:`status`), and cooperative cancellation
    (:meth:`cancel_remote`). All inherited Future behavior (``wait``,
    ``done``, ``then``, ``value``, ...) is unchanged.

    Parameters
    ----------
    call_state : CallState
        Typed per-call state backing this handle. Job ids, chunk counters,
        and backend status recorded during execution are read live from it.
    cancel_all : Callable[[], None]
        Processor-level callback cancelling all in-flight remote jobs; used
        by :meth:`cancel_remote`.
    """

    def __init__(self, call_state: CallState, cancel_all: Callable[[], None]) -> None:
        super().__init__()
        self._call_state = call_state
        self._cancel_all = cancel_all

    @property
    def job_ids(self) -> list[str]:
        """Remote job ids accumulated across chunks, in observation order.

        This is a live view of the underlying :class:`CallState` list: ids
        recorded while chunks are polling appear here immediately.
        """
        return self._call_state.job_ids

    def status(self) -> dict:
        """Return the current progress and state of this call.

        Returns
        -------
        dict
            ``{"state", "progress", "message", "chunks_total", "chunks_done",
            "active_chunks"}``. ``state`` is ``"IDLE"`` before the first
            backend poll, the backend-reported state while polling, and
            ``"COMPLETE"`` once the future resolves without a recorded
            backend status.
        """
        return self._call_state.status_snapshot(future_done=self.done())

    def cancel_remote(self) -> None:
        """Cooperatively cancel this call and its in-flight remote jobs.

        Requests cancellation on the per-call state (observed by chunk and
        polling threads), cancels all active remote jobs best-effort, and
        resolves this future with ``concurrent.futures.CancelledError`` if
        it is not already done. Awaiting the future afterwards raises that
        error.
        """
        from concurrent.futures import CancelledError

        self._call_state.request_cancel()
        self._cancel_all()
        if not self.done():
            self.set_exception(CancelledError("Remote call was cancelled"))


_ALLOWED_STATE_TYPES = (
    pcvl.StateVector,
    pcvl.FockState,
    pcvl.NoisyFockState,
    pcvl.BasicState,
    pcvl.LogicalState,
)


def check_sequence(input: Any) -> Sequence[Any] | None:
    """
    Check whether an object can be treated as a sequence.

    Parameters
    ----------
    input : Any
        Object to validate.

    Returns
    -------
    Sequence | None
        The original object if it is an instance of
        ``collections.abc.Sequence``.

        Otherwise, if the object is iterable, a tuple containing its
        elements.

        Returns None if the object is not iterable.

    Notes
    -----
    This helper accepts objects that are not instances of
    ``collections.abc.Sequence`` but can be iterated over, such as
    NumPy arrays and PyTorch tensors. Such objects are converted to
    tuples before being returned.

    Examples
    --------
    >>> check_sequence([1, 2, 3])
    [1, 2, 3]

    >>> check_sequence((1, 2, 3))
    (1, 2, 3)

    >>> check_sequence(np.array([1, 2, 3]))
    (1, 2, 3)

    >>> check_sequence(42)
    None
    """

    if isinstance(input, Sequence) and not isinstance(input, (str, bytes)):
        return input
    try:
        return tuple(input)
    except TypeError:
        return None


class ValidatedLayerConfig:
    """
    Validate and normalize the configuration dictionary returned by
    ``export_config()``.

    Parameters
    ----------
    config_to_verify : dict
        Configuration dictionary containing the layer definition.

    Attributes
    ----------
    circuit : pcvl.ACircuit
        Perceval circuit associated with the layer.

    input_state : Sequence[Integral] | pcvl.BasicState | pcvl.StateVector | pcvl.BSDistribution | pcvl.SVDistribution | None
        Input state for the circuit. May be ``None``, a sequence of integers,
        or one of the supported Perceval state objects. Sequence-like inputs
        are normalized through ``check_sequence()``.

    input_param_order : Sequence[str] | None
        Ordered names of the circuit parameters expected by the layer.
        Sequence-like inputs are normalized through ``check_sequence()``.

    Raises
    ------
    KeyError
        If one of the required configuration keys is missing:

        - ``"circuit"``
        - ``"input_state"``
        - ``"input_param_order"``

    ValueError
        If:

        - ``circuit`` is not a ``pcvl.ACircuit``.
        - ``input_state`` is neither ``None``, a supported Perceval state
          object, nor a sequence.
        - ``input_state`` is a sequence containing non-integer elements.
        - ``input_param_order`` is neither ``None`` nor a sequence.
        - ``input_param_order`` contains non-string elements.

    Notes
    -----
    Sequence validation relies on ``check_sequence()``. Accepted sequence
    implementations may include Python sequences as well as array-like objects
    supported by that helper.
    """

    def __init__(self, config_to_verify: dict):
        """
        Validate and normalize a layer configuration dictionary.

        Parameters
        ----------
        config_to_verify : dict
            Configuration dictionary containing the following required keys:

            - ``"circuit"``: a ``pcvl.ACircuit`` instance.
            - ``"input_state"``: ``None``, a sequence of integers, or a supported
            Perceval state object.
            - ``"input_param_order"``: ``None`` or a sequence of strings.

        Raises
        ------
        KeyError
            If one of the required keys is missing from ``config_to_verify``.

        ValueError
            If:

            - ``config_to_verify["circuit"]`` is not a ``pcvl.ACircuit``.
            - ``config_to_verify["input_state"]`` is neither ``None``, a valid
            Perceval state object, nor a sequence.
            - ``config_to_verify["input_state"]`` contains non-integer elements.
            - ``config_to_verify["input_param_order"]`` is neither ``None`` nor a
            sequence.
            - ``config_to_verify["input_param_order"]`` contains non-string
            elements.

        Notes
        -----
        Sequence-like inputs are normalized using ``check_sequence()``. Objects
        that are iterable but not instances of ``collections.abc.Sequence``
        (e.g. NumPy arrays or PyTorch tensors) may therefore be accepted and
        converted to tuples.
        """
        # circuit
        try:
            self.circuit: pcvl.ACircuit = config_to_verify["circuit"]
        except KeyError:
            raise KeyError(
                "There must be a key 'circuit' in the configs dictionary that is associated with a perceval.ACircuit."
            )
        if not isinstance(self.circuit, pcvl.ACircuit):
            raise ValueError(
                f"The 'circuit' key of the config dictionary must be a perceval.ACircuit, got {type(self.circuit)}."
            )

        # input_state
        try:
            self.input_state: (
                Sequence[Integral]
                | pcvl.BasicState
                | pcvl.StateVector
                | pcvl.BSDistribution
                | pcvl.SVDistribution
                | None
            ) = config_to_verify["input_state"]
        except KeyError:
            raise KeyError(
                "There must be a key 'input_state' in the configs dictionary that is associated with a Sequence[Integral], a Perceval State object or None."
            )
        if self.input_state is not None:
            if isinstance(self.input_state, _ALLOWED_STATE_TYPES):
                pass

            else:
                input_state_sequence: Sequence[Integral] | None = check_sequence(
                    self.input_state
                )
                if input_state_sequence is None:
                    raise ValueError(
                        "'input_state' must be None, a sequence of integers, "
                        "or an Perceval state object "
                        f"(got {type(self.input_state).__name__})."
                    )
                self.input_state = input_state_sequence
                bad_types = {
                    type(x).__name__
                    for x in self.input_state
                    if not isinstance(x, Integral)
                }

                if bad_types:
                    raise ValueError(
                        f"'input_state' must contain only integers when it is a sequence. "
                        f"Got sequence type {type(self.input_state).__name__} "
                        f"with non-integer element types: {sorted(bad_types)}."
                    )

        # input_param_order
        try:
            self.input_param_order: Sequence[str] | None = config_to_verify[
                "input_param_order"
            ]
        except KeyError:
            raise KeyError(
                "There must be a key 'input_param_order' in the configs dictionary that is associated with a Sequence[str] or None."
            )
        if self.input_param_order is not None:
            input_param_order_sequence: Sequence[str] | None = check_sequence(
                self.input_param_order
            )
            if input_param_order_sequence is None:
                raise ValueError(
                    f"'input_param_order' must be a sequence of strings or None, got {type(self.input_param_order).__name__}."
                )
            self.input_param_order = input_param_order_sequence
            bad_types = {
                type(x).__name__
                for x in self.input_param_order
                if not isinstance(x, str)
            }

            if bad_types:
                raise ValueError(
                    f"'input_param_order' must contain only strings. "
                    f"Got sequence type {type(self.input_param_order).__name__} "
                    f"with non-integer element types: {sorted(bad_types)}."
                )


@runtime_checkable
class SupportsExportConfig(Protocol):
    """
    Protocol for objects that can export their configuration as a dictionary.

    Implementations must provide an ``export_config()`` method returning a
    dictionary containing the information required to reconstruct or validate
    the object's configuration.

    Notes
    -----
    This protocol is marked as ``@runtime_checkable``, allowing runtime checks
    with ``isinstance()`` and ``issubclass()``.

    Examples
    --------
    >>> isinstance(obj, SupportsExportConfig)
    True
    """

    def export_config(self) -> dict:
        """
        Export the object's configuration.

        Returns
        -------
        dict
            Dictionary containing the configuration of the object.
        """
        ...


class MerlinProcessor:
    """RPC-style processor for quantum execution.

    Offloads :class:`~merlin.algorithms.module.MerlinModule` leaves (e.g.
    QuantumLayer) to a Perceval backend while keeping classical layers local.
    Automatically handles batching, remote chunking, concurrency control,
    timeouts, and cooperative cancellation.

    **Key Features**

    - Torch-friendly asynchronous execution via ``Future[torch.Tensor]``.
    - Perceval backend offload of quantum leaves only; non-quantum leaves run locally.
    - Remote batch **chunking** (``microbatch_size``) and **parallel** submission
      per leaf (``chunk_concurrency``).
    - Cancellation support, both per future and globally.
    - Global timeouts that cancel in-flight remote jobs and check local jobs
      before and after synchronous execution.
    - Isolated backend object per execution: local processors are rebuilt from
      copied non-circuit experiment state, and remote processors are cloned or
      built from a session.
    - Descriptive cloud job names (<= 50 chars) for remote chunk traceability.

    **Execution Model**

    The processor automatically selects the execution strategy based on backend
    capabilities:

    - If the backend exposes ``"probs"`` command and ``nsample`` is None or 0: computes **exact probabilities**.
    - Otherwise: uses **sampling** with ``"sample_count"`` or ``"samples"`` command.
      Samples per input = ``nsample`` if provided, else
      ``min(DEFAULT_SHOTS_PER_CALL, max_shots_per_call)``.

    Backend capabilities are extracted once at initialization and stored in
    :attr:`backend_capabilities`.

    Parameters
    ----------
    remote_processor : RemoteProcessor | None
        Deprecated Perceval remote processor entry point. Pass the same
        RemoteProcessor through ``processor=`` instead. Cloned per chunk for
        thread safety.
    session : ISession | None
        Perceval session (e.g. Scaleway) used to build remote processors.
        ``session.build_remote_processor()`` is called per chunk. Exactly one of
        ``processor``, ``remote_processor``, or ``session`` must be provided.
    microbatch_size : int
        Maximum number of inputs submitted in a single remote backend chunk.
        Ignored for local processors.
        Default: 32.
    timeout : float
        Default wall-time limit in seconds for backend calls. Can be overridden
        per call via ``timeout=...``. Default: 3600.0.
    max_shots_per_call : int | None
        Hard cap on shots per backend sampler call (only used when sampling,
        not with exact probabilities). If ``nsample`` exceeds this cap,
        ``nsample`` is clamped to this value with a warning. If ``None``,
        defaults are used internally. Default: None.
    chunk_concurrency : int
        Maximum number of concurrent remote chunk submissions per quantum layer.
        Ignored for local processors.
        Default: 1 (serial).
    token : str | None
        Optional authentication token forwarded to cloned remote processors.
        If not provided, extracted from the processor's RPC handler. Ignored
        for local processors.
    processor : AProcessor | None
        Keyword-only Perceval processor entry point. Local, non-remote processors
        are stored for the local backend path, rebuilt for each local execution,
        and do not require remote token extraction. RemoteProcessor instances
        passed here are normalized to the remote processor path.

    Attributes
    ----------
    backend_kind : str
        Active backend route: ``"local_processor"``, ``"remote_processor"``,
        or ``"session"``.
    processor : AProcessor | None
        Local, non-remote Perceval processor used by the local backend route.
        ``None`` for remote backend routes.
    backend_capabilities : BackendCapabilities
        Backend name and command snapshot extracted at initialization.
    """

    DEFAULT_MAX_SHOTS: int = 100_000
    _MAX_CHUNK_RETRIES: int = 3
    _MAX_ESTIMATOR_RETRIES: int = 3
    DEFAULT_SHOTS_PER_CALL: int = 10_000
    _JOB_NAME_MAX: int = 50

    def __init__(
        self,
        remote_processor: RemoteProcessor | None = None,
        session: ISession | None = None,
        microbatch_size: int = 32,
        timeout: float = 3600.0,
        max_shots_per_call: int | None = None,
        chunk_concurrency: int = 1,
        token: str | None = None,
        *,
        processor: AProcessor | None = None,
    ):
        """Initialize the Merlin processor backend.

        Backend capabilities (available commands) are extracted once at initialization
        and stored in :attr:`backend_capabilities` for the lifetime of the processor.
        These determine whether execution uses exact probabilities or sampling.

        **Backend Paths**

        The processor supports three backend entry points:

        1. **AProcessor path** (``processor`` provided):
            Primary Perceval entry point. Local processors are stored as the
            local backend, rebuilt for each local execution, and used without
            remote token extraction. RemoteProcessor instances are normalized
            to the RemoteProcessor path.
        2. **RemoteProcessor path** (``remote_processor`` provided):
            Direct RemoteProcessor backend. The RP is stored and cloned per chunk.
        3. **ISession path** (``session`` provided):
            Preferred for Scaleway and future session-based providers.
            ``session.build_remote_processor()`` is called per chunk.
        All paths expose backend capabilities via :attr:`backend_capabilities`,
        which drive the probability vs sampling decision. The active route is
        stored in :attr:`backend_kind`.

        Parameters
        ----------
        remote_processor : RemoteProcessor | None
            Deprecated Perceval ``RemoteProcessor`` entry point (simulator or
            QPU-backed). Pass the same object through ``processor=`` instead.
            Exactly one of ``processor``, ``remote_processor``, or ``session``
            must be provided. Default: None.
        session : ISession | None
            Perceval session (e.g. ``pcvl.providers.scaleway.Session``).
            Exactly one of ``processor``, ``remote_processor``, or ``session``
            must be provided. Default: None.
        microbatch_size : int
            Maximum number of inputs submitted in a single remote backend
            chunk. Ignored for local processors. Default: 32.
        timeout : float
            Default wall-time limit (seconds) for backend calls. Per-call
            override via ``timeout=...`` on API methods. Default: 3600.0.
        max_shots_per_call : int | None
            Hard cap on shots per backend sampler call (only applies when
            sampling; ignored for exact probabilities). If ``nsample`` exceeds
            this value in :meth:`forward` or :meth:`forward_async`, ``nsample``
            is clamped with a warning. If it is ``None``, it is set to
            100 000. Default: None.
        chunk_concurrency : int
            Max number of remote chunk jobs in flight per quantum leaf during
            a single call. Ignored for local processors. Default: 1 (serial).
        token : str | None
            Optional authentication token forwarded to cloned remote processors.
            If not provided, extracted from the processor's RPC handler.
            Ignored for local processors. Default: None.
        processor : AProcessor | None
            Keyword-only Perceval ``AProcessor``. Local processors use the
            local backend and do not require remote token extraction.
            RemoteProcessor instances passed here use the existing remote
            backend. Exactly one of ``processor``, ``remote_processor``, or
            ``session`` must be provided. Default: None.

        Raises
        ------
        TypeError
            If exactly one backend is not provided, if a backend type is
            invalid, or if ``processor`` is a remote AProcessor subclass other
            than RemoteProcessor.
        ValueError
            If no token can be resolved from the RemoteProcessor path or
            explicitly provided.

        Warns
        -----
        DeprecationWarning
            If ``remote_processor`` is provided instead of ``processor``.
        """
        n_backends = sum(
            backend is not None for backend in (processor, remote_processor, session)
        )
        if n_backends == 0:
            raise TypeError(
                "Exactly one of 'processor', 'remote_processor', or 'session' "
                "must be provided."
            )
        if n_backends > 1:
            raise TypeError(
                "'processor', 'remote_processor', and 'session' are mutually "
                "exclusive; provide exactly one."
            )
        if remote_processor is not None:
            warnings.warn(
                "The 'remote_processor' argument is deprecated and will be "
                "removed in a future release. Pass the RemoteProcessor through "
                "'processor=' instead; the processor argument will become the "
                "source of truth.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.processor: AProcessor | None = None
        self.session: ISession | None = None
        self.remote_processor: RemoteProcessor | None = None
        self.backend_kind: str
        self._token: str | None = token
        capability_processor: AProcessor | None = None

        if processor is not None:
            if not isinstance(processor, AProcessor):
                raise TypeError(f"Expected AProcessor, got {type(processor)}")
            if isinstance(processor, RemoteProcessor):
                remote_processor = processor
            elif processor.is_remote:
                raise TypeError(
                    "Unsupported remote AProcessor subclass "
                    f"{type(processor)}; pass a RemoteProcessor or ISession instead."
                )
            else:
                self.processor = processor
                self.backend_kind = "local_processor"
                capability_processor = processor

        if self.processor is None:
            if session is not None:
                # ── ISession path ──
                if not isinstance(session, ISession):
                    raise TypeError(f"Expected ISession, got {type(session)}")
                self.session = session
                self.backend_kind = "session"

                # Build ONE initial processor to extract metadata (backend name, available commands).
                # Fresh processors will be created per chunk via _create_fresh_rp().
                _init_rp = PercevalAdapter.build_from_session(self.session)
                remote_processor = _init_rp
            else:
                self.backend_kind = "remote_processor"

            assert remote_processor is not None  # for type checker
            if not isinstance(remote_processor, RemoteProcessor):
                raise TypeError(
                    f"Expected RemoteProcessor, got {type(remote_processor)}"
                )

            # Store RemoteProcessor only for the non-session path.
            # Session path will call _create_fresh_rp() to build per-chunk processors.
            if self.session is None:
                self.remote_processor = remote_processor
            capability_processor = remote_processor

        assert capability_processor is not None

        # Extract backend capabilities (name and available commands)
        backend_name, available_cmds = PercevalAdapter.get_backend_capabilities(
            capability_processor
        )
        self.backend_capabilities = BackendCapabilities(
            name=backend_name,
            available_commands=available_cmds,
        )

        # Check if commands list is empty and warn
        if not self.backend_capabilities.available_commands:
            backend_label = (
                "Local processor"
                if self.backend_kind == "local_processor"
                else "Remote processor"
            )
            warnings.warn(
                f"{backend_label} has no available commands. "
                "Ensure the platform is properly configured.",
                stacklevel=2,
            )

        if self.backend_kind == "remote_processor":
            # Auto-extract the token from the RP's handler when not
            # explicitly provided, so cloned RPs inherit it.
            if self._token is None:
                self._token = self._extract_rp_token(remote_processor)

            if self._token is None:
                raise TokenExtractionError(
                    "Could not extract auth token from RemoteProcessor. "
                    "Either pass token= to MerlinProcessor or call "
                    "RemoteConfig.set_token() before constructing the "
                    "RemoteProcessor."
                )

        self.microbatch_size = microbatch_size
        self.default_timeout = float(timeout)
        self.max_shots_per_call = (
            self.DEFAULT_MAX_SHOTS
            if max_shots_per_call is None
            else int(max_shots_per_call)
        )

        # Concurrency of chunk submissions inside a single quantum leaf
        self.chunk_concurrency = max(1, int(chunk_concurrency))

        # Caches & global tracking
        self._layer_cache: dict[uuid.UUID, dict[str, Any]] = {}
        self._job_history: list[RemoteJob] = []

        # Lifecycle/cancellation
        self._lock = threading.Lock()
        self._active_jobs: set[RemoteJob] = set()
        self._closed = False

    # ─── Backward compatibility properties ───

    @property
    def backend_name(self) -> str:
        """Backend platform name (e.g., "sim:slos").

        This is a backward-compatibility property. Use `backend_capabilities.name` directly.
        """
        return self.backend_capabilities.name

    @property
    def available_commands(self) -> tuple[str]:
        """Snapshot of supported remote commands (e.g., ("probs", "sample_count")).

        This is a backward-compatibility property. Use `backend_capabilities.available_commands` directly.
        """
        return self.backend_capabilities.available_commands

    # ---------------- Small compatibility helpers ----------------

    def _get_computation_scheme(self, layer: MerlinModule) -> str:
        """Return the Combinadics scheme string for a layer's computation space.

        Returns one of ``"fock"``, ``"unbunched"``, ``"dual_rail"``.
        """
        cs = getattr(layer, "computation_space", None)
        if cs is not None:
            # ComputationSpace.value is the scheme string
            val = getattr(cs, "value", None)
            if isinstance(val, str) and val in ("fock", "unbunched", "dual_rail"):
                return val
            # Fallback: match by enum name
            name = getattr(cs, "name", "")
            if name == "UNBUNCHED":
                return "unbunched"
            if name == "DUAL_RAIL":
                return "dual_rail"

        return "fock"

    def _effective_sample_count(self, nsample: int | None) -> int:
        """Return the sample count submitted to a sampling command.

        Parameters
        ----------
        nsample : int | None
            Requested samples per input. If ``None``, the processor default is
            used.

        Returns
        -------
        int
            Sample count capped by ``max_shots_per_call``.
        """
        requested_shots = (
            self.DEFAULT_SHOTS_PER_CALL if nsample is None else int(nsample)
        )
        return min(requested_shots, self.max_shots_per_call)

    # ---------------- Public APIs ----------------

    def __enter__(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("MerlinProcessor is closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        suppress_exception = False
        try:
            self.cancel_all()
        finally:
            # End session lifecycle if provided
            with self._lock:
                self._closed = True
        return suppress_exception

    def cancel_all(self) -> None:
        """Cancel all in-flight jobs across all futures."""
        with self._lock:
            jobs = list(self._active_jobs)
        for job in jobs:
            PercevalAdapter.cancel_job(job)

    def forward(
        self,
        module: nn.Module,
        input: torch.Tensor,
        *,
        nsample: int | None = None,
        timeout: float | None = None,
    ) -> torch.Tensor:
        """Synchronously execute a module against the configured Perceval backend.

        Convenience wrapper around :meth:`forward_async` that blocks until completion.
        Classic layers run locally; quantum leaves (those with ``export_config()`` and
        ``should_offload()`` returning ``True``) are submitted to the configured backend.

        **Execution Strategy**

        The backend determines whether results are exact probabilities or samples:

        - If backend exposes ``"probs"`` command: uses exact probabilities if sample is None or 0.
        - Otherwise: uses sampling; shots = ``nsample`` if provided, else
          ``min(DEFAULT_SHOTS_PER_CALL, max_shots_per_call)``. If ``nsample``
          exceeds ``max_shots_per_call``, a warning is issued and ``nsample``
          is clamped.

        Parameters
        ----------
        module : nn.Module
            Module tree to evaluate. Must be in ``.eval()`` mode.
        input : torch.Tensor
            Input batch ``[B, D]`` or shape required by the first layer.
            Moved to CPU for backend execution; output is moved back to original
            device/dtype.
        nsample : int | None
            Requested samples per input when using sampling. Ignored if backend
            supports exact probabilities. If ``None``,
            ``min(DEFAULT_SHOTS_PER_CALL, max_shots_per_call)`` is used.
            Default: None.
        timeout : float | None
            Per-call override of the default timeout (seconds). ``None`` or ``0``
            means unlimited. Default: None (uses ``default_timeout``).

        Returns
        -------
        torch.Tensor
            Output tensor from the module. Batch dimension ``B`` and distribution
            dimension depend on the leaf output shape.

        Raises
        ------
        RuntimeError
            If the processor is closed or ``module`` is in training mode.
        TimeoutError
            If global timeout is exceeded. Remote jobs are cancelled
            best-effort; local synchronous jobs are checked before and after
            execution.
        """
        fut = self.forward_async(module, input, nsample=nsample, timeout=timeout)
        return fut.wait()

    def forward_async(
        self,
        module: nn.Module,
        input: torch.Tensor,
        *,
        nsample: int | None = None,
        timeout: float | None = None,
    ) -> MerlinFuture:
        """Asynchronously execute a module against the configured Perceval backend.

        Returns a :class:`MerlinFuture` (a ``torch.futures.Future`` subclass)
        that resolves to the output tensor.
        Remote batches are automatically chunked and submitted with limited
        concurrency. Local processor inputs are kept as one Merlin-level batch
        and represented as Perceval sampler iterations using an isolated
        backend object.

        **Execution Strategy**

        The backend determines whether results are exact probabilities or samples:

        - If backend exposes ``"probs"`` command: uses exact probabilities; ``nsample``
          is ignored. Results are already normalized probabilities.
        - Otherwise: uses sampling; shots = ``nsample`` if provided, else
          ``min(DEFAULT_SHOTS_PER_CALL, max_shots_per_call)``. If ``nsample``
          exceeds ``max_shots_per_call``, a warning is issued and ``nsample``
          is clamped.

        Parameters
        ----------
        module : nn.Module
            Module tree to evaluate. Must be in ``.eval()`` mode.
        input : torch.Tensor
            Input batch ``[B, D]`` or shape required by the first layer.
            Moved to CPU for backend execution; output is moved back to original
            device/dtype.
        nsample : int | None
            Requested samples per input when using sampling. Ignored if backend
            supports exact probabilities. If ``None``,
            ``min(DEFAULT_SHOTS_PER_CALL, max_shots_per_call)`` is used.
            Default: None.
        timeout : float | None
            Per-call override of the default timeout (seconds). ``None`` or ``0``
            means unlimited. Default: None (uses ``default_timeout``).

        Returns
        -------
        MerlinFuture
            Typed ``torch.futures.Future[torch.Tensor]`` subclass exposing:

            - ``future.job_ids: list[str]`` — accumulates remote job IDs
              across chunks.
            - ``future.status() -> dict`` — current progress and state:
              ``{"state", "progress", "message", "chunks_total", "chunks_done", "active_chunks"}``.
            - ``future.cancel_remote() -> None`` — cooperatively cancel; awaiting
              the future raises ``CancelledError``.

        Raises
        ------
        RuntimeError
            If the processor is closed or ``module`` is in training mode.
        TimeoutError
            If global timeout is exceeded. Remote jobs are cancelled
            best-effort; local synchronous jobs are checked before and after
            execution.
        concurrent.futures.CancelledError
            If :meth:`future.cancel_remote` is called.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("MerlinProcessor is closed")

        if module.training:
            raise RuntimeError(
                "Backend quantum execution requires `.eval()` mode because no gradient"
                "can flow through the QPU runs. Gradient computation methods in the MerlinProcessor"
                " will come in v0.5. Call `module.eval()` before forward."
            )

        if nsample is not None and nsample > self.max_shots_per_call:
            warnings.warn(
                f"Number of samples requested ({nsample}) exceeds max_shots_per_call "
                f"({self.max_shots_per_call}). This is a hard cap and will be applied. "
                f"nsample will be capped to {self.max_shots_per_call}.",
                UserWarning,
                stacklevel=2,
            )
            nsample = self.max_shots_per_call

        effective_timeout = self.default_timeout if timeout is None else timeout
        deadline: float | None = (
            None
            if effective_timeout in (None, 0)
            else time.time() + float(effective_timeout)
        )

        original_device = input.device
        original_dtype = input.dtype
        layers: list[Any] = list(self._iter_layers_in_order(module))

        state = CallState.new()
        fut = MerlinFuture(state, self.cancel_all)

        def _run_pipeline():
            try:
                x = input
                for layer in layers:
                    # Policy: offload MerlinModule leaves; else run locally
                    if isinstance(layer, MerlinModule):
                        try:
                            # Preferred (new) signature
                            should_offload = bool(layer.should_offload())

                        except Exception:
                            should_offload = False
                    else:
                        should_offload = False

                    if state.cancel_requested:
                        raise self._cancelled_error()

                    if should_offload:
                        x = self._offload_quantum_layer_with_chunking(
                            layer, x, nsample, state, deadline
                        )
                    else:
                        with torch.no_grad():
                            x = layer(x)

                if not fut.done():
                    fut.set_result(x.to(device=original_device, dtype=original_dtype))
            except BaseException as e:
                if not fut.done():
                    fut.set_exception(e)

        threading.Thread(target=_run_pipeline, daemon=True).start()
        return fut

    # ---------------- Chunked offload per quantum leaf ----------------

    def _offload_quantum_layer_with_chunking(
        self,
        layer: MerlinModule,
        input_tensor: torch.Tensor,
        nsample: int | None,
        state: CallState,
        deadline: float | None,
    ) -> torch.Tensor:
        """Execute a quantum layer through the selected backend route.

        Local processors receive the full input as one Merlin-level batch and
        convert rows into Perceval sampler iterations without microbatch
        splitting. Remote processors split the batch into ``microbatch_size``
        chunks, submit up to ``chunk_concurrency`` jobs concurrently, and
        stitch the chunk outputs.
        """
        if input_tensor.is_cuda:
            input_tensor = input_tensor.cpu()

        cache = self._layer_cache.get(layer.uid)
        if cache is None:
            if not isinstance(layer, SupportsExportConfig):
                raise TypeError(
                    "The layer must have a export_config() method returning a dictionary of this type: {'circuit':perceval.ACircuit, 'input_state': Sequence[Integral]|'perceval state object'|None, 'input_param_order': Sequence[str]|None}."
                )
            config = ValidatedLayerConfig(layer.export_config())
            self._layer_cache[layer.uid] = {"config": config}
        else:
            config = cache["config"]

        B = input_tensor.shape[0]

        if self.backend_kind == "local_processor":
            return self._run_chunk_local(
                layer, config, input_tensor, nsample, state, deadline
            )

        chunks = BatchChunker.split_batch(B, self.microbatch_size)
        return self._run_chunks_pooled(
            layer, config, input_tensor, chunks, nsample, state, deadline
        )

    def _run_chunks_pooled(
        self,
        layer: MerlinModule,
        config: ValidatedLayerConfig,
        input_tensor: torch.Tensor,
        chunks: list[tuple[int, int]],
        nsample: int | None,
        state: CallState,
        deadline: float | None,
    ) -> torch.Tensor:
        """Submit chunk jobs with limited concurrency and stitch results.

        Delegates chunk orchestration to :class:`BatchChunker`; kept as the
        processor-level entry point so callers and tests interact with the
        coordinator rather than the execution unit directly.
        """
        return self._make_batch_chunker().run_chunks(
            layer, config, input_tensor, chunks, nsample, state, deadline
        )

    # ---------------- Execution unit factories ----------------

    def _make_batch_chunker(self) -> BatchChunker:
        """Build the chunk orchestration unit wired to this processor.

        ``run_chunk`` is bound at call time so monkeypatched processor
        methods remain observable, matching pre-extraction behavior.
        """
        return BatchChunker(
            run_chunk=self._run_chunk,
            chunk_concurrency=self.chunk_concurrency,
            cancel_all=self.cancel_all,
        )

    def _make_job_runner(self) -> RemoteJobRunner:
        """Build the remote chunk execution unit wired to this processor.

        Dependencies are injected as bound methods and deferred lambdas so
        that live attribute mutation (e.g. ``max_shots_per_call``) and
        monkeypatched processor methods keep working exactly as before the
        extraction.
        """
        return RemoteJobRunner(
            create_processor=self._create_fresh_rp,
            get_available_commands=lambda: self.available_commands,
            effective_sample_count=self._effective_sample_count,
            get_max_shots_per_call=lambda: self.max_shots_per_call,
            default_shots_per_call=self.DEFAULT_SHOTS_PER_CALL,
            map_results=self._process_batch_results,
            register_job=self._register_job,
            unregister_job=self._unregister_job,
            get_microbatch_limit=(
                lambda: None if self.session is not None else self.microbatch_size
            ),
            max_retries=self._MAX_CHUNK_RETRIES,
            job_name_max=self._JOB_NAME_MAX,
        )

    def _register_job(self, job: RemoteJob) -> None:
        """Track a submitted job for cancellation and history."""
        with self._lock:
            self._active_jobs.add(job)
            self._job_history.append(job)

    def _unregister_job(self, job: RemoteJob) -> None:
        """Remove a job from active cancellation tracking."""
        with self._lock:
            self._active_jobs.discard(job)

    def _run_chunk(
        self,
        layer: MerlinModule,
        config: ValidatedLayerConfig,
        input_chunk: torch.Tensor,
        nsample: int | None,
        state: CallState,
        deadline: float | None,
        job_base_label: str | None = None,
    ) -> torch.Tensor:
        """Submit a single chunk job with retries and return the mapped tensor.

        Local backends are dispatched to :meth:`_run_chunk_local`; remote
        chunk execution is delegated to :class:`RemoteJobRunner`.
        """
        if self.backend_kind == "local_processor":
            return self._run_chunk_local(
                layer, config, input_chunk, nsample, state, deadline
            )

        return self._make_job_runner().run_chunk(
            layer,
            config,
            input_chunk,
            nsample,
            state,
            deadline,
            job_base_label=job_base_label,
        )

    def _create_fresh_local_processor(
        self,
    ) -> tuple[AProcessor, LocalExperimentSnapshot]:
        """Create an isolated local Perceval processor for one execution.

        Delegates to :meth:`PercevalAdapter.rebuild_local_processor`, which
        snapshots the experiment metadata and rebuilds the processor from a
        copied experiment and fresh backend.

        Returns
        -------
        tuple[AProcessor, LocalExperimentSnapshot]
            Fresh local processor (copied non-circuit experiment state and a
            fresh backend instance) and the experiment snapshot to restore after
            the execution circuit is installed.

        Raises
        ------
        TypeError
            If the configured local processor cannot be reconstructed safely.
        """
        assert self.processor is not None
        return PercevalAdapter.rebuild_local_processor(self.processor)

    def _run_chunk_local(
        self,
        layer: MerlinModule,
        config: ValidatedLayerConfig,
        input_chunk: torch.Tensor,
        nsample: int | None,
        state: CallState,
        deadline: float | None,
    ) -> torch.Tensor:
        """Execute a local AProcessor batch with an isolated processor.

        The local processor is rebuilt before each execution so that
        ``set_circuit``, ``with_input``, and ``min_detected_photons_filter``
        calls do not mutate the shared ``self.processor`` instance.

        Cancellation and deadline are checked *before* execution (to skip work
        the caller no longer needs) and *after* execution (to avoid returning
        stale results when the deadline has passed during a long synchronous
        run).

        Parameters
        ----------
        layer : MerlinModule
            The quantum leaf being executed.  Used by
            :meth:`_process_batch_results` to map raw sampler output back to
            a tensor.
        config : ValidatedLayerConfig
            Validated circuit and input configuration extracted from ``layer``.
        input_chunk : torch.Tensor
            2D tensor of shape ``(batch_size, n_params)`` containing the
            circuit parameter values for the local batch.
        nsample : int | None
            Number of samples per row.  ``None`` or ``<= 0`` triggers exact
            probability computation when the backend supports ``"probs"``.
        state : CallState
            Typed per-call state; its ``cancel_requested`` flag is checked
            cooperatively before and after execution.
        deadline : float | None
            Absolute wall-clock deadline (``time.time()`` seconds).  ``None``
            means no deadline.

        Returns
        -------
        torch.Tensor
            2D output tensor of shape ``(batch_size, n_outputs)`` produced by
            :meth:`_process_batch_results`.

        Raises
        ------
        concurrent.futures.CancelledError
            If ``state.cancel_requested`` is ``True`` before or after
            execution.
        TimeoutError
            If ``deadline`` has elapsed before or after execution.
        """
        from concurrent.futures import CancelledError

        if state.cancel_requested:
            raise CancelledError("Local call was cancelled")
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Local call timed out")

        assert self.processor is not None

        batch_size = input_chunk.shape[0]
        input_param_names = self._extract_input_params(config)
        input_np = input_chunk.detach().cpu().numpy()

        iteration_params: list[dict[str, float]] = []
        for i in range(batch_size):
            circuit_params = {}
            for j, param_name in enumerate(input_param_names):
                circuit_params[param_name] = (
                    float(input_np[i, j]) if j < input_chunk.shape[1] else 0.0
                )
            iteration_params.append(circuit_params)

        processor, experiment_snapshot = self._create_fresh_local_processor()
        PercevalAdapter.set_circuit(
            processor, PercevalAdapter.copy_circuit(config.circuit)
        )
        PercevalAdapter.restore_experiment(processor.experiment, experiment_snapshot)
        PercevalAdapter.set_input(processor, config.input_state)

        sampler = PercevalAdapter.create_sampler(
            processor, self.max_shots_per_call, iteration_params
        )

        is_probability = ("probs" in self.available_commands) and (
            nsample is None or int(nsample) <= 0
        )

        if is_probability:
            raw_results = PercevalAdapter.execute_sync(sampler, "probs")
        else:
            use_shots = self._effective_sample_count(nsample)
            if "sample_count" in self.available_commands:
                cmd = "sample_count"
            elif "samples" in self.available_commands:
                cmd = "samples"
            else:
                cmd = "sample_count"
            raw_results = PercevalAdapter.execute_sync(
                sampler, cmd, max_samples=use_shots
            )

        if state.cancel_requested:
            raise CancelledError("Local call was cancelled")
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Local call timed out")

        return self._process_batch_results(
            raw_results, batch_size, layer, nsample, is_probability
        )

    # ---------------- Per-call RP pool helpers ----------------

    def _create_fresh_rp(self) -> RemoteProcessor:
        """Build a fresh RemoteProcessor for each chunk/attempt.

        Creates a new, independent RemoteProcessor to ensure thread-safe execution
        per chunk. Consumed by :class:`~merlin.core.execution.RemoteJobRunner`,
        which submits and polls the job for exact probabilities or sampling.

        **Dual-Path Strategy**

        - **ISession path**: Each call to ``session.build_remote_processor()`` returns
          an independent RP with its own RPC handler state, which is safe for
          concurrent chunk execution and clean retries.
        - **RemoteProcessor path**: Clones the stored RP with a new RPC handler to
          achieve thread-safety. The clone inherits the token forwarded from init.

        The fresh RP is then passed to ``Sampler`` to submit jobs with backend
        capabilities already extracted in ``backend_capabilities``. Backend commands
        (``"probs"`` vs. ``"sample_count"``/``"samples"``) are selected during
        :meth:`~merlin.core.execution.RemoteJobRunner.submit_job` based on
        ``nsample`` and available capabilities.

        Returns
        -------
        RemoteProcessor
            A new, independent ``RemoteProcessor`` instance ready to set circuit,
            configure iterations, and submit sampler jobs.

        Raises
        ------
        RuntimeError
            If called for a local processor backend.
        """
        if self.session is not None:
            # Session path: create a fresh processor from the session
            return PercevalAdapter.build_from_session(self.session)
        if self.remote_processor is None:
            raise RuntimeError(
                "Fresh RemoteProcessor creation is only available for remote "
                "processor or session backends."
            )
        # RemoteProcessor path: clone the stored processor
        return self._clone_remote_processor(self.remote_processor)

    # ---------------- Utilities & mapping ----------------

    def _clone_remote_processor(self, rp: RemoteProcessor) -> RemoteProcessor:
        """Create a sibling RemoteProcessor with its own RPC handler (thread-safe).

        Forwards the token extracted at init time so that inline-token
        RemoteProcessors are cloned correctly. Delegates the Perceval
        handler access to :class:`PercevalAdapter`.
        """
        return PercevalAdapter.clone_remote_processor(rp, self._token)

    @staticmethod
    def _extract_rp_token(rp: RemoteProcessor) -> str | None:
        """Extract the auth token from a RemoteProcessor.

        Delegates to :meth:`PercevalAdapter.extract_token`, which probes the
        RPC handler token attributes, the Authorization header, and the
        global ``RemoteConfig`` fallback. Returns ``None`` only if every
        strategy fails.
        """
        return PercevalAdapter.extract_token(rp)

    def _iter_layers_in_order(self, module: nn.Module) -> Iterable[nn.Module]:
        """Yield execution leaves in deterministic order.

        MerlinModule instances are treated as single leaves (not recursed into).
        """
        if isinstance(module, MerlinModule):
            yield module
            return
        children = list(module.children())
        if not children:
            yield module
            return
        for child in children:
            yield from self._iter_layers_in_order(child)

    def _extract_input_params(self, config: ValidatedLayerConfig) -> list[str]:
        """Extract circuit parameter names that correspond to model inputs."""
        return list(config.input_param_order)

    def _process_batch_results(
        self,
        raw_results: Any,
        batch_size: int,
        layer: MerlinModule,
        nsample: int | None = None,
        is_probability: bool = False,
    ) -> torch.Tensor:
        """Map raw cloud results dict into a [B, dist_size] probability tensor.

        Parameters
        ----------
        is_probability : bool
            Whether results are probabilities (True) or sample counts (False).
            Determined at submit time by
            :meth:`~merlin.core.execution.RemoteJobRunner.submit_job` to avoid
            recalculation.
        """
        if raw_results is None:
            raise RuntimeError(
                "Remote job returned no results. This may indicate a job execution failure "
                "or an issue with the remote platform."
            )

        if not isinstance(raw_results, dict):
            raise RuntimeError(
                f"Unexpected remote results type: {type(raw_results)} (expected dict)."
            )

        dist_size, state_to_index, valid_states = self._get_state_mapping(layer)
        output_tensors: list[torch.Tensor] = []

        if "results_list" in raw_results:
            results_list = raw_results["results_list"]
            for i, result_item in enumerate(results_list):
                if i >= batch_size:
                    break
                if "results" in result_item:
                    state_counts = self._normalize_state_result_counts(
                        result_item["results"], is_probability
                    )
                    probs = torch.zeros(dist_size)
                    if state_counts:
                        if valid_states is not None:
                            filtered_counts = {}
                            for state_str, count in state_counts.items():
                                state_tuple = self._parse_perceval_state(state_str)
                                if state_tuple in valid_states:
                                    filtered_counts[state_str] = count
                            state_counts = filtered_counts

                        if not state_counts:
                            output_tensors.append(torch.zeros(dist_size))
                            continue

                        total = 1.0 if is_probability else sum(state_counts.values())

                        for state_str, value in state_counts.items():
                            state_tuple = self._parse_perceval_state(state_str)
                            if not state_tuple:
                                continue
                            if state_to_index is not None:
                                if state_tuple not in state_to_index:
                                    continue
                                idx = state_to_index[state_tuple]
                            else:
                                continue
                            if idx < dist_size:
                                probs[idx] = (
                                    value
                                    if is_probability
                                    else (value / total if total > 0 else 0)
                                )

                        prob_sum = probs.sum()
                        if prob_sum > 0 and abs(float(prob_sum) - 1.0) > 1e-6:
                            probs = probs / prob_sum
                        output_tensors.append(probs)
                else:
                    output_tensors.append(torch.zeros(dist_size))

        while len(output_tensors) < batch_size:
            output_tensors.append(torch.zeros(dist_size))

        return torch.stack(output_tensors[:batch_size])

    def _normalize_state_result_counts(
        self, state_results: Any, is_probability: bool
    ) -> dict[Any, float | int]:
        """Normalize Perceval state results to a state-count mapping.

        Parameters
        ----------
        state_results : Any
            Result object returned by a Perceval sampler command. Probability
            and ``sample_count`` commands return mapping-like objects, while
            ``samples`` returns a sequence-like collection of sampled states.
        is_probability : bool
            Whether ``state_results`` came from a probability command.

        Returns
        -------
        dict[Any, float | int]
            Mapping from state representation to probability or sample count.

        Raises
        ------
        RuntimeError
            If probability results are not mapping-like, or sample results are
            neither mapping-like nor iterable state samples.
        """
        result_items = getattr(state_results, "items", None)
        if callable(result_items):
            return dict(result_items())

        if is_probability:
            raise RuntimeError(
                "Probability results must be a mapping of states to probabilities."
            )

        if isinstance(state_results, (str, bytes)):
            raise RuntimeError(
                "Sample results must be a mapping of states to counts or an "
                "iterable of sampled states."
            )

        try:
            sample_iterator = iter(state_results)
        except TypeError as exc:
            raise RuntimeError(
                "Sample results must be a mapping of states to counts or an "
                "iterable of sampled states."
            ) from exc

        state_counts: dict[Any, float | int] = {}
        for sampled_state in sample_iterator:
            state_tuple = self._parse_perceval_state(sampled_state)
            if not state_tuple:
                continue
            state_counts[state_tuple] = state_counts.get(state_tuple, 0) + 1

        return state_counts

    def _get_state_mapping(
        self, layer: MerlinModule
    ) -> tuple[int, dict | None, set | None]:
        """Determine the output distribution size and Fock-state-to-index mapping."""
        scheme = self._get_computation_scheme(layer)
        needs_filter = scheme != "fock"

        if hasattr(layer, "computation_process") and hasattr(
            layer.computation_process, "simulation_graph"
        ):
            graph: Any = layer.computation_process.simulation_graph

            final_keys = getattr(graph, "final_keys", None)
            if final_keys:
                keys = list(final_keys)
                dist_size = len(keys)
                state_to_index = {state: idx for idx, state in enumerate(keys)}
                valid_states = set(keys) if needs_filter else None
                return dist_size, state_to_index, valid_states

            # Prefer mapped_keys if present (newer graphs)
            mapped_keys = getattr(graph, "mapped_keys", None)
            if mapped_keys:
                keys = list(mapped_keys)
                dist_size = len(keys)
                state_to_index = {state: idx for idx, state in enumerate(keys)}
                valid_states = set(keys) if needs_filter else None
                return dist_size, state_to_index, valid_states

            if hasattr(layer, "circuit") and hasattr(layer.circuit, "m"):
                n_modes = int(layer.circuit.m)  # type: ignore[arg-type]
            else:
                n_modes = int(graph.m)  # type: ignore[arg-type]

            if hasattr(layer, "input_state"):
                input_state = layer.input_state
                n_photons = int(sum(input_state))  # type: ignore[arg-type]
            else:
                n_photons = int(graph.n_photons)  # type: ignore[arg-type]

            keys = Combinadics(scheme, n_photons, n_modes).enumerate_states()
            dist_size = len(keys)
            state_to_index = {state: idx for idx, state in enumerate(keys)}
            valid_states = set(keys) if needs_filter else None

            return dist_size, state_to_index, valid_states

        if hasattr(layer, "circuit") and hasattr(layer, "input_state"):
            circuit = cast(Any, layer.circuit)
            input_state = cast(Any, layer.input_state)

            n_modes = int(circuit.m)
            n_photons = int(sum(input_state))

            keys = Combinadics(scheme, n_photons, n_modes).enumerate_states()
            dist_size = len(keys)
            state_to_index = {state: idx for idx, state in enumerate(keys)}
            valid_states = set(keys) if needs_filter else None

            return dist_size, state_to_index, valid_states

        raise RuntimeError(
            f"Cannot infer state mapping for layer of type {type(layer)!r}. "
            "Expected a MerlinModule with either a 'computation_process' + 'simulation_graph' "
            "or 'circuit' and 'input_state' attributes."
        )

    # ---- Shot estimation (no remote jobs submitted) ----

    def estimate_required_shots_per_input(
        self,
        layer: MerlinModule,
        input: torch.Tensor,
        desired_samples_per_input: int,
    ) -> list[int]:
        """Estimate required shots per input row using the platform estimator.

        Parameters
        ----------
        layer : MerlinModule
            Layer providing ``export_config()`` for remote estimation.
        input : torch.Tensor
            Input tensor with one or more rows to estimate.
        desired_samples_per_input : int
            Target number of usable samples per input row.

        Returns
        -------
        list[int]
            Estimated shots per input row. ``0`` means the target is not
            considered viable.

        Raises
        ------
        TypeError
            If ``layer`` does not provide ``export_config()``.
        ValueError
            If ``input`` is not one- or two-dimensional.
        RuntimeError
            If called on a local processor backend. Shot estimation uses the
            Perceval remote estimator and is remote-only.
        """
        if self.backend_kind == "local_processor":
            raise RuntimeError(
                "estimate_required_shots_per_input() is only available for "
                "remote processor or session backends; local processor backends "
                "do not expose the Perceval remote estimator."
            )

        if not isinstance(layer, SupportsExportConfig):
            raise TypeError(
                "For shot estimation, the layer must have a export_config() method returning a dictionary of this type: {'circuit':perceval.ACircuit, 'input_state': Sequence[Integral]|'perceval state object'|None, 'input_param_order': Sequence[str]|None}."
            )
        config = ValidatedLayerConfig(layer.export_config())

        if input.dim() == 1:
            x = input.unsqueeze(0)
        elif input.dim() == 2:
            x = input
        else:
            raise ValueError("input must be 1D or 2D tensor")

        if not isinstance(layer, SupportsExportConfig):
            raise TypeError(
                "The layer must have a export_config() method returning a dictionary of this type: {'circuit':perceval.ACircuit, Sequence[Integral]|'perceval state object'|None, 'input_param_order': Sequence[str]|None}."
            )
        config = ValidatedLayerConfig(layer.export_config())
        child_rp = self._create_fresh_rp()
        PercevalAdapter.configure_processor(
            child_rp, config.circuit, config.input_state
        )

        input_param_names = self._extract_input_params(config)

        import requests  # type: ignore[import-untyped]

        x_np = x.detach().cpu().numpy()
        estimates: list[int] = []
        for i in range(x_np.shape[0]):
            row = x_np[i]
            param_values: dict[str, float] = {}
            for j, pname in enumerate(input_param_names):
                param_values[pname] = float(row[j] * np.pi) if j < row.shape[0] else 0.0

            # Retry on transient read timeouts from the cloud estimator.
            est = None
            last_ex: Exception | None = None
            for _attempt in range(self._MAX_ESTIMATOR_RETRIES):
                try:
                    est = PercevalAdapter.estimate_required_shots(
                        child_rp, desired_samples_per_input, param_values
                    )
                    break
                except requests.exceptions.ReadTimeout as ex:
                    last_ex = ex
                    time.sleep(0.2)
            if est is None and last_ex is not None:
                raise last_ex
            estimates.append(int(est) if est is not None else 0)

        return estimates

    # ---- Misc ----

    def _parse_perceval_state(self, state_str: Any) -> tuple:
        """Parse a Perceval state string like '|1,0,1>' into a tuple of ints."""
        if isinstance(state_str, str):
            if "|" in state_str and ">" in state_str:
                state_str = state_str.strip("|>")
                try:
                    return tuple(int(v) for v in state_str.split(","))
                except Exception:
                    return ()
            elif "," in state_str:
                try:
                    return tuple(int(v) for v in state_str.split(","))
                except Exception:
                    return ()
        elif hasattr(state_str, "__iter__"):
            return tuple(state_str)
        return ()

    def get_job_history(self) -> list[RemoteJob]:
        """Return all jobs observed or submitted by this instance.

        Returns
        -------
        list[RemoteJob]
            Recorded remote jobs.
        """
        return self._job_history

    def clear_job_history(self) -> None:
        """Clear the internal job history list."""
        self._job_history = []

    def _cancelled_error(self):
        """Create a CancelledError with a standard message."""
        from concurrent.futures import CancelledError

        return CancelledError("Remote call was cancelled")
