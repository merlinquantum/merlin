"""Execution units extracted from MerlinProcessor (PML-305).

This module hosts the focused components that own remote chunk execution:

- :class:`BatchChunker` — splits Merlin-level batches into microbatch chunks,
  runs them with bounded concurrency, and stitches the outputs back together.
- :class:`RemoteJobRunner` — executes one remote chunk end to end: builds a
  fresh remote processor, prepares sampler iterations, submits a job, polls it,
  and maps the raw results into a tensor.

Both units receive their dependencies (fresh-processor factory, backend
capabilities, result mapping, job tracking) as injected callables so they are
independently testable with no-cloud fakes. :class:`~merlin.core.merlin_processor.MerlinProcessor`
remains the public entry point and coordinates these units without owning the
execution details itself.

The threading semantics (daemon chunk threads, cooperative cancellation,
deadline checks, polling backoff) are intentionally identical to the
pre-extraction MerlinProcessor implementation.
"""

from __future__ import annotations

import logging
import threading
import time
import zlib
from collections.abc import Callable
from concurrent.futures import CancelledError
from typing import TYPE_CHECKING

import torch
from perceval.runtime import RemoteJob, RemoteProcessor

from .perceval_adapter import PercevalAdapter, RemoteJobFailedError

if TYPE_CHECKING:
    from perceval.algorithm import Sampler

    from ..algorithms.module import MerlinModule
    from .merlin_processor import CallState, ValidatedLayerConfig

logger = logging.getLogger(__name__)


class BatchChunker:
    """Split input batches into chunks and run them with bounded concurrency.

    Parameters
    ----------
    run_chunk : Callable
        Callable executing one chunk with the signature
        ``(layer, config, input_chunk, nsample, state, deadline, job_base_label)``
        and returning a ``torch.Tensor``. Injected so chunk orchestration is
        testable with fakes and so monkeypatched processor methods stay
        observable.
    get_chunk_concurrency : Callable[[], int]
        Returns the maximum number of chunk jobs allowed in flight at once. Read
        once per scheduling pass (rather than captured at construction) so a
        mid-call change to the processor's ``chunk_concurrency`` takes effect,
        matching the pre-extraction loop.
    cancel_all : Callable[[], None]
        Invoked when the deadline elapses so in-flight remote jobs are
        cancelled best-effort before raising ``TimeoutError``.
    """

    def __init__(
        self,
        *,
        run_chunk: Callable[..., torch.Tensor],
        get_chunk_concurrency: Callable[[], int],
        cancel_all: Callable[[], None],
    ) -> None:
        self._run_chunk = run_chunk
        self._get_chunk_concurrency = get_chunk_concurrency
        self._cancel_all = cancel_all

    @staticmethod
    def split_batch(batch_size: int, microbatch_size: int) -> list[tuple[int, int]]:
        """Split ``batch_size`` rows into ``[start, end)`` microbatch chunks.

        Parameters
        ----------
        batch_size : int
            Total number of rows to split.
        microbatch_size : int
            Maximum number of rows per chunk. Must be strictly positive.

        Returns
        -------
        list[tuple[int, int]]
            ``[start, end)`` half-open index ranges covering all ``batch_size``
            rows in order.

        Raises
        ------
        ValueError
            If ``microbatch_size`` is not strictly positive. A non-positive size
            would never advance the split and loop forever.
        """
        if microbatch_size <= 0:
            raise ValueError(
                f"microbatch_size must be strictly positive, got {microbatch_size}."
            )
        chunks: list[tuple[int, int]] = []
        start = 0
        while start < batch_size:
            end = min(start + microbatch_size, batch_size)
            chunks.append((start, end))
            start = end
        return chunks

    def run_chunks(
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

        Parameters
        ----------
        layer : MerlinModule
            Quantum leaf whose backend execution produces each chunk output.
        config : ValidatedLayerConfig
            Validated layer configuration (circuit, input state, param order).
        input_tensor : torch.Tensor
            Full input batch to split across ``chunks``.
        chunks : list[tuple[int, int]]
            ``[start, end)`` row ranges produced by :meth:`split_batch`.
        nsample : int | None
            Requested sample count, or ``None``/``<= 0`` for exact probabilities.
        state : CallState
            Per-call state used for chunk counters and cooperative cancellation.
        deadline : float | None
            Absolute ``time.time()`` deadline, or ``None`` for no timeout.

        Returns
        -------
        torch.Tensor
            The per-chunk outputs concatenated along the batch dimension.

        Raises
        ------
        TimeoutError
            If ``deadline`` elapses before all chunks finish; in-flight remote
            jobs are cancelled best-effort first.
        BaseException
            The first error raised by any chunk, re-raised once the remaining
            in-flight chunks have settled.
        """
        state.add_planned_chunks(len(chunks))
        outputs: list[torch.Tensor | None] = [None] * len(chunks)
        errors: list[BaseException] = []

        total_chunks = len(chunks)
        layer_name = getattr(layer, "name", layer.__class__.__name__)

        def _call(s: int, e: int, idx: int):
            try:
                base_label = (
                    f"mer:{layer_name}:{state.call_id}:{idx + 1}/{total_chunks}"
                )
                t = self._run_chunk(
                    layer,
                    config,
                    input_tensor[s:e],
                    nsample,
                    state,
                    deadline,
                    job_base_label=base_label,
                )
                outputs[idx] = t
            except BaseException as ex:
                errors.append(ex)

        in_flight = 0
        idx = 0
        futures: list[threading.Thread] = []
        while in_flight > 0 or (not errors and idx < len(chunks)):
            concurrency = max(1, int(self._get_chunk_concurrency()))
            while not errors and idx < len(chunks) and in_flight < concurrency:
                s, e = chunks[idx]
                state.mark_chunk_started()
                th = threading.Thread(target=_call, args=(s, e, idx), daemon=True)
                th.start()
                futures.append(th)
                idx += 1
                in_flight += 1

            for th in list(futures):
                if not th.is_alive():
                    futures.remove(th)
                    in_flight -= 1
                    state.mark_chunk_finished()

            if deadline is not None and time.time() >= deadline:
                self._cancel_all()
                raise TimeoutError("Remote call timed out (remote cancel issued)")

            time.sleep(0.01)

        if errors:
            raise errors[0]

        return torch.cat(outputs, dim=0)  # type: ignore[arg-type]


class RemoteJobRunner:
    """Execute a single remote chunk: fresh processor, submit, poll, map.

    Parameters
    ----------
    create_processor : Callable[[], RemoteProcessor]
        Factory returning a fresh, independent RemoteProcessor per attempt.
    get_available_commands : Callable[[], tuple[str, ...]]
        Returns the backend command snapshot driving probs-vs-sampling.
    extract_input_params : Callable[[ValidatedLayerConfig], list[str]]
        Returns the ordered circuit-parameter names that receive model inputs.
        Injected (rather than reading ``config.input_param_order`` directly) so
        the remote and local paths share the single param-routing seam.
    effective_sample_count : Callable[[int | None], int]
        Maps a requested ``nsample`` to the capped shot count to submit.
    get_max_shots_per_call : Callable[[], int | None]
        Returns the current hard cap on shots per sampler call.
    default_shots_per_call : int
        Fallback shots value used when the cap is unset.
    map_results : Callable
        Maps a raw results dict to a tensor with the signature
        ``(raw_results, batch_size, layer, nsample, is_probability)``.
    register_job : Callable[[RemoteJob], None]
        Records a submitted job for cancellation tracking and history.
    unregister_job : Callable[[RemoteJob], None]
        Removes a job from active-cancellation tracking.
    get_microbatch_limit : Callable[[], int | None]
        Returns the per-chunk size guard, or ``None`` when chunk sizes are
        not bounded (session backends).
    max_retries : int
        Number of submission attempts per chunk.
    job_name_max : int
        Maximum length of remote job names.
    """

    def __init__(
        self,
        *,
        create_processor: Callable[[], RemoteProcessor],
        get_available_commands: Callable[[], tuple[str, ...]],
        extract_input_params: Callable[[ValidatedLayerConfig], list[str]],
        effective_sample_count: Callable[[int | None], int],
        get_max_shots_per_call: Callable[[], int | None],
        default_shots_per_call: int,
        map_results: Callable[..., torch.Tensor],
        register_job: Callable[[RemoteJob], None],
        unregister_job: Callable[[RemoteJob], None],
        get_microbatch_limit: Callable[[], int | None],
        max_retries: int = 3,
        job_name_max: int = 50,
    ) -> None:
        self._create_processor = create_processor
        self._get_available_commands = get_available_commands
        self._extract_input_params = extract_input_params
        self._effective_sample_count = effective_sample_count
        self._get_max_shots_per_call = get_max_shots_per_call
        self._default_shots_per_call = default_shots_per_call
        self._map_results = map_results
        self._register_job = register_job
        self._unregister_job = unregister_job
        self._get_microbatch_limit = get_microbatch_limit
        self._max_retries = max_retries
        self._job_name_max = job_name_max

    def run_chunk(
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

        Builds a fresh remote processor and sampler on each attempt (so a
        corrupted processor cannot poison retries), submits the job, polls it to
        completion, and maps the raw results into a tensor. Cancellation and
        deadline are checked cooperatively before every attempt.

        Parameters
        ----------
        layer : MerlinModule
            Quantum leaf whose backend execution produces the chunk output.
        config : ValidatedLayerConfig
            Validated layer configuration (circuit, input state, param order).
        input_chunk : torch.Tensor
            Rows of the batch assigned to this chunk.
        nsample : int | None
            Requested sample count, or ``None``/``<= 0`` for exact probabilities.
        state : CallState
            Per-call state observed for cooperative cancellation and job ids.
        deadline : float | None
            Absolute ``time.time()`` deadline, or ``None`` for no timeout.
        job_base_label : str | None
            Base label for the remote job name, or ``None`` to leave it unset.

        Returns
        -------
        torch.Tensor
            The mapped ``[chunk_size, ...]`` output tensor for this chunk.

        Raises
        ------
        ValueError
            If the chunk exceeds the microbatch guard (an internal invariant).
        CancelledError
            If cancellation is requested during the attempt loop.
        TimeoutError
            If ``deadline`` elapses during the attempt loop.
        RuntimeError
            If every submission attempt fails; chained to the last error.
        """
        batch_size = input_chunk.shape[0]
        microbatch_limit = self._get_microbatch_limit()
        if microbatch_limit is not None and batch_size > microbatch_limit:
            raise ValueError(
                f"Chunk size {batch_size} exceeds microbatch {microbatch_limit}. "
                "Please report this bug."
            )

        input_param_names = self._extract_input_params(config)
        input_np = input_chunk.detach().cpu().numpy()

        # Pre-compute iteration params (cheap, only done once).
        iteration_params: list[dict[str, float]] = []
        for i in range(batch_size):
            circuit_params = {}
            for j, param_name in enumerate(input_param_names):
                circuit_params[param_name] = (
                    float(input_np[i, j]) if j < input_chunk.shape[1] else 0.0
                )
            iteration_params.append(circuit_params)

        last_error: BaseException | None = None
        for attempt in range(self._max_retries):
            if state.cancel_requested:
                raise CancelledError("Remote call was cancelled")
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("Remote call timed out (remote cancel issued)")

            # Build a fresh RemoteProcessor and Sampler on each attempt so that
            # a corrupted RP doesn't poison retries.
            rp = self._create_processor()
            PercevalAdapter.configure_processor(rp, config.circuit, config.input_state)

            max_shots_per_call = self._get_max_shots_per_call()
            max_shots_arg = (
                self._default_shots_per_call
                if max_shots_per_call is None
                else int(max_shots_per_call)
            )
            sampler = PercevalAdapter.create_sampler(
                rp, max_shots_arg, iteration_params
            )

            job = None
            try:
                job, is_probability = self.submit_job(sampler, nsample, job_base_label)
                self._register_job(job)

                return self.poll_job(
                    job, state, deadline, batch_size, layer, nsample, is_probability
                )
            except (CancelledError, TimeoutError, KeyboardInterrupt):
                raise
            except Exception as exc:
                last_error = exc
                if job is not None:
                    self._unregister_job(job)
                logger.warning(
                    "Chunk attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(min(1.0 * (2**attempt), 5.0))

        raise RuntimeError(
            f"Chunk failed after {self._max_retries} attempts"
        ) from last_error

    def _capped_name(self, base: str, cmd: str) -> str:
        """Return a sanitized remote job name capped at ``job_name_max``."""
        name = f"{base}:{cmd}"
        name = "".join(ch if ch.isalnum() or ch in "-_:/=." else "_" for ch in name)
        if len(name) <= self._job_name_max:
            return name
        h = f"{zlib.adler32(name.encode()):08x}"
        keep = self._job_name_max - 1 - len(h)
        if keep < 1:
            return h[: self._job_name_max]
        return name[:keep] + "~" + h

    def submit_job(
        self,
        sampler: Sampler,
        nsample: int | None,
        job_base_label: str | None,
    ) -> tuple[RemoteJob, bool]:
        """Submit a job to the sampler, selecting command based on backend capabilities.

        **Command Selection Strategy**

        1. **Exact Probabilities** (``"probs"`` command):
           - Used if backend exposes ``"probs"`` AND (``nsample`` is None or ``nsample <= 0``).
           - Returns normalized probability distribution; ``nsample`` is ignored.

        2. **Sampling** (``"sample_count"`` or ``"samples"`` commands):
           - Used if exact probabilities are not available or ``nsample > 0``.
           - Tries ``"sample_count"`` first, falls back to ``"samples"``.
           - Number of samples = ``effective_sample_count(nsample)``.

        Job names are sanitized and capped through :meth:`_capped_name`.

        Parameters
        ----------
        sampler : perceval.algorithm.Sampler
            Perceval Sampler instance configured with circuit and iterations.
        nsample : int | None
            Number of samples requested. If ``None`` or ``<= 0``, triggers
            exact probability computation (if available).
        job_base_label : str | None
            Base label for the remote job name, or ``None`` to leave it unset.

        Returns
        -------
        tuple[RemoteJob, bool]
            The submitted job handle and the ``is_probability`` execution flag.
        """
        available_commands = self._get_available_commands()
        is_probability = ("probs" in available_commands) and (
            nsample is None or int(nsample) <= 0
        )

        if is_probability:
            cmd = "probs"
            max_samples = None
        else:
            if "sample_count" in available_commands:
                cmd = "sample_count"
            elif "samples" in available_commands:
                cmd = "samples"
            else:
                cmd = "sample_count"
            max_samples = self._effective_sample_count(nsample)

        name = self._capped_name(job_base_label, cmd) if job_base_label else None
        job = PercevalAdapter.submit_async(
            sampler, cmd, name=name, max_samples=max_samples
        )
        return job, is_probability

    def poll_job(
        self,
        job: RemoteJob,
        state: CallState,
        deadline: float | None,
        batch_size: int,
        layer: MerlinModule,
        nsample: int | None,
        is_probability: bool = False,
    ) -> torch.Tensor:
        """Poll a submitted job until complete/failed/timeout and return results.

        Continuously polls the job status, updating call state and handling
        timeouts, cancellation, and failures. Upon completion, maps results to
        a tensor through the injected ``map_results`` dependency.

        Parameters
        ----------
        job : perceval.runtime.RemoteJob
            Submitted job to poll.
        state : CallState
            Per-call state updated with status and job ids, and observed for
            cooperative cancellation.
        deadline : float | None
            Absolute ``time.time()`` deadline, or ``None`` for no timeout.
        batch_size : int
            Number of input rows in this chunk, forwarded to ``map_results``.
        layer : MerlinModule
            Quantum leaf forwarded to ``map_results`` for output extraction.
        nsample : int | None
            Original sample-count request, forwarded to ``map_results``.
        is_probability : bool
            Whether the job runs in exact-probability mode. Default value is
            False.

        Returns
        -------
        torch.Tensor
            The mapped ``[batch_size, ...]`` output tensor.

        Raises
        ------
        CancelledError
            If cancellation is requested or the backend reports a cancel.
        TimeoutError
            If ``deadline`` elapses while polling.
        RemoteJobFailedError
            If the backend reports the job as failed.
        RuntimeError
            If a completed job never yields a dict payload within the bounded
            re-poll window.
        """
        _MAX_NON_DICT_RETRIES = 60  # 60 * 0.1s = 6s
        non_dict_retries = 0
        sleep_ms = 50
        while True:
            if state.cancel_requested:
                PercevalAdapter.cancel_job(job)
                raise CancelledError("Remote call was cancelled")

            if deadline is not None and time.time() >= deadline:
                PercevalAdapter.cancel_job(job)
                raise TimeoutError("Remote call timed out (remote cancel issued)")

            snapshot = PercevalAdapter.job_snapshot(job)
            state.set_current_status(
                state=snapshot.state,
                progress=snapshot.progress,
                message=snapshot.stop_message,
            )

            job_id = snapshot.job_id
            if job_id is not None:
                state.record_job_id(job_id)

            if snapshot.is_failed:
                msg = snapshot.stop_message
                if msg and "Cancel requested" in str(msg):
                    self._unregister_job(job)
                    raise CancelledError("Remote call was cancelled")
                self._unregister_job(job)
                raise RemoteJobFailedError(
                    f"Remote job failed: {msg or 'unknown error'} (job_id={job_id!r})"
                )

            if snapshot.is_complete:
                try:
                    raw = PercevalAdapter.get_results(job)
                except RuntimeError as ex:
                    msg = str(ex)
                    if "Results are not available" in msg:
                        time.sleep(0.05)
                        continue
                    if "Cancel requested" in msg:
                        self._unregister_job(job)
                        raise CancelledError("Remote call was cancelled")
                    raise

                if isinstance(raw, dict):
                    self._unregister_job(job)
                    return self._map_results(
                        raw, batch_size, layer, nsample, is_probability
                    )

                # The backend sometimes reports completion before the dict
                # payload is actually available.  Re-poll the same job for a
                # bounded window before giving up to the outer retry loop.
                non_dict_retries += 1
                if non_dict_retries >= _MAX_NON_DICT_RETRIES:
                    self._unregister_job(job)
                    raise RuntimeError(
                        f"Job complete but results were not a dict after "
                        f"{_MAX_NON_DICT_RETRIES} re-polls; "
                        f"job_id={job_id!r}, type={type(raw)}, value={raw!r}"
                    )
                time.sleep(0.1)
                continue

            time.sleep(sleep_ms / 1000.0)
            sleep_ms = min(sleep_ms * 2, 400)
