"""Benchmark QuantumLayer GPU time and memory across modes, photons and batch size.

This script measures three phases for a plain :class:`merlin.QuantumLayer`
(MZI entangling layers around angle encoding) on CUDA:

* **graph building** -- constructing the ``QuantumLayer`` (and, with it, the
  SLOS computation graph) directly on the target device;
* **forward pass** -- ``layer(x)``;
* **backward pass** -- ``output.sum().backward()``.

The main sweep varies the number of modes (up to 24), the number of photons
(up to 12, and never more than ``n_modes // 2``), the batch size, and the
computation space (``FOCK`` or ``UNBUNCHED``). A second, smaller sweep repeats
a subset of these configurations with a :class:`perceval.NoiseModel` attached
(source indistinguishability and transmittance) and compares noisy versus
noiseless time and memory.

Cases whose computation-space system size exceeds ``--max-basis-size`` are
skipped by default since ``FOCK`` at 24 modes / 12 photons has 834,451,800
basis states and will not fit on any current GPU. Any case that still raises
``torch.cuda.OutOfMemoryError`` is caught, recorded, and skipped so one large
case cannot abort the whole run.

All measured values are written to JSON. Console tables are printed as each
sweep completes.

Example
-------
PYTHONPATH=$PWD PCVL_PERSISTENT_PATH=.pcvl_home \\
python benchmarks/benchmark_gpu_memory.py \\
    --json-out benchmarks/results/gpu_memory.json

Reduced, faster run:

PYTHONPATH=$PWD python benchmarks/benchmark_gpu_memory.py \\
    --modes 8,16,24 --batch-sizes 1,8 --repetitions 3
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess  # noqa: S404
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import perceval as pcvl
    import torch

    import merlin as ML
else:
    pcvl = None
    torch = None
    ML = None

BYTES_PER_MIB = 1024 * 1024
MAX_SUPPORTED_MODES = 24
MAX_SUPPORTED_PHOTONS = 12

DEFAULT_MODES = "8,12,16,20,24"
DEFAULT_PHOTON_FRACTIONS = "0.25,0.5"
DEFAULT_BATCH_SIZES = "1,8,32,64"
DEFAULT_SPACES = "FOCK,UNBUNCHED"
DEFAULT_MAX_BASIS_SIZE = 3_000_000

DEFAULT_NOISE_MODES = "8,12,16,20,24"
DEFAULT_NOISE_BATCH_SIZES = "1,8"


def _ensure_runtime_dependencies() -> None:
    """Import benchmark runtime dependencies needed for CUDA execution."""
    global ML, pcvl, torch

    if torch is None:
        import torch as torch_module

        torch = torch_module
    if pcvl is None:
        import perceval as pcvl_module

        pcvl = pcvl_module
    if ML is None:
        import merlin as merlin_module

        ML = merlin_module


@dataclass(frozen=True)
class LayerCase:
    """One QuantumLayer benchmark case.

    Parameters
    ----------
    name : str
        Stable case name.
    curve_name : str
        Name of the sweep that owns this case.
    n_modes : int
        Number of photonic modes.
    n_photons : int
        Number of photons.
    computation_space : merlin.ComputationSpace
        Measurement computation space.
    batch_size : int
        Input batch size.
    noisy : bool
        Whether a source :class:`perceval.NoiseModel` is attached.
    x_value : float
        Numeric x-axis value for plotting.
    x_label : str
        Human-readable x-axis label.
    """

    name: str
    curve_name: str
    n_modes: int
    n_photons: int
    computation_space: ML.ComputationSpace
    batch_size: int
    noisy: bool
    x_value: float
    x_label: str


def _basis_size(computation_space: ML.ComputationSpace, n_modes: int, n_photons: int) -> int:
    """Return the computation-space system size."""
    if computation_space is ML.ComputationSpace.FOCK:
        return math.comb(n_modes + n_photons - 1, n_photons)
    if computation_space is ML.ComputationSpace.UNBUNCHED:
        return math.comb(n_modes, n_photons)
    raise ValueError(f"Unsupported computation space: {computation_space}.")


def _validate_mode_photon_count(n_modes: int, n_photons: int) -> None:
    """Enforce the mode/photon limits this benchmark is allowed to explore."""
    if n_modes <= 0 or n_photons <= 0:
        raise ValueError("n_modes and n_photons must be positive.")
    if n_modes > MAX_SUPPORTED_MODES:
        raise ValueError(f"n_modes={n_modes} exceeds the {MAX_SUPPORTED_MODES}-mode limit.")
    if n_photons > MAX_SUPPORTED_PHOTONS:
        raise ValueError(
            f"n_photons={n_photons} exceeds the {MAX_SUPPORTED_PHOTONS}-photon limit."
        )
    if n_photons > n_modes // 2:
        raise ValueError(
            f"n_photons={n_photons} exceeds n_modes // 2 ({n_modes // 2}) for n_modes={n_modes}."
        )


def _parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list."""
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer.")
    if any(item <= 0 for item in items):
        raise ValueError("All integer values must be positive.")
    return items


def _parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated float list."""
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one float.")
    if any(item <= 0 for item in items):
        raise ValueError("All fraction values must be positive.")
    return items


def _parse_space_list(value: str) -> list[ML.ComputationSpace]:
    """Parse a comma-separated computation-space list."""
    spaces = [_computation_space_from_name(item) for item in value.split(",") if item.strip()]
    if not spaces:
        raise ValueError("Expected at least one computation space.")
    return spaces


def _computation_space_from_name(name: str) -> ML.ComputationSpace:
    """Return a supported computation-space enum from a CLI name."""
    normalized_name = name.strip().upper()
    if normalized_name == "FOCK":
        return ML.ComputationSpace.FOCK
    if normalized_name == "UNBUNCHED":
        return ML.ComputationSpace.UNBUNCHED
    raise ValueError(f"Unsupported computation space: {name}.")


def _photon_counts_for_modes(n_modes: int, fractions: list[float]) -> list[int]:
    """Return distinct, valid photon counts for one mode count and fill fractions."""
    max_photons = min(n_modes // 2, MAX_SUPPORTED_PHOTONS)
    counts = set()
    for fraction in fractions:
        candidate = max(1, round(n_modes * fraction))
        counts.add(min(candidate, max_photons))
    return sorted(counts)


def _build_layer(
    *,
    n_modes: int,
    n_photons: int,
    computation_space: ML.ComputationSpace,
    latent_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    noise: pcvl.NoiseModel | None = None,
) -> ML.QuantumLayer:
    """Build one QuantumLayer directly on the target device.

    Building on-device (rather than building on CPU and moving with ``.to``)
    is what this benchmark calls "graph building": constructing the layer
    also constructs its SLOS computation graph.
    """
    builder = ML.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer(trainable=True, model="mzi", name="pre")
    builder.add_angle_encoding(modes=list(range(latent_dim)), name="input")
    builder.add_entangling_layer(trainable=True, model="mzi", name="post")

    return ML.QuantumLayer(
        input_size=latent_dim,
        builder=builder,
        n_photons=n_photons,
        noise=noise,
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=computation_space
        ),
        dtype=dtype,
        device=device,
    )


def _cuda_event_elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    """Return elapsed CUDA event time in milliseconds."""
    return float(start.elapsed_time(end))


def _summarize_float_samples(samples: list[float]) -> dict[str, Any]:
    """Return JSON-safe summary statistics for float samples."""
    return {
        "samples": samples,
        "mean": mean(samples),
        "std": stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
    }


def _summarize_int_samples(samples: list[int]) -> dict[str, Any]:
    """Return JSON-safe summary statistics for integer samples."""
    summary = _summarize_float_samples([float(sample) for sample in samples])
    summary["samples"] = samples
    summary["mean"] = int(round(summary["mean"]))
    summary["min"] = min(samples)
    summary["max"] = max(samples)
    return summary


def _cleanup_cuda(device: torch.device) -> None:
    """Collect Python and CUDA cached memory before or after a case."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def _measure_graph_build(
    *,
    n_modes: int,
    n_photons: int,
    computation_space: ML.ComputationSpace,
    latent_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    noise: pcvl.NoiseModel | None,
    repetitions: int,
) -> tuple[ML.QuantumLayer, dict[str, Any]]:
    """Build a layer ``repetitions`` times, timing each construction.

    Returns the last built layer (kept for the forward/backward stage) and
    JSON-ready timing/memory statistics.
    """
    build_times_ms = []
    peak_allocated_bytes = []
    layer = None
    for _ in range(repetitions):
        if layer is not None:
            del layer
        _cleanup_cuda(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        layer = _build_layer(
            n_modes=n_modes,
            n_photons=n_photons,
            computation_space=computation_space,
            latent_dim=latent_dim,
            dtype=dtype,
            device=device,
            noise=noise,
        )
        torch.cuda.synchronize(device)
        build_times_ms.append((time.perf_counter() - start) * 1000.0)
        peak_allocated_bytes.append(
            max(0, torch.cuda.max_memory_allocated(device) - baseline_allocated)
        )

    stats = {
        "graph_build_time_ms": _summarize_float_samples(build_times_ms),
        "graph_build_peak_delta_allocated_bytes": _summarize_int_samples(
            peak_allocated_bytes
        ),
    }
    return layer, stats


def _run_forward_backward_once(
    layer: ML.QuantumLayer,
    x: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Measure one forward/backward pass with CUDA events and peak memory."""
    layer.zero_grad(set_to_none=True)

    forward_baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    forward_start.record()
    output = layer(x)
    forward_end.record()
    torch.cuda.synchronize(device)
    if output.device.type != "cuda":
        raise RuntimeError(f"Layer output is on {output.device}, not CUDA.")

    forward_peak_allocated = torch.cuda.max_memory_allocated(device)

    backward_baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    backward_start.record()
    loss = output.sum()
    loss.backward()
    backward_end.record()
    torch.cuda.synchronize(device)

    backward_peak_allocated = torch.cuda.max_memory_allocated(device)
    loss_value = float(loss.detach().cpu())
    output_shape = list(output.shape)
    del output
    del loss

    return {
        "forward_time_ms": _cuda_event_elapsed_ms(forward_start, forward_end),
        "backward_time_ms": _cuda_event_elapsed_ms(backward_start, backward_end),
        "forward_peak_allocated_bytes": int(forward_peak_allocated),
        "forward_peak_delta_allocated_bytes": int(
            max(0, forward_peak_allocated - forward_baseline_allocated)
        ),
        "backward_peak_allocated_bytes": int(backward_peak_allocated),
        "backward_peak_delta_allocated_bytes": int(
            max(0, backward_peak_allocated - backward_baseline_allocated)
        ),
        "loss": loss_value,
        "output_shape": output_shape,
    }


def _run_case(
    case: LayerCase,
    *,
    latent_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    noise: pcvl.NoiseModel | None,
    warmup_steps: int,
    repetitions: int,
) -> dict[str, Any] | None:
    """Run one benchmark case and return a JSON-ready result row.

    Returns ``None`` (after printing a warning) if the case raises
    ``torch.cuda.OutOfMemoryError``.
    """
    _validate_mode_photon_count(case.n_modes, case.n_photons)
    _cleanup_cuda(device)

    try:
        layer, build_stats = _measure_graph_build(
            n_modes=case.n_modes,
            n_photons=case.n_photons,
            computation_space=case.computation_space,
            latent_dim=latent_dim,
            dtype=dtype,
            device=device,
            noise=noise,
            repetitions=repetitions,
        )
        allocated_after_build = torch.cuda.memory_allocated(device)

        x = torch.randn(case.batch_size, latent_dim, dtype=dtype, device=device)
        torch.cuda.synchronize(device)

        for _ in range(warmup_steps):
            _ = _run_forward_backward_once(layer, x, device)
        torch.cuda.synchronize(device)

        measurements = [
            _run_forward_backward_once(layer, x, device) for _ in range(repetitions)
        ]
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"  SKIP {case.name}: CUDA out of memory ({exc}).", flush=True)
        _cleanup_cuda(device)
        return None

    parameter_count = sum(parameter.numel() for parameter in layer.parameters())
    result = {
        "case_name": case.name,
        "curve_name": case.curve_name,
        "x_value": case.x_value,
        "x_label": case.x_label,
        "n_modes": case.n_modes,
        "n_photons": case.n_photons,
        "computation_space": case.computation_space.name,
        "batch_size": case.batch_size,
        "noisy": case.noisy,
        "basis_size": _basis_size(case.computation_space, case.n_modes, case.n_photons),
        "output_size": layer.output_size,
        "parameter_count": parameter_count,
        "allocated_after_build_bytes": int(allocated_after_build),
        **build_stats,
        "forward_time_ms": _summarize_float_samples([
            row["forward_time_ms"] for row in measurements
        ]),
        "backward_time_ms": _summarize_float_samples([
            row["backward_time_ms"] for row in measurements
        ]),
        "forward_peak_allocated_bytes": _summarize_int_samples([
            row["forward_peak_allocated_bytes"] for row in measurements
        ]),
        "backward_peak_allocated_bytes": _summarize_int_samples([
            row["backward_peak_allocated_bytes"] for row in measurements
        ]),
        "forward_peak_delta_allocated_bytes": _summarize_int_samples([
            row["forward_peak_delta_allocated_bytes"] for row in measurements
        ]),
        "backward_peak_delta_allocated_bytes": _summarize_int_samples([
            row["backward_peak_delta_allocated_bytes"] for row in measurements
        ]),
        "loss": _summarize_float_samples([row["loss"] for row in measurements]),
        "output_shape": measurements[-1]["output_shape"],
    }

    del layer
    del x
    _cleanup_cuda(device)
    return result


def _build_main_cases(args: argparse.Namespace) -> list[LayerCase]:
    """Build the main modes/photons/batch/space sweep, skipping oversized cases."""
    modes_list = _parse_int_list(args.modes)
    fractions = _parse_float_list(args.photon_fractions)
    batch_sizes = _parse_int_list(args.batch_sizes)
    spaces = _parse_space_list(args.spaces)

    for n_modes in modes_list:
        if n_modes > MAX_SUPPORTED_MODES:
            raise ValueError(f"--modes contains {n_modes}, above the {MAX_SUPPORTED_MODES} limit.")

    cases = []
    for n_modes in modes_list:
        for n_photons in _photon_counts_for_modes(n_modes, fractions):
            for computation_space in spaces:
                basis_size = _basis_size(computation_space, n_modes, n_photons)
                if basis_size > args.max_basis_size:
                    print(
                        f"  SKIP m{n_modes}_p{n_photons}_{computation_space.name}: "
                        f"basis size {basis_size:,} exceeds --max-basis-size "
                        f"{args.max_basis_size:,}.",
                        flush=True,
                    )
                    continue
                for batch_size in batch_sizes:
                    cases.append(
                        LayerCase(
                            name=(
                                f"m{n_modes}_p{n_photons}_"
                                f"{computation_space.name.lower()}_b{batch_size}"
                            ),
                            curve_name="mode_photon_batch_sweep",
                            n_modes=n_modes,
                            n_photons=n_photons,
                            computation_space=computation_space,
                            batch_size=batch_size,
                            noisy=False,
                            x_value=batch_size,
                            x_label=(
                                f"{computation_space.name} m={n_modes} n={n_photons} "
                                f"batch={batch_size}"
                            ),
                        )
                    )
    return cases


def _build_noise_cases(args: argparse.Namespace) -> list[LayerCase]:
    """Build the noisy-vs-noiseless FOCK sweep at half mode fill."""
    if args.skip_noise:
        return []

    modes_list = _parse_int_list(args.noise_modes)
    batch_sizes = _parse_int_list(args.noise_batch_sizes)

    cases = []
    for n_modes in modes_list:
        n_photons = min(n_modes // 2, MAX_SUPPORTED_PHOTONS)
        basis_size = _basis_size(ML.ComputationSpace.FOCK, n_modes, n_photons)
        if basis_size > args.max_basis_size:
            print(
                f"  SKIP noise m{n_modes}_p{n_photons}: basis size {basis_size:,} "
                f"exceeds --max-basis-size {args.max_basis_size:,}.",
                flush=True,
            )
            continue
        for batch_size in batch_sizes:
            for noisy in (False, True):
                cases.append(
                    LayerCase(
                        name=(
                            f"noise_m{n_modes}_p{n_photons}_b{batch_size}_"
                            f"{'noisy' if noisy else 'noiseless'}"
                        ),
                        curve_name="noise_model_sweep",
                        n_modes=n_modes,
                        n_photons=n_photons,
                        computation_space=ML.ComputationSpace.FOCK,
                        batch_size=batch_size,
                        noisy=noisy,
                        x_value=batch_size,
                        x_label=(
                            f"m={n_modes} n={n_photons} batch={batch_size} "
                            f"{'noisy' if noisy else 'noiseless'}"
                        ),
                    )
                )
    return cases


def _git_value(args: list[str], repo: Path) -> str:
    """Return git metadata when available."""
    try:
        return subprocess.check_output(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _device_metadata(device: torch.device) -> dict[str, Any]:
    """Return CUDA device metadata."""
    props = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "name": torch.cuda.get_device_name(device),
        "index": torch.cuda.current_device(),
        "total_memory_bytes": int(props.total_memory),
        "major": int(props.major),
        "minor": int(props.minor),
        "multi_processor_count": int(props.multi_processor_count),
    }


def _dtype_from_name(name: str) -> torch.dtype:
    """Return a supported floating dtype from a CLI name."""
    normalized_name = name.strip().lower()
    if normalized_name in {"float32", "fp32"}:
        return torch.float32
    if normalized_name in {"float64", "fp64"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}.")


def _benchmark_metadata(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Return run metadata for the output JSON."""
    repo = Path(__file__).resolve().parents[1]
    return {
        "schema_version": 1,
        "benchmark": "gpu_memory",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"], repo),
            "branch": _git_value(["branch", "--show-current"], repo),
        },
        "device": _device_metadata(device),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    }


def _run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run both sweeps and return JSON-ready results."""
    _ensure_runtime_dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    requested_device = torch.device(args.device)
    if requested_device.type != "cuda":
        raise ValueError("This benchmark requires a CUDA device.")
    if requested_device.index is not None:
        torch.cuda.set_device(requested_device)
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype_from_name(args.dtype)

    result = _benchmark_metadata(args, device)
    result["cases"] = []

    print("Building main sweep cases...", flush=True)
    main_cases = _build_main_cases(args)
    for case in main_cases:
        print(f"Running {case.name}...", flush=True)
        row = _run_case(
            case,
            latent_dim=args.latent_dim,
            dtype=dtype,
            device=device,
            noise=None,
            warmup_steps=args.warmup_steps,
            repetitions=args.repetitions,
        )
        if row is not None:
            result["cases"].append(row)
            _write_json(result, args.json_out)

    print("\nBuilding NoiseModel sweep cases...", flush=True)
    noise_model = pcvl.NoiseModel(
        indistinguishability=args.noise_indistinguishability,
        transmittance=args.noise_transmittance,
    )
    noise_cases = _build_noise_cases(args)
    for case in noise_cases:
        print(f"Running {case.name}...", flush=True)
        row = _run_case(
            case,
            latent_dim=args.latent_dim,
            dtype=dtype,
            device=device,
            noise=noise_model if case.noisy else None,
            warmup_steps=args.warmup_steps,
            repetitions=args.repetitions,
        )
        if row is not None:
            result["cases"].append(row)
            _write_json(result, args.json_out)

    return result


def _format_ms(value: float) -> str:
    """Format a millisecond value for console tables."""
    return f"{value:,.2f}"


def _format_mib(value_bytes: float) -> str:
    """Format a byte value in MiB for console tables."""
    return f"{value_bytes / BYTES_PER_MIB:,.1f}"


def _print_summary(result: dict[str, Any]) -> None:
    """Print a compact console summary of every recorded case."""
    cases = result["cases"]
    if not cases:
        print("No cases completed.")
        return

    print("\n" + "=" * 118)
    print(
        f"{'case':<34} {'space':<10} {'basis size':>12} {'batch':>6} "
        f"{'build_ms':>10} {'fwd_ms':>9} {'bwd_ms':>9} {'fwd_MiB':>9} {'bwd_MiB':>9}"
    )
    print("-" * 118)
    for row in cases:
        print(
            f"{row['case_name']:<34} "
            f"{row['computation_space']:<10} "
            f"{row['basis_size']:>12,} "
            f"{row['batch_size']:>6} "
            f"{_format_ms(row['graph_build_time_ms']['mean']):>10} "
            f"{_format_ms(row['forward_time_ms']['mean']):>9} "
            f"{_format_ms(row['backward_time_ms']['mean']):>9} "
            f"{_format_mib(row['forward_peak_delta_allocated_bytes']['mean']):>9} "
            f"{_format_mib(row['backward_peak_delta_allocated_bytes']['mean']):>9}"
        )
    print("=" * 118)

    noise_rows = [row for row in cases if row["curve_name"] == "noise_model_sweep"]
    if not noise_rows:
        return

    print("\nNoiseModel comparison (noisy vs noiseless):")
    print(
        f"{'config':<28} {'batch':>6} {'fwd_ratio':>10} {'bwd_ratio':>10} "
        f"{'fwd_MiB_ratio':>14} {'bwd_MiB_ratio':>14}"
    )
    by_key: dict[tuple[int, int, int, bool], dict[str, Any]] = {
        (row["n_modes"], row["n_photons"], row["batch_size"], row["noisy"]): row
        for row in noise_rows
    }
    seen = set()
    for row in noise_rows:
        key = (row["n_modes"], row["n_photons"], row["batch_size"])
        if key in seen:
            continue
        seen.add(key)
        noiseless = by_key.get((*key, False))
        noisy = by_key.get((*key, True))
        if noiseless is None or noisy is None:
            continue
        label = f"m={key[0]} n={key[1]}"
        print(
            f"{label:<28} {key[2]:>6} "
            f"{noisy['forward_time_ms']['mean'] / noiseless['forward_time_ms']['mean']:>10.2f} "
            f"{noisy['backward_time_ms']['mean'] / noiseless['backward_time_ms']['mean']:>10.2f} "
            f"{noisy['forward_peak_delta_allocated_bytes']['mean'] / max(1, noiseless['forward_peak_delta_allocated_bytes']['mean']):>14.2f} "
            f"{noisy['backward_peak_delta_allocated_bytes']['mean'] / max(1, noiseless['backward_peak_delta_allocated_bytes']['mean']):>14.2f}"
        )


def _plot_results(result: dict[str, Any], plot_dir: Path) -> list[Path]:
    """Write simple time and memory plots for the main sweep. Requires matplotlib."""
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    main_rows = [row for row in result["cases"] if row["curve_name"] == "mode_photon_batch_sweep"]
    if not main_rows:
        return []

    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in main_rows:
        key = (row["n_modes"], row["n_photons"], row["computation_space"])
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["batch_size"])

    written = []

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    for key, rows in sorted(groups.items()):
        label = f"{key[2]} m={key[0]} n={key[1]}"
        batch_sizes = [row["batch_size"] for row in rows]
        axes[0].plot(
            batch_sizes,
            [row["forward_time_ms"]["mean"] for row in rows],
            marker="o",
            label=label,
        )
        axes[1].plot(
            batch_sizes,
            [row["backward_time_ms"]["mean"] for row in rows],
            marker="o",
            label=label,
        )
    axes[0].set_ylabel("Forward time (ms)")
    axes[1].set_ylabel("Backward time (ms)")
    for axis in axes:
        axis.set_xlabel("batch size")
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("QuantumLayer GPU forward/backward time")
    time_path = plot_dir / "gpu_memory_time_vs_batch.png"
    fig.savefig(time_path, dpi=160)
    plt.close(fig)
    written.append(time_path)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    colors = {"FOCK": "tab:blue", "UNBUNCHED": "tab:orange"}
    for key, rows in sorted(groups.items()):
        color = colors.get(key[2], "tab:gray")
        basis_sizes = [row["basis_size"] for row in rows]
        peak_mib = [
            max(
                row["forward_peak_delta_allocated_bytes"]["max"],
                row["backward_peak_delta_allocated_bytes"]["max"],
            )
            / BYTES_PER_MIB
            for row in rows
        ]
        axis.scatter(basis_sizes, peak_mib, color=color, alpha=0.7)
    axis.set_xscale("log")
    axis.set_xlabel("computation-space system size")
    axis.set_ylabel("Peak allocated delta (MiB)")
    axis.grid(True, alpha=0.3)
    axis.set_title("QuantumLayer GPU peak allocated memory vs system size")
    memory_path = plot_dir / "gpu_memory_vs_basis_size.png"
    fig.savefig(memory_path, dpi=160)
    plt.close(fig)
    written.append(memory_path)

    return written


def _write_json(result: dict[str, Any], path: Path) -> None:
    """Write benchmark JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark QuantumLayer GPU graph-build, forward and backward "
            "time/memory across modes, photons, batch size and computation "
            "space, plus a NoiseModel comparison."
        )
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmarks/results/gpu_memory.json"),
        help="Path where JSON results are written.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional directory where summary PNG plots are written.",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device string.")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Floating dtype used by layers and input tensors.",
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument(
        "--modes",
        default=DEFAULT_MODES,
        help=f"Comma-separated mode counts, each <= {MAX_SUPPORTED_MODES}.",
    )
    parser.add_argument(
        "--photon-fractions",
        default=DEFAULT_PHOTON_FRACTIONS,
        help="Comma-separated fill fractions of n_modes used to pick photon counts.",
    )
    parser.add_argument(
        "--batch-sizes",
        default=DEFAULT_BATCH_SIZES,
        help="Comma-separated batch sizes for the main sweep.",
    )
    parser.add_argument(
        "--spaces",
        default=DEFAULT_SPACES,
        help="Comma-separated computation spaces: FOCK, UNBUNCHED.",
    )
    parser.add_argument(
        "--max-basis-size",
        type=int,
        default=DEFAULT_MAX_BASIS_SIZE,
        help="Skip cases whose computation-space system size exceeds this.",
    )
    parser.add_argument(
        "--skip-noise",
        action="store_true",
        help="Do not run the NoiseModel comparison sweep.",
    )
    parser.add_argument(
        "--noise-modes",
        default=DEFAULT_NOISE_MODES,
        help="Comma-separated mode counts for the NoiseModel sweep (half-fill photons).",
    )
    parser.add_argument(
        "--noise-batch-sizes",
        default=DEFAULT_NOISE_BATCH_SIZES,
        help="Comma-separated batch sizes for the NoiseModel sweep.",
    )
    parser.add_argument(
        "--noise-indistinguishability",
        type=float,
        default=0.9,
        help="Source indistinguishability used by the NoiseModel sweep.",
    )
    parser.add_argument(
        "--noise-transmittance",
        type=float,
        default=0.95,
        help="Transmittance used by the NoiseModel sweep.",
    )
    args = parser.parse_args()

    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if args.latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")
    if args.max_basis_size <= 0:
        raise ValueError("max_basis_size must be positive.")
    return args


def main() -> int:
    """Run the benchmark from the command line."""
    args = _parse_args()
    result = _run_benchmark(args)
    _write_json(result, args.json_out)
    print(f"\nWrote JSON results to {args.json_out}")
    _print_summary(result)
    if args.plot_dir is not None:
        written = _plot_results(result, args.plot_dir)
        for path in written:
            print(f"Wrote plot to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
