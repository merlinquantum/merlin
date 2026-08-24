"""Plot GPU memory/time benchmark results produced by ``benchmark_gpu_memory.py``.

Reads the JSON written by that script and produces eight PNGs (one FOCK/one
UNBUNCHED for each of four views):

1. Peak allocated memory vs. batch size, one line per mode count, with
   ``n_photons = n_modes // 2`` (``memory_vs_batch_<space>.png``).
2. Peak allocated memory vs. photon count, one line per mode count, at a
   fixed batch size (``memory_vs_photons_<space>.png``).
3. Graph-build ("compilation") time vs. photon count, one line per mode
   count, at a fixed batch size (``build_time_vs_photons_<space>.png``).
4. Forward and backward pass time vs. photon count, one line per mode
   count, at a fixed batch size, forward on top / backward on bottom of the
   same figure (``fwd_bwd_time_vs_photons_<space>.png``).

Only noiseless cases from the main ``mode_photon_batch_sweep`` curve are used
for those eight. Pass ``--noise`` to additionally plot the ``noise_model_sweep``
curve (FOCK, photons = modes // 2), comparing noisy vs. noiseless runs:

5. ``noise_overhead_ratio.png`` -- noisy/noiseless ratio of forward time,
   backward time, and peak memory vs. mode count, one line style per batch
   size available in the sweep.
6. ``noise_absolute_b<batch>.png`` -- grouped bars of absolute forward time,
   backward time, and peak memory, noiseless vs. noisy, at
   ``--noise-batch-size``.

Example
-------
python benchmarks/plot_gpu_memory_results.py \\
    --json benchmarks/results/gpu_memory.json \\
    --output-dir benchmarks/results/gpu_memory_plots \\
    --batch-size 8 \\
    --noise --noise-batch-size 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

BYTES_PER_MIB = 1024 * 1024
SPACES = ("FOCK", "UNBUNCHED")


def _load_main_cases(json_path: Path) -> list[dict[str, Any]]:
    """Load noiseless cases from the main mode/photon/batch sweep."""
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [
        case
        for case in data["cases"]
        if case["curve_name"] == "mode_photon_batch_sweep" and not case["noisy"]
    ]


def _mode_colors(cases: list[dict[str, Any]]) -> dict[int, Any]:
    """Assign a stable color to each mode count, shared across all plots."""
    modes = sorted({case["n_modes"] for case in cases})
    cmap = plt.get_cmap("tab10")
    return {n_modes: cmap(i % 10) for i, n_modes in enumerate(modes)}


def _peak_memory_mib(case: dict[str, Any]) -> float:
    """Return the larger of forward/backward peak allocated delta, in MiB."""
    forward = case["forward_peak_delta_allocated_bytes"]["mean"]
    backward = case["backward_peak_delta_allocated_bytes"]["mean"]
    return max(forward, backward) / BYTES_PER_MIB


def _load_noise_cases(json_path: Path) -> list[dict[str, Any]]:
    """Load cases from the NoiseModel comparison sweep."""
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [case for case in data["cases"] if case["curve_name"] == "noise_model_sweep"]


def _noise_pairs(
    cases: list[dict[str, Any]], batch_size: int
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    """Return (n_modes, noiseless_row, noisy_row) triples for one batch size."""
    by_key = {
        (row["n_modes"], row["noisy"]): row
        for row in cases
        if row["batch_size"] == batch_size
    }
    pairs = []
    for n_modes in sorted({row["n_modes"] for row in cases if row["batch_size"] == batch_size}):
        noiseless = by_key.get((n_modes, False))
        noisy = by_key.get((n_modes, True))
        if noiseless is not None and noisy is not None:
            pairs.append((n_modes, noiseless, noisy))
    return pairs


def _plot_memory_vs_batch(
    cases: list[dict[str, Any]],
    space: str,
    colors: dict[int, Any],
    output_dir: Path,
) -> Path | None:
    """Plot peak memory vs. batch size at n_photons = n_modes // 2."""
    rows = [
        case
        for case in cases
        if case["computation_space"] == space and case["n_photons"] == case["n_modes"] // 2
    ]
    if not rows:
        return None

    fig, axis = plt.subplots(figsize=(8, 5.5))
    for n_modes in sorted({row["n_modes"] for row in rows}):
        series = sorted(
            (row for row in rows if row["n_modes"] == n_modes),
            key=lambda row: row["batch_size"],
        )
        axis.plot(
            [row["batch_size"] for row in series],
            [_peak_memory_mib(row) for row in series],
            marker="o",
            color=colors[n_modes],
            label=f"m={n_modes}, n={n_modes // 2}",
        )
    axis.set_xlabel("batch size")
    axis.set_ylabel("Peak allocated delta (MiB)")
    axis.set_yscale("log")
    axis.set_title(f"{space}: peak memory vs. batch size (photons = modes / 2)")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"memory_vs_batch_{space.lower()}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_memory_vs_photons(
    cases: list[dict[str, Any]],
    space: str,
    colors: dict[int, Any],
    output_dir: Path,
    batch_size: int,
) -> Path | None:
    """Plot peak memory vs. photon count at a fixed batch size."""
    rows = [
        case
        for case in cases
        if case["computation_space"] == space and case["batch_size"] == batch_size
    ]
    if not rows:
        return None

    fig, axis = plt.subplots(figsize=(8, 5.5))
    for n_modes in sorted({row["n_modes"] for row in rows}):
        series = sorted(
            (row for row in rows if row["n_modes"] == n_modes),
            key=lambda row: row["n_photons"],
        )
        axis.plot(
            [row["n_photons"] for row in series],
            [_peak_memory_mib(row) for row in series],
            marker="o",
            color=colors[n_modes],
            label=f"m={n_modes}",
        )
    axis.set_xlabel("number of photons")
    axis.set_ylabel("Peak allocated delta (MiB)")
    axis.set_yscale("log")
    axis.set_title(f"{space}: peak memory vs. photon count (batch={batch_size})")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"memory_vs_photons_{space.lower()}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_build_time_vs_photons(
    cases: list[dict[str, Any]],
    space: str,
    colors: dict[int, Any],
    output_dir: Path,
    batch_size: int,
) -> Path | None:
    """Plot graph-build ("compilation") time vs. photon count at a fixed batch size."""
    rows = [
        case
        for case in cases
        if case["computation_space"] == space and case["batch_size"] == batch_size
    ]
    if not rows:
        return None

    fig, axis = plt.subplots(figsize=(8, 5.5))
    for n_modes in sorted({row["n_modes"] for row in rows}):
        series = sorted(
            (row for row in rows if row["n_modes"] == n_modes),
            key=lambda row: row["n_photons"],
        )
        axis.plot(
            [row["n_photons"] for row in series],
            [row["graph_build_time_ms"]["mean"] for row in series],
            marker="o",
            color=colors[n_modes],
            label=f"m={n_modes}",
        )
    axis.set_xlabel("number of photons")
    axis.set_ylabel("Graph build time (ms)")
    axis.set_title(f"{space}: graph build time vs. photon count (batch={batch_size})")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"build_time_vs_photons_{space.lower()}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_forward_backward_time_vs_photons(
    cases: list[dict[str, Any]],
    space: str,
    colors: dict[int, Any],
    output_dir: Path,
    batch_size: int,
) -> Path | None:
    """Plot forward (top) and backward (bottom) pass time vs. photon count."""
    rows = [
        case
        for case in cases
        if case["computation_space"] == space and case["batch_size"] == batch_size
    ]
    if not rows:
        return None

    fig, (forward_axis, backward_axis) = plt.subplots(
        2, 1, figsize=(8, 9), sharex=True, constrained_layout=True
    )
    for n_modes in sorted({row["n_modes"] for row in rows}):
        series = sorted(
            (row for row in rows if row["n_modes"] == n_modes),
            key=lambda row: row["n_photons"],
        )
        photon_counts = [row["n_photons"] for row in series]
        color = colors[n_modes]
        forward_axis.plot(
            photon_counts,
            [row["forward_time_ms"]["mean"] for row in series],
            marker="o",
            color=color,
            label=f"m={n_modes}",
        )
        backward_axis.plot(
            photon_counts,
            [row["backward_time_ms"]["mean"] for row in series],
            marker="o",
            color=color,
            label=f"m={n_modes}",
        )
    forward_axis.set_ylabel("Forward time (ms)")
    backward_axis.set_ylabel("Backward time (ms)")
    backward_axis.set_xlabel("number of photons")
    for axis in (forward_axis, backward_axis):
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle(f"{space}: forward/backward time vs. photon count (batch={batch_size})")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"fwd_bwd_time_vs_photons_{space.lower()}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_noise_ratio(cases: list[dict[str, Any]], output_dir: Path) -> Path | None:
    """Plot noisy/noiseless ratio of forward time, backward time and memory."""
    batch_sizes = sorted({row["batch_size"] for row in cases})
    line_styles = ["-", "--", "-.", ":"]
    metric_colors = {
        "forward time": "tab:blue",
        "backward time": "tab:orange",
        "peak memory": "tab:green",
    }

    fig, axis = plt.subplots(figsize=(8, 5.5))
    plotted = False
    for batch_index, batch_size in enumerate(batch_sizes):
        pairs = _noise_pairs(cases, batch_size)
        if not pairs:
            continue
        style = line_styles[batch_index % len(line_styles)]
        modes = [n_modes for n_modes, _, _ in pairs]
        ratios = {
            "forward time": [
                noisy["forward_time_ms"]["mean"] / noiseless["forward_time_ms"]["mean"]
                for _, noiseless, noisy in pairs
            ],
            "backward time": [
                noisy["backward_time_ms"]["mean"] / noiseless["backward_time_ms"]["mean"]
                for _, noiseless, noisy in pairs
            ],
            "peak memory": [
                _peak_memory_mib(noisy) / max(1e-9, _peak_memory_mib(noiseless))
                for _, noiseless, noisy in pairs
            ],
        }
        for metric_name, values in ratios.items():
            axis.plot(
                modes,
                values,
                marker="o",
                linestyle=style,
                color=metric_colors[metric_name],
                label=f"{metric_name} (batch={batch_size})",
            )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("number of modes (photons = modes / 2)")
    axis.set_ylabel("noisy / noiseless ratio")
    axis.set_title("FOCK: NoiseModel overhead vs. noiseless baseline")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "noise_overhead_ratio.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_noise_absolute(
    cases: list[dict[str, Any]], output_dir: Path, batch_size: int
) -> Path | None:
    """Plot grouped noiseless-vs-noisy bars for time and memory at one batch size."""
    pairs = _noise_pairs(cases, batch_size)
    if not pairs:
        return None

    modes = [n_modes for n_modes, _, _ in pairs]
    positions = list(range(len(modes)))
    width = 0.35

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), constrained_layout=True)
    time_specs = [
        ("forward_time_ms", "Forward time (ms)"),
        ("backward_time_ms", "Backward time (ms)"),
    ]
    for axis, (metric_key, ylabel) in zip(axes[:2], time_specs, strict=True):
        noiseless_values = [noiseless[metric_key]["mean"] for _, noiseless, _ in pairs]
        noisy_values = [noisy[metric_key]["mean"] for _, _, noisy in pairs]
        axis.bar(
            [p - width / 2 for p in positions],
            noiseless_values,
            width,
            label="noiseless",
            color="tab:gray",
        )
        axis.bar(
            [p + width / 2 for p in positions],
            noisy_values,
            width,
            label="noisy",
            color="tab:red",
        )
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions)
        axis.set_xticklabels([f"m={n_modes}" for n_modes in modes])
        axis.grid(True, alpha=0.3, axis="y")
        axis.legend(fontsize=8)

    memory_axis = axes[2]
    noiseless_memory = [_peak_memory_mib(noiseless) for _, noiseless, _ in pairs]
    noisy_memory = [_peak_memory_mib(noisy) for _, _, noisy in pairs]
    memory_axis.bar(
        [p - width / 2 for p in positions],
        noiseless_memory,
        width,
        label="noiseless",
        color="tab:gray",
    )
    memory_axis.bar(
        [p + width / 2 for p in positions],
        noisy_memory,
        width,
        label="noisy",
        color="tab:red",
    )
    memory_axis.set_ylabel("Peak allocated delta (MiB)")
    memory_axis.set_xticks(positions)
    memory_axis.set_xticklabels([f"m={n_modes}" for n_modes in modes])
    memory_axis.grid(True, alpha=0.3, axis="y")
    memory_axis.legend(fontsize=8)

    fig.suptitle(f"FOCK: NoiseModel absolute cost, noiseless vs. noisy (batch={batch_size})")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"noise_absolute_b{batch_size}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot benchmark_gpu_memory.py JSON results."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("benchmarks/results/gpu_memory.json"),
        help="Path to the benchmark_gpu_memory.py JSON output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/gpu_memory_plots"),
        help="Directory where PNG plots are written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used for the 'vs. photon count' plots.",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Also plot the NoiseModel comparison sweep (noise_overhead_ratio.png, noise_absolute_b<batch>.png).",
    )
    parser.add_argument(
        "--noise-batch-size",
        type=int,
        default=8,
        help="Batch size used for the noisy-vs-noiseless grouped bar plot.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate all plots from the command line."""
    args = _parse_args()
    cases = _load_main_cases(args.json)
    if not cases:
        raise ValueError(f"No noiseless mode_photon_batch_sweep cases found in {args.json}.")

    colors = _mode_colors(cases)
    written = []
    for space in SPACES:
        for plot_fn in (
            lambda c, s, col, out: _plot_memory_vs_batch(c, s, col, out),
            lambda c, s, col, out: _plot_memory_vs_photons(c, s, col, out, args.batch_size),
            lambda c, s, col, out: _plot_build_time_vs_photons(c, s, col, out, args.batch_size),
            lambda c, s, col, out: _plot_forward_backward_time_vs_photons(
                c, s, col, out, args.batch_size
            ),
        ):
            path = plot_fn(cases, space, colors, args.output_dir)
            if path is None:
                print(f"  SKIP {space}: no matching cases.")
            else:
                written.append(path)

    if args.noise:
        noise_cases = _load_noise_cases(args.json)
        if not noise_cases:
            print("  SKIP noise: no noise_model_sweep cases found.")
        else:
            ratio_path = _plot_noise_ratio(noise_cases, args.output_dir)
            if ratio_path is None:
                print("  SKIP noise ratio plot: no matching noisy/noiseless pairs.")
            else:
                written.append(ratio_path)

            absolute_path = _plot_noise_absolute(
                noise_cases, args.output_dir, args.noise_batch_size
            )
            if absolute_path is None:
                print(
                    "  SKIP noise absolute plot: no matching noisy/noiseless pairs "
                    f"at batch={args.noise_batch_size}."
                )
            else:
                written.append(absolute_path)

    print(f"\nWrote {len(written)} plots to {args.output_dir}:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
