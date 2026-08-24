"""Shared test helpers for measurement readout tests."""

import torch


def reference_occupancy_readout(
    output_keys: list[tuple[int, ...]],
    probabilities: torch.Tensor,
) -> tuple[tuple[tuple[int, ...], ...], torch.Tensor]:
    """Group probabilities by occupancy key without renormalizing."""
    occupancy_keys = [
        tuple(1 if count > 0 else 0 for count in key) for key in output_keys
    ]
    grouped_keys = tuple(sorted(set(occupancy_keys)))
    key_to_group = {key: index for index, key in enumerate(grouped_keys)}
    group_indices = torch.tensor(
        [key_to_group[key] for key in occupancy_keys],
        dtype=torch.long,
        device=probabilities.device,
    )
    grouped = torch.zeros(
        probabilities.shape[0],
        len(grouped_keys),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    grouped.index_add_(1, group_indices, probabilities)
    return grouped_keys, grouped
