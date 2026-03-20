from perceval import PostSelect, BasicState
import torch


def post_select_probs(
    post_select: PostSelect | str,
    keys: list[tuple],
    probs: torch.Tensor,
    same_keys: bool = True
) -> tuple[list[tuple], torch.Tensor]:
    """
    Given a batch of probabilities and corresponding keys, perform
    post-selection in the style of Perceval.

    Args:
        post_select: Post-selection object
        keys: List of states representing the basis the probabilities
            are written in.
        probs: Batch of probabilities to be post-selected on.
        same_keys: Determines whether to write the probability vectors
            in the new post-selected basis or the original basis.
    """
    if len(keys) != probs.shape[-1]:
        raise ValueError("Probabilities do not match keys shape.")

    if isinstance(post_select, str):
        post_select = PostSelect(post_select)

    was_1d = probs.ndim == 1
    if was_1d:
        probs = probs.unsqueeze(0)

    new_keys = [] if not same_keys else keys

    keep = []
    for key in keys:
        kept = post_select(BasicState(key))
        keep.append(kept)

        if not same_keys and kept:
            new_keys.append(key)

    mask = torch.tensor(keep, dtype=torch.bool, device=probs.device)

    if same_keys:
        new_probs = mask * probs
    else:
        new_probs = probs[:, mask]

    # Normalize vectors that are not zero vectors.
    norm = new_probs.sum(dim=-1, keepdim=True)
    new_probs = torch.where(norm > 0, new_probs / norm, new_probs)

    if was_1d:
        new_probs = new_probs.squeeze(0)

    assert len(new_keys) == new_probs.shape[-1], (
        "Output probabilities do not match output keys shape."
    )
    return new_keys, new_probs
