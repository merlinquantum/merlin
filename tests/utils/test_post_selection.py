import pytest
import torch
from perceval import PostSelect
from merlin.utils import post_select_probs


@pytest.fixture
def uniform_probs():
    return torch.tensor([0.25, 0.25, 0.25, 0.25])

@pytest.fixture
def keys():
    return [(0, 2), (1, 1), (2, 0), (0, 0)]



def test_keep_all(keys, uniform_probs):
    """Post-selection that accepts everything leaves probs unchanged."""
    ps = "[0,1]>=0"
    new_keys, new_probs = post_select_probs(ps, keys, uniform_probs)

    assert new_keys == keys
    torch.testing.assert_close(new_probs, uniform_probs)


def test_keep_none_same_keys(keys, uniform_probs):
    """Rejecting everything with same_keys yields a zero vector."""
    ps = PostSelect("[0]==99")  # never true
    _, new_probs = post_select_probs(ps, keys, uniform_probs, same_keys=True)
    torch.testing.assert_close(new_probs, torch.zeros_like(uniform_probs))


def test_normalization(keys):
    """Surviving probabilities are renormalized to sum to 1."""
    probs = torch.tensor([0.1, 0.4, 0.4, 0.1])
    ps = PostSelect("[0]==1")  # keeps only (1,1)
    _, new_probs = post_select_probs(ps, keys, probs, same_keys=False)

    assert pytest.approx(new_probs.sum().item(), abs=1e-6) == 1.0


def test_same_keys_true_zeros_rejected(keys, uniform_probs):
    """same_keys=True: rejected entries become 0, key list is unchanged."""
    ps = PostSelect("[0]==1")
    new_keys, new_probs = post_select_probs(ps, keys, uniform_probs, same_keys=True)

    assert new_keys == keys
    assert new_probs.shape[-1] == len(keys)

    # Only the (1,1) slot survives
    kept_idx = keys.index((1, 1))
    for i, p in enumerate(new_probs):
        if i != kept_idx:
            assert p.item() == pytest.approx(0.0)


def test_same_keys_false_shrinks_keys(keys, uniform_probs):
    """same_keys=False: new_keys contains only accepted states."""
    ps = PostSelect("[0]==1")  # accepts (1,1) only
    new_keys, new_probs = post_select_probs(ps, keys, uniform_probs, same_keys=False)

    assert new_keys == [(1, 1)]
    assert new_probs.shape[-1] == 1


def test_batch_probs(keys):
    """2-D input (batch × states) is handled correctly."""
    probs = torch.tensor([[0.5, 0.5, 0.0, 0.0],
                          [0.0, 0.0, 0.5, 0.5]])
    ps = PostSelect("[0]==1")  # keeps index 1: (1,1)
    _, new_probs = post_select_probs(ps, keys, probs, same_keys=True)

    assert new_probs.shape == probs.shape

    # First batch row should sum to 1 (had prob on kept state)
    assert new_probs[0].sum().item() == pytest.approx(1.0)

    # Second batch row is all-zero (no prob on kept state)
    torch.testing.assert_close(new_probs[1], torch.zeros(len(keys)))


def test_1d_input_returns_1d(keys, uniform_probs):
    """1-D input is returned as 1-D."""
    _, new_probs = post_select_probs("[0]==1", keys, uniform_probs)

    assert new_probs.ndim == 1


def test_single_state_kept():
    """Works correctly with a single-element key list."""
    keys = [(1,)]
    probs = torch.tensor([1.0])
    ps = PostSelect("[0]==1")
    new_keys, new_probs = post_select_probs(ps, keys, probs, same_keys=False)

    assert new_keys == [(1,)]
    assert new_probs.item() == pytest.approx(1.0)


def test_device_preserved(keys, uniform_probs):
    """Output tensor lives on the same device as the input."""
    _, new_probs = post_select_probs("[0]==1", keys, uniform_probs)

    assert new_probs.device == uniform_probs.device
