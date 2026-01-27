import perceval as pcvl
import pytest
import torch

from merlin.core.computation_space import ComputationSpace
from merlin.core.probability_distribution import ProbabilityDistribution
from merlin.core.state_vector import StateVector


def test_from_tensor_and_normalize_dense():
    probs = torch.tensor([0.2, 0.3, 0.5])  # basis size for (2 modes, 2 photons) is 3
    pd = ProbabilityDistribution.from_tensor(probs, n_modes=2, n_photons=2)
    assert pd.is_normalized
    dense = pd.to_dense()
    assert torch.isclose(dense.sum(), torch.tensor(1.0))


def test_from_state_vector_builds_probabilities():
    coef = 1 / torch.sqrt(torch.tensor(2.0))
    sv = StateVector.from_tensor(
        torch.tensor([coef, coef], dtype=torch.complex64), n_modes=2, n_photons=1
    )
    pd = ProbabilityDistribution.from_state_vector(sv)
    dense = pd.to_dense()
    assert torch.allclose(dense, torch.tensor([0.5, 0.5]))


def test_filter_unbunched_dense_performance():
    probs = torch.tensor([
        0.5,
        0.25,
        0.25,
    ])  # basis size for (n_modes=2, n_photons=2) is 3
    pd = ProbabilityDistribution.from_tensor(probs, n_modes=2, n_photons=2)
    filtered = pd.filter(ComputationSpace.UNBUNCHED)
    # states are (2,0),(1,1),(0,2); unbunched keeps only (1,1)
    assert torch.allclose(filtered.to_dense(), torch.tensor([1.0]))
    assert filtered.basis_size == 1
    assert filtered.computation_space is ComputationSpace.UNBUNCHED
    assert torch.allclose(filtered.logical_performance, torch.tensor(0.25 / 1.0))


def test_filter_with_allowed_list_sparse():
    indices = torch.tensor([[0, 1, 2]])
    values = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float32)
    sparse_probs = torch.sparse_coo_tensor(indices, values, (3,))
    pd = ProbabilityDistribution.from_tensor(sparse_probs, n_modes=2, n_photons=2)
    allowed = [(2, 0), (0, 2)]
    filtered = pd.filter(allowed)
    dense = filtered.to_dense()
    assert torch.allclose(dense, torch.tensor([0.14285715, 0.85714287]), atol=1e-6)
    assert torch.allclose(filtered.logical_performance, torch.tensor(0.7))


def test_filter_outputs_are_normalized_dense_and_sparse():
    # dense case
    probs = torch.tensor([0.5, 0.25, 0.25])
    pd = ProbabilityDistribution.from_tensor(probs, n_modes=2, n_photons=2)
    filtered_dense = pd.filter(ComputationSpace.UNBUNCHED)
    assert torch.isclose(filtered_dense.to_dense().sum(), torch.tensor(1.0))

    # sparse case
    indices = torch.tensor([[0, 1, 2]])
    values = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float32)
    sparse_probs = torch.sparse_coo_tensor(indices, values, (3,))
    pd_sparse = ProbabilityDistribution.from_tensor(
        sparse_probs, n_modes=2, n_photons=2
    )
    filtered_sparse = pd_sparse.filter([(2, 0), (0, 2)])
    assert torch.isclose(filtered_sparse.to_dense().sum(), torch.tensor(1.0))


def test_filter_dual_rail_dense_shrinks_basis_and_tracks_perf():
    # Build distribution over 4 modes, 2 photons with mass split between valid/invalid dual-rail states
    probe = ProbabilityDistribution.from_tensor(torch.ones(10), n_modes=4, n_photons=2)
    basis = probe.basis
    probs = torch.zeros(len(basis))
    valid_states = {(1, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0), (0, 1, 0, 1)}
    for i, state in enumerate(basis):
        if state in valid_states:
            probs[i] = 0.2  # total valid mass 0.8
        else:
            probs[i] = 0.2 / (len(basis) - len(valid_states))  # spread remaining 0.2
    pd = ProbabilityDistribution.from_tensor(probs, n_modes=4, n_photons=2)
    filtered = pd.filter(ComputationSpace.DUAL_RAIL)
    assert filtered.computation_space is ComputationSpace.DUAL_RAIL
    assert filtered.basis_size == 4
    assert torch.isclose(filtered.to_dense().sum(), torch.tensor(1.0))
    assert torch.isclose(filtered.logical_performance, torch.tensor(0.8))


def test_filter_dual_rail_invalid_geometry_raises():
    probs = torch.tensor([
        0.5,
        0.25,
        0.25,
    ])  # basis size for (n_modes=3, n_photons=1) is 3
    pd = ProbabilityDistribution.from_tensor(probs, n_modes=3, n_photons=1)
    with pytest.raises(ValueError):
        _ = pd.filter(ComputationSpace.DUAL_RAIL)


def test_filter_combined_space_and_predicate():
    # start with uniform mass over Fock(4 modes, 2 photons); 10 states total
    probe = ProbabilityDistribution.from_tensor(torch.ones(10), n_modes=4, n_photons=2)

    def first_pair_occupied(state: tuple[int, ...]) -> bool:
        return state[0] == 1  # dual-rail states with photon in first rail of pair 0

    filtered = probe.filter((ComputationSpace.DUAL_RAIL, first_pair_occupied))
    # dual_rail keeps 4 states; predicate keeps 2 of them
    assert filtered.computation_space is ComputationSpace.DUAL_RAIL
    assert filtered.basis_size == 2
    # perf: 2 kept out of 10 original mass
    assert torch.isclose(filtered.logical_performance, torch.tensor(0.2))
    assert torch.isclose(filtered.to_dense().sum(), torch.tensor(1.0))


def test_filter_predicate_sparse_rejects_all():
    indices = torch.tensor([[0, 1]])
    values = torch.tensor([0.4, 0.6], dtype=torch.float32)
    sparse_probs = torch.sparse_coo_tensor(indices, values, (2,))
    pd = ProbabilityDistribution.from_tensor(sparse_probs, n_modes=2, n_photons=1)
    filtered = pd.filter(lambda _state: False)
    dense = filtered.to_dense()
    assert torch.allclose(dense, torch.zeros_like(dense))
    assert torch.allclose(filtered.logical_performance, torch.zeros(()))


def test_to_perceval_sparse_batch():
    # shape (batch=2, basis=2), sparse representation
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])  # (batch_idx, basis_idx)
    values = torch.tensor([0.7, 0.3, 0.4, 0.6], dtype=torch.float32)
    tensor = torch.sparse_coo_tensor(indices, values, (2, 2))
    pd = ProbabilityDistribution.from_tensor(tensor, n_modes=2, n_photons=1)
    pcvl_list = pd.to_perceval()
    assert isinstance(pcvl_list, list) and len(pcvl_list) == 2
    assert pytest.approx(pcvl_list[0][pcvl.BasicState([1, 0])]) == pytest.approx(0.7)
    assert pytest.approx(pcvl_list[1][pcvl.BasicState([0, 1])]) == pytest.approx(0.6)


def test_to_perceval_and_back_sparse_vs_dense():
    dist = pcvl.BSDistribution()
    dist[pcvl.BasicState([1, 0])] = 0.8
    dist[pcvl.BasicState([0, 1])] = 0.2
    pd_sparse = ProbabilityDistribution.from_perceval(dist, sparse=True)
    assert pd_sparse.is_sparse
    pcvl_back = pd_sparse.to_perceval()
    assert pytest.approx(pcvl_back[pcvl.BasicState([1, 0])]) == pytest.approx(0.8)
    assert pytest.approx(pcvl_back[pcvl.BasicState([0, 1])]) == pytest.approx(0.2)

    pd_dense = ProbabilityDistribution.from_perceval(dist, sparse=False)
    assert not pd_dense.is_sparse
    assert torch.allclose(pd_dense.to_dense(), torch.tensor([0.8, 0.2]))


def test_batched_to_perceval():
    tensor = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32)
    pd = ProbabilityDistribution.from_tensor(tensor, n_modes=2, n_photons=1)
    pcvl_list = pd.to_perceval()
    assert isinstance(pcvl_list, list)
    assert len(pcvl_list) == 2
    assert pytest.approx(pcvl_list[0][pcvl.BasicState([1, 0])]) == pytest.approx(0.7)
    assert pytest.approx(pcvl_list[1][pcvl.BasicState([0, 1])]) == pytest.approx(0.6)


def test_getitem_returns_probability():
    probs = torch.tensor([0.1, 0.9])
    pd = ProbabilityDistribution.from_tensor(probs, n_modes=2, n_photons=1)
    assert torch.isclose(pd[pcvl.BasicState([0, 1])], torch.tensor(0.9))
    with pytest.raises(KeyError):
        _ = pd[[2, 0]]


def test_probability_distribution_to_propagates_metadata_and_lp():
    pd = ProbabilityDistribution.from_tensor(
        torch.tensor([0.5, 0.5]), n_modes=2, n_photons=1
    )
    pd.logical_performance = torch.tensor(0.25, dtype=torch.float32)
    converted = pd.to(dtype=torch.float64)
    assert converted is not pd
    assert converted.n_modes == pd.n_modes
    assert converted.n_photons == pd.n_photons
    assert converted.tensor.dtype == torch.float64
    assert converted.logical_performance is not None
    assert converted.logical_performance.dtype == torch.float64


def test_probability_distribution_view_and_reshape_not_supported():
    pd = ProbabilityDistribution.from_tensor(
        torch.tensor([0.5, 0.5]), n_modes=2, n_photons=1
    )
    with pytest.raises(AttributeError):
        _ = pd.view
    with pytest.raises(AttributeError):
        _ = pd.reshape


def test_probability_distribution_delegated_attrs_and_clone_detach_requires_grad():
    tensor = torch.tensor([0.5, 0.5], dtype=torch.float32)
    pd = ProbabilityDistribution.from_tensor(tensor, n_modes=2, n_photons=1)
    pd.logical_performance = torch.tensor(0.2, dtype=torch.float32)

    assert pd.shape == (2,)
    assert pd.device == pd.tensor.device
    assert pd.dtype == pd.tensor.dtype
    assert pd.requires_grad is False

    cloned = pd.clone()
    assert cloned is not pd
    assert torch.allclose(cloned.tensor, pd.tensor)
    assert cloned.logical_performance is not None
    assert torch.allclose(cloned.logical_performance, pd.logical_performance)

    detached = pd.detach()
    assert detached.tensor.requires_grad is False
    if detached.logical_performance is not None:
        assert detached.logical_performance.requires_grad is False

    pd.requires_grad_(True)
    assert pd.tensor.requires_grad is True
    if pd.logical_performance is not None:
        assert pd.logical_performance.requires_grad is True
