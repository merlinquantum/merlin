import perceval as pcvl
import pytest
import torch

from merlin.core.state_vector import StateVector
from merlin.utils.combinadics import Combinadics


def test_from_basic_state_dense_and_sparse():
    sv_sparse = StateVector.from_basic_state([1, 0], sparse=True)
    assert sv_sparse.tensor.is_sparse
    assert torch.allclose(sv_sparse.to_dense(), torch.tensor([1 + 0j, 0 + 0j]))

    sv_dense = StateVector.from_basic_state([1, 0], sparse=False)
    assert not sv_dense.tensor.is_sparse
    assert torch.allclose(sv_dense.tensor, torch.tensor([1 + 0j, 0 + 0j]))
    assert sv_dense.tensor.dtype.is_complex


def test_from_perceval_sparse_vs_dense_heuristic():
    # n_modes=3, n_photons=2 => basis size = 6
    sv_pcvl_sparse = pcvl.StateVector(pcvl.BasicState([1, 1, 0]))
    sv_sparse = StateVector.from_perceval(sv_pcvl_sparse)
    assert sv_sparse.tensor.is_sparse

    # Build a superposition via addition (perceval auto-normalizes)
    sv_pcvl_dense = pcvl.StateVector(pcvl.BasicState([2, 0, 0])) + pcvl.StateVector(
        pcvl.BasicState([1, 1, 0])
    )
    sv_dense = StateVector.from_perceval(sv_pcvl_dense)
    assert not sv_dense.tensor.is_sparse


def test_tensor_product_dense_overrides_sparse():
    left = StateVector.from_basic_state([1, 0], sparse=False)
    right = StateVector.from_basic_state([0, 1], sparse=True)
    combined = left.tensor_product(right)
    expected = StateVector.from_basic_state([1, 0, 0, 1], sparse=False)
    assert not combined.tensor.is_sparse
    assert torch.allclose(combined.tensor, expected.to_dense())


def test_tensor_product_normalizes_result():
    left = 2 * StateVector.from_basic_state([1, 0], sparse=False)
    right = 3 * StateVector.from_basic_state([0, 1], sparse=False)
    combined = left.tensor_product(right)
    # single term should normalize back to amplitude 1 at the correct basis index
    target_idx = combined.basis.index((1, 0, 0, 1))
    dense = combined.to_dense()
    assert torch.isclose(torch.linalg.vector_norm(dense), torch.tensor(1.0))
    assert torch.isclose(dense[target_idx].real, torch.tensor(1.0))


def test_tensor_product_at_operator_matches_method():
    left = StateVector.from_basic_state([1, 0], sparse=False)
    right = StateVector.from_basic_state([0, 1], sparse=True)
    via_operator = left @ right
    via_method = left.tensor_product(right)
    assert torch.allclose(via_operator.to_dense(), via_method.to_dense())
    assert via_operator.is_sparse == via_method.is_sparse


def test_tensor_product_at_operator_with_basic_state_left():
    sv = StateVector.from_basic_state([1, 0], sparse=False)
    combined = [0, 1] @ sv
    expected = StateVector.from_basic_state([0, 1], sparse=False).tensor_product(sv)
    assert torch.allclose(combined.to_dense(), expected.to_dense())


def test_memory_bytes_dense_and_sparse():
    dense_sv = StateVector.from_basic_state([1, 0], sparse=False)
    expected_dense = dense_sv.tensor.numel() * dense_sv.tensor.element_size()
    assert dense_sv.memory_bytes() == expected_dense

    sparse_sv = StateVector.from_basic_state([1, 0], sparse=True)
    coalesced = sparse_sv.tensor.coalesce()
    expected_sparse = (
        coalesced.indices().numel() * coalesced.indices().element_size()
        + coalesced.values().numel() * coalesced.values().element_size()
    )
    assert sparse_sv.memory_bytes() == expected_sparse


def test_normalized_flag_flow():
    sv = StateVector.from_basic_state([1, 0], sparse=False)
    assert sv.is_normalized
    scaled = 2 * sv
    assert not scaled.is_normalized
    renorm = scaled.normalize()
    assert scaled.is_normalized
    assert renorm is scaled
    added = sv + sv
    assert not added.is_normalized
    added.to_dense()
    assert added.is_normalized
    tp = sv.tensor_product(sv)
    assert tp.is_normalized


def test_addition_and_scalar_mul():
    sv = StateVector.from_basic_state([1, 0], sparse=False)
    doubled = sv + sv
    scaled = 2 * sv
    # addition keeps raw amplitudes (lazy norm)
    assert torch.allclose(doubled.tensor, torch.tensor([2 + 0j, 0 + 0j]))
    # scalar multiplication does not renormalize
    assert torch.allclose(scaled.tensor, torch.tensor([2 + 0j, 0 + 0j]))
    # explicit normalization available (in-place)
    doubled.normalize()
    assert torch.allclose(doubled.tensor, torch.tensor([1 + 0j, 0 + 0j]))

    # weighted sum preserves coefficients
    sv_alt = StateVector.from_basic_state([0, 1], sparse=False)
    weighted = (1 / 3) * sv + (2 / 3) * sv_alt
    assert torch.allclose(weighted.tensor, torch.tensor([1 / 3 + 0j, 2 / 3 + 0j]))
    weighted.normalize()
    normed = weighted.tensor
    assert torch.isclose(torch.linalg.vector_norm(normed), torch.tensor(1.0))


def test_subtraction_and_normalized_str():
    sv_a = StateVector.from_basic_state([1, 0], sparse=False)
    sv_b = StateVector.from_basic_state([0, 1], sparse=False)
    diff = sv_a - sv_b
    assert torch.allclose(diff.tensor, torch.tensor([1 + 0j, -1 + 0j]))

    # normalized_str should reflect normalized amplitudes
    s = diff.normalized_str()
    assert "tensor=" in s
    assert "StateVector" in s
    assert str(diff) == s

    sv_sparse = StateVector.from_basic_state([1, 0], sparse=True)
    added_sparse = sv_sparse + sv_sparse
    assert added_sparse.tensor.is_sparse
    # to_dense normalizes before returning
    assert torch.allclose(added_sparse.to_dense(), torch.tensor([1 + 0j, 0 + 0j]))
    added_sparse.normalize()
    assert torch.allclose(added_sparse.to_dense(), torch.tensor([1 + 0j, 0 + 0j]))


def test_addition_mismatched_basis_raises():
    sv_left = StateVector.from_basic_state([1, 0], sparse=False)
    sv_right = StateVector.from_basic_state([1, 0, 0], sparse=False)
    with pytest.raises(ValueError):
        _ = sv_left + sv_right


def test_index_and_getitem():
    sv = StateVector.from_basic_state([1, 0], sparse=False)
    assert sv.index([1, 0]) == 0
    assert torch.allclose(sv[[1, 0]], torch.tensor(1 + 0j))
    assert sv.index([0, 1]) == 1
    with pytest.raises(KeyError):
        _ = sv[[2, 0]]

    # Sparse with zero amplitude should yield None for index
    zero_sparse = StateVector.from_tensor(
        torch.sparse_coo_tensor(size=(2,)), n_modes=2, n_photons=1
    )
    assert zero_sparse.index([1, 0]) is None


def test_to_perceval_single_and_batch_dense():
    sv = StateVector.from_basic_state([1, 0], sparse=False)
    pv = sv.to_perceval()
    assert isinstance(pv, pcvl.StateVector)
    assert pv[pcvl.BasicState([1, 0])] == 1

    # Superposition is forwarded with amplitudes preserved
    coef = 1 / torch.sqrt(torch.tensor(2.0))
    sup = StateVector.from_tensor(
        torch.tensor([coef, coef], dtype=torch.complex64), n_modes=2, n_photons=1
    )
    pv_sup = sup.to_perceval()
    assert pytest.approx(pv_sup[pcvl.BasicState([1, 0])].real) == pytest.approx(
        float(coef)
    )
    assert pytest.approx(pv_sup[pcvl.BasicState([0, 1])].real) == pytest.approx(
        float(coef)
    )

    batch_tensor = torch.tensor(
        [[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j], [coef, coef]], dtype=torch.complex64
    )
    batch_sv = StateVector.from_tensor(batch_tensor, n_modes=2, n_photons=1)
    pv_list = batch_sv.to_perceval()
    assert isinstance(pv_list, list)
    assert len(pv_list) == 3
    assert pv_list[0][pcvl.BasicState([1, 0])] == 1
    assert pv_list[1][pcvl.BasicState([0, 1])] == 1
    assert pytest.approx(pv_list[2][pcvl.BasicState([1, 0])].real) == pytest.approx(
        float(coef)
    )
    assert pytest.approx(pv_list[2][pcvl.BasicState([0, 1])].real) == pytest.approx(
        float(coef)
    )


def test_to_perceval_preserves_amplitudes_without_renorm():
    tensor = torch.tensor([2.0 + 0j, 0.5 + 0j], dtype=torch.complex64)
    sv = StateVector.from_tensor(tensor, n_modes=2, n_photons=1)
    pv = sv.to_perceval()
    assert pytest.approx(pv[pcvl.BasicState([1, 0])].real) == pytest.approx(2.0)
    assert pytest.approx(pv[pcvl.BasicState([0, 1])].real) == pytest.approx(0.5)
    assert pv.is_normalized is False


def test_to_perceval_sparse_batch():
    # Shape (2, 2): batch=2, basis=2
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    values = torch.tensor([1 + 0j, 1 + 0j, 0.5 + 0j, 0.5 + 0j], dtype=torch.complex64)
    sparse_batch = torch.sparse_coo_tensor(indices, values, (2, 2))
    sv = StateVector.from_tensor(sparse_batch, n_modes=2, n_photons=1)
    pv_list = sv.to_perceval()
    assert len(pv_list) == 2
    assert pv_list[0][pcvl.BasicState([1, 0])] == 1
    assert pv_list[0][pcvl.BasicState([0, 1])] == 1
    assert pytest.approx(pv_list[1][pcvl.BasicState([1, 0])].real) == pytest.approx(0.5)
    assert pytest.approx(pv_list[1][pcvl.BasicState([0, 1])].real) == pytest.approx(0.5)


def test_sparse_batch_normalization_preserves_per_batch_norms():
    # batch 0 amplitudes: [3, 4] => norm 5; batch 1 amplitudes: [0, 2] => norm 2
    indices = torch.tensor([[0, 0, 1], [0, 1, 1]])
    values = torch.tensor([3.0 + 0j, 4.0 + 0j, 2.0 + 0j], dtype=torch.complex64)
    sparse_batch = torch.sparse_coo_tensor(indices, values, (2, 2))
    sv = StateVector.from_tensor(sparse_batch, n_modes=2, n_photons=1)
    sv.normalize()
    dense = sv.to_dense()
    assert torch.allclose(dense[0], torch.tensor([0.6 + 0j, 0.8 + 0j]))
    assert torch.allclose(dense[1], torch.tensor([0.0 + 0j, 1.0 + 0j]))


def test_tensor_coalesced_persists_on_state():
    indices = torch.tensor([[0, 0]])
    values = torch.tensor([0.5 + 0j, 0.5 + 0j], dtype=torch.complex64)
    tensor = torch.sparse_coo_tensor(indices, values, (2,))
    sv = StateVector.from_tensor(tensor, n_modes=2, n_photons=1)
    coalesced = sv._tensor_coalesced()
    assert coalesced.is_coalesced()
    assert sv.tensor is coalesced
    assert torch.allclose(
        coalesced.values(), torch.tensor([1.0 + 0j], dtype=torch.complex64)
    )


def test_from_perceval_validations():
    with pytest.raises(ValueError):
        StateVector.from_perceval(pcvl.StateVector())

    inconsistent = pcvl.StateVector(pcvl.BasicState([1, 0])) + pcvl.StateVector(
        pcvl.BasicState([1, 1])
    )
    with pytest.raises(ValueError):
        StateVector.from_perceval(inconsistent)


def test_tensor_product_sparse_basic_fastpath_and_empty_sparse():
    left = StateVector.from_basic_state([1, 0], sparse=True)
    right_values = torch.tensor([2.0 + 0j, 3.0 + 0j], dtype=torch.complex64)
    right_sparse = torch.sparse_coo_tensor(torch.tensor([[0, 1]]), right_values, (2,))
    right = StateVector.from_tensor(right_sparse, n_modes=2, n_photons=1)

    combined = left.tensor_product(right, sparse=True)
    assert combined.tensor.is_sparse
    comb = Combinadics("fock", 2, 4)
    idx_a = comb.fock_to_index((1, 0, 1, 0))
    idx_b = comb.fock_to_index((1, 0, 0, 1))
    expected = torch.zeros(comb.compute_space_size(), dtype=torch.complex64)
    expected[idx_a] = 2.0
    expected[idx_b] = 3.0
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(combined.to_dense(), expected)

    empty_other = StateVector.from_tensor(
        torch.sparse_coo_tensor(size=(2,)), n_modes=2, n_photons=1
    )
    empty_product = left.tensor_product(empty_other, sparse=True)
    assert torch.allclose(empty_product.to_dense(), torch.zeros_like(expected))


def test_getitem_sparse_batch_gathers_per_batch():
    indices = torch.tensor([[0, 1], [0, 1]])
    values = torch.tensor([2.0 + 0j, 3.0 + 0j], dtype=torch.complex64)
    sparse_batch = torch.sparse_coo_tensor(indices, values, (2, 2))
    sv = StateVector.from_tensor(sparse_batch, n_modes=2, n_photons=1)
    amp_state0 = sv[[1, 0]]
    amp_state1 = sv[[0, 1]]
    assert torch.allclose(amp_state0, torch.tensor([1.0 + 0j, 0.0 + 0j]))
    assert torch.allclose(amp_state1, torch.tensor([0.0 + 0j, 1.0 + 0j]))
