import math

import perceval as pcvl
import pytest
import torch

from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer
from merlin.bridge.quantum_bridge import QuantumBridge


def make_identity_layer(
    m: int, n_photons: int, *, no_bunching: bool = True
) -> QuantumLayer:
    c = pcvl.Circuit(m)  # identity unitary
    layer = QuantumLayer(
        circuit=c,
        n_photons=n_photons,
        device=torch.device("cpu"),
        dtype=torch.float32,
        amplitude_encoding=True,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=ComputationSpace.default(no_bunching=no_bunching)
        ),
    )
    return layer


def find_key_index(layer: QuantumLayer, basic_state: pcvl.BasicState) -> int:
    keys = layer.computation_process.simulation_graph.mapped_keys
    target = tuple(basic_state)
    for i, k in enumerate(keys):
        kt = tuple(k)
        if kt == target:
            return i
    raise AssertionError(f"BasicState {target} not found in mapped_keys")


@pytest.mark.parametrize(
    "basis_index,bits", [(0, "00"), (1, "01"), (2, "10"), (3, "11")]
)
def test_basis_state_mapping_little_endian(basis_index: int, bits: str):
    groups = [1, 1]  # 2 qubits -> two 1-qubit groups
    m = sum(2**g for g in groups)
    n_photons = len(groups)
    layer = make_identity_layer(m, n_photons)

    bridge = QuantumBridge(
        qubit_groups=groups,
        n_modes=m,
        n_photons=n_photons,
        wires_order="little",
        normalize=True,
    )

    psi = torch.zeros(2 ** sum(groups), dtype=torch.complex64)
    psi[basis_index] = 1.0 + 0.0j

    payload = bridge(psi)
    out = layer(payload)
    assert out.shape[0] == 1

    # Expected Fock state for this basis (little-endian reverses the bitstring before grouping)
    expected_bs = bridge.qubit_to_fock_state(bits)
    idx = find_key_index(layer, expected_bs)

    # Identity circuit should route basis |bits> to the same Fock key with prob ~1
    assert torch.isclose(out[0, idx], torch.tensor(1.0, dtype=out.dtype), atol=1e-6)
    assert torch.isclose(out[0].sum(), torch.tensor(1.0, dtype=out.dtype), atol=1e-6)


def test_superposition_and_normalization():
    groups = [1, 1]
    m = sum(2**g for g in groups)
    layer = make_identity_layer(m, n_photons=len(groups))

    bridge = QuantumBridge(
        qubit_groups=groups,
        n_modes=m,
        n_photons=len(groups),
        wires_order="little",
        normalize=True,
    )

    psi = torch.zeros(4, dtype=torch.complex64)
    psi[0] = 1.0 + 0.0j
    psi[3] = 0.0 + 1.0j

    psi_batch = psi.unsqueeze(0).expand(2, -1).clone()

    payload = bridge(psi_batch)
    out = layer(payload)
    assert out.shape[0] == 2

    idx_00 = find_key_index(layer, bridge.qubit_to_fock_state("00"))
    idx_11 = find_key_index(layer, bridge.qubit_to_fock_state("11"))

    # After normalization, each component should carry 0.5 probability on identity circuit
    for b in range(2):
        assert torch.isclose(
            out[b, idx_00], torch.tensor(0.5, dtype=out.dtype), atol=1e-5
        )
        assert torch.isclose(
            out[b, idx_11], torch.tensor(0.5, dtype=out.dtype), atol=1e-5
        )
        assert torch.isclose(
            out[b].sum(), torch.tensor(1.0, dtype=out.dtype), atol=1e-6
        )


def test_bridge_with_pennylane_qnode():
    qml = pytest.importorskip("pennylane")

    groups = [1]
    layer = make_identity_layer(m=2, n_photons=len(groups))

    dev = qml.device("default.qubit", wires=sum(groups), shots=None)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def pl_state(theta):
        qml.RY(theta, wires=0)
        return qml.state()

    bridge = QuantumBridge(
        qubit_groups=groups,
        n_modes=2,
        n_photons=len(groups),
        wires_order="little",
        normalize=True,
    )

    angle = math.pi / 3
    inputs = torch.tensor([[angle]], dtype=torch.get_default_dtype())
    theta = inputs.squeeze().to(torch.get_default_dtype())
    psi = pl_state(theta)
    psi = psi.to(torch.complex64)

    payload = bridge(psi)
    out = layer(payload)

    idx_0 = find_key_index(layer, bridge.qubit_to_fock_state("0"))
    idx_1 = find_key_index(layer, bridge.qubit_to_fock_state("1"))

    expected_zero = math.cos(angle / 2) ** 2
    expected_one = math.sin(angle / 2) ** 2

    assert torch.isclose(
        out[0, idx_0], torch.tensor(expected_zero, dtype=out.dtype), atol=1e-6
    )
    assert torch.isclose(
        out[0, idx_1], torch.tensor(expected_one, dtype=out.dtype), atol=1e-6
    )
    assert torch.isclose(out[0].sum(), torch.tensor(1.0, dtype=out.dtype), atol=1e-6)


def test_wires_order_big_endian_changes_mapping():
    groups = [1, 1]
    m = sum(2**g for g in groups)
    layer = make_identity_layer(m, n_photons=len(groups))

    # State |01> (index 1)
    # Big-endian: do not reverse bits
    bridge_big = QuantumBridge(
        qubit_groups=groups,
        n_modes=m,
        n_photons=len(groups),
        wires_order="big",
    )
    psi = torch.zeros(4, dtype=torch.complex64)
    psi[1] = 1.0 + 0.0j

    payload_big = bridge_big(psi)
    out_big = layer(payload_big)
    idx_big = find_key_index(layer, bridge_big.qubit_to_fock_state("01"))

    # Little-endian: reverse bits before grouping
    bridge_little = QuantumBridge(
        qubit_groups=groups,
        n_modes=m,
        n_photons=len(groups),
        wires_order="little",
    )
    payload_little = bridge_little(psi)
    out_little = layer(payload_little)
    idx_little = find_key_index(layer, bridge_little.qubit_to_fock_state("01"))

    assert torch.isclose(
        out_big[0, idx_big], torch.tensor(1.0, dtype=out_big.dtype), atol=1e-6
    )
    assert torch.isclose(
        out_little[0, idx_little], torch.tensor(1.0, dtype=out_little.dtype), atol=1e-6
    )
    # Ensure they target different keys for the same computational basis
    assert idx_big != idx_little


def test_transition_matrix_dual_rail_permutation():
    groups = [1, 1]
    bridge = QuantumBridge(
        qubit_groups=groups,
        n_modes=sum(2**g for g in groups),
        n_photons=len(groups),
        computation_space=ComputationSpace.DUAL_RAIL,
        normalize=False,
    )
    dense = bridge.transition_matrix().to_dense()
    col_sums = dense.real.sum(dim=0)
    torch.testing.assert_close(col_sums, torch.ones_like(col_sums))
    row_sums = dense.abs().sum(dim=1)
    assert torch.all(row_sums <= 1.0 + 1e-6)
    output_index = {occ: idx for idx, occ in enumerate(bridge.output_basis)}
    for col, occ in enumerate(bridge.basis_occupancies):
        row = output_index[occ]
        assert torch.isclose(dense[row, col], torch.tensor(1.0, dtype=dense.dtype))


def test_transition_matrix_unbunched_subset():
    groups = [1, 1]
    bridge = QuantumBridge(
        qubit_groups=groups,
        n_modes=sum(2**g for g in groups),
        n_photons=len(groups),
        computation_space=ComputationSpace.UNBUNCHED,
        normalize=False,
    )
    matrix = bridge.transition_matrix().to_dense()
    assert matrix.shape == (6, 4)  # C(4, 2) x 2^2
    col_sums = matrix.real.sum(dim=0)
    torch.testing.assert_close(col_sums, torch.ones_like(col_sums))
    row_sums = matrix.abs().sum(dim=1)
    assert torch.all(row_sums <= 1.0 + 1e-6)


def test_transition_matrix_fock_subset():
    groups = [1, 1]
    bridge = QuantumBridge(
        qubit_groups=groups,
        n_modes=sum(2**g for g in groups),
        n_photons=len(groups),
        computation_space=ComputationSpace.FOCK,
        normalize=False,
    )
    matrix = bridge.transition_matrix().to_dense()
    assert matrix.shape == (10, 4)  # C(4 + 2 - 1, 2) x 2^2
    col_sums = matrix.real.sum(dim=0)
    torch.testing.assert_close(col_sums, torch.ones_like(col_sums))
    row_sums = matrix.abs().sum(dim=1)
    assert torch.all(row_sums <= 1.0 + 1e-6)


def test_bridge_properties_exposed():
    groups = [2, 1]
    m = sum(2**g for g in groups)
    bridge = QuantumBridge(
        qubit_groups=groups,
        n_modes=m,
        n_photons=len(groups),
        computation_space=ComputationSpace.UNBUNCHED,
    )
    assert bridge.n_modes == m
    assert bridge.n_photons == len(groups)
    expected_size = math.comb(m, len(groups))
    assert bridge.output_size == expected_size


def test_bridge_fock_space_matches_layer_keys():
    groups = [1, 1]
    m = sum(2**g for g in groups)
    layer = make_identity_layer(m, n_photons=len(groups), no_bunching=False)
    bridge = QuantumBridge(
        qubit_groups=groups,
        n_modes=m,
        n_photons=len(groups),
        computation_space=ComputationSpace.FOCK,
    )

    psi = torch.zeros(4, dtype=torch.complex64)
    psi[2] = 1.0 + 0.0j  # |10>

    payload = bridge(psi)
    out = layer(payload)

    expected_bs = bridge.qubit_to_fock_state("10")
    idx = find_key_index(layer, expected_bs)

    assert torch.isclose(out[0, idx], torch.tensor(1.0, dtype=out.dtype), atol=1e-6)
    assert torch.isclose(out[0].sum(), torch.tensor(1.0, dtype=out.dtype), atol=1e-6)


def test_large_qloq_encoding_basis_state():
    groups = [4, 4]  # 8 qubits grouped into two 4-qubit photons
    m = sum(2**g for g in groups)  # 32 modes
    layer = make_identity_layer(m, n_photons=len(groups))

    bridge = QuantumBridge(
        qubit_groups=groups,
        n_modes=m,
        n_photons=len(groups),
        wires_order="little",
        normalize=True,
    )

    basis_index = 42  # Arbitrary computational basis element
    bits = format(basis_index, f"0{sum(groups)}b")

    psi = torch.zeros(2 ** sum(groups), dtype=torch.complex64)
    psi[basis_index] = 1.0 + 0.0j

    payload = bridge(psi)
    out = layer(payload)

    expected_bs = bridge.qubit_to_fock_state(bits)
    idx = find_key_index(layer, expected_bs)

    assert torch.isclose(out[0, idx], torch.tensor(1.0, dtype=out.dtype), atol=1e-6)
    assert torch.isclose(out[0].sum(), torch.tensor(1.0, dtype=out.dtype), atol=1e-6)


def test_error_when_qubit_groups_do_not_match_state_length():
    # Provide a 2-qubit state but groups sum to 1
    bridge = QuantumBridge(qubit_groups=[1], n_modes=2, n_photons=1)

    with pytest.raises(ValueError):
        psi = torch.tensor(
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex64
        )
        _ = bridge(psi)


def test_error_when_merlin_layer_mismatch_modes():
    # groups imply m=4, but build a layer with m=6
    bad_layer = make_identity_layer(m=6, n_photons=2)

    bridge = QuantumBridge(qubit_groups=[1, 1], n_modes=4, n_photons=2)

    with pytest.raises(ValueError):
        psi = torch.zeros(4, dtype=torch.complex64)
        psi[0] = 1.0 + 0.0j
        payload = bridge(psi)
        _ = bad_layer(payload)


def test_quantum_bridge_sequential_backward_with_pennylane():
    qml = pytest.importorskip("pennylane")

    groups = [1]
    m = sum(2**g for g in groups)
    layer = make_identity_layer(m, n_photons=len(groups))

    dev = qml.device("default.qubit", wires=sum(groups), shots=None)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def pl_state(theta):
        qml.RY(theta, wires=0)
        return qml.state()

    class PennyLaneModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        def forward(self, *_args):
            return pl_state(self.theta)

    state_module = PennyLaneModule()
    bridge = QuantumBridge(
        qubit_groups=groups,
        n_modes=m,
        n_photons=len(groups),
        wires_order="little",
        normalize=True,
    )

    model = torch.nn.Sequential(state_module, bridge, layer)

    output = model(torch.zeros(1, dtype=torch.float32))
    idx_1 = find_key_index(layer, bridge.qubit_to_fock_state("1"))
    loss = output[0, idx_1]

    loss.backward()

    assert state_module.theta.grad is not None
    assert not torch.allclose(
        state_module.theta.grad,
        torch.zeros_like(state_module.theta.grad),
    )
