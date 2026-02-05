import os
import tempfile

import perceval as pcvl
import pytest
import torch

from merlin.core.computation_space import ComputationSpace
from merlin.pcvl_pytorch.slos_torchscript import (
    build_slos_distribution_computegraph,
    compute_slos_distribution,
    load_slos_distribution_computegraph,
)


def test_slos_compute_backward_supports_unitary_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float
    complex_dtype = torch.cfloat

    # small instance
    m = 4
    n_photons = 2

    # build graph
    graph = build_slos_distribution_computegraph(m, n_photons, dtype=dtype)

    # create batched random complex matrices and obtain unitary via QR
    B = 3
    # create random complex matrix with real and imag parts
    u = torch.randn((B, m, m), dtype=dtype, device=device) + 1j * torch.randn(
        (B, m, m), dtype=dtype, device=device
    )
    u = u.to(complex_dtype)
    # make unitary via batch QR
    q, r = torch.linalg.qr(u)

    # fix phases on r diagonal
    r_diag = torch.diagonal(r, dim1=-2, dim2=-1)
    safe_abs = torch.where(
        torch.abs(r_diag) == 0, torch.ones_like(r_diag), torch.abs(r_diag)
    )
    phases = r_diag / safe_abs
    q = q * phases.conj().unsqueeze(-2)

    # require grad on q
    q = q.requires_grad_(True)

    # simple input state: first n_photons modes are occupied
    input_state = [1] * n_photons + [0] * (m - n_photons)

    # compute probabilities for batch
    keys, amps = graph.compute(q, input_state)
    # amps shape: [B, K] where K is number of output states

    # create scalar loss (sum of probs) to allow backward
    probs = amps.real**2 + amps.imag**2
    loss = probs.sum()

    loss.backward()

    # check gradients exist on q
    assert q.grad is not None, "q.grad is None: backward did not populate gradients"
    # Check that gradient has non-zero entries (at least one)
    assert torch.any(q.grad.abs() > 1e-12).item(), "q.grad is all zeros"


# Fixture to create a temporary file (compatible with GitHub)
@pytest.fixture
def get_tmp_file():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        yield f.name
    os.unlink(f.name)


def test_slos_save_load_computation_graph(get_tmp_file):
    dtype = torch.float

    # small instance
    m = 4
    n_photons = 2

    # build graph
    graph = build_slos_distribution_computegraph(m, n_photons, dtype=dtype)

    # Save then load the graph
    graph.save(get_tmp_file)
    loaded_graph = load_slos_distribution_computegraph(get_tmp_file)

    sDoesntMatch = "save->load: Quantity does not match: "
    assert graph.m == loaded_graph.m, sDoesntMatch + "nb modes"
    assert graph.n_photons == loaded_graph.n_photons, sDoesntMatch + "n_photons"
    assert graph.computation_space == loaded_graph.computation_space, (
        sDoesntMatch + "computation_space"
    )

    assert graph.keep_keys == loaded_graph.keep_keys, sDoesntMatch + "keep_keys"
    assert graph.dtype == loaded_graph.dtype, sDoesntMatch + "dtype"
    assert graph.output_map_func == loaded_graph.output_map_func, (
        sDoesntMatch + "has_output_map_func"
    )

    # Vérification de vectorized_operations (liste de tuples de tenseurs)
    assert len(graph.vectorized_operations) == len(
        loaded_graph.vectorized_operations
    ), sDoesntMatch + "vectorized_operations (length mismatch)"

    for i, (tuple1, tuple2) in enumerate(
        zip(
            graph.vectorized_operations, loaded_graph.vectorized_operations, strict=True
        )
    ):
        assert len(tuple1) == len(tuple2), (
            f"{sDoesntMatch}vectorized_operations (tuple {i} length mismatch)"
        )
        for j, (tensor1, tensor2) in enumerate(zip(tuple1, tuple2, strict=True)):
            assert torch.equal(tensor1, tensor2), (
                f"{sDoesntMatch}vectorized_operations (tuple {i}, tensor {j})"
            )

    assert graph.final_keys == loaded_graph.final_keys, sDoesntMatch + "final_keys"
    assert graph.mapped_keys == loaded_graph.mapped_keys, sDoesntMatch + "mapped_keys"
    assert graph.total_mapped_keys == loaded_graph.total_mapped_keys, (
        sDoesntMatch + "total_mapped_keys"
    )


def test_slos_compute_slos_distribution_with_output_map_function():

    # small instance
    m = 4
    n_photons = 2

    # Create a meaningful unitary
    circuit = pcvl.Circuit(m)
    circuit.add(0, pcvl.components.BS())
    circuit.add(2, pcvl.components.BS())
    circuit.add(1, pcvl.components.PS(phi=1.5))
    circuit.add(3, pcvl.components.PS(phi=1.7))
    circuit.add(1, pcvl.components.BS())
    circuit.add(0, pcvl.components.BS())
    circuit.add(2, pcvl.components.BS())

    U = circuit.compute_unitary()
    U_torch = torch.from_numpy(U)
    # print(U)

    input_state = (
        [1, 0] * n_photons
        if n_photons == m // 2
        else [1] * n_photons + (m - n_photons) * [0]
    )

    # Can't output_map_func = reverse_state:
    # Error l.513 '@jit.script' : DeprecationWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`
    # def reverse_state(state):
    #    return state[::-1]

    keys, amplitudes = compute_slos_distribution(
        unitary=U_torch,
        input_state=input_state,
        output_map_func=None,
        keep_keys=True,
        computation_space=ComputationSpace.FOCK,
    )
    print(amplitudes)

    expected_keys = [
        (2, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 0, 1, 0),
        (1, 0, 0, 1),
        (0, 2, 0, 0),
        (0, 1, 1, 0),
        (0, 1, 0, 1),
        (0, 0, 2, 0),
        (0, 0, 1, 1),
        (0, 0, 0, 2),
    ]
    assert keys == expected_keys, (
        f"Keys do not match : expected {expected_keys}, calculated {keys}"
    )

    expected_amplitudes = torch.tensor(
        [
            [
                -0.2375 + 0.1763j,
                0.2494 - 0.0177j,
                0.0325 - 0.2582j,
                -0.2582 + 0.3210j,
                -0.2625 - 0.1763j,
                0.2376 + 0.3855j,
                0.0319 - 0.2376j,
                -0.2621 - 0.1909j,
                0.2494 - 0.0177j,
                -0.2371 + 0.1617j,
            ]
        ],
        dtype=torch.complex128,
    )

    assert torch.allclose(amplitudes, expected_amplitudes, atol=1e-4), (
        f"Amplitudes do not match :\n"
        f"expected : {expected_amplitudes}\n"
        f"calculated  : {amplitudes}"
    )
